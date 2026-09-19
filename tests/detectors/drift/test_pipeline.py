import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attacklab.photometric_shift import generate
from cva.core.capability import Availability, Capability
from cva.core.drift_scan import scan_drift
from cva.core.interfaces import DriftTest
from cva.core.scanid import new_scan_id
from cva.core.types import Dataset, Disposition, Nature, Sample
from cva.detectors.drift.command import run
from cva.detectors.drift.registry import build_checks
from cva.loaders.drift import (
    EmbeddingRows,
    FeatureTable,
    MeasuredDataset,
    load_embeddings,
    load_image_dataset,
)
from cva.loaders.safety import UnsafeArtifact

IDS = itertools.count()


def batch(name,features):
    n = len(next(iter(features.values())))
    samples = [Sample(str(i),hashlib.sha256(f'{name}-{i}'.encode()).hexdigest(),Path(f'{i}.png'),8,8) for i in range(n)]
    return MeasuredDataset(samples=samples,measurements=FeatureTable(name,features))


def scan(ref,inc,**kwargs):
    return scan_drift(ref,inc,checks=kwargs.pop('checks',build_checks()),
                       scan_id=new_scan_id(counter=next(IDS)),**kwargs)


def test_photometric_end_to_end_and_schema(tmp_path):
    import jsonschema
    generate(tmp_path/'corpus',n=60)
    result = run(tmp_path/'corpus/incoming',tmp_path/'corpus/reference',tmp_path/'report')
    f = next(f for f in result.findings if f.detector_id == 'drift.interpretable_axes')
    assert 'brightness' in f.reason and f.severity.value == 'medium'
    assert f.nature == Nature.QUALITY and f.disposition == Disposition.REVIEW
    report_dir = Path(result.report_dir)
    report = json.loads((report_dir/'drift.report.json').read_text())
    schema = json.loads((Path(__file__).resolve().parents[3]/'schemas/report.schema.json').read_text())
    jsonschema.Draft202012Validator(schema,format_checker=jsonschema.FormatChecker()).validate(report)
    assert report['drift_summary']['reference_n'] == 60
    assert 'distribution_shift' in report['coverage']['assessed']
    assert report['coverage']['not_assessed']['semantic_shift'] == ['drift.semantic_axis']
    assert report['coverage']['not_assessed']['suspicious_manipulation'] == ['drift.vs_manipulation']
    assert report['coverage']['counts_only_kind'] == 'attack'
    assert 'MODEL_' not in json.dumps(report['access_assumptions'])
    assert 'MODEL_' not in (report_dir/'drift.report.html').read_text()
    assert 'semantic_shift' in (report_dir/'drift.coverage.md').read_text()


def test_no_reference_small_and_overlapping_batches_are_not_clean():
    a,b = batch('a',{'brightness':np.arange(2)}),batch('b',{'brightness':np.arange(2)})
    for ref in (None,a,b):
        r = scan(ref,b)
        assert r.verdict == 'REVIEW'
        assert all(f.availability == Availability.UNAVAILABLE for f in r.findings)
        assert all(f.disposition_rule for f in r.findings)
    a = batch('same',{'brightness':np.arange(80)})
    r = scan(a,a)
    assert all('share image content' in f.reason for f in r.findings)


def test_embedding_version_and_known_shift():
    x = np.random.default_rng(10).normal(size=(120,4))
    a,b = batch('a',{'brightness':np.arange(120)}),batch('b',{'brightness':np.arange(120)})
    ids = tuple(s.sample_id for s in a.samples)
    ref = EmbeddingRows(ids,x,'test','1')
    inc = EmbeddingRows(ids,x+3,'test','1')
    r = scan(a,b,reference_dist=ref,incoming_dist=inc)
    f = next(f for f in r.findings if f.detector_id == 'drift.distribution')
    assert f.availability == Availability.OK and f.disposition == Disposition.REVIEW
    assert f.evidence[0].data['centroid']['euclidean'] == pytest.approx(6.)
    bad = EmbeddingRows(ids,x,'test','2')
    r = scan(a,b,reference_dist=ref,incoming_dist=bad)
    assert r.plan[0].resolution.state == Availability.UNAVAILABLE


def test_loader_measurements_and_bad_alignment(tmp_path):
    for i in range(2):
        Image.new('RGB',(16,16),(128,128,128)).save(tmp_path/f'{i}.png')
    b = load_image_dataset(tmp_path)
    assert isinstance(b,Dataset)
    assert b.measurements.features['brightness'] == pytest.approx([128/255]*2)
    assert b.measurements.features['rms_contrast'] == pytest.approx([0,0],abs=1e-12)
    np.savez(tmp_path/'bad.npz',sample_ids=['1.png','0.png'],embeddings=np.zeros((2,3)),
             extractor_id='e',extractor_version='1')
    with pytest.raises(ValueError,match='sample_ids'):
        load_embeddings(tmp_path/'bad.npz',b)


def test_corrupt_input_not_silently_skipped(tmp_path):
    (tmp_path/'broken.png').write_bytes(b'broken')
    with pytest.raises(UnsafeArtifact,match='S7'):
        load_image_dataset(tmp_path)


def test_generator_reproducibility(tmp_path):
    assert generate(tmp_path/'one',n=4) == generate(tmp_path/'two',n=4)
    with pytest.raises(ValueError,match='empty'):
        generate(tmp_path/'one',n=4)


def test_contract_capability_resolution_and_error():
    checks = build_checks()
    assert all(isinstance(c,DriftTest) for c in checks)
    a,b = batch('a',{'brightness':np.arange(20)}),batch('b',{'brightness':np.arange(20)})
    class NeedLabels:
        id = 'drift.test'
        version = '1'
        requires = frozenset({Capability.DATASET_LABELS})
        optional = frozenset()
        attack_classes = frozenset({'distribution_shift'})
        def assess(self,*args):
            raise AssertionError('Missing capability was ignored')
    r = scan(a,b,checks=[NeedLabels()])
    assert r.findings[0].availability == Availability.UNAVAILABLE
    assert Capability.DATASET_LABELS in r.plan[0].resolution.missing
    class Broken(NeedLabels):
        requires = frozenset({Capability.DATASET_IMAGES})
        def assess(self,*args):
            raise RuntimeError('injected defect')
    r = scan(a,b,checks=[Broken()])
    assert r.verdict == 'REVIEW' and r.findings[0].availability == Availability.ERROR
    assert r.plan[0].resolution.state == Availability.ERROR


def test_invalid_embeddings_rejected():
    with pytest.raises(ValueError,match='finite'):
        EmbeddingRows(('1',),np.array([[np.nan]]),'x','1')


@pytest.mark.parametrize('seed',[0,7,42])
def test_clean_independent_batches_and_offline_execution(tmp_path,monkeypatch,seed):
    import socket
    def no_network(*args,**kwargs):
        raise AssertionError('Network access attempted')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    generate(tmp_path/'clean',n=80,brightness=0,noise=0,seed=seed)
    result = run(tmp_path/'clean/incoming',tmp_path/'clean/reference',tmp_path/'report')
    f = next(f for f in result.findings if f.detector_id == 'drift.interpretable_axes')
    assert f.severity.value == 'info'
    assert f.disposition == Disposition.ACCEPT
    assert result.verdict == 'REVIEW'


def test_distinct_scan_ids_stable_finding_ids_and_selftest(tmp_path):
    generate(tmp_path/'corpus',n=20)
    args = [tmp_path/'corpus/incoming',tmp_path/'corpus/reference']
    a = run(*args,tmp_path/'reports')
    b = run(*args,tmp_path/'reports')
    assert a.scan_id != b.scan_id and a.report_dir != b.report_dir
    assert [f.finding_id for f in a.findings] == [f.finding_id for f in b.findings]
    c = run(*args,tmp_path/'selftest1',selftest=True)
    d = run(*args,tmp_path/'selftest2',selftest=True)
    assert c.scan_id == d.scan_id
    assert (Path(c.report_dir)/'drift.report.json').read_bytes() == (Path(d.report_dir)/'drift.report.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(*args,tmp_path/'selftest1',selftest=True)


def test_missing_feature_marks_partial_coverage():
    values = np.linspace(0,1,30)
    a = batch('a',{'brightness':values,'exif_iso':values+100})
    b = batch('b',{'brightness':values})
    f = next(f for f in scan(a,b).findings if f.detector_id == 'drift.interpretable_axes')
    assert f.availability == Availability.DEGRADED
    assert any('exif_iso' in s for s in f.limitations)


def test_shared_model_renderer_preserves_current_backend_capabilities(tmp_path):
    from cva.core.capability import CapabilitySet
    from cva.core.orchestrator import ScanResult
    from cva.report.render_html import render
    from cva.report.report_json import build
    result = ScanResult('scan','model','onnx',CapabilitySet(),[],[],{},'ACCEPT')
    render(result,tmp_path/'model.html')
    # Current modules intentionally shows every capability for model scans. Drift's
    # scoped rows must not alter that established model behavior.
    assert all(c.value in (tmp_path/'model.html').read_text() for c in Capability)
    assert {r['capability'] for r in build(result)['access_assumptions']['capabilities_absent']} == {c.value for c in Capability}


def test_partial_overlap_detects_renamed_images():
    a,b = batch('a',{'brightness':np.arange(30)}),batch('b',{'brightness':np.arange(30)})
    b.samples[0].content_sha256 = a.samples[10].content_sha256
    assert all(f.availability == Availability.UNAVAILABLE for f in scan(a,b).findings)


def test_existing_canonical_dataset_adapter(tmp_path):
    from cva.loaders.datasets.base import InMemoryDataset
    from cva.loaders.drift import measure_dataset
    Image.new('RGB',(8,8),(64,64,64)).save(tmp_path/'a.png')
    original = load_image_dataset(tmp_path)
    canonical = InMemoryDataset(samples=original.samples,root=tmp_path)
    table = measure_dataset(canonical)
    assert table.features['brightness'][0] == pytest.approx(64/255)


def test_detectors_leave_policy_to_shared_risk_engine(monkeypatch):
    from cva.risk import drift as risk
    a,b = batch('a',{'brightness':np.arange(30)}),batch('b',{'brightness':np.arange(30)+100})
    check = build_checks()[1]
    raw = check.assess(a,b,None,None)
    assert raw[0].disposition_rule == ''  # only the dataclass default, no detector policy
    called = []
    real = risk.assess
    def capture(*args,**kwargs):
        called.append(True)
        return real(*args,**kwargs)
    monkeypatch.setattr(risk,'assess',capture)
    result = scan(a,b)
    assert called and all(f.disposition_rule for f in result.findings)
    assert next(f for f in result.findings if f.detector_id == check.id).disposition_rule == 'DRIFT_MATERIAL'


def test_scan_output_allocation_is_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from cva.detectors.drift.command import reserve_output
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _:reserve_output(tmp_path,False,0),range(20)))
    assert len({row[0] for row in rows}) == 20
    assert all(row[1].is_dir() for row in rows)
