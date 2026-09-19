import json

import numpy as np
import pytest
from PIL import Image

from attacklab.photometric_shift import generate
from cva.core.capability import Availability
from cva.core.drift import DriftBatch
from cva.core.drift_scan import scan_drift
from cva.core.registry import DRIFT_REGISTRY
from cva.core.types import Disposition, Nature
from cva.drift_cli import run
from cva.loaders.drift import load_image_batch


def test_photometric_end_to_end_and_reports(tmp_path):
    generate(tmp_path/'corpus',n=60)
    result = run(tmp_path/'corpus/incoming',tmp_path/'corpus/reference',tmp_path/'report')
    f = next(f for f in result.findings if f.detector_id == 'drift.interpretable_axes')
    assert 'brightness' in f.reason and f.severity.value == 'medium'
    assert f.nature == Nature.QUALITY and f.disposition == Disposition.REVIEW
    report = json.loads((tmp_path/'report/drift.report.json').read_text())
    assert report['reference']['n'] == 60 and report['dataset']['format'] == 'dataset'
    assert 'drift.distribution' in report['coverage']['not_assessed']
    assert 'brightness' in (tmp_path/'report/drift.report.html').read_text()
    assert (tmp_path/'report/drift.coverage.md').exists()


def test_no_reference_and_small_batches_are_not_clean_passes():
    b = DriftBatch('b',('1','2'),{'brightness':np.array([.1,.2])})
    for ref in (None,b):
        r = scan_drift(ref,b)
        assert r.verdict == 'REVIEW'
        assert all(f.availability == Availability.UNAVAILABLE for f in r.findings)


def test_embedding_version_and_known_shift():
    rng = np.random.default_rng(10)
    ids = tuple(map(str,range(120)))
    x = rng.normal(size=(120,4))
    a = DriftBatch('a',ids,embeddings=x,extractor_id='test',extractor_version='1')
    b = DriftBatch('b',ids,embeddings=x+3,extractor_id='test',extractor_version='1')
    r = scan_drift(a,b)
    f = next(f for f in r.findings if f.detector_id == 'drift.distribution')
    assert f.availability == Availability.OK and f.disposition == Disposition.REVIEW
    assert f.evidence[0].data['centroid']['euclidean'] == pytest.approx(6.)
    bad = DriftBatch('c',ids,embeddings=x,extractor_id='test',extractor_version='2')
    r = scan_drift(a,bad)
    row = next(r for r in r.plan if r.check_id == 'drift.distribution')
    assert row.resolution.state == Availability.UNAVAILABLE


def test_identical_features_have_no_shift():
    rng = np.random.default_rng(33)
    b = DriftBatch('same',tuple(map(str,range(80))),{'brightness':rng.random(80)})
    r = scan_drift(b,b)
    f = next(f for f in r.findings if f.detector_id == 'drift.interpretable_axes')
    assert f.score_raw == 0 and f.severity.value == 'info'


def test_loader_measurements_and_bad_alignment(tmp_path):
    for i in range(2):
        Image.new('RGB',(16,16),(128,128,128)).save(tmp_path/f'{i}.png')
    b = load_image_batch(tmp_path)
    assert b.features['brightness'] == pytest.approx([128/255]*2)
    assert b.features['rms_contrast'] == pytest.approx([0,0],abs=1e-12)
    assert b.features['sharpness'] == pytest.approx([0,0])
    np.savez(tmp_path/'bad.npz',sample_ids=['1.png','0.png'],embeddings=np.zeros((2,3)),
             extractor_id='e',extractor_version='1')
    with pytest.raises(ValueError,match='sample_ids'):
        load_image_batch(tmp_path,tmp_path/'bad.npz')


def test_corrupt_input_not_silently_skipped(tmp_path):
    (tmp_path/'broken.png').write_bytes(b'broken')
    with pytest.raises(ValueError,match='Unreadable'):
        load_image_batch(tmp_path)


def test_generator_reproducibility(tmp_path):
    assert generate(tmp_path/'one',n=4) == generate(tmp_path/'two',n=4)
    with pytest.raises(ValueError,match='empty'):
        generate(tmp_path/'one',n=4)


def test_plugin_exception_is_error_and_review(monkeypatch):
    class Broken:
        def assess(self,*args):
            raise RuntimeError('injected defect')
    monkeypatch.setitem(DRIFT_REGISTRY,'drift.broken',Broken)
    b = DriftBatch('b',tuple(map(str,range(20))),{'brightness':np.zeros(20)})
    r = scan_drift(b,b)
    assert r.verdict == 'REVIEW'
    f = next(f for f in r.findings if f.detector_id == 'drift.broken')
    assert f.availability == Availability.ERROR
    assert next(p for p in r.plan if p.check_id == 'drift.broken').resolution.state == Availability.ERROR


def test_invalid_embeddings_rejected():
    with pytest.raises(ValueError,match='finite'):
        DriftBatch('a',('1',),embeddings=np.array([[np.nan]]),extractor_id='x',extractor_version='1')


def test_clean_independent_batches_and_offline_execution(tmp_path, monkeypatch):
    import socket
    def no_network(*args, **kwargs):
        raise AssertionError('Network access attempted')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    generate(tmp_path/'clean',n=80,brightness=0,noise=0)
    result = run(tmp_path/'clean/incoming',tmp_path/'clean/reference',tmp_path/'report')
    f = next(f for f in result.findings if f.detector_id == 'drift.interpretable_axes')
    assert f.severity.value == 'info'
    assert f.disposition == Disposition.ACCEPT
    # Whole scan remains REVIEW because other checks were not assessed.
    assert result.verdict == 'REVIEW'


def test_valid_embedding_cache_and_repeated_scan_ids(tmp_path):
    for i in range(20):
        Image.new('RGB',(8,8),(i,i,i)).save(tmp_path/f'{i:02}.png')
    np.savez(tmp_path/'cache.npz',sample_ids=[f'{i:02}.png' for i in range(20)],
             embeddings=np.arange(60).reshape(20,3),extractor_id='example',extractor_version='v1')
    batch = load_image_batch(tmp_path,tmp_path/'cache.npz')
    a,b = scan_drift(batch,batch),scan_drift(batch,batch)
    assert a.scan_id == b.scan_id
    assert [f.to_dict() for f in a.findings] == [f.to_dict() for f in b.findings]


def test_missing_feature_marks_partial_coverage():
    values = np.linspace(0,1,30)
    ids = tuple(map(str,range(30)))
    a = DriftBatch('a',ids,{'brightness':values,'exif_iso':values+100})
    b = DriftBatch('b',ids,{'brightness':values})
    f = next(f for f in scan_drift(a,b).findings if f.detector_id == 'drift.interpretable_axes')
    assert f.availability == Availability.DEGRADED
    assert any('exif_iso' in s for s in f.limitations)
