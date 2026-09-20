import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attacklab.photometric_shift import generate, inject
from cva.core.capability import Availability, Capability
from cva.core.orchestrator import profile_hash_of
from cva.core.types import ExclusionReason
from cva.detectors.drift.command import main, run
from cva.detectors.drift.config import DriftConfig, load_profile
from cva.detectors.drift.registry import build_checks
from cva.loaders.drift import EmbeddingRows, load_image_dataset
from tests.detectors.drift.test_pipeline import batch, scan


def test_complete_clean_scan_accepts_and_deferred_gaps_do_not_change_it():
    x = np.linspace(0, 1, 80)
    a,b = batch('a', {'brightness':x}),batch('b', {'brightness':x})
    ids = tuple(s.sample_id for s in a.samples)
    rows = EmbeddingRows(ids, x[:,None], 'independent-test', '1')
    for checks in (build_checks()[:2], build_checks()):
        r = scan(a,b,checks=checks,reference_dist=rows,incoming_dist=rows)
        assert r.verdict == 'ACCEPT'
        assert all(f.availability == Availability.OK for f in r.findings[:2])
        assert r.findings[0].disposition_rule == 'D7'
        assert any('MMD and energy distance' in line for line in r.findings[0].limitations)


def test_exclusion_reasons_distinguish_access_from_data_and_implementation():
    x = np.arange(40)
    a,b = batch('a',{'brightness':x}),batch('b',{'brightness':x})
    r = scan(None,b)
    assert r.findings[0].exclusion_reason == ExclusionReason.CAPABILITY
    assert build_checks()[1].requires == {Capability.DATASET_IMAGES}
    r = scan(a,b)
    assert all(f.exclusion_reason is None for f in r.findings)
    assert all(row.resolution.exclusion_reason is None for row in r.plan)
    profile = load_profile()
    profile['disabled_checks'] = ['drift.interpretable_axes']
    r = scan(a,b,profile=profile)
    assert r.findings[1].exclusion_reason == ExclusionReason.BUDGET


def test_profile_values_schema_hash_and_cli_precedence(tmp_path):
    profile = load_profile()
    profile['psi'].update(n_bins=4,alpha=.02,min_reference_n=12,min_incoming_n=9,
                          min_samples=8,min_ks_effect=.3,permutations=299,seed=19,
                          axis_thresholds={'brightness':.4})
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(profile))
    resolved = load_profile(path,overrides={'alpha':.04})
    cfg = DriftConfig.from_profile(resolved)
    assert (cfg.bins,cfg.alpha,cfg.seed,cfg.permutations,cfg.min_ks_effect) == (4,.04,19,299,.3)
    assert cfg.axis_thresholds == {'brightness':.4}
    assert resolved['psi']['min_reference_n'] == 12
    assert resolved['psi']['min_incoming_n'] == 9
    altered = copy.deepcopy(resolved)
    altered['disposition']['rules'][-1]['description'] = 'policy changed'
    assert profile_hash_of(altered) != profile_hash_of(resolved)
    for block in ({'bad_typo':1},{'permutations':3},{'axis_thresholds':{'brightness':2}}):
        bad = copy.deepcopy(profile)
        bad['psi'].update(block)
        with pytest.raises(ValueError):
            load_profile(bad)


def test_profile_sample_floors_and_per_axis_effect_gate():
    x = np.linspace(0,1,30)
    a,b = batch('a',{'brightness':x}),batch('b',{'brightness':x+.5})
    profile = load_profile()
    profile['psi'].update(min_reference_n=31,min_incoming_n=2)
    assert scan(a,b,profile=profile).findings[1].availability == Availability.UNAVAILABLE
    from cva.detectors.drift.common import compare_features
    profile['psi']['axis_thresholds'] = {'brightness':1.}
    f = compare_features(a.measurements,b.measurements,DriftConfig.from_profile(profile),
                         'drift.interpretable_axes',a.measurements.features,b.measurements.features)[0]
    assert not f.evidence[0].data['axes']['brightness']['material_shift']


def test_failed_scan_and_report_remove_only_reserved_directory(tmp_path,monkeypatch):
    from cva.detectors.drift import command
    generate(tmp_path/'data',n=20)
    output = tmp_path/'out'
    sentinel = output/'existing'
    sentinel.mkdir(parents=True)
    (sentinel/'keep').write_text('previous scan')
    def fail(*args,**kwargs):
        raise RuntimeError('injected failure')
    for function in ('scan_drift','render'):
        with monkeypatch.context() as patch:
            patch.setattr(command,function,fail)
            with pytest.raises(RuntimeError,match='injected failure'):
                run(tmp_path/'data/incoming',tmp_path/'data/reference',output)
        assert not list(output.glob('s-*'))
        assert (sentinel/'keep').read_text() == 'previous scan'


def test_cli_headline_reports_gaps_and_profile_is_replayable(tmp_path,capsys,monkeypatch):
    generate(tmp_path/'data',n=25,brightness=0,noise=0)
    args = ['--incoming',str(tmp_path/'data/incoming'),'--reference',str(tmp_path/'data/reference'),
            '--out',str(tmp_path/'out'),'--bins','4']
    main(args)
    headline = capsys.readouterr().out
    assert 'checks assessed' in headline and 'coverage gaps/errors' in headline
    directory = next((tmp_path/'out').glob('s-*'))
    effective = json.loads((directory/'effective.profile.json').read_text())
    assert effective['psi']['n_bins'] == 4
    report = json.loads((directory/'drift.report.json').read_text())
    assert report['produced_by']['profile_hash'] == profile_hash_of(effective)
    monkeypatch.chdir(directory)
    replay = run(tmp_path/'data/incoming',tmp_path/'data/reference',tmp_path/'replay',profile='effective.profile.json')
    assert replay.profile_hash == report['produced_by']['profile_hash']


def test_unsupported_files_are_counted(tmp_path):
    Image.new('RGB',(4,4),(30,40,50)).save(tmp_path/'image.png')
    Image.new('RGB',(4,4)).save(tmp_path/'ignored.gif')
    (tmp_path/'note.txt').write_text('not an image')
    dataset = load_image_dataset(tmp_path)
    assert len(dataset.samples) == 1
    assert any('2 files with unsupported suffixes' in s for s in dataset.measurements.limitations)


def test_hand_computed_axes_and_metadata(tmp_path):
    pixels = np.zeros((3,3,3),dtype='uint8')
    pixels[1,1] = 255
    exif = Image.Exif()
    exif[34855] = 200
    Image.fromarray(pixels).save(tmp_path/'impulse.png',exif=exif)
    features = load_image_dataset(tmp_path).measurements.features
    assert features['brightness'][0] == pytest.approx(1/9)
    assert features['rms_contrast'][0] == pytest.approx(np.sqrt(8)/9)
    # Laplacian: four +1 neighbours, -4 centre, four zeros; mean zero.
    assert features['sharpness'][0] == pytest.approx(20/9)
    assert features['noise_residual'][0] == 0
    for channel in ('red','green','blue'):
        assert features[f'{channel}_hist_0'][0] == pytest.approx(8/9)
        assert features[f'{channel}_hist_7'][0] == pytest.approx(1/9)
    assert features['exif_iso'][0] == 200
    jpeg = tmp_path/'jpeg'
    jpeg.mkdir()
    Image.new('RGB',(8,8),(128,128,128)).save(jpeg/'tables.jpg',qtables=[[10]*64,[20]*64],subsampling=0)
    assert load_image_dataset(jpeg).measurements.features['jpeg_quantization_mean'][0] == 15


def test_injector_fraction_reproducibility_and_numeric_truth(tmp_path,clean_photo_corpus):
    one = inject(clean_photo_corpus,tmp_path/'one',fraction=.6,brightness=.12,noise=0,seed=3)
    two = inject(clean_photo_corpus,tmp_path/'two',fraction=.6,brightness=.12,noise=0,seed=3)
    assert one == two
    assert one['transformed_n'] == 36
    records = one['files']
    assert len({r['source_sha256'] for r in records}) == 120
    deltas = [r['brightness_after']-r['brightness_before'] for r in records if r['transformed']]
    assert deltas == pytest.approx([.12]*36,abs=1/255)
    assert all(r['source_sha256']==r['sha256'] for r in records if not r['transformed'])
    result = run(tmp_path/'one/incoming',tmp_path/'one/reference',tmp_path/'report')
    axes = result.findings[1].evidence[0].data['axes']
    row = axes['brightness']
    ref_mean = np.mean([r['brightness_after'] for r in records if r['output'].startswith('reference/')])
    inc_mean = np.mean([r['brightness_after'] for r in records if r['output'].startswith('incoming/')])
    assert row['reference_mean'] == pytest.approx(ref_mean)
    assert row['incoming_mean'] == pytest.approx(inc_mean)
    assert row['incoming_mean']-row['reference_mean'] == pytest.approx(.6*.12,abs=.005)
    assert row['material_shift'] and .5 <= row['ks'] <= .8 and row['psi'] > 0
    assert row['ks_q'] < .05 and row['psi_q'] < .05
    assert 'MMD and energy distance' in (Path(result.report_dir)/'drift.coverage.md').read_text()


@pytest.mark.parametrize('options,axis,direction',[
    ({'contrast':1.7},'rms_contrast',1),
    ({'noise':.07},'noise_residual',1),
    ({'jpeg_quality':15},'jpeg_quantization_mean',1),
])
def test_injector_axes_against_known_transforms(tmp_path,clean_photo_corpus,options,axis,direction):
    inject(clean_photo_corpus,tmp_path/'injected',fraction=1,**options)
    result = run(tmp_path/'injected/incoming',tmp_path/'injected/reference',tmp_path/'report')
    rows = result.findings[1].evidence[0].data['axes']
    assert rows[axis]['material_shift']
    assert direction*(rows[axis]['incoming_mean']-rows[axis]['reference_mean']) > 0
    if 'contrast' in options:
        assert rows[axis]['incoming_mean']/rows[axis]['reference_mean'] == pytest.approx(1.7,abs=.05)
    if 'noise' in options:
        assert rows['sharpness']['incoming_mean'] > rows['sharpness']['reference_mean']
    assert not rows['exif_iso']['material_shift']
