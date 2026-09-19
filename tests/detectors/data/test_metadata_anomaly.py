from __future__ import annotations

from attacklab.script_batch_attack import script_generate_batch
from cva.core.capability import Availability
from cva.detectors.data.metadata_anomaly import MetadataAnomaly
from tests.detectors.data.helpers import detect, resolve


def test_declares_registry_row():
    assert MetadataAnomaly.id == "data.metadata_anomaly"
    assert {str(c) for c in MetadataAnomaly.requires} == {"DATASET_IMAGES"}
    assert not MetadataAnomaly.optional


def test_clean_dataset_zero_findings(clean):
    assert resolve(MetadataAnomaly, clean).runnable
    assert detect(MetadataAnomaly, clean, None, None) == []


def test_script_generated_batch_is_found(clean, workdir):
    ds, m = script_generate_batch(clean, workdir / "sb", seed=3, target_contributor="A")
    fs = detect(MetadataAnomaly, ds, None, None)
    assert len(fs) == 1, [f.reason for f in fs]
    f = fs[0]
    assert (f.target_type, f.target_ref) == ("contributor", "A")
    assert f.attack_class == "script_generated_batch"
    assert f.nature.value == "quality" and f.severity.value == "medium"
    assert f.availability == Availability.OK
    for needle in ("EXIF", "quantisation", "ImageMagick", "ms", "sidecar"):
        assert needle in f.reason, needle
    assert any("forgeable" in l for l in f.limitations)
    assert f.disposition_rule == "pending_risk_engine"
    assert f.score_raw >= f.threshold


def test_exif_cluster_attribution_reads_as_hypothesis(clean, workdir):
    import dataclasses
    ds, _ = script_generate_batch(clean, workdir / "sb2", seed=4, target_contributor="B")
    ds = type(ds)([dataclasses.replace(s, contributor_source="exif_cluster") if s.contributor == "B"
                   else s for s in ds.samples], ds.categories)
    f = detect(MetadataAnomaly, ds, None, None)[0]
    assert "HYPOTHESIS" in f.reason


def test_finding_id_stable_and_scan_independent(clean, workdir):
    ds, _ = script_generate_batch(clean, workdir / "sb3", seed=3, target_contributor="A")
    a = detect(MetadataAnomaly, ds, None, None)[0]
    b = detect(MetadataAnomaly, ds, None, None)[0]
    assert a.finding_id == b.finding_id and len(a.finding_id) == 16


def test_unknown_param_in_own_namespace_is_an_error_but_other_keys_are_not(clean):
    import pytest
    with pytest.raises(ValueError, match="min_grup"):
        detect(MetadataAnomaly, clean, params={"min_grup": 5})
    # another plug-in's keys in the shared profile are none of this detector's business
    assert detect(MetadataAnomaly, clean, profile={"nc_top_k": 3, "data.near_dup": {"cosine": 0.9}}) == []


def test_scan_id_is_stamped_from_ctx_and_does_not_change_the_finding_id(clean, workdir):
    from tests.detectors.data.helpers import make_ctx
    ds, _ = script_generate_batch(clean, workdir / "sb4", seed=3, target_contributor="A")
    a = MetadataAnomaly().detect(ds, None, None, make_ctx(scan_id="scan-1"))[0]
    b = MetadataAnomaly().detect(ds, None, None, make_ctx(scan_id="scan-2"))[0]
    assert (a.scan_id, b.scan_id) == ("scan-1", "scan-2") and a.finding_id == b.finding_id


# ---- camera signals, tested on exact metadata records ------------------------------------------
def _rec(cam=None, exif=None):
    return {"exif": bool(cam) if exif is None else exif, "camera": cam, "quant": None,
            "software": "", "dims": (64, 64), "mtime": 1.0}


def _varied(n_each=20, cams=("c1", "c2", "c3", "c4")):
    return [_rec(c) for c in cams for _ in range(n_each)]


def test_single_camera_needs_files_that_actually_name_a_camera():
    """A14: 199 files with EXIF stripped and ONE that names a camera is no pattern."""
    det = MetadataAnomaly()
    from cva.detectors.data.base import Params
    from cva.detectors.data.metadata_anomaly import DEFAULTS
    det.p = Params(DEFAULTS)
    stripped_but_one = [_rec("only")] + [_rec(None, exif=False) for _ in range(199)]
    assert "single_camera" not in det._signals(stripped_but_one, _varied())
    genuinely_single = [_rec("only") for _ in range(40)]
    sig = det._signals(genuinely_single, _varied())
    assert "single_camera" in sig and "all 40 files that name a camera" in sig["single_camera"]["text"]


def test_pooled_cameras_is_implemented_and_judged_against_every_other_group():
    """A8: many cameras in ONE group while every other group has <= 1."""
    det = MetadataAnomaly()
    from cva.detectors.data.base import Params
    from cva.detectors.data.metadata_anomaly import DEFAULTS
    det.p = Params(DEFAULTS)
    pooled = _varied(10, ("a", "b", "c", "d"))
    cohort = [_rec("x") for _ in range(30)] + [_rec("y") for _ in range(30)]
    sig = det._signals(pooled, cohort, peer_cams=[{"x"}, {"y"}])
    assert "pooled_cameras" in sig and "4 different cameras" in sig["pooled_cameras"]["text"]
    # normal: peers are varied too -> a varied group is not anomalous
    assert "pooled_cameras" not in det._signals(pooled, _varied(), peer_cams=[{"a", "b"}, {"c", "d"}])
    # with a single peer there is no baseline to compare against
    assert "pooled_cameras" not in det._signals(pooled, cohort, peer_cams=[{"x"}])
    # too few files actually name a camera
    thin = [_rec("a"), _rec("b"), _rec("c")] + [_rec(None, exif=False) for _ in range(60)]
    assert "pooled_cameras" not in det._signals(thin, cohort, peer_cams=[{"x"}, {"y"}])


def test_reports_not_performed_when_no_group_is_large_enough(clean):
    """A10: an empty list would read as 'checked, nothing found'."""
    tiny = type(clean)(clean.samples[:30], clean.categories)          # 4 contributors, each < 20 files
    fs = detect(MetadataAnomaly, tiny)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
    assert "at least 20 files" in fs[0].reason
