from __future__ import annotations

from attacklab.script_batch_attack import script_generate_batch
from cva.core.capability import Availability
from cva.detectors.data.metadata_anomaly import MetadataAnomaly
from conftest import resolve


def test_declares_registry_row():
    assert MetadataAnomaly.id == "data.metadata_anomaly"
    assert {str(c) for c in MetadataAnomaly.requires} == {"DATASET_IMAGES"}
    assert not MetadataAnomaly.optional


def test_clean_dataset_zero_findings(clean):
    assert resolve(MetadataAnomaly, clean).runnable
    assert MetadataAnomaly().detect(clean, None, None) == []


def test_script_generated_batch_is_found(clean, workdir):
    ds, m = script_generate_batch(clean, workdir / "sb", seed=3, target_contributor="A")
    fs = MetadataAnomaly().detect(ds, None, None)
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
    f = MetadataAnomaly().detect(ds, None, None)[0]
    assert "HYPOTHESIS" in f.reason


def test_finding_id_stable_and_scan_independent(clean, workdir):
    ds, _ = script_generate_batch(clean, workdir / "sb3", seed=3, target_contributor="A")
    a = MetadataAnomaly().detect(ds, None, None)[0]
    b = MetadataAnomaly().detect(ds, None, None)[0]
    assert a.finding_id == b.finding_id and len(a.finding_id) == 16


def test_unknown_param_is_an_error():
    import pytest
    with pytest.raises(ValueError):
        MetadataAnomaly({"min_grup": 5})
