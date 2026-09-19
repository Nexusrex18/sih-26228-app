from cva.core.capability import Availability
from cva.core.finding import Disposition, Severity
from cva.core.model import Manifest, ModelBattery
from cva.detectors.base import CheckContext
from cva.detectors.model.weight_digest import WeightDigestCheck


def _ctx(**kw):
    return CheckContext(scan_id="t", **kw)


def test_matching_digest_accepts(clean_model, manifest_for):
    f = WeightDigestCheck().check(
        clean_model, _ctx(battery=ModelBattery(manifest=manifest_for(clean_model))))[0]
    assert f.disposition is Disposition.ACCEPT
    assert f.attack_class == "model_substitution"


def test_mismatched_digest_quarantines_and_names_the_benign_explanation(clean_model):
    bad = Manifest(model_id="clean", weights_sha256="0" * 64)
    f = WeightDigestCheck().check(clean_model, _ctx(battery=ModelBattery(manifest=bad)))[0]
    assert f.disposition is Disposition.QUARANTINE
    assert f.severity is Severity.CRITICAL
    # a hash cannot distinguish malice from a re-export; the finding must say so
    assert any("benign" in l.lower() or "quantis" in l.lower() for l in f.limitations)


def test_no_manifest_degrades_and_offers_to_emit_one(clean_model):
    f = WeightDigestCheck().check(clean_model, _ctx(battery=ModelBattery()))[0]
    assert f.availability is Availability.DEGRADED
    assert "manifest" in f.reason.lower()
