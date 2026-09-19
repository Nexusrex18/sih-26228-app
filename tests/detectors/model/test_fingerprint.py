import numpy as np

from cva.core.finding import Severity
from cva.core.model import Manifest, ModelBattery
from cva.detectors.base import CheckContext
from cva.detectors.model.fingerprint import FingerprintCheck, fingerprint


def test_fingerprint_is_deterministic(clean_model):
    assert np.allclose(fingerprint(clean_model), fingerprint(clean_model))


def test_matching_fingerprint_accepts(clean_model, manifest_for):
    f = FingerprintCheck().check(clean_model, CheckContext(
        battery=ModelBattery(manifest=manifest_for(clean_model)), scan_id="t"))[0]
    assert f.severity is Severity.INFO


def test_divergent_fingerprint_is_hostile(clean_model, backdoored_model, manifest_for):
    f = FingerprintCheck().check(backdoored_model, CheckContext(
        battery=ModelBattery(manifest=manifest_for(clean_model)), scan_id="t"))[0]
    assert f.severity.rank >= Severity.LOW.rank
