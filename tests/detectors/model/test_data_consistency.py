from cva.core.capability import Availability
from cva.detectors.base import CheckContext
from cva.detectors.model.data_consistency import DataConsistencyCheck


def test_registers_as_a_detector_not_a_modelcheck():
    """It needs the CONTRIBUTED dataset; ModelCheck.check() has no parameter for it."""
    from cva.detectors.base import DETECTOR_REGISTRY, REGISTRY
    assert "model.data_consistency" in DETECTOR_REGISTRY
    assert "model.data_consistency" not in REGISTRY


def test_degrades_without_a_trigger_candidate(clean_model, probes):
    x, _ = probes
    f = DataConsistencyCheck().detect(x, None, clean_model, CheckContext(scan_id="t"))[0]
    assert f.availability is Availability.DEGRADED
    assert "trigger candidate" in f.reason
