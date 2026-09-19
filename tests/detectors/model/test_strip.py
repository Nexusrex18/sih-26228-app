from cva.core.capability import Availability, Capability
from cva.detectors.base import CheckContext
from cva.detectors.model.strip import StripCheck


def test_strip_declares_suspect_inputs_as_required():
    """STRIP is an INPUT-level detector. Measured: from clean probes alone it scores the
    model's general confidence, not a backdoor, at a 100% false-alarm rate."""
    assert Capability.SUSPECT_INPUTS in StripCheck.requires


def test_unavailable_without_suspect_inputs_and_points_elsewhere(clean_model, probes):
    x, y = probes
    f = StripCheck().check(clean_model, CheckContext(probes_x=x, probes_y=y, scan_id="t"))[0]
    assert f.availability is Availability.UNAVAILABLE
    assert "universal_margin" in f.reason
