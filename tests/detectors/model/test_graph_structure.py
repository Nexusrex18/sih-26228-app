from cva.core.capability import Availability
from cva.detectors.base import CheckContext
from cva.detectors.model.graph_structure import GraphStructureCheck


def test_degrades_on_non_onnx_rather_than_crashing(clean_model):
    f = GraphStructureCheck().check(clean_model, CheckContext(scan_id="t"))[0]
    assert f.availability is Availability.DEGRADED
    assert "onnx" in f.reason.lower()
