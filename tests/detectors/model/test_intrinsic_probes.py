from cva.core.finding import Disposition
from cva.detectors.base import CheckContext
from cva.detectors.model.intrinsic_probes import IntrinsicProbeCheck


def test_produces_a_ranking_for_neural_cleanse_topk(clean_model, probes):
    x, y = probes
    f = IntrinsicProbeCheck().check(clean_model, CheckContext(
        probes_x=x, probes_y=y, profile={"noise_probes": 32}, scan_id="t"))[0]
    assert f.produced_by.startswith("ranking=")


def test_never_quarantines_on_its_own(clean_model, backdoored_model, probes):
    """A prior, not a verdict — disposition is capped at review by design."""
    x, y = probes
    for m in (clean_model, backdoored_model):
        f = IntrinsicProbeCheck().check(m, CheckContext(
            probes_x=x, probes_y=y, profile={"noise_probes": 32}, scan_id="t"))[0]
        assert f.disposition is not Disposition.QUARANTINE
