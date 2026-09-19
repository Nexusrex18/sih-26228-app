import numpy as np
import pytest

from cva.core.capability import Availability, Capability
from cva.detectors.base import CheckContext
from cva.detectors.model.neural_cleanse import NeuralCleanseCheck, mad_anomaly_index


def test_mad_flags_only_small_side_outliers():
    """A LARGE trigger is not suspicious; a small one is. The index must be one-sided."""
    ai = mad_anomaly_index(np.array([50.0, 52.0, 48.0, 5.0, 51.0]))
    assert ai[3] > 2.0                      # the small outlier is flagged
    ai2 = mad_anomaly_index(np.array([50.0, 52.0, 48.0, 500.0, 51.0]))
    assert ai2[3] < 0                       # the large one is never flagged


def test_tight_cluster_cannot_manufacture_an_outlier():
    """Measured: without a relative floor, [50,52,48,500,51] gave class 2 an index of 2.02
    purely from a 3-unit gap, because MAD collapses on a tight cluster."""
    ai = mad_anomaly_index(np.array([50.0, 52.0, 48.0, 500.0, 51.0]))
    assert ai.max() < 2.0


def test_identical_norms_produce_no_outlier():
    ai = mad_anomaly_index(np.array([40.0, 40.0, 40.0, 40.0]))
    assert ai.max() < 2.0


def test_gradients_are_optional_not_required():
    """On ONNX it must DEGRADE to the gradient-free path, never resolve UNAVAILABLE."""
    assert Capability.MODEL_GRADIENTS in NeuralCleanseCheck.optional
    assert Capability.MODEL_GRADIENTS not in NeuralCleanseCheck.requires


def test_degrades_without_gradients_and_states_the_penalty(backdoored_model, probes):
    x, y = probes
    f = NeuralCleanseCheck().check(backdoored_model, CheckContext(
        probes_x=x, probes_y=y, profile={"nes_steps": 3}, scan_id="t"))[0]
    assert f.availability is Availability.DEGRADED
    assert "NO GRADIENTS" in " ".join(f.access_assumptions)
    assert any("confidence capped" in l.lower() for l in f.limitations)


@pytest.mark.slow
def test_gtsrb_feasibility_spike():
    """§2's mandatory spike: Neural Cleanse wall-clock per class, extrapolated to the demo
    class count. Slow/manual by design — not CI-gated, but it must exist and be runnable."""
    pytest.skip("run manually: attacklab/scenarios/backdoor_gtsrb_spike.yaml")
