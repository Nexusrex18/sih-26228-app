from cva.core.types import Disposition
from cva.detectors.base import CheckContext
from cva.detectors.model.anomalous import AnomalousBehaviourCheck

from .conftest import SyntheticModel


def test_collapsed_model_is_flagged(probes):
    """PS 2.2.2's third verdict: neither substituted nor backdoored, just wrong."""
    class Collapsed(SyntheticModel):
        def logits(self, x):
            import numpy as np
            z = np.full((len(self._n(x)), self._k), -10.0)
            z[:, 1] = 10.0
            return z
    x, y = probes
    f = AnomalousBehaviourCheck().check(
        Collapsed("collapsed"), CheckContext(probes_x=x, probes_y=y, scan_id="t"))[0]
    assert f.disposition is Disposition.REVIEW
    assert f.attack_class == "model_anomalous"
