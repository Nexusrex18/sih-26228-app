from cva.core.capability import Availability, Capability
from cva.core.model import ModelBattery
from cva.detectors.base import CheckContext
from cva.detectors.model.weight_stats import WeightStatisticsCheck

from .conftest import SyntheticModel


def test_small_battery_degrades_rather_than_reporting_a_meaningless_z(clean_model, probes):
    """n=2 makes the sample std a terrible estimator; measured 170 sigma on a clean layer."""
    x, y = probes
    small = ModelBattery(models=[SyntheticModel("r0"), SyntheticModel("r1")])
    fs = WeightStatisticsCheck().check(clean_model, CheckContext(
        probes_x=x, probes_y=y, battery=small, scan_id="t"))
    weight_half = fs[0]
    assert weight_half.availability is Availability.DEGRADED
    assert "at least" in weight_half.reason


def test_emits_both_halves_of_ps_2_2_2(clean_model, probes, battery):
    """PS names 'parameter OR activation statistics'; one check, two findings."""
    x, y = probes
    m = SyntheticModel("c", caps={Capability.MODEL_PREDICT, Capability.MODEL_WEIGHTS,
                                  Capability.MODEL_ACTIVATIONS})
    fs = WeightStatisticsCheck().check(m, CheckContext(
        probes_x=x, probes_y=y, battery=battery, scan_id="t"))
    assert {f.attack_class for f in fs} == {"weight_anomaly", "activation_anomaly"}


def test_activation_half_flags_dead_units(probes, battery):
    x, y = probes
    dead = SyntheticModel("dead", dead_frac=0.9,
                          caps={Capability.MODEL_PREDICT, Capability.MODEL_WEIGHTS,
                                Capability.MODEL_ACTIVATIONS})
    fs = WeightStatisticsCheck().check(dead, CheckContext(
        probes_x=x, probes_y=y, battery=battery, scan_id="t"))
    act = [f for f in fs if f.attack_class == "activation_anomaly"][0]
    assert act.score_raw > 0.35
