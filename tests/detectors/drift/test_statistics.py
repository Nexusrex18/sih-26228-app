import numpy as np
import pytest

from cva.core.drift import DriftConfig
from cva.detectors.drift.statistics import (
    bh_adjust,
    compare_axis,
    psi_critical_value,
    psi_pvalue,
)


def test_psi_scaled_null_regression():
    # chi-square_9(0.95) = 16.918977604620448; N=100, M=200.
    assert psi_critical_value(.05,100,200,10) == pytest.approx(.2537846640693067)
    assert psi_pvalue(.2537846640693067,100,200,10) == pytest.approx(.05)
    assert psi_pvalue(.1,1000,1000,10) < psi_pvalue(.1,100,100,10)


def test_bh_known_values():
    assert bh_adjust([.01,.04,.03,.002]) == pytest.approx([.02,.04,.04,.008])


def test_identical_constant_and_tail_shift():
    cfg = DriftConfig()
    same = compare_axis(np.zeros(80),np.zeros(80),cfg)
    assert same['psi'] == 0 and same['psi_p'] == 1 and same['ks'] == 0
    shifted = compare_axis(np.zeros(80),np.ones(80)*100,cfg)
    assert shifted['ks'] == 1 and shifted['psi_p'] < .05


def test_sparse_permutation_is_seeded():
    cfg = DriftConfig(bins=20)
    x,y = np.arange(12),np.arange(12)+40
    a,b = compare_axis(x,y,cfg),compare_axis(x,y,cfg)
    assert a == b
    assert a['psi_method'] == 'permutation (sparse bins)'
    assert a['psi_p'] >= 1/(cfg.permutations+1)


def test_clean_null_false_alarm_smoke():
    # A loose repeated-sampling regression, not a field validation claim.
    rng = np.random.default_rng(381)
    flags = 0
    cfg = DriftConfig(bins=5)
    for _ in range(100):
        row = compare_axis(rng.normal(size=200),rng.normal(size=200),cfg)
        flags += row['psi_p'] < .05
    assert flags <= 15


@pytest.mark.parametrize('x,y', [([1,np.nan],[2,3]),([], [2,3]),([1,np.inf],[2,3])])
def test_invalid_observations_rejected(x,y):
    with pytest.raises(ValueError):
        compare_axis(x,y,DriftConfig())
