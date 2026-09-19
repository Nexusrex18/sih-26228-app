"""Property-based tests for the reporting statistics.

These are invariants, so they get properties rather than examples: the interesting cases
are the ones nobody thinks to write. Every failure found here would otherwise have shown up
as a confidently-wrong number in a report.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cva.bench.stats import (BOOTSTRAP_MIN_N, bootstrap_rate, mcnemar, required_n,
                             separation, wilson)


@given(n=st.integers(1, 500), f=st.floats(0, 1))
def test_wilson_contains_the_point_estimate_and_stays_in_unit_range(n, f):
    k = int(round(f * n))
    r = wilson(k, n)
    assert 0.0 <= r.lo <= r.point <= r.hi <= 1.0


@given(n=st.integers(1, 200))
def test_wilson_is_never_degenerate_at_the_extremes(n):
    """Wald returns zero width at k=0 and k=n — certainty from no evidence."""
    assert wilson(0, n).hi > 0.0
    assert wilson(n, n).lo < 1.0


@given(k=st.integers(0, 50), extra=st.integers(1, 200))
def test_more_evidence_never_widens_the_interval(k, extra):
    """Same proportion, more observations, must not become less certain."""
    n1 = k + extra
    n2 = n1 * 4
    w1 = wilson(k, n1)
    w2 = wilson(int(round(w1.point * n2)), n2)
    assert (w2.hi - w2.lo) <= (w1.hi - w1.lo) + 1e-9


@given(hits=st.lists(st.booleans(), min_size=1, max_size=8))
def test_small_n_never_uses_the_bootstrap(hits):
    """The guard that stopped [1.00, 1.00] from two observations."""
    r = bootstrap_rate(np.array(hits))
    assert "wilson" in r.method
    assert not (r.lo == r.hi == 1.0) or len(hits) == 0


@given(p0=st.floats(0.05, 0.45), gap=st.floats(0.1, 0.5))
def test_required_n_grows_as_the_effect_shrinks(p0, gap):
    p1 = min(0.95, p0 + gap)
    smaller_gap = min(0.95, p0 + gap / 2)
    if smaller_gap <= p0 + 1e-6:
        pytest.skip("degenerate gap")
    assert required_n(p0, smaller_gap) >= required_n(p0, p1)


@given(a=st.lists(st.floats(-50, 50), min_size=2, max_size=30),
       b=st.lists(st.floats(-50, 50), min_size=2, max_size=30))
@settings(max_examples=50)
def test_auroc_is_in_range_and_symmetric(a, b):
    s1 = separation(a, b)["auroc"]
    s2 = separation(b, a)["auroc"]
    assert 0.0 <= s1 <= 1.0
    assert abs((s1 + s2) - 1.0) < 1e-6      # swapping the classes must mirror it


@given(v=st.lists(st.floats(-10, 10), min_size=3, max_size=20))
def test_identical_distributions_score_one_half(v):
    """No separation must read as chance, not as a weak positive."""
    assert abs(separation(v, list(v))["auroc"] - 0.5) < 1e-9


@given(hits=st.lists(st.booleans(), min_size=1, max_size=40))
def test_mcnemar_of_a_detector_against_itself_is_never_significant(hits):
    a = np.array(hits)
    b01, b10, p = mcnemar(a, a)
    assert b01 == b10 == 0 and p == 1.0
