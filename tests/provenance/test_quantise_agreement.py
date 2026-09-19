"""Backend's `core/quantise.py` and the seal's `quantise.py` are two independent implementations of one
normative rule (Module C plan §5.2: floor(float64(x) * SCALE + 0.5)). They must never drift apart: a
disagreement means a record hashed by one side cannot be recomputed by the other — a false alarm on a
clean record. This test lives in the seal's suite because the seal is what breaks."""
from __future__ import annotations

import math
import random

import pytest

from cva.provenance.seal import quantise as seal

core = pytest.importorskip("cva.core.quantise")


def halves() -> list[float]:
    """Values that sit exactly on, and a hair either side of, rounding boundaries."""
    out = []
    for n in range(-6, 400):
        for scale in (1e-6, 1 / 64, 1.0):
            x = (n + 0.5) * scale
            out += [x, math.nextafter(x, math.inf), math.nextafter(x, -math.inf)]
    return out


CASES = halves() + [random.Random(5).uniform(-2000, 2000) for _ in range(3000)] + [0.0, 1.0, 1e-7, 0.999999, 1.5]


def test_confidence_agrees_everywhere():
    for x in CASES:
        assert seal.q_conf(x) == core.q_confidence(x), x


def test_fine_box_agrees_everywhere():
    for x in CASES:
        assert seal.q_box64(x) == core.q_box(x), x


def test_coarse_box_agrees_everywhere():
    for x in CASES:
        assert seal.q_px(x) == core.q_box_coarse(x), x


def test_both_reject_non_finite_values():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            seal.q_conf(bad)
        with pytest.raises(ValueError):
            core.q_confidence(bad)
