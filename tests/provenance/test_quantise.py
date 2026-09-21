"""Quantisation (plan §5.2, §7.2): floor(float64(x) * SCALE + 0.5), always, before canonicalising."""
from __future__ import annotations

import math
import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cva.provenance.seal.constants import INT_MAX
from cva.provenance.seal.errors import NonFiniteValue, QuantiseError
from cva.provenance.seal.quantise import q_box64, q_conf, q_e6, q_px


def f32(x: float) -> float:
    """The float64 value of x rounded to float32 — what a float32 model output looks like widened."""
    return struct.unpack("f", struct.pack("f", x))[0]


# (input, expected conf_e6). Includes the exact halves the plan calls out (§7.2).
CONF_TABLE = [
    (0.0, 0), (1.0, 1_000_000), (0.5, 500_000), (1e-7, 0),
    (0.0000005, 1), (0.0000015, 2), (0.0000025, 3), (0.0000035, 4),   # half-up; banker's rounding gives 0,2,2,4
    (0.871204, 871_204), (0.9999995, 1_000_000), (0.9999994, 999_999),
    (-0.2, 0), (-1e-9, 0),                       # clamped up to 0
    (1.5, 1_000_000), (7.0, 1_000_000),          # clamped down to 1_000_000
]


@pytest.mark.parametrize("x,expected", CONF_TABLE)
def test_q_conf_table(x, expected):
    assert q_conf(x) == expected


def test_it_is_half_up_not_bankers_rounding():
    # Python's round() gives 0, 2, 2 at these halves; the wire format must give 1, 2, 3.
    assert [round(0.5), round(1.5), round(2.5)] == [0, 2, 2]
    assert [q_px(0.5), q_px(1.5), q_px(2.5)] == [1, 2, 3]


def test_half_up_means_toward_positive_infinity_for_negatives():
    assert [q_px(-0.5), q_px(-1.5), q_px(-2.5)] == [0, -1, -2]


@pytest.mark.parametrize("x,expected", [
    (0.0, 0), (1.0, 64), (0.0078125, 1),        # 1/128 px * 64 = 0.5 exactly -> rounds up
    (0.0234375, 2),                              # 3/128 * 64 = 1.5 -> 2
    (1204.0, 77_056), (-3.0, -192), (0.00390625, 0),
])
def test_q_box64_table(x, expected):
    assert q_box64(x) == expected


@pytest.mark.parametrize("x,expected", [(0.485, 485_000), (0.229, 229_000), (255.0, 255_000_000),
                                        (-0.5, -500_000), (0.0, 0)])
def test_q_e6_is_unclamped_for_preprocessing_constants(x, expected):
    assert q_e6(x) == expected


def test_the_rule_is_applied_to_the_widened_float32_value():
    """float32 -> float64 widening is exact, so the answer is defined by the float32 the model
    emitted, not by the decimal it was printed as. 5e-7 in float32 is slightly BELOW one half."""
    assert f32(5e-7) < 5e-7
    assert q_conf(5e-7) == 1
    assert q_conf(f32(5e-7)) == 0
    assert q_conf(f32(0.871204)) == 871_204      # ordinary values are unaffected


def test_accepts_int_and_numpy_style_scalars():
    assert q_conf(1) == 1_000_000
    assert q_conf(0) == 0

    class Scalar:                                # duck-types a numpy scalar
        def __float__(self) -> float:
            return 0.25

    assert q_conf(Scalar()) == 250_000


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("fn", [q_conf, q_e6, q_box64, q_px])
def test_non_finite_is_rejected_by_every_quantiser(fn, bad):
    with pytest.raises(NonFiniteValue):
        fn(bad)


@pytest.mark.parametrize("fn", [q_conf, q_e6, q_box64, q_px])
@pytest.mark.parametrize("bad", [True, False, None, "0.5", [0.5], b"1"])
def test_non_numbers_and_bools_are_rejected(fn, bad):
    with pytest.raises(QuantiseError):
        fn(bad)


def test_out_of_range_results_are_rejected_not_wrapped():
    with pytest.raises(QuantiseError):
        q_box64(1e300)
    with pytest.raises(QuantiseError):
        q_e6(float(INT_MAX))                     # * 1e6 overflows the +/-(2**53-1) integer range
    with pytest.raises(QuantiseError):
        q_px(1e17)


def test_a_huge_int_does_not_crash_with_an_overflow_error():
    with pytest.raises(QuantiseError):
        q_conf(10**400)


def test_results_are_plain_ints_the_profile_accepts():
    for v in (q_conf(0.5), q_box64(3.3), q_px(3.3), q_e6(0.5)):
        assert type(v) is int


_FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(_FINITE, _FINITE)
def test_quantisation_is_monotonic_non_decreasing(a, b):
    lo, hi = sorted((a, b))
    assert q_box64(lo) <= q_box64(hi)
    assert q_px(lo) <= q_px(hi)
    assert q_e6(lo) <= q_e6(hi)
    assert q_conf(lo) <= q_conf(hi)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_confidence_is_always_within_range(p):
    assert 0 <= q_conf(p) <= 1_000_000


@given(_FINITE)
def test_error_is_at_most_half_a_quantum(x):
    assert abs(q_box64(x) / 64.0 - x) <= 0.5 / 64.0 + 1e-9
    assert abs(q_px(x) - x) <= 0.5 + 1e-9
    assert math.isclose(q_e6(x) / 1e6, x, abs_tol=0.5e-6 + 1e-9)
