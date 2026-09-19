"""Quantisation (plan §5.2, C-10/S2): every number becomes an integer BEFORE it is canonicalised.

The multiply-add-floor sequence below is normative and must not be "simplified":

    q = floor(float64(x) * SCALE + 0.5)

  * NOT `round()` — Python's is banker's rounding (round(0.5) == 0, round(1.5) == 2), which depends
    on the parity of the neighbouring integer and which no other language reproduces at an exact half.
  * float32 -> float64 widening is exact, so the rule is bit-deterministic across languages given
    the same input float.
  * A C/C++ port MUST disable fused multiply-add contraction (e.g. `-ffp-contract=off`): fusing
    `x * SCALE + 0.5` into one rounding step changes results at exact halves and silently forks the
    hash. This is a wire-format requirement, recorded in the published spec at C8.
"""
from __future__ import annotations

import math
from typing import Any

from .constants import INT_MAX
from .errors import NonFiniteValue, QuantiseError

E6 = 1_000_000.0
Q64 = 64.0
CONF_MAX = 1_000_000


def _as_float(x: Any) -> float:
    # float() would happily coerce "0.5" or b"1" — a mistyped value must never reach a signed record.
    if isinstance(x, (bool, str, bytes, bytearray)):
        raise QuantiseError(f"{type(x).__name__} is not a number")
    try:
        f = float(x)                     # exact widening for float32 / numpy scalars
    except (TypeError, ValueError, OverflowError) as e:
        raise QuantiseError(f"not a number: {x!r}") from e
    if not math.isfinite(f):
        raise NonFiniteValue(f"non-finite value {f!r}")
    return f


def _floor_half_up(v: float) -> int:
    q = math.floor(v + 0.5)
    if abs(q) > INT_MAX:
        raise QuantiseError(f"quantised value {q} outside +/-(2**53-1)")
    return q


def q_e6(x: Any) -> int:
    """x * 1e6, half-up, unclamped. Used for preprocessing mean/std/scale."""
    return _floor_half_up(_as_float(x) * E6)


def q_conf(p: Any) -> int:
    """Confidence / probability p in [0, 1] -> integer 'conf_e6', clamped to [0, 1_000_000]."""
    return min(CONF_MAX, max(0, q_e6(p)))


def q_box64(x: Any) -> int:
    """Box coordinate in pixels -> integer 1/64 px (the fine hash)."""
    return _floor_half_up(_as_float(x) * Q64)


def q_px(x: Any) -> int:
    """Box coordinate in pixels -> whole pixels (the coarse `decision_sha256`, D6)."""
    return _floor_half_up(_as_float(x))
