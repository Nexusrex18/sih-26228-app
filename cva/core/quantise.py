"""The float/hash quantisation boundary — backend_plan.md §7.7.

Backend produces six fields of the seal record (§9.2) and must respect the hashing rules
Module C's hardening established.

    Why no float ever enters hashed JSON: JCS was adopted specifically because "the same
    logical output hashes differently across Python versions, dict orderings and float
    formatting" — but JCS mandates ECMAScript Number::toString semantics, and Python's
    repr() switches to exponential notation at a DIFFERENT threshold than ECMAScript does.
    Our outputs are full of floats — confidences, box coordinates — so the exact failure
    JCS was adopted to prevent re-enters through JCS's own numeric rules.
    Fix: quantise before canonicalising, ALWAYS.

**`floor(x + 0.5)` is normative and must not be "simplified" back to `round()`.** Python's
`round()` is banker's rounding: `round(0.5) == 0` while `round(1.5) == 2`, so at an exact
half the result depends on the parity of the neighbouring integer. That is not
reproducible in any other language, and cross-implementation divergence is precisely what
quantisation exists to eliminate. `floor(x + 0.5)` is unconditional and portable. The
multiply-add-floor sequence was corrected by the Module C plan §5.2 after this plan had it
wrong; the correction is the spec.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

CONFIDENCE_SCALE = 1_000_000.0   # declared 6 dp
BOX_SCALE = 64.0                 # 1/64 px, fixed-point integer


class NonFiniteValue(ValueError):
    """NaN and ±inf are REJECTED — never quantised, never canonicalised.

    There is no honest integer for them, and JCS has no encoding for them either. A record
    that silently mapped NaN to 0 would hash cleanly while meaning something else.
    """


def _floor_half(x: float) -> int:
    if not math.isfinite(x):
        raise NonFiniteValue(f"{x!r} cannot enter hashed JSON")
    return math.floor(float(x) + 0.5)


def q_confidence(p: float) -> int:
    """`floor(float64(p) * 1e6 + 0.5)`, clamped to [0, 1_000_000]."""
    v = _floor_half(float(p) * CONFIDENCE_SCALE)
    return max(0, min(1_000_000, v))


def q_box(x: float) -> int:
    """`floor(float64(x) * 64.0 + 0.5)` — 1/64 px."""
    return _floor_half(float(x) * BOX_SCALE)


def q_box_coarse(x: float) -> int:
    """`floor(float64(x) + 0.5)` — whole px, for `output.decision_sha256`.

    The coarse form exists because the decision-relevant comparison is EXACT BY HASH. A
    tolerance band is a place to hide a nudged confidence: 0.51 -> 0.49 flips a threshold
    decision while sitting inside any sane float tolerance.
    """
    return _floor_half(float(x))


def q_preprocess(value: Any) -> Any:
    """Quantise a resolved preprocessing spec recursively.

    `mean`/`std` take the same x1e6 integer rule as confidence — which is why
    `preprocess_hash` is taken over the QUANTISED form, not the raw one.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return q_confidence(value) if -1.0 <= value <= 1.0 else _floor_half(
            value * CONFIDENCE_SCALE)
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {k: q_preprocess(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [q_preprocess(v) for v in value]
    return value


def canonical_bytes(obj: Any) -> bytes:
    """JCS (RFC 8785) if available, else a deterministic sorted-compact fallback.

    `rfc8785` is a Mode C dependency and one of only two third-party imports CI invariant
    4 admits, so it is present wherever the seal runs. The fallback exists so that Mode A
    tooling does not hard-fail without it — and it is NOT claimed to be JCS: anything
    hashed for the chain goes through the real thing.
    """
    try:
        import rfc8785
        return rfc8785.dumps(obj)
    except ImportError:
        import json
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")


def preprocess_hash(spec: dict[str, Any]) -> str:
    """§9.2 — sha256 of the JCS canonical form of the QUANTISED resolved spec.

    Paired always with `preprocess_ref`, a pointer to the STORED spec: you cannot re-run
    preprocessing from a digest, and `prov.recompute` is the strongest check in the
    system. A record carrying `preprocess_hash` alone makes its own strongest check
    impossible to perform.
    """
    return hashlib.sha256(canonical_bytes(q_preprocess(spec))).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
