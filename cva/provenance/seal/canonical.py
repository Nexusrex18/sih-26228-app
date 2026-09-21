"""Canonical bytes: the §5.1 profile on top of RFC 8785 (decision D3).

The profile is a strict SUBSET of JCS, so `rfc8785.dumps` yields byte-identical output for every
accepted object — but a C/C++/Rust implementer (Module C plan A-1) needs ~150 lines, not a Unicode
library, because there are no floats, no non-ASCII and no exotic escapes to get right.

Profile:
  * object keys        ASCII  [a-z0-9_]+
  * string values      printable ASCII 0x20–0x7E only  (use `ascii_encode` for anything else)
  * numbers            integers, |n| <= 2**53-1; NO floats, ever
  * arrays             homogeneous (all int, all str, all bool, all object, or all array); no null
  * `null`             permitted by this layer only as an object value; records.py restricts it further
  * depth              <= MAX_DEPTH;  size <= MAX_RECORD_BYTES (records) — payloads pass max_bytes=None

Nothing here "repairs" input. A violation raises `NonCanonical` naming the JSON path.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import rfc8785

from .constants import INT_MAX, MAX_DEPTH, MAX_RECORD_BYTES
from .errors import NonCanonical

_KEY = re.compile(r"[a-z0-9_]+")
_PRINTABLE = re.compile(r"[\x20-\x7e]*")
_PCT = re.compile(r"%([0-9A-F]{2})")


def _kind(v: Any) -> str:
    if isinstance(v, bool):                      # bool before int: bool is an int subclass
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    if isinstance(v, Mapping):
        return "object"
    if isinstance(v, (list, tuple)):
        return "array"
    if v is None:
        return "null"
    if isinstance(v, float):
        return "float"
    return type(v).__name__


def validate_profile(obj: Any, path: str = "$", _depth: int = 0) -> None:
    """Raise NonCanonical on the first §5.1 violation. Iteration order of dicts is irrelevant."""
    kind = _kind(obj)
    if kind in ("object", "array") and _depth + 1 > MAX_DEPTH:
        raise NonCanonical(f"nesting deeper than {MAX_DEPTH}", path=path)
    if kind == "object":
        for k, v in obj.items():
            if not isinstance(k, str) or not _KEY.fullmatch(k):
                raise NonCanonical(f"key {k!r} is not ASCII [a-z0-9_]+", path=path)
            validate_profile(v, f"{path}.{k}", _depth + 1)
    elif kind == "array":
        kinds = {_kind(x) for x in obj}
        if "null" in kinds:
            raise NonCanonical("null inside an array", path=path)
        if len(kinds) > 1:
            raise NonCanonical(f"array of mixed types {sorted(kinds)}", path=path)
        for i, x in enumerate(obj):
            validate_profile(x, f"{path}[{i}]", _depth + 1)
    elif kind == "str":
        if not _PRINTABLE.fullmatch(obj):
            raise NonCanonical("string outside printable ASCII 0x20-0x7E (use ascii_encode)", path=path)
    elif kind == "int":
        if abs(obj) > INT_MAX:
            raise NonCanonical(f"integer {obj} outside +/-(2**53-1)", path=path)
    elif kind in ("bool", "null"):
        pass
    elif kind == "float":
        raise NonCanonical("float — quantise to an integer first (plan §5.2)", path=path)
    else:
        raise NonCanonical(f"unsupported type {kind}", path=path)


def canonical_bytes(obj: Mapping[str, Any], *, max_bytes: int | None = MAX_RECORD_BYTES) -> bytes:
    """Validate against the §5.1 profile, then return `rfc8785.dumps(obj)`.

    `max_bytes` defaults to the record cap. Output payloads (a raw pre-filter detector output can
    legitimately hold thousands of boxes) pass `max_bytes=None` — the cap exists to reject oversized
    *records*, not large payloads.
    """
    if not isinstance(obj, Mapping):
        raise NonCanonical("top level must be an object")
    validate_profile(obj)
    out = rfc8785.dumps(_plain(obj))
    if max_bytes is not None and len(out) > max_bytes:
        raise NonCanonical(f"canonical form is {len(out)} bytes, over the {max_bytes}-byte cap")
    return out


def _plain(v: Any) -> Any:
    """Tuples -> lists, Mappings -> dicts, so the library sees only plain JSON containers."""
    if isinstance(v, Mapping):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def parse_strict(data: bytes, *, max_bytes: int | None = MAX_RECORD_BYTES,
                 require_canonical: bool = True) -> dict[str, Any]:
    """Parse stored/exported record bytes. Rejects, rather than tolerates:

      duplicate keys (the classic parser-differential attack) · floats · NaN/Infinity · integers
      beyond 2**53-1 · non-ASCII bytes · a top level that is not an object · anything outside the
      profile · and — last — any input whose bytes are not EXACTLY the canonical bytes of what it
      parses to (different key order, whitespace, escapes, a trailing newline, ...).

    That last check is what makes "stored record == its canonical bytes" (§5.1 rule 7) enforceable:
    re-serialisation is itself a finding, code "non_canonical_encoding".

    `require_canonical=False` skips only that last byte-equality check (every other rule still holds).
    It exists for human-handled files such as the trust root, which are not hashed as bytes; records
    and ledger exports must always use the default.
    """
    raw = bytes(data)
    if max_bytes is not None and len(raw) > max_bytes:
        raise NonCanonical(f"{len(raw)} bytes exceeds the {max_bytes}-byte cap")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise NonCanonical(f"non-ASCII byte at offset {e.start}") from None

    def no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            dup = next(k for k in keys if keys.count(k) > 1)
            raise NonCanonical(f"duplicate key {dup!r}")
        return dict(pairs)

    def no_float(s: str) -> Any:
        raise NonCanonical(f"float literal {s!r}")

    def no_const(s: str) -> Any:
        raise NonCanonical(f"{s} is not permitted")

    def bounded_int(s: str) -> int:
        n = int(s)
        if abs(n) > INT_MAX:
            raise NonCanonical(f"integer {s} outside +/-(2**53-1)")
        return n

    try:
        obj = json.loads(text, object_pairs_hook=no_dupes, parse_float=no_float,
                         parse_constant=no_const, parse_int=bounded_int)
    except NonCanonical:
        raise
    except RecursionError:
        raise NonCanonical("nesting too deep to parse") from None
    except ValueError as e:                      # json.JSONDecodeError
        raise NonCanonical(f"not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise NonCanonical("top level must be an object")
    validate_profile(obj)
    if require_canonical and canonical_bytes(obj, max_bytes=max_bytes) != raw:
        raise NonCanonical("valid JSON but not in canonical form (key order, whitespace, escapes, "
                           "or trailing bytes differ)", code="non_canonical_encoding")
    return obj


# --- text that is not printable ASCII must be encoded by the caller (plan §5.1 rule 3) ----------

def ascii_encode(text: str) -> str:
    """Percent-encode `text` into the printable-ASCII profile: every UTF-8 byte outside 0x20-0x7E,
    and '%' itself, becomes %XX with UPPERCASE hex (RFC 3986 §2.1's normal form — one canonical
    spelling, so two implementations encode identically). Space and other printable ASCII pass through.

    Raises NonCanonical for text that is not valid Unicode (e.g. a lone surrogate).
    """
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise NonCanonical(f"text is not valid Unicode: {e.reason}") from None
    return "".join(chr(b) if 0x20 <= b <= 0x7E and b != 0x25 else f"%{b:02X}" for b in data)


def ascii_decode(text: str) -> str:
    """Inverse of `ascii_encode`, for DISPLAY. Strict: a '%' not followed by two uppercase hex digits,
    or a non-printable byte in the input, is an error rather than passed through."""
    if not _PRINTABLE.fullmatch(text):
        raise NonCanonical("encoded text must be printable ASCII")
    out = bytearray()
    i = 0
    while i < len(text):
        c = text[i]
        if c == "%":
            m = _PCT.fullmatch(text[i:i + 3])
            if not m:
                raise NonCanonical(f"malformed percent-escape at offset {i}")
            out.append(int(m.group(1), 16))
            i += 3
        else:
            out.append(ord(c))
            i += 1
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        raise NonCanonical("percent-escapes do not decode to valid UTF-8") from None
