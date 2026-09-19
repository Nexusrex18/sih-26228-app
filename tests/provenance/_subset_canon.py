"""An independent reference canonicaliser for the §5.1 profile — for the differential test only.

Deliberately shares NO code with `rfc8785`. The profile has no floats and no non-ASCII, so JCS
collapses to: sorted keys, no whitespace, integers as decimal, and only two string escapes (`"` and
`\\`). If the library and this ~15-line function ever disagree on a profile-valid object, one of them
is wrong. It is also the seed of the strict-subset canonicaliser the C core will need (plan A-1/C9).
"""
from __future__ import annotations

from typing import Any


def ref_canon(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(ref_canon(x) for x in v) + "]"
    if isinstance(v, dict):                      # ASCII keys: code-unit order == Python str order
        return "{" + ",".join(f"{ref_canon(k)}:{ref_canon(x)}" for k, x in sorted(v.items())) + "}"
    raise TypeError(type(v))
