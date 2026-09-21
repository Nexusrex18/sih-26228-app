"""The `cva-ledgerd` wire protocol: one JSON object per line, over a Unix socket.

Deliberately boring. Newline-delimited JSON with a hard frame limit, a closed set of ops and
a typed error for every refusal. The daemon that holds the signing key parses as little as
possible, and what it does parse it parses with the stdlib.

Every response carries `ok`. A refusal carries `error` (a stable machine code the browser is
shown, plan §5.3) and `detail` (prose for the analyst). The web process never invents a
reason of its own for a failure that happened here.
"""
from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from typing import Any

#: Requests are small by construction; a justification is capped at 8 KiB encoded.
MAX_FRAME_BYTES = 128 * 1024
#: Responses carry record batches, so they get more room — still bounded.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024

OPS = ("status", "records", "append_analyst_event", "append_scan_record")

#: Stable refusal codes. `ledger_unavailable`, `stale_state` and `not_permitted` are the
#: three the plan names in §5.3; the rest are the specific cases behind them.
ERROR_CODES = (
    "ledger_unavailable", "stale_state", "not_permitted", "four_eyes", "no_such_pending",
    "unknown_finding", "justification_required", "invalid_request", "invalid_field",
    "invalid_reason_code", "not_quarantined", "frame_too_large", "read_only", "internal",
)


class ProtocolError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def request(op: str, **fields: Any) -> dict[str, Any]:
    if op not in OPS:
        raise ProtocolError("invalid_request", f"unknown op {op!r}")
    return {"op": op, **fields}


def ok(**fields: Any) -> dict[str, Any]:
    return {"ok": True, **fields}


def error(code: str, detail: str) -> dict[str, Any]:
    if code not in ERROR_CODES:
        code = "internal"
    return {"ok": False, "error": code, "detail": detail}


def encode(obj: Mapping[str, Any]) -> bytes:
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return data + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_FRAME_BYTES:
        raise ProtocolError("frame_too_large",
                            f"{len(line)} bytes exceeds the {MAX_FRAME_BYTES}-byte frame limit")
    try:
        obj = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError("invalid_request", f"not a JSON object ({e})") from None
    if not isinstance(obj, dict):
        raise ProtocolError("invalid_request", "expected a JSON object")
    return obj


def read_frame(sock: socket.socket, *, limit: int = MAX_FRAME_BYTES) -> bytes | None:
    """Read one newline-terminated frame. `None` at a clean EOF."""
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            b = sock.recv(4096)
        except (TimeoutError, OSError) as e:
            raise ProtocolError("ledger_unavailable", f"socket read failed: {e}") from None
        if not b:
            if not chunks:
                return None
            raise ProtocolError("invalid_request", "connection closed mid-frame")
        nl = b.find(b"\n")
        if nl >= 0:
            chunks.append(b[:nl])
            return b"".join(chunks)
        chunks.append(b)
        total += len(b)
        if total > limit:
            raise ProtocolError("frame_too_large",
                                f"more than {limit} bytes with no newline")


def send(sock: socket.socket, obj: Mapping[str, Any]) -> None:
    sock.sendall(encode(obj))


__all__ = ["ERROR_CODES", "MAX_FRAME_BYTES", "MAX_RESPONSE_BYTES", "OPS", "ProtocolError",
           "decode", "encode", "error", "ok", "read_frame", "request", "send"]
