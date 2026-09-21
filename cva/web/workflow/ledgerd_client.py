"""The ONLY code in `cva-web` that talks to `cva-ledgerd` (plan §6).

Fail-closed (D-E4): a failure raises `LedgerUnreachable` or `LedgerRefused` and the caller
changes nothing. There is no retry queue, no optimistic update and no local mirror of an
action that the ledger did not accept.
"""
from __future__ import annotations

import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cva.ledgerd import protocol as p


class LedgerUnreachable(Exception):
    """The socket is not there, or the daemon did not answer. Code: `ledger_unavailable`."""

    code = "ledger_unavailable"


class LedgerRefused(Exception):
    """The daemon validated the request and said no. Nothing was written."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Appended:
    seq: int
    record_hash: str
    deduped: bool = False


@dataclass(frozen=True)
class LedgerStatus:
    capabilities: tuple[str, ...]
    size: int
    read_only: bool
    four_eyes: bool

    @property
    def can_sign(self) -> bool:
        return "SIGNING_KEY" in self.capabilities


class LedgerdClient:
    def __init__(self, socket_path: Path, *, timeout: float = 10.0) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    def _call(self, req: Mapping[str, Any]) -> dict[str, Any]:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect(str(self.socket_path))
                p.send(s, req)
                line = p.read_frame(s, limit=p.MAX_RESPONSE_BYTES)
        except (OSError, p.ProtocolError) as e:
            raise LedgerUnreachable(
                f"cva-ledgerd is not answering on {self.socket_path}: {e}. The workflow is "
                "unavailable and nothing was recorded — a decision that exists in the UI but "
                "not in the ledger is exactly what this design refuses to produce.") from None
        if line is None:
            raise LedgerUnreachable("cva-ledgerd closed the connection without answering")
        try:
            resp = p.decode(line)
        except p.ProtocolError as e:
            raise LedgerUnreachable(f"cva-ledgerd sent an unreadable answer: {e}") from None
        if not resp.get("ok"):
            raise LedgerRefused(str(resp.get("error", "internal")),
                                str(resp.get("detail", "")))
        return resp

    # -- reads ---------------------------------------------------------------------------
    def status(self) -> LedgerStatus:
        r = self._call(p.request("status"))
        return LedgerStatus(capabilities=tuple(r.get("capabilities") or []),
                            size=int(r.get("size", 0)),
                            read_only=bool(r.get("read_only", True)),
                            four_eyes=bool(r.get("four_eyes", True)))

    def records(self, *, types: Sequence[str] | None = None, scan_id: str | None = None,
                since_seq: int | None = None) -> list[dict[str, Any]]:
        fields: dict[str, Any] = {}
        if types:
            fields["types"] = list(types)
        if scan_id:
            fields["scan_id"] = scan_id
        if since_seq is not None:
            fields["since_seq"] = since_seq
        return list(self._call(p.request("records", **fields)).get("records") or [])

    # -- writes ---------------------------------------------------------------------------
    def append_analyst_event(self, body: Mapping[str, Any]) -> Appended:
        r = self._call(p.request("append_analyst_event", event=dict(body)))
        return Appended(seq=int(r["seq"]), record_hash=str(r.get("record_hash", "")),
                        deduped=bool(r.get("deduped", False)))

    def append_scan_record(self, body: Mapping[str, Any]) -> Appended:
        r = self._call(p.request("append_scan_record", scan=dict(body)))
        return Appended(seq=int(r["seq"]), record_hash=str(r.get("record_hash", "")))


class LedgerdAuditLedger:
    """`core.ledger.AuditLedger` over the socket — the O2 handoff.

    `cva scan --audit-ledger-socket` seals its `scan_record` through `cva-ledgerd` under its
    own uid, which the daemon's allowlist grants `scan_record` and nothing else. The scanner
    therefore never holds the signing key either.

    `capabilities()` is an ACTIVE probe: it asks the daemon, which answers from the real
    `SealedLedger.capabilities()`. A socket that connects is not a signing key, and this
    reports `SIGNING_KEY` only when the daemon says a key loaded AND produced a signature
    that verified.
    """

    def __init__(self, socket_path: Path, *, timeout: float = 10.0) -> None:
        self.client = LedgerdClient(Path(socket_path), timeout=timeout)

    def append(self, record: Mapping[str, Any]) -> str:
        rtype = record.get("type")
        if rtype != "scan_record":
            raise LedgerRefused(
                "not_permitted",
                f"this handle appends scan_record only, not {rtype!r}. Analyst decisions "
                "are written by cva-web through the same daemon, under a different uid.")
        return str(self.client.append_scan_record(record.get("scan") or {}).seq)

    def capabilities(self) -> set[Any]:
        from cva.core.capability import Capability
        try:
            status = self.client.status()
        except (LedgerUnreachable, LedgerRefused):
            return set()
        out: set[Any] = set()
        if status.can_sign and not status.read_only:
            out.add(Capability.SIGNING_KEY)
        return out


__all__ = ["Appended", "LedgerRefused", "LedgerStatus", "LedgerUnreachable",
           "LedgerdAuditLedger", "LedgerdClient"]
