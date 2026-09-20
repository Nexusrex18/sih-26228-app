"""`cva-ledgerd` — the only process that holds the signing key (plan D-E3, §5.1).

It accepts typed requests on a Unix socket, checks them against `SO_PEERCRED`, the record
schema, the role table and the folded ledger, and only then signs. A compromised `cva-web`
can spam requests; it cannot forge, edit or delete history.

Everything fails CLOSED (D-E4). Any error on any step leaves the ledger untouched and the
caller gets the reason.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cva.web.workflow.events import is_wire_safe
from cva.web.workflow.fold import FoldRejection, check_admissible, fold
from cva.web.workflow.states import LedgerEvent

from . import protocol as p
from .peercred import PeerCredUnavailable, peer_credentials
from .policy import ANALYST_EVENT, SCAN_RECORD, Policy, decide

log = logging.getLogger("cva.ledgerd")

SOCKET_MODE = 0o660


class LedgerBackend:
    """The daemon's view of the ledger. `SealedLedger` is imported HERE and nowhere in web.

    A `records()` that returns dicts with `seq` is all the fold needs, and it is all the web
    process is ever given: raw signed bytes stay on this side of the socket.
    """

    def __init__(self, ledger: Any, *, read_only: bool = False) -> None:
        self._ledger = ledger
        self.read_only = read_only
        self._lock = threading.Lock()

    @classmethod
    def open(cls, ledger_path: Path, key_path: Path | None) -> LedgerBackend:
        from cva.provenance.seal import FileKeyProvider, SealedLedger
        if key_path is None:
            return cls(SealedLedger.open(ledger_path, read_only=True), read_only=True)
        key = FileKeyProvider(key_path)
        return cls(SealedLedger.open(ledger_path, key=key))

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._ledger.records()]

    def capabilities(self) -> list[str]:
        with self._lock:
            return sorted(str(c) for c in self._ledger.capabilities())

    def size(self) -> int:
        with self._lock:
            return int(self._ledger.size())

    def append(self, rtype: str, body: Mapping[str, Any]) -> tuple[int, str]:
        if self.read_only:
            raise p.ProtocolError(
                "read_only",
                "this daemon opened the ledger without a signing key, so it can serve reads "
                "and refuse writes, and it will not pretend otherwise")
        with self._lock:
            written = self._ledger.append_typed(rtype, dict(body))
        first = next((w for w in written if w.type == rtype), written[0])
        return int(first.seq), str(first.record_hash)

    def close(self) -> None:
        with self._lock:
            close = getattr(self._ledger, "close", None)
            if callable(close):
                close()


class Daemon:
    """The request handler, free of socket plumbing so it can be tested directly."""

    def __init__(self, backend: LedgerBackend, policy: Policy, *,
                 four_eyes: bool = True) -> None:
        self.backend = backend
        self.policy = policy
        self.four_eyes = four_eyes
        self._seen: dict[str, tuple[int, str]] = {}
        self._rebuild_idempotency()

    def _rebuild_idempotency(self) -> None:
        """Idempotency must survive a restart (plan test 10.5).

        The `request_id` is IN the record, so the map is rebuilt from the ledger rather than
        kept only in memory. A retry after a crash then still records once.
        """
        try:
            for r in self.backend.records():
                a = r.get("analyst") or {}
                rid = a.get("request_id")
                if r.get("type") == ANALYST_EVENT and isinstance(rid, str):
                    self._seen[rid] = (int(r.get("seq", 0)), "")
        except Exception as e:                            # noqa: BLE001 - a ledger we cannot
            log.warning("could not rebuild the idempotency map: %s", e)   # read is reported,
            self._seen = {}                                               # never assumed empty

    # -- ops ---------------------------------------------------------------------------
    def handle(self, req: Mapping[str, Any], uid: int) -> dict[str, Any]:
        op = req.get("op")
        if op not in p.OPS:
            return p.error("invalid_request", f"unknown op {op!r}; expected one of {list(p.OPS)}")
        allowed, detail = decide(self.policy, uid, str(op))
        if not allowed:
            return p.error("not_permitted", detail)
        try:
            if op == "status":
                return self._status()
            if op == "records":
                return self._records(req)
            if op == "append_analyst_event":
                return self._append_analyst(req, uid)
            return self._append_scan(req)
        except p.ProtocolError as e:
            return p.error(e.code, e.detail)
        except Exception as e:                            # noqa: BLE001 - fail closed, say why
            log.exception("ledgerd: unhandled error on %s", op)
            return p.error("internal", f"{type(e).__name__}: {e}")

    def _status(self) -> dict[str, Any]:
        return p.ok(capabilities=self.backend.capabilities(), size=self.backend.size(),
                    read_only=self.backend.read_only, four_eyes=self.four_eyes)

    def _records(self, req: Mapping[str, Any]) -> dict[str, Any]:
        types = req.get("types")
        scan_id = req.get("scan_id")
        since = req.get("since_seq")
        out = []
        for r in self.backend.records():
            if types and r.get("type") not in types:
                continue
            if since is not None and int(r.get("seq", 0)) <= int(since):
                continue
            if scan_id:
                body = r.get("analyst") or r.get("scan") or {}
                if body.get("scan_id") != scan_id:
                    continue
            out.append(r)
        return p.ok(records=out, size=self.backend.size())

    def _append_analyst(self, req: Mapping[str, Any], uid: int) -> dict[str, Any]:
        body = req.get("event")
        if not isinstance(body, Mapping):
            return p.error("invalid_request", "`event` must be the analyst section object")
        body = dict(body)

        rid = body.get("request_id")
        if isinstance(rid, str) and rid in self._seen:
            seq, rh = self._seen[rid]
            return p.ok(seq=seq, record_hash=rh, deduped=True)

        # The daemon VALIDATES the encoding; it never repairs it. Re-encoding "helpfully" is
        # not idempotent — `%20` becomes `%2520` — so a client that forgot to encode and a
        # client that remembered are indistinguishable to a guess. Refusing is the only
        # answer that cannot silently corrupt an analyst's words.
        for field in ("justification", "target_ref"):
            v = body.get(field)
            if isinstance(v, str) and not is_wire_safe(v):
                return p.error(
                    "invalid_field",
                    f"`{field}` is not in the ledger's wire form. Module C §5.1 allows "
                    "printable ASCII only, so text is percent-encoded by the client and "
                    "decoded for display. cva-ledgerd does not re-encode: doing so would "
                    "turn an already-encoded justification into a corrupted one.")

        ev = LedgerEvent.from_record({"seq": self.backend.size(), "analyst": body})
        state = self._fold_for(ev.scan_id)
        try:
            check_admissible(state, ev, four_eyes=self.four_eyes, findings_known=False)
        except FoldRejection as e:
            return p.error(e.code, e.detail)

        try:
            seq, rh = self.backend.append(ANALYST_EVENT, {"analyst": body})
        except p.ProtocolError:
            raise
        except Exception as e:                            # noqa: BLE001 - includes InvalidRecord
            return p.error("invalid_field",
                           f"the seal refused this record: {type(e).__name__}: {e}")
        if isinstance(rid, str):
            self._seen[rid] = (seq, rh)
        return p.ok(seq=seq, record_hash=rh, deduped=False)

    def _append_scan(self, req: Mapping[str, Any]) -> dict[str, Any]:
        body = req.get("scan")
        if not isinstance(body, Mapping):
            return p.error("invalid_request", "`scan` must be the scan section object")
        try:
            seq, rh = self.backend.append(SCAN_RECORD, {"scan": dict(body)})
        except p.ProtocolError:
            raise
        except Exception as e:                            # noqa: BLE001
            return p.error("invalid_field",
                           f"the seal refused this scan record: {type(e).__name__}: {e}")
        return p.ok(seq=seq, record_hash=rh)

    def _fold_for(self, scan_id: str) -> Any:
        """Fold this scan's events with NO originals — deliberately.

        The daemon has no `report.json` and must not read one: that would make the process
        holding the signing key a second consumer of the report with its own opinion of a
        disposition, which D-E2 forbids. So it cannot tell a raise from a lowering, and it
        does not try. What it enforces from the ledger alone is everything that matters here:
        the role table, `expected_prev_seq`, and that an `approve` names a real change made
        by somebody else. Whether that change was immediate or pending is decided by
        `cva-web`'s fold, which does have the originals.
        """
        events = [LedgerEvent.from_record(r) for r in self.backend.records()
                  if r.get("type") == ANALYST_EVENT
                  and (r.get("analyst") or {}).get("scan_id") == scan_id]
        return fold({}, events, four_eyes=self.four_eyes, track_unknown=True)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        sock.settimeout(30.0)
        daemon: Daemon = self.server.daemon           # type: ignore[attr-defined]
        try:
            cred = peer_credentials(sock)
        except PeerCredUnavailable as e:
            p.send(sock, p.error("not_permitted", str(e)))
            return
        while True:
            try:
                line = p.read_frame(sock)
            except p.ProtocolError as e:
                p.send(sock, p.error(e.code, e.detail))
                return
            if line is None:
                return
            try:
                req = p.decode(line)
            except p.ProtocolError as e:
                p.send(sock, p.error(e.code, e.detail))
                continue
            resp = daemon.handle(req, cred.uid)
            if not resp.get("ok"):
                log.info("refused op=%s uid=%s error=%s", req.get("op"), cred.uid,
                         resp.get("error"))
            p.send(sock, resp)


class LedgerdServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, socket_path: Path, daemon: Daemon) -> None:
        self.socket_path = Path(socket_path)
        self.daemon = daemon
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        super().__init__(str(self.socket_path), _Handler)
        os.chmod(self.socket_path, SOCKET_MODE)

    def server_close(self) -> None:
        super().server_close()
        self.socket_path.unlink(missing_ok=True)


def serve(socket_path: Path, ledger_path: Path, key_path: Path | None,
          policy: Policy, *, four_eyes: bool = True) -> LedgerdServer:
    backend = LedgerBackend.open(Path(ledger_path), Path(key_path) if key_path else None)
    server = LedgerdServer(Path(socket_path), Daemon(backend, policy, four_eyes=four_eyes))
    log.info("cva-ledgerd listening on %s (ledger=%s, read_only=%s)", socket_path,
             ledger_path, backend.read_only)
    return server


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="cva-ledgerd",
        description="The privilege-separation boundary: holds the signing key so the web "
                    "process never does (Module E plan D-E3).")
    ap.add_argument("--socket", default="/run/cva/ledgerd.sock")
    ap.add_argument("--ledger", required=True, help="an EXISTING ledger (cva-seal init)")
    ap.add_argument("--key", default=None,
                    help="the signing key file, mode 0600. Omit for a read-only daemon, "
                         "which serves the dashboard and refuses every write.")
    ap.add_argument("--policy", default=None,
                    help="JSON/YAML: {writers: {uid: [record types]}, readers: [uid], "
                         "same_uid_ok: bool}. Omitted means the single-uid development "
                         "policy, and the daemon says so at start-up.")
    ap.add_argument("--no-four-eyes", action="store_true",
                    help="record approve events but stop requiring them (policy, O3)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if a.policy:
        from .policy import from_mapping
        text = Path(a.policy).read_text()
        raw = json.loads(text) if text.lstrip().startswith("{") else _yaml(text)
        pol = from_mapping(raw)
    else:
        from .policy import development_policy
        pol = development_policy()
        log.warning("no --policy given: running the SINGLE-UID development policy. Every "
                    "process sharing uid %d may append analyst_event and scan_record. A "
                    "deployment with separate accounts must pass --policy.", os.getuid())

    server = serve(Path(a.socket), Path(a.ledger),
                   Path(a.key) if a.key else None, pol, four_eyes=not a.no_four_eyes)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.daemon.backend.close()
    return 0


def _yaml(text: str) -> dict[str, Any]:
    import yaml
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("the policy file must hold a mapping")
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
