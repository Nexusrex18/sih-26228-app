"""`cva-ledgerd`: the only process that holds the signing key (Module E decision D-E3; plan §5.7; gate C9).

The web app parses attacker-influenced strings for a living. If it is compromised it must be able to spam requests
and nothing else. So it never touches the key: it sends a typed append request over a Unix socket, and this daemon

  1. reads the peer's uid from the kernel (SO_PEERCRED — not from anything the client claims),
  2. checks that uid's record-type allowlist (web uid -> analyst_event; scanner uid -> scan_record; nobody may write
     genesis, key_rotation, checkpoint, anchor_event, model_registration, inference or degraded_marker through here),
  3. re-validates the record against its schema AND against the folded workflow state (an approval must name a real
     override by a different actor; a write based on a stale view is refused; a retried request_id is idempotent),
  4. and only then signs and appends, through the ordinary `SealedLedger.append` least-privilege path.

`Ledgerd.handle` is the whole policy as a plain function (testable without sockets); `serve` puts it on a socket.
Requests and replies are one JSON object per line.
"""
from __future__ import annotations

import json
import os
import socket
import socketserver
import struct
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditFold, apply_event, check_event
from .canonical import parse_strict
from .errors import InvalidRecord, LedgerUnavailable, NonCanonical, SealError
from .store import AUDIT_APPEND_TYPES, SealedLedger

NEVER_VIA_DAEMON = ("genesis", "key_rotation", "checkpoint", "anchor_event", "model_registration", "inference", "degraded_marker")


@dataclass
class Policy:
    """uid -> the record types that uid may request. Anything not listed is denied."""

    allow: dict[int, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for uid, types in self.allow.items():
            bad = set(types) - set(AUDIT_APPEND_TYPES)
            if bad:
                raise ValueError(f"uid {uid} may not be granted {sorted(bad)}: only {AUDIT_APPEND_TYPES} go through the daemon")

    def permits(self, uid: int | None, rtype: str) -> bool:
        return uid is not None and rtype in self.allow.get(uid, frozenset())


class Ledgerd:
    def __init__(self, ledger: SealedLedger, policy: Policy) -> None:
        self._ledger = ledger
        self.policy = policy
        self._lock = threading.Lock()
        self.fold = AuditFold()
        self._results: dict[str, tuple[dict[str, Any], str]] = {}   # request_id -> (reply given, digest of what was asked)
        self._replay()

    def _replay(self) -> None:
        """Rebuild the workflow state from the chain: the ledger is the only source of truth, so a restart loses nothing."""
        from .audit import fold_analyst_events
        self.fold = fold_analyst_events(parse_strict(d) for d in self._ledger.stored_records())
        for rec in (parse_strict(d) for d in self._ledger.stored_records()):
            if rec["type"] == "analyst_event":
                self._results[rec["analyst"]["request_id"]] = ({"ok": True, "seq": rec["seq"], "replayed": True},
                                                               json.dumps(rec["analyst"], sort_keys=True))

    @staticmethod
    def _err(code: str, detail: str) -> dict[str, Any]:
        return {"ok": False, "error": code, "detail": detail}

    def handle(self, request: Mapping[str, Any], peer_uid: int | None) -> dict[str, Any]:
        rtype = request.get("type")
        body = request.get("body")
        if not isinstance(rtype, str) or not isinstance(body, dict):
            return self._err("bad_request", "a request is {\"type\": <str>, \"body\": <object>}")
        if rtype in NEVER_VIA_DAEMON:
            return self._err("forbidden_type", f"{rtype} is never written through the daemon")
        if rtype not in AUDIT_APPEND_TYPES:
            return self._err("unknown_type", f"unknown record type {rtype!r}")
        if not self.policy.permits(peer_uid, rtype):
            return self._err("not_permitted", f"uid {peer_uid} may not request {rtype}")
        with self._lock:
            if rtype == "analyst_event":
                a = body.get("analyst")
                if not isinstance(a, dict):
                    return self._err("invalid", "analyst_event needs an 'analyst' section")
                rid = a.get("request_id")
                if isinstance(rid, str) and rid in self._results:
                    reply0, asked = self._results[rid]
                    if asked != json.dumps(a, sort_keys=True):
                        return self._err("request_id_conflict", "that request_id was already used for a different request")
                    return {**reply0, "idempotent": True}
            record = {"type": rtype, **body}
            try:
                self._validate(record)
            except InvalidRecord as e:
                return self._err("invalid", str(e))
            if rtype == "analyst_event":
                # the request_id retry case was answered above; anything else in `check_event` is a real violation
                fresh = AuditFold(findings=self.fold.findings, request_ids={})       # request ids are handled by idempotency
                bad = check_event(fresh, self._ledger.size(), record["analyst"])
                if bad:
                    return self._err(bad[0].code, bad[0].detail)
            try:
                # `append_typed` reports every record the call wrote — including a checkpoint the cadence triggers AFTER
                # ours — so the seq we hand back (and fold into workflow state) is the event's own, never size()-1.
                written = self._ledger.append_typed(rtype, {k: v for k, v in record.items() if k != "type"})
            except (InvalidRecord, NonCanonical, ValueError) as e:
                return self._err("invalid", str(e))
            except LedgerUnavailable as e:
                return self._err("ledger_unavailable", str(e))
            except SealError as e:
                return self._err("refused", str(e))
            mine = next(w for w in written if w.type == rtype)
            seq, rh = mine.seq, mine.record_hash
            reply: dict[str, Any] = {"ok": True, "seq": seq, "record_hash": rh}
            if rtype == "analyst_event":
                apply_event(self.fold, seq, record["analyst"])
                self.fold.request_ids[record["analyst"]["request_id"]] = seq
                self._results[record["analyst"]["request_id"]] = ({"ok": True, "seq": seq, "record_hash": rh},
                                                                  json.dumps(record["analyst"], sort_keys=True))
            else:
                self.fold.scans.append(seq)
            return reply

    @staticmethod
    def _validate(record: Mapping[str, Any]) -> None:
        from .records import SECTIONS, validate_section
        rtype = record["type"]
        extra = set(record) - {"type"} - set(SECTIONS[rtype])
        if extra or any(s not in record for s in SECTIONS[rtype]):
            raise InvalidRecord("$", f"{rtype} takes exactly the sections {list(SECTIONS[rtype])}")
        for s in SECTIONS[rtype]:
            validate_section(s, record[s])


def peer_uid(sock: socket.socket) -> int | None:
    """The connecting process's uid, from the kernel (SO_PEERCRED). None where the platform cannot say."""
    try:
        pid, uid, gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        return int(uid)
    except (OSError, AttributeError):
        return None


MAX_LINE = 64 * 1024            # one request; the largest legitimate one (a justification) is ~8 KiB
MAX_CONNECTIONS = 64            # concurrent clients; more are told "busy" and dropped, not queued without bound
IDLE_TIMEOUT = 10.0             # seconds a client may hold a connection without completing a request


class _Handler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        self.timeout = self.server.idle_timeout              # type: ignore[attr-defined,misc]
        super().setup()

    def handle(self) -> None:
        daemon: Ledgerd = self.server.daemon               # type: ignore[attr-defined]
        uid = peer_uid(self.request)
        try:
            while True:
                # bounded read: a client that streams bytes and never sends a newline cannot grow this process's memory
                line = self.rfile.readline(MAX_LINE + 1)
                if not line:
                    return
                if len(line) > MAX_LINE and not line.endswith(b"\n"):
                    self._send(Ledgerd._err("request_too_large", f"a request is at most {MAX_LINE} bytes"))
                    return
                try:
                    req = json.loads(line)
                    reply = daemon.handle(req, uid) if isinstance(req, dict) else Ledgerd._err("bad_request", "not an object")
                except ValueError:
                    reply = Ledgerd._err("bad_request", "not JSON")
                self._send(reply)
        except (TimeoutError, OSError):
            return                                             # idle or vanished client: drop it, say nothing

    def _send(self, reply: Mapping[str, Any]) -> None:
        self.wfile.write(json.dumps(reply, separators=(",", ":")).encode("ascii") + b"\n")


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128          # the default of 5 refuses (EAGAIN) a burst of analysts on a Unix socket

    def __init__(self, path: str, daemon: Ledgerd, *, max_connections: int = MAX_CONNECTIONS,
                 idle_timeout: float = IDLE_TIMEOUT) -> None:
        if os.path.exists(path):
            os.unlink(path)
        self._slots = threading.BoundedSemaphore(max_connections)
        self.idle_timeout = idle_timeout
        old = os.umask(0o117)                                # the socket is CREATED 0660, never briefly world-accessible
        try:
            super().__init__(path, _Handler)
        finally:
            os.umask(old)
        self.daemon = daemon
        self.socket_path = path

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(b'{"ok":false,"error":"busy","detail":"too many concurrent connections"}\n')
            except OSError:
                pass
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def serve(path: str, daemon: Ledgerd, **limits: Any) -> Server:
    """Start serving in a background thread; returns the server (`shutdown()` + `server_close()` to stop).
    `limits`: `max_connections`, `idle_timeout`."""
    srv = Server(path, daemon, **limits)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def request(path: str, req: Mapping[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    """A minimal client: one request, one reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall(json.dumps(req, separators=(",", ":")).encode("ascii") + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)          # type: ignore[no-any-return]
