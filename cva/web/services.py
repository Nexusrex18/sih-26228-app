"""Per-request access to the report index, the ledger and the verification state.

One place that knows how to assemble "what does this scan look like right now", so no view
re-derives it — and so the ledger being unreachable degrades in exactly one way everywhere.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from flask import current_app

from .audit.verify_bridge import VerifyState, verify
from .config import WebConfig
from .reports import seal_check
from .reports.index import ReportIndex
from .reports.loader import LoadedReport, load
from .workflow.fold import fold
from .workflow.ledgerd_client import (
    LedgerdClient,
    LedgerRefused,
    LedgerStatus,
    LedgerUnreachable,
)
from .workflow.states import FoldResult, LedgerEvent


@dataclass(frozen=True)
class LedgerHealth:
    """What the ledger-trust banner shows on every page (plan §5.4).

    The workflow is read-only unless `writable` — and `writable` requires BOTH that the
    daemon answers with a signing key AND that verification has passed. A fold we cannot
    verify is not a record anyone may act on.
    """

    reachable: bool
    detail: str
    status: LedgerStatus | None = None
    verify_state: VerifyState | None = None

    @property
    def verified(self) -> bool:
        return bool(self.verify_state and self.verify_state.ok)

    @property
    def writable(self) -> bool:
        return (self.reachable and self.status is not None and self.status.can_sign
                and not self.status.read_only and self.verified)

    @property
    def banner_kind(self) -> str:
        if not self.reachable:
            return "unavailable"
        if self.verify_state is None or self.verify_state.state == "UNAVAILABLE":
            return "unverified"
        if not self.verify_state.ok:
            return "failed"
        return "verified"

    @property
    def banner_text(self) -> str:
        if not self.reachable:
            return ("Workflow unavailable — the ledger is not writable. "
                    "No decision can be recorded, so none is offered.")
        if self.verify_state is None or self.verify_state.state == "UNAVAILABLE":
            return (f"Audit ledger NOT VERIFIED — {self.verify_state.detail if self.verify_state else 'no verification has run'}. "
                    "Recorded decisions are shown but cannot currently be trusted.")
        if not self.verify_state.ok:
            return (f"Audit ledger verification FAILED — {self.verify_state.summary}. "
                    "The workflow is read-only and every recorded decision below is "
                    "marked unverified.")
        return (f"Audit ledger verified at seq {self.verify_state.records_checked} "
                f"({self.verify_state.checked_at_label}).")


class VerifyCache:
    """The verify bridge shells out; doing that per request would make triage crawl.

    The result is cached with the ledger's size as part of the key, so an appended record
    invalidates it immediately. A stale "verified" badge is the one thing this cache must
    never produce.
    """

    def __init__(self, ttl_s: float = 120.0) -> None:
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._value: tuple[int, float, VerifyState] | None = None

    def get(self, cfg: WebConfig, ledger_size: int) -> VerifyState:
        now = time.monotonic()
        with self._lock:
            if self._value is not None:
                size, at, state = self._value
                if size == ledger_size and now - at < self.ttl_s:
                    return state
        state = verify(cfg.ledger_path, cfg.trust_root, timeout_s=cfg.subprocess_timeout_s)
        with self._lock:
            self._value = (ledger_size, now, state)
        return state

    def invalidate(self) -> None:
        with self._lock:
            self._value = None


def config() -> WebConfig:
    return current_app.config["CVA"]


def index() -> ReportIndex:
    return current_app.extensions["cva_index"]


def client() -> LedgerdClient:
    return current_app.extensions["cva_ledgerd"]


def verify_cache() -> VerifyCache:
    return current_app.extensions["cva_verify"]


def ledger_health() -> LedgerHealth:
    """Asked once per request by the base template. Never raises."""
    try:
        status = client().status()
    except (LedgerUnreachable, LedgerRefused) as e:
        return LedgerHealth(reachable=False, detail=str(e))
    state = verify_cache().get(config(), status.size)
    return LedgerHealth(reachable=True, detail="", status=status, verify_state=state)


def ledger_records(scan_id: str | None = None,
                   types: tuple[str, ...] = ("analyst_event",)) -> list[dict[str, Any]] | None:
    """`None` means "could not read", which is NOT the same as "there are none"."""
    try:
        return client().records(types=list(types), scan_id=scan_id)
    except (LedgerUnreachable, LedgerRefused):
        return None


@dataclass
class ScanView:
    """A report plus its folded human state plus its seal state — what a page renders."""

    report: LoadedReport
    state: FoldResult
    seal: seal_check.SealState
    events_available: bool

    @property
    def scan_id(self) -> str:
        return self.report.scan_id

    def finding(self, finding_id: str) -> dict[str, Any] | None:
        return self.report.findings_by_id.get(finding_id)

    def state_of(self, finding_id: str):
        return self.state.findings.get(finding_id)


def load_scan(scan_id: str) -> ScanView:
    report = load(config().reports_dir, scan_id)
    analyst = ledger_records(scan_id=scan_id, types=("analyst_event",))
    all_scan_records = ledger_records(scan_id=scan_id, types=("scan_record",))
    state = fold(report.originals(),
                 [LedgerEvent.from_record(r) for r in (analyst or [])],
                 four_eyes=config().four_eyes)
    seal = seal_check.check(report.sha256, scan_id, all_scan_records,
                            ledger_available=all_scan_records is not None)
    return ScanView(report=report, state=state, seal=seal, events_available=analyst is not None)


__all__ = ["LedgerHealth", "ScanView", "VerifyCache", "client", "config", "index",
           "ledger_health", "ledger_records", "load_scan", "verify_cache"]
