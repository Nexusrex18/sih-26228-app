"""The states the fold produces. Pure data; no Flask, no ledger, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import DISPOSITION_RANK, decode_text


@dataclass(frozen=True)
class LedgerEvent:
    """One `analyst_event` as the fold sees it, with `seq` from the ledger."""

    seq: int
    actor_id: str
    role: str
    action: str
    scan_id: str
    target_type: str
    target_ref: str
    finding_id: str
    justification: str = ""
    new_disposition: str | None = None
    reason_code: str | None = None
    assignee: str | None = None
    refs_seq: int | None = None
    expected_prev_seq: int | None = None
    request_id: str = ""
    created_at_utc: str = ""

    @property
    def target_ref_text(self) -> str:
        return decode_text(self.target_ref)

    @property
    def justification_text(self) -> str:
        return decode_text(self.justification)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> LedgerEvent:
        a = record.get("analyst") or {}
        return cls(
            seq=int(record.get("seq", 0)),
            actor_id=str(a.get("actor_id", "")),
            role=str(a.get("role", "")),
            action=str(a.get("action", "")),
            scan_id=str(a.get("scan_id", "")),
            target_type=str(a.get("target_type", "")),
            target_ref=str(a.get("target_ref", "")),
            finding_id=str(a.get("finding_id", "")),
            justification=str(a.get("justification", "")),
            new_disposition=a.get("new_disposition"),
            reason_code=a.get("reason_code"),
            assignee=a.get("assignee"),
            refs_seq=a.get("refs_seq"),
            expected_prev_seq=a.get("expected_prev_seq"),
            request_id=str(a.get("request_id", "")),
            created_at_utc=str(record.get("created_at_utc", "")),
        )


@dataclass(frozen=True)
class PendingChange:
    """An override or release awaiting a second, different approver (D-E7)."""

    seq: int
    requested_by: str
    action: str
    new_disposition: str | None
    reason_code: str | None
    justification: str

    @property
    def justification_text(self) -> str:
        return decode_text(self.justification)


@dataclass
class FindingState:
    """Original (tool) · effective (after humans) · pending (awaiting a second person).

    `cited_seq` is the ledger record that produced `effective`. The UI never shows an
    effective state without it (plan §5.4).
    """

    finding_id: str
    original: str
    original_rule: str = ""
    effective: str = ""
    cited_seq: int | None = None
    pending: PendingChange | None = None
    superseded: list[PendingChange] = field(default_factory=list)
    assignee: str | None = None
    acknowledged_by: tuple[str, ...] = ()
    last_seq: int = 0
    events: list[LedgerEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.effective:
            self.effective = self.original

    @property
    def changed_by_human(self) -> bool:
        return self.effective != self.original

    @property
    def rank(self) -> int:
        return DISPOSITION_RANK.get(self.effective, 0)


@dataclass
class TargetState:
    """A contributor / batch / model / dataset held or released as a whole."""

    target_type: str
    target_ref: str
    status: str = "active"                  # active | quarantined
    cited_seq: int | None = None
    pending_release: PendingChange | None = None
    last_seq: int = 0
    events: list[LedgerEvent] = field(default_factory=list)

    @property
    def target_ref_text(self) -> str:
        return decode_text(self.target_ref)


@dataclass
class FoldResult:
    findings: dict[str, FindingState] = field(default_factory=dict)
    targets: dict[tuple[str, str], TargetState] = field(default_factory=dict)
    #: Events the fold saw but could not attach to a known finding — kept so the timeline
    #: never silently drops a record that exists in the ledger.
    orphan_events: list[LedgerEvent] = field(default_factory=list)
    last_seq: int = 0
    #: Every `override` and `release`, by its own `seq`. This is what an `approve` names.
    #:
    #: It is kept SEPARATELY from `FindingState.pending` because the two questions are not
    #: the same one. "Is this change still awaiting a second person?" needs the report's
    #: original disposition, which only `cva-web` has. "Is there a change at seq N that this
    #: actor did not request?" needs nothing but the ledger — and that is the question
    #: `cva-ledgerd` must answer, because it deliberately never reads `report.json` (doing so
    #: would make the key-holding daemon a second consumer of the report with an opinion of
    #: its own about a disposition, which D-E2 forbids).
    approvable: dict[int, LedgerEvent] = field(default_factory=dict)
    #: Seqs already blessed, so a second `approve` of the same change is refused.
    approved: dict[int, str] = field(default_factory=dict)

    @property
    def pending_count(self) -> int:
        return (sum(1 for s in self.findings.values() if s.pending)
                + sum(1 for t in self.targets.values() if t.pending_release))

    def worst_effective(self) -> str:
        return max((s.effective for s in self.findings.values()),
                   key=lambda d: DISPOSITION_RANK.get(d, 0), default="accept")


__all__ = ["FindingState", "FoldResult", "LedgerEvent", "PendingChange", "TargetState"]
