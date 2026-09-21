"""The audit trail: every record in ledger order (plan §7.4).

Ordering is by `seq`, never by time. The host clock is untrusted and every displayed time
says so — a clock that steps backwards changes what is *shown*, never what is *ordered*.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..workflow.events import AUDIT_FLAG_PREFIX, decode_text


@dataclass(frozen=True)
class TimelineRow:
    seq: int
    kind: str                       # scan_record | analyst_event | other
    created_at_utc: str
    actor: str
    role: str
    action: str
    target: str
    detail: str
    justification: str = ""
    reason_code: str = ""
    refs_seq: int | None = None
    scan_id: str = ""
    audit_flagged: bool = False
    key_id: str = ""

    @property
    def is_decision(self) -> bool:
        return self.kind == "analyst_event"


def rows(records: Iterable[Mapping[str, Any]]) -> list[TimelineRow]:
    out: list[TimelineRow] = []
    for r in records:
        seq = int(r.get("seq", 0))
        kind = str(r.get("type", "other"))
        when = str(r.get("created_at_utc", ""))
        key_id = str(r.get("key_id", ""))
        if kind == "analyst_event":
            a = r.get("analyst") or {}
            just = decode_text(str(a.get("justification", "")))
            flagged = just.startswith(AUDIT_FLAG_PREFIX)
            out.append(TimelineRow(
                seq=seq, kind=kind, created_at_utc=when,
                actor=str(a.get("actor_id", "")), role=str(a.get("role", "")),
                action=str(a.get("action", "")),
                target=f"{a.get('target_type', '')} {decode_text(str(a.get('target_ref', '')))}",
                detail=str(a.get("new_disposition") or ""),
                justification=just[len(AUDIT_FLAG_PREFIX):] if flagged else just,
                reason_code=str(a.get("reason_code") or ""),
                refs_seq=a.get("refs_seq"), scan_id=str(a.get("scan_id", "")),
                audit_flagged=flagged, key_id=key_id))
        elif kind == "scan_record":
            s = r.get("scan") or {}
            counts = s.get("finding_counts") or {}
            out.append(TimelineRow(
                seq=seq, kind=kind, created_at_utc=when, actor="cva scan", role="tool",
                action="scan sealed", target=str(s.get("scan_id", "")),
                detail=f"report {str(s.get('report_sha256', ''))[:16]}…, "
                       + (", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                          or "no findings"),
                scan_id=str(s.get("scan_id", "")), key_id=key_id))
        else:
            out.append(TimelineRow(seq=seq, kind="other", created_at_utc=when,
                                   actor="—", role="—", action=kind,
                                   target="", detail="", key_id=key_id))
    return sorted(out, key=lambda r: r.seq)


def decision_count(rows_: Iterable[TimelineRow]) -> int:
    return sum(1 for r in rows_ if r.is_decision)


__all__ = ["TimelineRow", "decision_count", "rows"]
