"""The analyst audit trail read back from the ledger (plan §5.7, decision D14; gate C9).

`scan_record` and `analyst_event` share the inference chain, so a tamper with an analyst decision is caught by the
same arithmetic as a tamper with an inference. What the chain cannot say by itself is whether the *workflow* was
followed — whether an approval names a real override, whether the approver is a different person, whether two
analysts overwrote each other. `fold_analyst_events` replays the events in chain order and reports exactly that.

Declared limit (plan §14): `actor_id` is a tool-attributed string, not a signature by the analyst. Four-eyes here
means "two different actor ids", and is only as strong as the tool that assigned them.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Violation:
    code: str            # approve_without_override | self_approval | stale_write | request_id_conflict | bad_reference
    seq: int
    finding: str
    detail: str


@dataclass
class FindingState:
    scan_id: str
    finding_id: str
    last_seq: int = 0                                  # 0 = no analyst event yet (seq 0 is genesis, never an event)
    disposition: str | None = None
    assignee: str | None = None
    timeline: list[int] = field(default_factory=list)  # seqs of its events, in order
    overrides: dict[int, str] = field(default_factory=dict)      # override seq -> actor_id
    approved: dict[int, str] = field(default_factory=dict)       # override seq -> approver actor_id


@dataclass
class AuditFold:
    findings: dict[tuple[str, str], FindingState] = field(default_factory=dict)
    scans: list[int] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    request_ids: dict[str, int] = field(default_factory=dict)    # request_id -> seq of the event that used it

    def state(self, scan_id: str, finding_id: str) -> FindingState:
        return self.findings.setdefault((scan_id, finding_id), FindingState(scan_id, finding_id))

    @property
    def clean(self) -> bool:
        return not self.violations


def check_event(fold: AuditFold, seq: int, a: Mapping[str, Any]) -> list[Violation]:
    """The workflow rules for ONE event against the folded state so far (used to fold, and by `ledgerd` to refuse
    a write before it is signed). Returns violations; does not mutate `fold`."""
    out: list[Violation] = []
    st = fold.findings.get((a["scan_id"], a["finding_id"]))
    last = st.last_seq if st else 0
    fid = f"{a['scan_id']}/{a['finding_id']}"
    prior = fold.request_ids.get(a["request_id"])
    if prior is not None:
        out.append(Violation("request_id_conflict", seq, fid, f"request_id already used by the event at seq {prior}"))
    if a["action"] in ("override", "approve", "quarantine", "release") and "expected_prev_seq" in a and a["expected_prev_seq"] != last:
        out.append(Violation("stale_write", seq, fid, f"expected the finding's last event to be seq {a['expected_prev_seq']} "
                                                      f"but it is seq {last}: another decision landed first"))
    if a["action"] == "approve":
        ref = a.get("refs_seq")
        if st is None or ref not in st.overrides:
            out.append(Violation("approve_without_override", seq, fid, f"refs_seq {ref} is not an override of this finding"))
        elif st.overrides[ref] == a["actor_id"]:
            out.append(Violation("self_approval", seq, fid, f"actor '{a['actor_id']}' approved their own override (seq {ref})"))
    return out


def apply_event(fold: AuditFold, seq: int, a: Mapping[str, Any]) -> None:
    st = fold.state(a["scan_id"], a["finding_id"])
    st.last_seq = seq
    st.timeline.append(seq)
    fold.request_ids.setdefault(a["request_id"], seq)
    if a["action"] == "override":
        st.overrides[seq] = a["actor_id"]
        st.disposition = a["new_disposition"]
    elif a["action"] == "approve" and a.get("refs_seq") in st.overrides:
        st.approved[a["refs_seq"]] = a["actor_id"]
    elif a["action"] == "quarantine":
        st.disposition = "quarantine"
    elif a["action"] == "release":
        st.disposition = "accept"
    elif a["action"] == "assign":
        st.assignee = a.get("assignee")


def fold_analyst_events(records: Iterable[Mapping[str, Any]]) -> AuditFold:
    """Replay `scan_record` and `analyst_event` records (parsed, in chain order) and report workflow violations."""
    fold = AuditFold()
    for rec in records:
        if rec["type"] == "scan_record":
            fold.scans.append(rec["seq"])
        elif rec["type"] == "analyst_event":
            a = rec["analyst"]
            fold.violations.extend(check_event(fold, rec["seq"], a))
            apply_event(fold, rec["seq"], a)
    return fold
