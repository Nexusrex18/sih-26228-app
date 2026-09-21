"""Effective-state fold (plan §5.4, Appendix A). Pure: inputs in, state out, no I/O.

Both `cva-web` and `cva-ledgerd` call it — the web process to render, the daemon to decide
whether an incoming request is legal against the history it already holds. One implementation
means the two can never disagree about what the ledger says.

The fold NEVER computes a disposition of its own (D-E2). It starts from the risk engine's
value and applies human events over it.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .events import DISPOSITION_RANK, TARGET_LEVEL_ACTIONS
from .states import FindingState, FoldResult, LedgerEvent, PendingChange, TargetState

#: Who may do what (plan §7.3). Policy — the team ratifies it (O3) — but the *shape* is fixed.
ROLE_ACTIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset(),
    "analyst": frozenset({"assign", "acknowledge", "override", "quarantine", "release"}),
    "approver": frozenset({"assign", "acknowledge", "override", "quarantine", "release",
                           "approve"}),
    "admin": frozenset(),
}

#: D-E9: `admin` manages accounts and cannot approve. Stated separately so a reader looking
#: for the rule finds it, rather than inferring it from an empty set above.
ADMIN_HAS_NO_WORKFLOW_RIGHTS = True


class FoldRejection(Exception):
    """Why an event may not apply. `code` is what the browser is shown (plan §5.3)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def raises(current: str, new: str) -> bool:
    return DISPOSITION_RANK.get(new, 0) > DISPOSITION_RANK.get(current, 0)


def _target_key(ev: LedgerEvent) -> tuple[str, str]:
    return (ev.target_type, ev.target_ref)


def _apply_finding_event(st: FindingState, ev: LedgerEvent, *, four_eyes: bool) -> None:
    st.events.append(ev)
    st.last_seq = max(st.last_seq, ev.seq)

    if ev.action == "assign":
        st.assignee = ev.assignee
    elif ev.action == "acknowledge":
        if ev.actor_id not in st.acknowledged_by:
            st.acknowledged_by = (*st.acknowledged_by, ev.actor_id)
    elif ev.action == "override":
        new = str(ev.new_disposition or st.effective)
        if raises(st.effective, new) or not four_eyes:
            st.effective = new
            st.cited_seq = ev.seq
            st.pending = None
        else:
            if st.pending is not None:
                st.superseded.append(st.pending)
            st.pending = PendingChange(seq=ev.seq, requested_by=ev.actor_id, action="override",
                                       new_disposition=new, reason_code=ev.reason_code,
                                       justification=ev.justification)
    elif ev.action == "approve":
        p = st.pending
        if p is None or ev.refs_seq != p.seq:
            # ledgerd refuses these, so reaching here means the ledger already holds one
            # (a pre-policy record, or a tampered file). Keep it in the timeline, do not act.
            return
        if ev.actor_id == p.requested_by:
            return
        st.effective = str(p.new_disposition or st.effective)
        st.cited_seq = ev.seq
        st.pending = None


def _apply_target_event(t: TargetState, ev: LedgerEvent, *, four_eyes: bool) -> None:
    t.events.append(ev)
    t.last_seq = max(t.last_seq, ev.seq)

    if ev.action == "quarantine":
        t.status = "quarantined"
        t.cited_seq = ev.seq
        t.pending_release = None
    elif ev.action == "release":
        if not four_eyes:
            t.status = "active"
            t.cited_seq = ev.seq
            t.pending_release = None
        else:
            t.pending_release = PendingChange(seq=ev.seq, requested_by=ev.actor_id,
                                              action="release", new_disposition=None,
                                              reason_code=ev.reason_code,
                                              justification=ev.justification)
    elif ev.action == "approve":
        p = t.pending_release
        if p is None or ev.refs_seq != p.seq or ev.actor_id == p.requested_by:
            return
        t.status = "active"
        t.cited_seq = ev.seq
        t.pending_release = None


def fold(originals: Mapping[str, tuple[str, str]],
         events: Iterable[LedgerEvent],
         *, four_eyes: bool = True, track_unknown: bool = False) -> FoldResult:
    """Fold `events` over the report's dispositions.

    `originals` maps `finding_id -> (disposition, disposition_rule)` exactly as the risk
    engine wrote them. Events are sorted by ledger `seq` here, so the result is order-stable
    however the caller supplied them (plan test 10.1).

    `track_unknown` is for `cva-ledgerd`, which has no report: it makes a state for any
    finding an event mentions so that `expected_prev_seq` still has something to compare
    against. `cva-web` leaves it False, so an event naming a finding the report does not
    contain lands in `orphan_events` and is shown as such rather than inventing a row.
    """
    result = FoldResult()
    for fid, (disp, rule) in originals.items():
        result.findings[fid] = FindingState(finding_id=fid, original=disp, original_rule=rule)

    for ev in sorted(events, key=lambda e: (e.seq, e.finding_id)):
        result.last_seq = max(result.last_seq, ev.seq)
        if ev.action in ("override", "release"):
            result.approvable[ev.seq] = ev
        elif ev.action == "approve" and ev.refs_seq is not None:
            result.approved[int(ev.refs_seq)] = ev.actor_id
        if ev.action in TARGET_LEVEL_ACTIONS or (
                ev.action == "approve"
                and int(ev.refs_seq or -1) in result.approvable
                and result.approvable[int(ev.refs_seq or -1)].action == "release"):
            key = _target_key(ev)
            t = result.targets.get(key)
            if t is None:
                t = TargetState(target_type=ev.target_type, target_ref=ev.target_ref)
                result.targets[key] = t
            _apply_target_event(t, ev, four_eyes=four_eyes)
            continue
        st = result.findings.get(ev.finding_id)
        if st is None:
            if not track_unknown:
                result.orphan_events.append(ev)
                continue
            st = FindingState(finding_id=ev.finding_id, original="accept")
            result.findings[ev.finding_id] = st
        _apply_finding_event(st, ev, four_eyes=four_eyes)
    return result


# --- the gate ledgerd applies before it signs -------------------------------------------------

def check_admissible(state: FoldResult, ev: LedgerEvent, *, four_eyes: bool = True,
                     findings_known: bool = True) -> None:
    """Raise `FoldRejection` unless `ev` may be recorded against `state`.

    Called by `cva-ledgerd` before it signs, and by `cva-web` before it asks, so the browser
    gets a reason it can act on rather than a bare refusal. Each rejection leaves the ledger
    untouched — Appendix A's "Recorded? no".

    `findings_known` is False for the daemon, which has no `report.json` and must not have
    one (D-E2). It can still enforce roles, staleness and four-eyes from the ledger alone;
    "is this a finding that exists?" is the web process's check, where the report lives.
    """
    allowed = ROLE_ACTIONS.get(ev.role, frozenset())
    if ev.action not in allowed:
        raise FoldRejection(
            "not_permitted",
            f"role '{ev.role}' may not '{ev.action}'"
            + (" — admin manages accounts and holds no workflow rights by default (D-E9), "
               "so that one person never controls both identity and decisions"
               if ev.role == "admin" else ""))

    is_target = ev.action in TARGET_LEVEL_ACTIONS
    st = state.findings.get(ev.finding_id)
    t = state.targets.get(_target_key(ev))

    if ev.expected_prev_seq is not None:
        seen = t.last_seq if (is_target and t is not None) else (st.last_seq if st else 0)
        if ev.expected_prev_seq != seen:
            raise FoldRejection(
                "stale_state",
                f"this was decided against event seq {ev.expected_prev_seq}, but seq {seen} "
                "is the latest one touching it. Reload, read what changed, and decide again.")

    if ev.action == "approve":
        refs = int(ev.refs_seq) if ev.refs_seq is not None else -1
        requested = state.approvable.get(refs)
        if requested is None:
            raise FoldRejection(
                "no_such_pending",
                f"no override or release at ledger seq {refs} that an approval could "
                "refer to")
        if requested.finding_id != ev.finding_id:
            raise FoldRejection(
                "no_such_pending",
                f"the change at seq {refs} is against finding {requested.finding_id}, not "
                f"{ev.finding_id}. An approval names the exact event it blesses.")
        if refs in state.approved:
            raise FoldRejection("no_such_pending",
                                f"the change at seq {refs} was already approved by "
                                f"{state.approved[refs]}")
        if four_eyes and requested.actor_id == ev.actor_id:
            raise FoldRejection(
                "four_eyes",
                "the analyst who requested a change may not approve it. Lowering a "
                "disposition or releasing a quarantine is the dangerous direction in an "
                "acceptance workflow, and it takes a second person.")

    if ev.action == "override":
        if findings_known and st is None:
            raise FoldRejection("unknown_finding",
                                f"finding {ev.finding_id} is not in the indexed report for "
                                f"scan {ev.scan_id}")
        if not str(ev.justification or "").strip():
            raise FoldRejection("justification_required",
                                "an override without a recorded justification is how "
                                "assurance systems fail in practice")

    if ev.action == "release" and (t is None or t.status != "quarantined"):
        raise FoldRejection("not_quarantined",
                            f"{ev.target_type} {ev.target_ref} is not under quarantine")


__all__ = ["ROLE_ACTIONS", "FoldRejection", "check_admissible", "fold", "raises"]
