"""Plan test 10.1: the fold, property-tested. Appendix A is the specification.

The five properties, quoted: an approve never activates without a different approver; a raise
is immediate; a lower is pending; stale writes never apply; replaying the same events yields
the same state; and the fold is order-stable by ledger `seq`.
"""
from __future__ import annotations

import pytest

from cva.web.workflow.events import DISPOSITION_RANK, encode_text
from cva.web.workflow.fold import FoldRejection, check_admissible, fold
from cva.web.workflow.states import LedgerEvent

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

DISPOSITIONS = ("accept", "review", "quarantine")
ACTORS = ("a.sharma", "b.rao", "c.menon")
FID = "0123456789abcdef"


def _ev(seq: int, actor: str, action: str, **kw) -> LedgerEvent:
    return LedgerEvent(seq=seq, actor_id=actor, role=kw.pop("role", "analyst"),
                       action=action, scan_id="s-2026-09-19-0001", target_type="sample",
                       target_ref=encode_text("img.jpg"), finding_id=kw.pop("fid", FID),
                       justification=encode_text(kw.pop("justification", "because")), **kw)


@st.composite
def event_sequences(draw):
    """Plausible histories: overrides, approvals of real seqs, assignments, acknowledgements."""
    n = draw(st.integers(min_value=0, max_value=12))
    events: list[LedgerEvent] = []
    override_seqs: list[int] = []
    for seq in range(1, n + 1):
        actor = draw(st.sampled_from(ACTORS))
        action = draw(st.sampled_from(
            ["override", "acknowledge", "assign"]
            + (["approve"] if override_seqs else [])))
        if action == "override":
            events.append(_ev(seq, actor, "override",
                              new_disposition=draw(st.sampled_from(DISPOSITIONS))))
            override_seqs.append(seq)
        elif action == "approve":
            events.append(_ev(seq, actor, "approve", role="approver",
                              refs_seq=draw(st.sampled_from(override_seqs))))
        elif action == "assign":
            events.append(_ev(seq, actor, "assign", assignee=draw(st.sampled_from(ACTORS))))
        else:
            events.append(_ev(seq, actor, "acknowledge"))
    return events


@settings(max_examples=250, deadline=None)
@given(original=st.sampled_from(DISPOSITIONS), events=event_sequences())
def test_an_approve_never_activates_a_change_its_own_requester_made(original, events):
    state = fold({FID: (original, "D3")}, events)
    f = state.findings[FID]
    for ev in f.events:
        if ev.action != "approve":
            continue
        requested = next((e for e in f.events if e.seq == ev.refs_seq), None)
        if requested is not None and requested.actor_id == ev.actor_id:
            assert f.cited_seq != ev.seq, (
                "a self-approval activated a change; four-eyes is the one rule that must "
                "hold for every history the ledger can contain")


@settings(max_examples=250, deadline=None)
@given(original=st.sampled_from(DISPOSITIONS), events=event_sequences())
def test_the_effective_state_is_always_cited(original, events):
    f = fold({FID: (original, "D3")}, events).findings[FID]
    if f.effective != f.original:
        assert f.cited_seq is not None, \
            "the UI never shows an effective state without the event that produced it"
        assert any(e.seq == f.cited_seq for e in f.events)


@settings(max_examples=250, deadline=None)
@given(original=st.sampled_from(DISPOSITIONS), events=event_sequences())
def test_the_fold_is_order_stable_by_seq(original, events):
    import random

    shuffled = list(events)
    random.Random(len(events)).shuffle(shuffled)
    a = fold({FID: (original, "D3")}, events).findings[FID]
    b = fold({FID: (original, "D3")}, shuffled).findings[FID]
    assert (a.effective, a.cited_seq, a.assignee) == (b.effective, b.cited_seq, b.assignee)


@settings(max_examples=250, deadline=None)
@given(original=st.sampled_from(DISPOSITIONS), events=event_sequences())
def test_replaying_the_same_events_yields_the_same_state(original, events):
    a = fold({FID: (original, "D3")}, events).findings[FID]
    b = fold({FID: (original, "D3")}, list(events)).findings[FID]
    assert (a.effective, a.cited_seq, a.assignee, a.acknowledged_by) == \
           (b.effective, b.cited_seq, b.assignee, b.acknowledged_by)


@settings(max_examples=200, deadline=None)
@given(original=st.sampled_from(DISPOSITIONS), target=st.sampled_from(DISPOSITIONS))
def test_a_raise_is_immediate_and_a_lower_is_pending(original, target):
    ev = _ev(1, "a.sharma", "override", new_disposition=target)
    f = fold({FID: (original, "D3")}, [ev]).findings[FID]
    if DISPOSITION_RANK[target] > DISPOSITION_RANK[original]:
        assert f.effective == target and f.pending is None
    elif DISPOSITION_RANK[target] < DISPOSITION_RANK[original]:
        assert f.effective == original and f.pending is not None
    else:
        assert f.effective == original


@settings(max_examples=200, deadline=None)
@given(seen=st.integers(min_value=0, max_value=20),
       claimed=st.integers(min_value=0, max_value=20))
def test_a_stale_write_never_applies(seen, claimed):
    history = [_ev(s, "a.sharma", "acknowledge") for s in range(1, seen + 1)]
    state = fold({FID: ("quarantine", "D3")}, history)
    ev = _ev(seen + 1, "b.rao", "override", new_disposition="review")
    ev = LedgerEvent(**{**ev.__dict__, "expected_prev_seq": claimed})
    if claimed == seen:
        check_admissible(state, ev)
    else:
        with pytest.raises(FoldRejection) as exc:
            check_admissible(state, ev)
        assert exc.value.code == "stale_state"


def test_a_newer_override_supersedes_an_earlier_pending_one_without_losing_it():
    """Appendix A: 'earlier pending superseded (kept in the timeline)'."""
    events = [_ev(1, "a.sharma", "override", new_disposition="review"),
              _ev(2, "b.rao", "override", new_disposition="accept")]
    f = fold({FID: ("quarantine", "D3")}, events).findings[FID]
    assert f.pending is not None and f.pending.seq == 2
    assert [p.seq for p in f.superseded] == [1]
    assert [e.seq for e in f.events] == [1, 2], "nothing is dropped from the timeline"


def test_an_event_for_a_finding_the_report_does_not_contain_is_never_invented():
    events = [_ev(1, "a.sharma", "override", fid="f" * 16, new_disposition="accept")]
    state = fold({FID: ("review", "D5")}, events)
    assert FID in state.findings and len(state.findings) == 1
    assert [e.seq for e in state.orphan_events] == [1], \
        "an event the report cannot explain is shown as such, never rendered as a row"


def test_with_four_eyes_disabled_a_lowering_is_immediate_and_still_recorded():
    """O3 is policy, not contract: the thresholds change without the fold changing shape."""
    ev = _ev(1, "a.sharma", "override", new_disposition="accept")
    f = fold({FID: ("quarantine", "D3")}, [ev], four_eyes=False).findings[FID]
    assert f.effective == "accept" and f.pending is None and f.cited_seq == 1
    assert len(f.events) == 1
