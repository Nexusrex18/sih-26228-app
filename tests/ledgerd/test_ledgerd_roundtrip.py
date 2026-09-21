"""Gate N3/N4: every action round-trips through ledgerd to a real `SealedLedger`."""
from __future__ import annotations

import pytest

from cva.web.workflow import events as E
from cva.web.workflow.fold import fold
from cva.web.workflow.ledgerd_client import LedgerRefused, LedgerUnreachable
from cva.web.workflow.states import LedgerEvent

SCAN = "s-2026-09-19-0001"
REF = "img_04471.jpg"
FID = E.synthetic_finding_id("sample", REF)


def _override(actor="a.sharma", **kw):
    kw.setdefault("justification",
                  "Duplicate cluster confirmed as a scanner re-export; the batch note "
                  "records the same finding under a different id.")
    return E.build(actor_id=actor, role="analyst", action="override", scan_id=SCAN,
                   target_type="sample", target_ref=REF, finding_id=FID,
                   new_disposition=kw.pop("new_disposition", "review"),
                   reason_code=kw.pop("reason_code", "quality_issue"),
                   expected_prev_seq=kw.pop("expected_prev_seq", 0), **kw)


def _approve(actor, refs_seq):
    return E.build(actor_id=actor, role="approver", action="approve", scan_id=SCAN,
                   target_type="sample", target_ref=REF, finding_id=FID, refs_seq=refs_seq)


def _state(client, original="quarantine"):
    records = client.records(types=["analyst_event"], scan_id=SCAN)
    return fold({FID: (original, "D3")}, [LedgerEvent.from_record(r) for r in records])


def test_status_reports_a_real_signing_key(client):
    st = client.status()
    assert st.can_sign, "the daemon holds the key, so SIGNING_KEY must be reported"
    assert not st.read_only


def test_lowering_is_pending_until_a_different_approver_acts(client):
    written = client.append_analyst_event(_override().as_wire())
    st = _state(client)
    f = st.findings[FID]
    assert f.effective == "quarantine", "a lowering override must not take effect alone"
    assert f.pending is not None and f.pending.seq == written.seq

    client.append_analyst_event(_approve("b.rao", written.seq).as_wire())
    f = _state(client).findings[FID]
    assert f.effective == "review"
    assert f.pending is None
    assert f.cited_seq is not None, "an effective state is never shown without its citation"


def test_approving_your_own_override_is_refused_and_unrecorded(client):
    written = client.append_analyst_event(_override().as_wire())
    before = len(client.records(types=["analyst_event"]))
    with pytest.raises(LedgerRefused) as exc:
        client.append_analyst_event(_approve("a.sharma", written.seq).as_wire())
    assert exc.value.code == "four_eyes"
    assert len(client.records(types=["analyst_event"])) == before, \
        "Appendix A: a refused approve is NOT recorded"
    assert _state(client).findings[FID].effective == "quarantine"


def test_raising_a_disposition_is_immediate_and_needs_no_approval(client):
    w = client.append_analyst_event(
        _override(new_disposition="quarantine", expected_prev_seq=0).as_wire())
    f = _state(client, original="review").findings[FID]
    assert f.effective == "quarantine"
    assert f.pending is None
    assert f.cited_seq == w.seq


def test_stale_expected_prev_seq_is_refused(client):
    client.append_analyst_event(_override().as_wire())
    with pytest.raises(LedgerRefused) as exc:
        client.append_analyst_event(
            _override(actor="b.rao", expected_prev_seq=0,
                      justification="A second analyst acting on the state they last saw, "
                                    "which is no longer the current one.").as_wire())
    assert exc.value.code == "stale_state"


def test_idempotency_a_retried_request_records_once(client):
    req = _override()
    first = client.append_analyst_event(req.as_wire())
    second = client.append_analyst_event(req.as_wire())
    assert second.deduped and second.seq == first.seq
    assert len(client.records(types=["analyst_event"])) == 1


def test_idempotency_survives_a_daemon_restart(ledger_dir, ledger_path, signing_key, client):
    """Plan test 10.5: restart, retry with the same request_id, record ONCE."""
    import threading

    from cva.ledgerd.policy import development_policy
    from cva.ledgerd.server import serve
    from cva.web.workflow.ledgerd_client import LedgerdClient

    req = _override()
    first = client.append_analyst_event(req.as_wire())

    sock2 = ledger_dir / "ledgerd2.sock"
    server = serve(sock2, ledger_path, signing_key, development_policy())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        again = LedgerdClient(sock2).append_analyst_event(req.as_wire())
        assert again.deduped and again.seq == first.seq
        assert len(LedgerdClient(sock2).records(types=["analyst_event"])) == 1
    finally:
        server.shutdown()
        server.server_close()
        server.daemon.backend.close()
        thread.join(timeout=5)


def test_unicode_justification_round_trips(client):
    """Plan test 10.4: the ledger is ASCII-only; the analyst's prose is not."""
    prose = ("Confirmé par l'équipe — доступ к исходному батчу подтверждён; "
             "重复簇来自扫描仪重新导出。")
    client.append_analyst_event(_override(justification=prose).as_wire())
    stored = client.records(types=["analyst_event"])[0]["analyst"]["justification"]
    assert stored.isascii(), "Module C §5.1: printable ASCII only on the wire"
    ev = LedgerEvent.from_record(client.records(types=["analyst_event"])[0])
    assert ev.justification_text == prose


def test_an_override_with_no_justification_never_reaches_the_socket():
    with pytest.raises(E.EventError) as exc:
        _override(justification="   ")
    assert exc.value.code == "justification_required"


def test_punctuation_is_not_a_justification():
    with pytest.raises(E.EventError) as exc:
        _override(justification="." * 60)
    assert exc.value.code == "justification_not_prose"


def test_prov_findings_offer_only_the_enumerated_reason_codes():
    assert E.allowed_reason_codes(True) == E.PROV_REASON_CODES
    with pytest.raises(E.EventError) as exc:
        E.build(actor_id="a.sharma", role="analyst", action="override", scan_id=SCAN,
                target_type="record", target_ref="seq:412",
                finding_id=E.synthetic_finding_id("record", "seq:412"),
                new_disposition="review", reason_code="quality_issue",
                justification="The hash did not match but the data seemed fine to me.",
                expected_prev_seq=0, is_prov_finding=True)
    assert exc.value.code == "invalid_reason_code"


def test_false_positive_confirmed_on_a_prov_finding_is_audit_flagged():
    req = E.build(actor_id="a.sharma", role="analyst", action="override", scan_id=SCAN,
                  target_type="record", target_ref="seq:412",
                  finding_id=E.synthetic_finding_id("record", "seq:412"),
                  new_disposition="review", reason_code="false_positive_confirmed",
                  justification="Reconstructed the record from the field unit's own copy; "
                                "the canonical encoding differed only in key order.",
                  expected_prev_seq=0, is_prov_finding=True)
    assert E.decode_text(req.body["justification"]).startswith(E.AUDIT_FLAG_PREFIX)


def test_admin_holds_no_workflow_rights(client):
    with pytest.raises(LedgerRefused) as exc:
        client.append_analyst_event(
            E.build(actor_id="root.admin", role="admin", action="override", scan_id=SCAN,
                    target_type="sample", target_ref=REF, finding_id=FID,
                    new_disposition="accept", reason_code="accepted_risk",
                    justification="An account administrator should not be able to wave a "
                                  "finding through on their own authority.",
                    expected_prev_seq=0).as_wire())
    assert exc.value.code == "not_permitted"


def test_target_release_needs_a_second_approver(client):
    ref = "survey_team_north"
    fid = E.synthetic_finding_id("contributor", ref)
    q = E.build(actor_id="a.sharma", role="analyst", action="quarantine", scan_id=SCAN,
                target_type="contributor", target_ref=ref, finding_id=fid)
    client.append_analyst_event(q.as_wire())
    rel = E.build(actor_id="a.sharma", role="analyst", action="release", scan_id=SCAN,
                  target_type="contributor", target_ref=ref, finding_id=fid,
                  justification="The contributor supplied a signed manifest for the "
                                "affected batch and the duplicates are accounted for.")
    w = client.append_analyst_event(rel.as_wire())

    records = client.records(types=["analyst_event"], scan_id=SCAN)
    st = fold({}, [LedgerEvent.from_record(r) for r in records])
    target = st.targets[("contributor", E.encode_text(ref))]
    assert target.status == "quarantined", "a release is pending until approved"
    assert target.pending_release is not None

    client.append_analyst_event(
        E.build(actor_id="b.rao", role="approver", action="approve", scan_id=SCAN,
                target_type="contributor", target_ref=ref, finding_id=fid,
                refs_seq=w.seq).as_wire())
    records = client.records(types=["analyst_event"], scan_id=SCAN)
    st = fold({}, [LedgerEvent.from_record(r) for r in records])
    assert st.targets[("contributor", E.encode_text(ref))].status == "active"


def test_the_web_uid_may_not_append_a_genesis_record(client):
    """S10, by construction: `genesis` is not an op this protocol has."""
    from cva.ledgerd import protocol as p
    with pytest.raises(p.ProtocolError):
        p.request("append_genesis")


def test_fail_closed_when_the_daemon_is_gone(ledger_dir):
    from cva.web.workflow.ledgerd_client import LedgerdClient
    client = LedgerdClient(ledger_dir / "no-such.sock")
    with pytest.raises(LedgerUnreachable):
        client.append_analyst_event(_override().as_wire())


def test_a_killed_daemon_leaves_the_state_unchanged(ledger_dir, ledger_path, signing_key):
    """Plan test 10.5, first half: kill mid-session, nothing local changed."""
    import threading

    from cva.ledgerd.policy import development_policy
    from cva.ledgerd.server import serve
    from cva.web.workflow.ledgerd_client import LedgerdClient

    sock = ledger_dir / "kill.sock"
    server = serve(sock, ledger_path, signing_key, development_policy())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = LedgerdClient(sock)
    client.append_analyst_event(_override().as_wire())
    before = client.records(types=["analyst_event"])
    server.shutdown()
    server.server_close()
    server.daemon.backend.close()
    thread.join(timeout=5)

    with pytest.raises(LedgerUnreachable):
        client.append_analyst_event(
            _override(actor="b.rao", expected_prev_seq=before[-1]["seq"],
                      justification="A second decision attempted while the ledger daemon "
                                    "is down must not appear anywhere.").as_wire())

    server = serve(sock, ledger_path, signing_key, development_policy())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert LedgerdClient(sock).records(types=["analyst_event"]) == before
    finally:
        server.shutdown()
        server.server_close()
        server.daemon.backend.close()
        thread.join(timeout=5)
