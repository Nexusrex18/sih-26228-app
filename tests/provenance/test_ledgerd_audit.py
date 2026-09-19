"""Gate C9 — `scan_record` and `analyst_event` end to end (decision D14; Module E D-E3/D-E6): assignment,
acknowledgement, override with a MANDATORY justification, and an approver; through `cva-ledgerd` and the audit fold."""
from __future__ import annotations

import json
import os
import sqlite3
import threading

import pytest

from cva.provenance.seal.audit import fold_analyst_events
from cva.provenance.seal.errors import InvalidRecord, SealError
from cva.provenance.seal.ledgerd import Ledgerd, Policy, peer_uid, request, serve
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import verify_ledger

from ._chain_helpers import SEED_A, clock, provider, rng
from ._ledger_helpers import MANIFEST

UID = os.getuid()
RID = iter(range(1, 10_000))


def rid() -> str:
    return f"{next(RID):032x}"


def analyst(action="acknowledge", *, actor="a.khan", finding="F-1", scan="S-1", **kw):
    a = {"actor_id": actor, "role": "analyst", "action": action, "scan_id": scan, "target_type": "sample",
         "target_ref": "img_0042.png", "finding_id": finding, "justification": "", "request_id": rid()}
    a.update(kw)
    return {"analyst": a}


def override(prev, *, actor="a.khan", finding="F-1", why="Duplicate cluster confirmed benign", **kw):
    body = analyst("override", actor=actor, finding=finding, new_disposition="accept", reason_code="known_benign",
                   justification=why, expected_prev_seq=prev)
    body["analyst"].update(kw)
    return body


SCAN = {"scan": {"scan_id": "S-1", "report_sha256": "b" * 64, "profile_hash": "c" * 64, "code_commit": "0" * 40,
                 "finding_counts": {"critical": 1, "info": 3}}}


@pytest.fixture
def led(tmp_path):
    key = provider(SEED_A)
    l = SealedLedger.init_ledger(tmp_path / "l.db", key, {**MANIFEST, "checkpoint_every": 1000}, clock=clock(), rng=rng())
    yield l
    l.close()


@pytest.fixture
def d(led):
    return Ledgerd(led, Policy({UID: frozenset({"analyst_event", "scan_record"})}))


def ok(r):
    assert r["ok"], r
    return r["seq"]


# --- the SDK rule behind the gate --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("why", ["", "   ", "\t"])
def test_an_override_without_a_justification_is_rejected_at_the_sdk_and_never_reaches_the_chain(led, why):
    n = led.size()
    body = override(0, why=why)
    with pytest.raises((InvalidRecord, SealError, ValueError)):
        led.append({"type": "analyst_event", **body})
    assert led.size() == n


def test_an_override_missing_what_it_changes_why_or_the_stale_write_guard_is_rejected(led):
    for missing in ("new_disposition", "reason_code", "expected_prev_seq"):
        body = override(0)
        del body["analyst"][missing]
        with pytest.raises((InvalidRecord, SealError, ValueError)):
            led.append({"type": "analyst_event", **body})


def test_an_unknown_reason_code_and_a_non_ascii_justification_are_refused(led):
    with pytest.raises((InvalidRecord, ValueError)):
        led.append({"type": "analyst_event", **override(0, reason_code="because")})
    with pytest.raises((InvalidRecord, ValueError, SealError)):
        led.append({"type": "analyst_event", **override(0, why="Café — benign")})


# --- through the daemon ---------------------------------------------------------------------------------------------------------

def test_the_full_workflow_assign_acknowledge_override_approve_and_the_chain_verifies_clean(d, led):
    ok(d.handle({"type": "scan_record", "body": SCAN}, UID))
    ok(d.handle({"type": "analyst_event", "body": analyst("assign", assignee="a.khan", actor="lead.rao")}, UID))
    ack = ok(d.handle({"type": "analyst_event", "body": analyst("acknowledge")}, UID))
    ov = ok(d.handle({"type": "analyst_event", "body": override(ack)}, UID))
    ap = ok(d.handle({"type": "analyst_event", "body": analyst("approve", actor="lead.rao", refs_seq=ov, expected_prev_seq=ov,
                                                               justification="Reviewed the cluster")}, UID))
    fold = fold_analyst_events(led.records())
    st = fold.findings[("S-1", "F-1")]
    assert fold.clean and st.disposition == "accept" and st.overrides == {ov: "a.khan"} and st.approved == {ov: "lead.rao"}
    assert st.timeline[-1] == ap and fold.scans
    from cva.provenance.seal.keys import TrustKey, TrustRoot
    from cva.provenance.seal.records import genesis_prev_hash
    k = provider(SEED_A)
    tr = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(k.key_id, k.public_key, "ledger"),))
    led.flush()
    assert verify_ledger(led.path, trust_root=tr).clean


def test_an_approver_must_be_someone_other_than_the_overrider(d):
    ov = ok(d.handle({"type": "analyst_event", "body": override(0)}, UID))
    r = d.handle({"type": "analyst_event", "body": analyst("approve", actor="a.khan", refs_seq=ov, expected_prev_seq=ov)}, UID)
    assert not r["ok"] and r["error"] == "self_approval"


def test_an_approval_must_name_a_real_override_of_that_finding(d):
    ov = ok(d.handle({"type": "analyst_event", "body": override(0, finding="F-1")}, UID))
    other = d.handle({"type": "analyst_event", "body": analyst("approve", actor="lead.rao", finding="F-2", refs_seq=ov, expected_prev_seq=0)}, UID)
    assert not other["ok"] and other["error"] == "approve_without_override"
    nothing = d.handle({"type": "analyst_event", "body": analyst("approve", actor="lead.rao", refs_seq=9999, expected_prev_seq=ov)}, UID)
    assert not nothing["ok"] and nothing["error"] == "approve_without_override"
    missing = analyst("approve", actor="lead.rao")
    r = d.handle({"type": "analyst_event", "body": missing}, UID)
    assert not r["ok"] and r["error"] == "invalid"                                  # refs_seq is required by the schema


def test_a_write_based_on_a_stale_view_is_refused_so_two_analysts_cannot_silently_overwrite_each_other(d):
    first = ok(d.handle({"type": "analyst_event", "body": analyst("acknowledge", actor="a.khan")}, UID))
    ok(d.handle({"type": "analyst_event", "body": override(first, actor="a.khan")}, UID))
    stale = d.handle({"type": "analyst_event", "body": override(first, actor="b.rao", why="I disagree")}, UID)      # still thinks `first` is last
    assert not stale["ok"] and stale["error"] == "stale_write" and "another decision landed first" in stale["detail"]


def test_a_retried_request_id_is_idempotent_and_a_reused_one_with_different_content_is_a_conflict(d, led):
    body = analyst("acknowledge")
    a = d.handle({"type": "analyst_event", "body": body}, UID)
    n = led.size()
    b = d.handle({"type": "analyst_event", "body": body}, UID)
    assert b["ok"] and b["seq"] == a["seq"] and b["idempotent"] and led.size() == n         # no second record
    changed = json.loads(json.dumps(body))
    changed["analyst"]["actor_id"] = "someone.else"
    c = d.handle({"type": "analyst_event", "body": changed}, UID)
    assert not c["ok"] and c["error"] == "request_id_conflict"


def test_the_workflow_state_is_rebuilt_from_the_chain_after_a_restart(led):
    pol = Policy({UID: frozenset({"analyst_event"})})
    d1 = Ledgerd(led, pol)
    ov = ok(d1.handle({"type": "analyst_event", "body": override(0)}, UID))
    body = analyst("acknowledge", finding="F-9")
    s1 = ok(d1.handle({"type": "analyst_event", "body": body}, UID))
    d2 = Ledgerd(led, pol)                                                                   # a new process, same ledger
    again = d2.handle({"type": "analyst_event", "body": body}, UID)
    assert again["seq"] == s1 and again["idempotent"]
    assert ok(d2.handle({"type": "analyst_event", "body": analyst("approve", actor="lead.rao", refs_seq=ov, expected_prev_seq=ov)}, UID))
    assert not d2.handle({"type": "analyst_event", "body": override(0, why="stale after restart")}, UID)["ok"]


# --- the allowlist (D-E3) -----------------------------------------------------------------------------------------------------------

def test_a_uid_may_request_only_the_types_it_was_granted(led):
    d = Ledgerd(led, Policy({1001: frozenset({"analyst_event"}), 1002: frozenset({"scan_record"})}))
    assert d.handle({"type": "analyst_event", "body": analyst()}, 1002)["error"] == "not_permitted"
    assert d.handle({"type": "scan_record", "body": SCAN}, 1001)["error"] == "not_permitted"
    assert d.handle({"type": "analyst_event", "body": analyst()}, 4242)["error"] == "not_permitted"
    assert d.handle({"type": "analyst_event", "body": analyst()}, None)["error"] == "not_permitted"       # no credentials, no write
    assert d.handle({"type": "scan_record", "body": SCAN}, 1002)["ok"] and d.handle({"type": "analyst_event", "body": analyst()}, 1001)["ok"]


@pytest.mark.parametrize("rtype", ["genesis", "key_rotation", "checkpoint", "anchor_event", "model_registration", "inference", "degraded_marker"])
def test_the_structural_and_inference_records_can_never_be_written_through_the_daemon(d, rtype):
    r = d.handle({"type": rtype, "body": {}}, UID)
    assert not r["ok"] and r["error"] == "forbidden_type"


def test_a_policy_cannot_even_be_configured_to_grant_a_structural_type():
    for t in ("key_rotation", "genesis", "checkpoint", "anchor_event", "inference"):
        with pytest.raises(ValueError, match="only"):
            Policy({1: frozenset({t})})


def test_malformed_requests_and_bodies_are_refused_cleanly(d):
    assert d.handle({"body": {}}, UID)["error"] == "bad_request"
    assert d.handle({"type": "analyst_event", "body": "x"}, UID)["error"] == "bad_request"
    assert d.handle({"type": "nonsense", "body": {}}, UID)["error"] == "unknown_type"
    assert d.handle({"type": "analyst_event", "body": {"scan": {}}}, UID)["error"] == "invalid"
    bad = analyst("override", justification="", new_disposition="accept", reason_code="known_benign", expected_prev_seq=0)
    assert d.handle({"type": "analyst_event", "body": bad}, UID)["error"] == "invalid"
    extra = {**analyst(), "scan": SCAN["scan"]}
    assert d.handle({"type": "analyst_event", "body": extra}, UID)["error"] == "invalid"


# --- over a real Unix socket, with the kernel's word for who is calling -------------------------------------------------------------------

def test_over_a_unix_socket_the_uid_comes_from_the_kernel_and_concurrent_clients_all_land(led, tmp_path):
    sock = str(tmp_path / "ledgerd.sock")
    srv = serve(sock, Ledgerd(led, Policy({UID: frozenset({"analyst_event", "scan_record"})})))
    try:
        assert request(sock, {"type": "scan_record", "body": SCAN})["ok"]
        results = []

        def worker(i):
            results.append(request(sock, {"type": "analyst_event", "body": override(0, finding=f"F-{i}", actor=f"a{i}")}))
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert all(r["ok"] for r in results) and len({r["seq"] for r in results}) == 16
        assert not request(sock, {"type": "inference", "body": {}})["ok"]
        import socket as _s
        c = _s.socket(_s.AF_UNIX)
        c.connect(sock)
        assert peer_uid(c) == UID                                                        # what the server sees of us
        c.sendall(b"not json\n")
        assert json.loads(c.makefile().readline())["error"] == "bad_request"
        c.close()
    finally:
        srv.shutdown()
        srv.server_close()
    assert fold_analyst_events(led.records()).clean and len(fold_analyst_events(led.records()).findings) == 16


def test_a_socket_client_running_as_a_uid_the_policy_does_not_know_is_refused(led, tmp_path):
    sock = str(tmp_path / "s2.sock")
    srv = serve(sock, Ledgerd(led, Policy({UID + 1: frozenset({"analyst_event"})})))         # we are not that uid
    try:
        r = request(sock, {"type": "analyst_event", "body": analyst()})
        assert not r["ok"] and r["error"] == "not_permitted"
    finally:
        srv.shutdown()
        srv.server_close()


# --- the trail is covered by the same tamper-evidence as everything else ---------------------------------------------------------------------

def _tamper_db(led, tmp_path):
    from attacklab.tamper import TamperDb
    led.flush()
    return TamperDb(led.path, tmp_path / "t.db")


def test_editing_an_analyst_justification_after_the_fact_is_a_record_edit_and_the_fold_alone_would_not_see_it(d, led, tmp_path):
    from cva.provenance.seal.keys import TrustKey, TrustRoot
    from cva.provenance.seal.records import genesis_prev_hash
    ov = ok(d.handle({"type": "analyst_event", "body": override(0)}, UID))
    ok(d.handle({"type": "analyst_event", "body": analyst("acknowledge", finding="F-2")}, UID))
    t = _tamper_db(led, tmp_path)
    rec = t.rec(ov)
    rec["analyst"]["justification"] = "Nothing to see here"
    t.put(ov, rec)
    t.close()
    k = provider(SEED_A)
    tr = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(k.key_id, k.public_key, "ledger"),))
    rep = verify_ledger(tmp_path / "t.db", trust_root=tr)
    assert [f.attack_class for f in rep.findings if f.severity != "info"] == ["record_edit"] and rep.findings[0].seq == ov


def test_deleting_an_analyst_decision_is_a_record_delete(d, led, tmp_path):
    from cva.provenance.seal.keys import TrustKey, TrustRoot
    from cva.provenance.seal.records import genesis_prev_hash
    a = ok(d.handle({"type": "analyst_event", "body": analyst("acknowledge")}, UID))
    ok(d.handle({"type": "analyst_event", "body": override(a)}, UID))
    ok(d.handle({"type": "analyst_event", "body": analyst("acknowledge", finding="F-3")}, UID))
    t = _tamper_db(led, tmp_path)
    t.delete(a + 1)                                                                          # the override vanishes
    t.close()
    k = provider(SEED_A)
    tr = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(k.key_id, k.public_key, "ledger"),))
    rep = verify_ledger(tmp_path / "t.db", trust_root=tr)
    assert [f.attack_class for f in rep.findings if f.severity != "info"] == ["record_delete"]
    assert sqlite3 is not None


def test_the_fold_reports_workflow_violations_that_bypassed_the_daemon(led):
    """Someone with the key (or a buggy writer) can append a self-approval directly; the chain accepts it, the fold names it."""
    ov = led.append_typed("analyst_event", analyst() | override(0))[0].seq
    led.append_typed("analyst_event", analyst("approve", actor="a.khan", refs_seq=ov, expected_prev_seq=ov))
    led.append_typed("analyst_event", override(0, why="stale on purpose"))
    fold = fold_analyst_events(led.records())
    assert [v.code for v in fold.violations] == ["self_approval", "stale_write"]


# --- the C core enforces the same rule -------------------------------------------------------------------------------------------------------

def test_the_c_core_also_refuses_an_override_without_a_justification_and_its_analyst_events_fold_cleanly(tmp_path):
    import shutil
    import subprocess

    from .test_c_core import NATIVE
    if not (shutil.which("cc") and (NATIVE / "cvseal").exists()):
        pytest.skip("C core not built")
    exe, db, seed = str(NATIVE / "cvseal"), tmp_path / "c.db", SEED_A.hex()
    man = json.dumps({"device_id": "c", "unit": "u", "profile_hash": "0" * 64, "checkpoint_every": 1000})
    subprocess.run([exe, "init", str(db), seed, man], check=True)
    bad = subprocess.run([exe, "append", str(db), seed, "analyst_event", json.dumps(override(0, why="  "))], capture_output=True)
    assert bad.returncode != 0 and b"justification" in bad.stderr
    for missing in ("new_disposition", "reason_code", "expected_prev_seq"):
        b = override(0)
        del b["analyst"][missing]
        assert subprocess.run([exe, "append", str(db), seed, "analyst_event", json.dumps(b)], capture_output=True).returncode != 0
    ov = subprocess.run([exe, "append", str(db), seed, "analyst_event", json.dumps(override(0))], capture_output=True, check=True)
    seq = int(ov.stdout)
    subprocess.run([exe, "append", str(db), seed, "analyst_event",
                    json.dumps(analyst("approve", actor="lead.rao", refs_seq=seq, expected_prev_seq=seq))], check=True)
    led = SealedLedger.open(db, read_only=True)
    fold = fold_analyst_events(led.records())
    led.close()
    assert fold.clean and fold.findings[("S-1", "F-1")].approved == {seq: "lead.rao"}
