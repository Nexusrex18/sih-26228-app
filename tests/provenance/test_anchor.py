"""The anchoring ceremony (plan §5.9, §7.10; gate C7): anchors, cosignatures, witness statements, the logbook
medium, and what an anchor does and does not let a verifier conclude."""
from __future__ import annotations

import json
import sqlite3

import pytest

from cva.provenance.seal.anchor import (
    AnchorError,
    attest,
    build_anchor,
    cosign,
    encode_anchor,
    export_anchor,
    load_anchor,
    logbook_anchor,
    parse_anchor,
    print_anchor,
    print_checksum,
    verify_anchor,
)
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import export_records, verify_ledger

from ._anchor_helpers import AnchorEnv


def classes(rep):
    return [f.attack_class for f in rep.findings if f.severity != "info"]


@pytest.fixture
def env(tmp_path):
    e = AnchorEnv(tmp_path)
    for i in range(12):
        e.seal(i)
    return e


# --- making an anchor ------------------------------------------------------------------------------------

def test_an_anchor_is_a_signed_checkpoint_plus_an_anchor_event_in_the_chain(env):
    a = env.anchor()
    cp = a["checkpoint"]
    assert cp["type"] == "checkpoint" and cp["seq"] == cp["checkpoint"]["tree_size"]
    env.sealer.close()
    with SealedLedger.open(env.ledger_path, read_only=True) as led:
        types = [r["type"] for r in led.records()]
    assert types[-2:] == ["checkpoint", "anchor_event"]
    ev = [r for r in _records(env) if r["type"] == "anchor_event"][-1]["anchor"]
    assert (ev["checkpoint_seq"], ev["tree_size"], ev["root_hash"]) == (cp["seq"], cp["checkpoint"]["tree_size"],
                                                                        cp["checkpoint"]["root_hash"])


def _records(env):
    c = sqlite3.connect(env.ledger_path)
    try:
        return [json.loads(r[0]) for r in c.execute("SELECT rec FROM records ORDER BY seq")]
    finally:
        c.close()


def test_the_anchor_file_is_canonical_reproducible_and_never_overwritten(env):
    a = env.anchor()
    data = (env.tmp / "a0.json").read_bytes()
    assert data == encode_anchor(a) and data.endswith(b"\n") and parse_anchor(data) == a
    env.sealer.close()
    led = SealedLedger.open(env.ledger_path, key=env.key, clock=env.clock, rng=env.rng, background_flush=False)
    with pytest.raises(AnchorError, match="refusing to overwrite"):
        export_anchor(led, env.tmp / "a0.json")
    led.close()
    assert (env.tmp / "a0.json").read_bytes() == data


def test_exporting_twice_with_nothing_new_reuses_the_checkpoint_but_new_records_get_a_fresh_one(env):
    a1 = env.anchor("a1.json")
    a2 = env.anchor("a2.json")
    assert a1["checkpoint"] == a2["checkpoint"]                # nothing but an anchor_event followed it
    env.seal(99)
    a3 = env.anchor("a3.json")
    assert a3["checkpoint"]["checkpoint"]["tree_size"] > a1["checkpoint"]["checkpoint"]["tree_size"]


def test_an_anchor_of_a_ledger_that_has_only_ever_had_cadence_checkpoints_is_still_produced(tmp_path):
    e = AnchorEnv(tmp_path, checkpoint_every=5)
    for i in range(12):
        e.seal(i)
    a = e.anchor()
    e.close()
    rep = verify_ledger(e.ledger_path, trust_root=e.trust, anchors=[a])
    assert rep.clean and rep.anchors_verified == 1


# --- a valid anchor against its own ledger -----------------------------------------------------------------------

def test_an_anchored_ledger_verifies_clean_and_reports_what_the_anchor_fixes(env):
    a = env.anchor()
    for i in range(20, 25):
        env.seal(i)
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert rep.clean and rep.anchors_verified == 1 and rep.anchors_invalid == 0
    covered = a["checkpoint"]["checkpoint"]["tree_size"] + 1
    assert rep.anchored_records == covered
    assert rep.unwitnessed_records == rep.records_checked - covered
    assert rep.unwitnessed_records >= 6                         # 5 inferences + the anchor_event


def test_verification_needs_only_the_export_and_the_anchor_with_the_live_database_gone(env):
    """The C7 gate: re-verify from an exported signed root alone."""
    a = env.anchor()
    exp = env.export()
    env.ledger_path.unlink()
    for ext in ("-wal", "-shm"):
        (env.tmp / f"ledger.db{ext}").unlink(missing_ok=True)
    rep = verify_ledger(exp, trust_root=env.trust, anchors=[env.tmp / "a0.json"])
    assert rep.source_kind == "jsonl" and rep.clean and rep.anchors_verified == 1
    assert rep.anchored_records == a["checkpoint"]["seq"] + 1


def test_an_anchor_may_be_supplied_as_a_dict_bytes_or_a_path(env):
    a = env.anchor()
    env.sealer.close()
    for form in (a, encode_anchor(a), env.tmp / "a0.json", str(env.tmp / "a0.json")):
        assert verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[form]).anchors_verified == 1


def test_several_anchors_the_strongest_defines_the_window(env):
    a1 = env.anchor("a1.json")
    for i in range(30, 40):
        env.seal(i)
    a2 = env.anchor("a2.json")
    env.seal(50)
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a1, a2])
    assert rep.clean and rep.anchors_verified == 2
    assert rep.anchored_records == a2["checkpoint"]["seq"] + 1


# --- tail truncation and forks (D11) --------------------------------------------------------------------------------

def _delete_tail(env, keep: int):
    env.sealer.close()
    c = sqlite3.connect(env.ledger_path, isolation_level=None)
    for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        c.execute(f"DROP TRIGGER {name}")
    c.execute("DELETE FROM records WHERE seq>=?", (keep,))
    c.close()


def test_truncating_the_tail_is_invisible_without_an_anchor_and_precise_with_one(env):
    a = env.anchor()
    covered = a["checkpoint"]["seq"] + 1
    _delete_tail(env, covered - 4)
    silent = verify_ledger(env.ledger_path, trust_root=env.trust)
    assert silent.clean and not silent.findings                       # the honest half: nothing in-band
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert classes(rep) == ["tail_truncation"]
    f = rep.by_class("tail_truncation")[0]
    assert f.severity == "high" and f.primary_check == "anchor_size"
    assert "at least 4 record(s)" in f.reason


def test_dropping_only_the_anchor_event_does_not_hide_the_truncation_of_records_it_covered(env):
    a = env.anchor()
    covered = a["checkpoint"]["seq"] + 1
    _delete_tail(env, covered)                                        # the anchor_event alone is gone
    assert classes(verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])) == []      # all covered records remain
    _delete_tail(env, covered - 1)                                    # ... but the checkpoint itself too
    assert classes(verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])) == ["tail_truncation"]


def test_records_truncated_after_the_last_anchor_stay_invisible_and_the_report_says_so(env):
    """The residual risk, stated as a tested fact: an anchor cannot vouch for what came after it."""
    a = env.anchor()
    for i in range(40, 46):
        env.seal(i)
    env.sealer.close()
    total = len(_records(env))
    _delete_tail(env, total - 5)
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert rep.clean and rep.anchors_verified == 1
    assert rep.unwitnessed_records == rep.records_checked - rep.anchored_records


def test_an_anchor_from_a_different_history_is_a_ledger_fork_when_everything_else_verifies(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a_env, b_env = AnchorEnv(tmp_path / "a"), AnchorEnv(tmp_path / "b")
    for i in range(12):
        a_env.seal(i)
        b_env.seal(i if i < 6 else i + 1000)                          # the same first records, then a different history
    anchor_a = a_env.anchor()
    b_env.anchor()
    a_env.close(), b_env.close()
    rep = verify_ledger(b_env.ledger_path, trust_root=b_env.trust, anchors=[anchor_a])
    assert classes(rep) == ["ledger_fork"]
    f = rep.by_class("ledger_fork")[0]
    assert f.severity == "critical" and f.nature == "adversarial" and f.primary_check == "anchor_root"
    assert verify_ledger(b_env.ledger_path, trust_root=b_env.trust).clean       # ... and B is fine on its own


def test_an_edit_inside_the_anchored_prefix_is_one_finding_not_an_edit_plus_a_fork(env):
    a = env.anchor()
    env.sealer.close()
    c = sqlite3.connect(env.ledger_path, isolation_level=None)
    for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        c.execute(f"DROP TRIGGER {name}")
    (rec,) = c.execute("SELECT rec FROM records WHERE seq=5").fetchone()
    from cva.provenance.seal.canonical import canonical_bytes
    obj = json.loads(rec)
    obj["output"]["jcs_sha256"] = "0" * 64
    import hashlib
    unsigned = {k: v for k, v in obj.items() if k != "signature"}
    c.execute("UPDATE records SET rec=?, rec_hash=? WHERE seq=5", (canonical_bytes(obj).decode(),
                                                                   hashlib.sha256(canonical_bytes(unsigned)).digest()))
    c.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert classes(rep) == ["record_edit"] and rep.anchors_verified == 0
    assert rep.findings[0].cascade_suppressed >= 1


def test_a_checkpoint_re_signed_at_the_anchored_index_is_a_fork(env):
    """The prefix matches but the checkpoint record in the ledger is not the one the anchor carries."""
    a = env.anchor()
    env.close()
    a2 = json.loads(json.dumps(a))
    a2["checkpoint"]["created_at_utc"] = "2026-09-19T02:00:05.000000Z"        # not what is stored
    # re-sign under the ledger key so the anchor itself verifies
    from cva.provenance.seal.chain import sign_record
    body = {k: v for k, v in a2["checkpoint"].items() if k != "signature"}
    a2["checkpoint"] = sign_record(body, env.key)
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a2])
    assert classes(rep) == ["ledger_fork"] and rep.by_class("ledger_fork")[0].primary_check == "anchor_checkpoint"


# --- cosignatures and witness statements ----------------------------------------------------------------------------------

def test_a_cosignature_by_a_trust_root_witness_verifies_and_is_reported(env):
    a = cosign(env.anchor(), env.witness)
    r = verify_anchor(a, env.trust)
    assert r.valid and r.cosigners == [env.witness.key_id]


def test_a_cosignature_by_a_key_that_is_not_a_trust_root_witness_invalidates_the_anchor(env):
    from ._chain_helpers import provider
    stranger = provider(bytes(range(7, 39)))
    r = verify_anchor(cosign(env.anchor(), stranger), env.trust)
    assert not r.valid and "not a witness key in the trust root" in r.problems[0]


def test_a_boundary_key_cannot_cosign_and_a_witness_key_cannot_attest(env):
    """Roles are separate authorities: a key vouched for as one thing is not thereby the other."""
    a = env.anchor()
    assert not verify_anchor(cosign(a, env.boundary), env.trust).valid
    assert not verify_anchor(attest(a, env.witness, "2026-09-19T03:00:00.000000Z"), env.trust).valid


def test_a_tampered_cosignature_invalidates_the_whole_anchor(env):
    a = cosign(env.anchor(), env.witness)
    sig = a["cosignatures"][0]["sig"]
    a["cosignatures"][0]["sig"] = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    r = verify_anchor(a, env.trust)
    assert not r.valid and r.cosigners == []


def test_a_cosignature_over_one_checkpoint_does_not_verify_on_another(env):
    a1 = cosign(env.anchor("a1.json"), env.witness)
    env.seal(60)
    a2 = env.anchor("a2.json")
    a2["cosignatures"] = a1["cosignatures"]
    assert not verify_anchor(a2, env.trust).valid


def test_a_witness_statement_bounds_absolute_time_and_the_earliest_wins(env):
    a = attest(env.anchor(), env.boundary, "2026-09-19T03:00:00.000000Z")
    r = verify_anchor(a, env.trust)
    assert r.valid and r.time_bound_utc == "2026-09-19T03:00:00.000000Z"
    assert r.attestations == [{"key_id": env.boundary.key_id, "observed_at_utc": "2026-09-19T03:00:00.000000Z"}]


def test_the_time_bound_reaches_the_ledger_report(env):
    a = attest(cosign(env.anchor(), env.witness), env.boundary, "2026-09-19T03:00:00.000000Z")
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert rep.clean and rep.attested_not_after == "2026-09-19T03:00:00.000000Z"


def test_a_witness_statement_about_another_root_or_with_an_edited_time_is_rejected(env):
    a = attest(env.anchor(), env.boundary, "2026-09-19T03:00:00.000000Z")
    edited = json.loads(json.dumps(a))
    edited["attestations"][0]["observed_at_utc"] = "2026-09-19T01:00:00.000000Z"      # claim it was seen earlier
    assert not verify_anchor(edited, env.trust).valid
    other = json.loads(json.dumps(a))
    other["attestations"][0]["root_hash"] = "0" * 64
    assert "different root" in verify_anchor(other, env.trust).problems[0]


def test_a_host_clock_ahead_of_the_witness_is_a_note_not_a_finding(env):
    a = attest(env.anchor(), env.boundary, "2026-09-19T01:00:00.000000Z")            # before the checkpoint's own time
    r = verify_anchor(a, env.trust)
    assert r.valid and any("host clock ran ahead" in n for n in r.notes)


# --- tampering with the anchor artefact itself -------------------------------------------------------------------------------

def test_an_anchor_whose_root_was_edited_fails_its_own_signature_and_is_not_used(env):
    a = env.anchor()
    a["checkpoint"]["checkpoint"]["root_hash"] = "0" * 64
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert classes(rep) == ["anchor_invalid"] and rep.anchors_invalid == 1 and rep.anchors_verified == 0
    f = rep.by_class("anchor_invalid")[0]
    assert f.severity == "medium" and "says nothing about the ledger" in f.reason


@pytest.mark.parametrize("junk", [b"not json", b"{}", b'{"v":"cva-seal/1"}', b"[]"])
def test_something_that_is_not_an_anchor_is_anchor_invalid_and_never_crashes_the_verifier(env, junk):
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[junk])
    assert classes(rep) == ["anchor_invalid"] and rep.anchors_invalid == 1


def test_an_unreadable_anchor_path_is_anchor_invalid(env, tmp_path):
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[tmp_path / "missing.json"])
    assert classes(rep) == ["anchor_invalid"]


def test_an_anchor_signed_by_a_key_the_trust_root_does_not_know_is_unusable(env):
    from cva.provenance.seal.keys import TrustKey, TrustRoot

    from ._chain_helpers import provider
    a = env.anchor()
    stranger = provider(bytes(range(3, 35)))
    tr = TrustRoot(env.trust.deployment_manifest_hash, (TrustKey(stranger.key_id, stranger.public_key, "ledger"),))
    r = verify_anchor(a, tr)
    assert not r.valid and "not a ledger key in the trust root" in r.problems[0]


def test_a_checkpoint_at_the_wrong_index_is_not_a_valid_anchor(env):
    a = env.anchor()
    from cva.provenance.seal.chain import sign_record
    cp = {k: v for k, v in a["checkpoint"].items() if k != "signature"}
    cp["seq"] += 1
    a["checkpoint"] = sign_record(cp, env.key)
    assert "sits at index tree_size" in " ".join(verify_anchor(a, env.trust).problems)


def test_building_an_anchor_around_anything_but_a_checkpoint_is_refused(env):
    rec = _records(env)[3]
    with pytest.raises(AnchorError, match="checkpoint"):
        build_anchor(rec)


def test_an_anchor_with_extra_or_missing_fields_is_rejected_by_the_parser():
    for bad in ({"v": "cva-seal/1", "checkpoint": {}, "cosignatures": []},
                {"v": "cva-seal/2", "checkpoint": {}, "cosignatures": [], "attestations": []},
                {"v": "cva-seal/1", "checkpoint": {}, "cosignatures": [], "attestations": [], "x": 1}):
        with pytest.raises(AnchorError):
            parse_anchor(json.dumps(bad).encode())


# --- the logbook medium --------------------------------------------------------------------------------------------------------

def test_the_printed_block_carries_size_root_and_a_checksum_in_readable_groups(env):
    a = env.anchor()
    text = print_anchor(a)
    cp = a["checkpoint"]["checkpoint"]
    assert f"tree size : {cp['tree_size']}" in text
    assert cp["root_hash"][:4] + " " + cp["root_hash"][4:8] in text
    assert print_checksum(cp["tree_size"], cp["root_hash"])[:4] in text
    assert "witnessed by" in text


def test_a_logbook_entry_typed_back_in_verifies_the_ledger_it_was_copied_from(env):
    a = env.anchor()
    cp = a["checkpoint"]["checkpoint"]
    lb = logbook_anchor(cp["tree_size"], " ".join(cp["root_hash"][i:i + 4] for i in range(0, 64, 4)),
                        print_checksum(cp["tree_size"], cp["root_hash"]))
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[lb])
    assert rep.clean and rep.anchors_verified == 1 and rep.anchored_records == cp["tree_size"]
    assert verify_anchor(lb, env.trust).notes and verify_anchor(lb, env.trust).medium == "logbook"


def test_a_mistyped_logbook_digit_is_caught_by_the_checksum_not_reported_as_a_fork(env):
    cp = env.anchor()["checkpoint"]["checkpoint"]
    good = print_checksum(cp["tree_size"], cp["root_hash"])
    bad_root = cp["root_hash"][:10] + ("0" if cp["root_hash"][10] != "0" else "1") + cp["root_hash"][11:]
    with pytest.raises(AnchorError, match="mistyped"):
        logbook_anchor(cp["tree_size"], bad_root, good)


def test_a_logbook_root_that_does_not_match_is_a_ledger_fork_and_a_larger_size_is_truncation(env):
    cp = env.anchor()["checkpoint"]["checkpoint"]
    env.sealer.close()
    wrong = logbook_anchor(cp["tree_size"], "0" * 64)
    assert classes(verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[wrong])) == ["ledger_fork"]
    far = logbook_anchor(cp["tree_size"] + 500, cp["root_hash"])
    assert classes(verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[far])) == ["tail_truncation"]


def test_load_anchor_reads_a_logbook_file_too(tmp_path):
    lb = logbook_anchor(5, "ab" * 32)
    (tmp_path / "lb.json").write_bytes(json.dumps({**lb, "kind": "logbook"}).encode())
    assert load_anchor(tmp_path / "lb.json")["tree_size"] == 5


# --- an anchor pins the history the anchor was taken over, at every size ----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_anchors_verify_at_odd_and_even_tree_sizes(tmp_path, n):
    e = AnchorEnv(tmp_path)
    for i in range(n):
        e.seal(i)
    a = e.anchor()
    e.close()
    rep = verify_ledger(e.ledger_path, trust_root=e.trust, anchors=[a])
    assert rep.clean and rep.anchors_verified == 1


def test_export_and_database_forms_give_the_same_anchor_verdict(env):
    a = env.anchor()
    exp = env.export()
    db = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    jl = verify_ledger(exp, trust_root=env.trust, anchors=[a])
    assert (db.anchors_verified, db.anchored_records, db.unwitnessed_records) == (
        jl.anchors_verified, jl.anchored_records, jl.unwitnessed_records)
    assert export_records is not None


def test_the_earliest_of_several_witness_statements_is_the_time_bound(tmp_path):
    from cva.provenance.seal.keys import TrustKey, TrustRoot

    from ._chain_helpers import provider
    e = AnchorEnv(tmp_path)
    for i in range(3):
        e.seal(i)
    second = provider(bytes(range(17, 49)))
    trust = TrustRoot(e.trust.deployment_manifest_hash, (*e.trust.keys, TrustKey(second.key_id, second.public_key, "boundary")))
    a = attest(attest(e.anchor(), e.boundary, "2026-09-19T05:00:00.000000Z"), second, "2026-09-19T04:00:00.000000Z")
    r = verify_anchor(a, trust)
    assert r.valid and len(r.attestations) == 2 and r.time_bound_utc == "2026-09-19T04:00:00.000000Z"
    e.close()


def test_a_tamper_after_the_anchored_prefix_does_not_hide_a_fork_inside_it(tmp_path):
    """The two findings have different causes: an edit AFTER the anchor cannot explain a root mismatch BEFORE it."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a_env, b_env = AnchorEnv(tmp_path / "a"), AnchorEnv(tmp_path / "b")
    for i in range(10):
        a_env.seal(i)
        b_env.seal(i if i < 5 else i + 1000)
    anchor_a = a_env.anchor()
    b_env.anchor()
    for i in range(30, 36):
        b_env.seal(i)                                                   # records AFTER b's own anchor
    b_env.close()
    a_env.close()
    from attacklab.tamper import TamperDb
    t = TamperDb(b_env.ledger_path, tmp_path / "t.db")
    victim = t.last_seq() - 2
    rec = t.rec(victim)
    rec["output"]["jcs_sha256"] = "0" * 64
    t.put(victim, rec)
    t.close()
    got = classes(verify_ledger(tmp_path / "t.db", trust_root=b_env.trust, anchors=[anchor_a]))
    assert "ledger_fork" in got and "record_edit" in got
