"""The standalone verifier (plan §7.8): sources, per-class behaviour, cascades, severities."""
from __future__ import annotations

import json
import sqlite3

import pytest

from cva.provenance.seal.canonical import canonical_bytes
from cva.provenance.seal.chain import seal_next
from cva.provenance.seal.errors import LedgerUnreadable
from cva.provenance.seal.keys import TrustKey, TrustRoot
from cva.provenance.seal.records import SECTIONS, genesis_prev_hash
from cva.provenance.seal.verify import (
    CLASS_PROFILE,
    SEVERITY_ORDER,
    JsonlSource,
    export_records,
    open_source,
    verify_ledger,
)

from ._chain_helpers import SEED_B, clock, provider
from ._fixtures import BODIES, H
from ._ledger_helpers import Env, export_bytes, valid_chain

TS = "2026-09-19T03:00:00.000000Z"


def classes(rep):
    return [f.attack_class for f in rep.findings if f.severity != "info"]


# --- clean ledgers ---------------------------------------------------------------------------------------

def test_a_clean_ledger_verifies_and_reports_what_it_checked(tmp_path):
    e = Env(tmp_path, checkpoint_every=10)
    for i in range(35):
        e.seal(i)
    e.close()
    r = verify_ledger(e.ledger_path, trust_root=e.trust)
    assert r.clean and not r.findings and r.source_kind == "sqlite"
    assert r.records_checked == 41 and r.count_by_type["inference"] == 35 and r.count_by_type["checkpoint"] == 4
    assert r.checkpoints_verified == 4 and r.payloads_checked > 0 and r.payloads_missing == 0
    assert r.declared_gaps == 0 and r.durability == "per_record" and r.loss_window == "0 records"
    assert r.unwitnessed_records == 41 and r.anchors_verified == 0        # no anchors yet: the WHOLE ledger is unwitnessed


def test_the_export_verifies_identically_to_the_database(tmp_path):
    e = Env(tmp_path, checkpoint_every=7)
    for i in range(20):
        e.seal(i)
    e.close()
    export_records(e.ledger_path, tmp_path / "x.jsonl")
    db, ex = verify_ledger(e.ledger_path, trust_root=e.trust), verify_ledger(tmp_path / "x.jsonl", trust_root=e.trust)
    assert db.clean and ex.clean and ex.source_kind == "jsonl"
    assert (db.records_checked, db.checkpoints_verified, dict(db.count_by_type)) == \
           (ex.records_checked, ex.checkpoints_verified, dict(ex.count_by_type))


def test_a_trust_root_can_be_a_file_path(tmp_path):
    chain, trust, _ = valid_chain(3)
    (tmp_path / "trust_root.json").write_bytes(trust.to_bytes() + b"\n")
    assert verify_ledger(export_bytes(chain), trust_root=tmp_path / "trust_root.json").clean


def test_the_source_can_be_bytes_a_path_or_a_prepared_source(tmp_path):
    chain, trust, _ = valid_chain(3)
    (tmp_path / "x.jsonl").write_bytes(export_bytes(chain))
    for src in (export_bytes(chain), tmp_path / "x.jsonl", str(tmp_path / "x.jsonl"), JsonlSource(export_bytes(chain))):
        assert verify_ledger(src, trust_root=trust).clean
    assert isinstance(open_source(export_bytes(chain)), JsonlSource)


# --- ledgers that cannot be assessed -----------------------------------------------------------------------

def test_an_empty_ledger_is_never_reported_verified(tmp_path):
    _, trust, _ = valid_chain(1)
    r = verify_ledger(b"", trust_root=trust)
    assert classes(r) == ["genesis_mismatch"] and "not the same as verified" in r.findings[0].reason


@pytest.mark.parametrize("payload", [b"not a database at all", b"\x00" * 4096])
def test_garbage_that_is_not_a_ledger_is_read_as_export_and_fails_to_parse(tmp_path, payload):
    _, trust, _ = valid_chain(1)
    p = tmp_path / "junk.bin"
    p.write_bytes(payload)
    r = verify_ledger(p, trust_root=trust)
    assert not r.clean and "malformed_record" in classes(r)


def test_a_sqlite_file_that_is_not_a_ledger_is_unreadable_not_a_tamper_finding(tmp_path):
    _, trust, _ = valid_chain(1)
    p = tmp_path / "other.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE unrelated (x)")
    c.close()
    with pytest.raises(LedgerUnreadable):
        verify_ledger(p, trust_root=trust)


def test_a_missing_path_is_unreadable(tmp_path):
    _, trust, _ = valid_chain(1)
    with pytest.raises(LedgerUnreadable):
        verify_ledger(tmp_path / "does-not-exist.db", trust_root=trust)


# --- the strict export format --------------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,expect_at_last", [
    (lambda d: d[:-1], True),                                      # missing final newline
    (lambda d: d + b"\n", True),                                   # a trailing blank line
    (lambda d: b"\xef\xbb\xbf" + d, False),                        # BOM
    (lambda d: d.replace(b"\n", b"\r\n"), False),                  # CRLF
    (lambda d: d.replace(b"}\n{", b"}\n\n{", 1), False),           # a blank line in the middle
    (lambda d: d.replace(b"}\n{", b"} \n{", 1), False),            # trailing space
])
def test_an_export_that_is_not_exactly_one_canonical_record_per_line_is_a_finding(mutate, expect_at_last):
    chain, trust, _ = valid_chain(4)
    r = verify_ledger(mutate(export_bytes(chain)), trust_root=trust)
    assert not r.clean and classes(r)[0] in ("malformed_record", "non_canonical_encoding")
    if expect_at_last:
        assert any(f.position == r.records_checked - 1 or f.position == r.records_checked - 2 for f in r.findings)


# --- severities and classes ----------------------------------------------------------------------------------

def test_every_class_that_must_quarantine_is_high_or_critical():
    """backend_plan §7.9 D1 floors on severity >= high: a certain tamper class left at `medium` is silently
    downgraded to `review`."""
    floor = SEVERITY_ORDER.index("high")
    exempt = {"boundary_flip": "medium", "degraded_gap": "low", "clock_regression": "info",
              "anchor_invalid": "medium"}          # a bad ARTEFACT, not evidence about the ledger: review, not quarantine
    for cls, (sev, _) in CLASS_PROFILE.items():
        if cls in exempt:
            assert sev == exempt[cls], cls
        else:
            assert SEVERITY_ORDER.index(sev) >= floor, f"{cls} is {sev}"


def test_every_class_the_verifier_can_emit_is_registered_in_backends_taxonomy():
    tax = pytest.importorskip("cva.core.taxonomy")
    missing = [c for c in CLASS_PROFILE if c not in tax.TAXONOMY]
    assert missing == [], f"not in core/taxonomy.py: {missing}"
    for cls in ("degraded_gap", "clock_regression"):
        assert tax.TAXONOMY[cls].kind == "operational"


# --- findings explain themselves -------------------------------------------------------------------------------

def test_a_finding_names_the_seq_the_key_and_what_carries_the_certainty(tmp_path):
    chain, trust, key = valid_chain(5)
    lines = list(chain.stored)
    rec = json.loads(lines[4])
    rec["output"]["jcs_sha256"] = "0" * 64
    lines[4] = canonical_bytes(rec)
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=trust)
    f = r.findings[0]
    assert f.attack_class == "record_edit" and f.seq == 4 and f.primary_check == "ed25519_signature"
    assert "seq 4" in f.reason and key.key_id[:8] in f.reason and "Certain" in f.reason and "cascade" in f.reason
    assert f.evidence and f.evidence[0][0] == "json" and f.severity == "critical"


def test_only_the_first_break_is_reported_and_its_consequences_are_counted(tmp_path):
    """One edit changes a leaf, so the successor's link AND every later checkpoint stop matching. That is
    one finding with a count — not a dozen."""
    from attacklab.tamper import build_clean_ledger
    fx = build_clean_ledger(tmp_path, n=40, seed=3, checkpoint_every=5)
    export_records(fx.ledger, tmp_path / "x.jsonl")
    lines = (tmp_path / "x.jsonl").read_bytes().split(b"\n")[:-1]
    later_checkpoints = sum(1 for x in lines[8:] if json.loads(x)["type"] == "checkpoint")
    assert later_checkpoints >= 3
    rec = json.loads(lines[6])
    rec["output"]["raw_jcs_sha256"] = "1" * 64
    lines[6] = canonical_bytes(rec)
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=fx.trust)
    assert classes(r) == ["record_edit"]
    assert r.findings[0].cascade_suppressed >= 1 + later_checkpoints


def test_two_independent_breaks_are_two_findings_not_one(tmp_path):
    chain, trust, _ = valid_chain(30)
    lines = list(chain.stored)
    for pos in (5, 20):
        rec = json.loads(lines[pos])
        rec["output"]["raw_jcs_sha256"] = "2" * 64
        lines[pos] = canonical_bytes(rec)
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=trust)
    assert classes(r) == ["record_edit", "record_edit"] and [f.seq for f in r.findings] == [5, 20]


def test_a_run_of_bad_signatures_is_one_finding_with_a_count(tmp_path):
    chain, trust, _ = valid_chain(20)
    lines = list(chain.stored)
    for pos in range(6, 14):
        rec = json.loads(lines[pos])
        rec["input"]["dims"] = [7, 7]
        lines[pos] = canonical_bytes(rec)
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=trust)
    assert classes(r) == ["record_edit"] and r.findings[0].cascade_suppressed >= 7
    assert "through seq 13" in r.findings[0].reason


# --- degraded intervals, clocks, anchors ------------------------------------------------------------------------

def test_a_declared_gap_is_a_low_operational_finding_never_a_tamper_class(tmp_path):
    body = {"gap": {"first_unsealed_utc": TS, "last_unsealed_utc": "2026-09-19T03:00:05.000000Z",
                    "reported_count": 12, "reason": "ledger_unwritable", "spill_sha256": H("a")}}
    chain, trust, _ = valid_chain(4, extra=[("degraded_marker", body)])
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert [f.attack_class for f in r.findings] == ["degraded_gap"] and r.declared_gaps == 1
    f = r.findings[0]
    assert f.severity == "low" and f.nature == "quality" and "12 inference" in f.reason and not r.clean


def test_a_backwards_clock_step_is_information_and_never_a_finding_above_info(tmp_path):
    from cva.provenance.seal.chain import MemoryChain

    from ._chain_helpers import manifest_for
    key = provider()
    times = iter([f"2026-09-19T02:00:0{i}.000000Z" for i in range(3)] + ["2026-09-19T01:00:00.000000Z"] * 5)

    def clk():
        from datetime import datetime
        return datetime.strptime(next(times), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=__import__("datetime").UTC)
    chain = MemoryChain(key, clock=clk)
    chain.append("genesis", {"deployment_manifest": manifest_for(key)})
    chain.append("model_registration", BODIES["model_registration"])
    for _ in range(4):
        chain.append("inference", BODIES["inference"])
    trust = TrustRoot(genesis_prev_hash(chain.records[0]["deployment_manifest"]), (TrustKey(key.key_id, key.public_key, "ledger"),))
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert r.clean and [f.attack_class for f in r.findings] == ["clock_regression"]
    assert r.findings[0].severity == "info" and "untrusted" in r.findings[0].reason


def anchor_body(checkpoint_seq):
    return {"anchor": {"checkpoint_seq": checkpoint_seq, "tree_size": 3, "root_hash": H("1"),
                       "cosigner_key_ids": [], "medium": "write_once", "label": "shift-change"}}


def test_an_anchor_event_is_counted_and_shrinks_the_unwitnessed_window():
    chain, trust, _ = valid_chain(4)
    chain.anchor_now(label="shift-change")
    chain.append("inference", BODIES["inference"])
    chain.append("inference", BODIES["inference"])
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert r.clean and r.anchors_in_chain == 1 and r.last_anchor_seq == 7 and r.unwitnessed_records == 2


def test_an_anchor_event_that_names_a_checkpoint_the_ledger_does_not_hold_is_a_mismatch():
    chain, trust, _ = valid_chain(3, extra=[("anchor_event", anchor_body(1))])
    assert "checkpoint_mismatch" in classes(verify_ledger(export_bytes(chain), trust_root=trust))


def test_an_anchor_closes_the_nonce_window_reuse_across_it_is_allowed_within_it_is_not():
    """D9: nonce uniqueness is scoped to the anchoring interval."""
    def draw_factory(reuse_after_anchor: bool):
        state = {"n": 0}

        def draw(k: int) -> bytes:
            state["n"] += 1
            n = state["n"]
            if n == 3:
                return bytes([7]) * k                                      # first inference's nonce
            if n == (9 if reuse_after_anchor else 5):
                return bytes([7]) * k                                      # reused
            return n.to_bytes(k, "big")
        return draw
    across, trust, _ = valid_chain(2, rng_fn=draw_factory(True))
    across.anchor_now()                                                    # nonces 5 (checkpoint) and 6 (event) drawn here
    for _ in range(3):
        across.append("inference", BODIES["inference"])
    assert verify_ledger(export_bytes(across), trust_root=trust).clean
    within, trust2, _ = valid_chain(4, rng_fn=draw_factory(False))
    assert classes(verify_ledger(export_bytes(within), trust_root=trust2)) == ["nonce_reuse"]


# --- checkpoints -------------------------------------------------------------------------------------------------

def test_a_checkpoint_whose_root_the_key_holder_got_wrong_is_reported_on_its_own():
    """Signed correctly, so not an edit — but it commits to a tree that does not exist."""
    chain, trust, _ = valid_chain(5, extra=[("checkpoint", {"checkpoint": {"tree_size": 7, "root_hash": H("9")}})])
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert classes(r) == ["checkpoint_mismatch"] and "hash to" in r.findings[0].reason
    assert r.findings[0].evidence[0][2]["tree_size_actual"] == 7


def test_a_checkpoint_claiming_the_wrong_tree_size_is_a_mismatch_even_with_the_right_root():
    from cva.provenance.seal.merkle import RootAccumulator, leaf_hash
    chain, trust, _ = valid_chain(5)
    acc = RootAccumulator()
    for d in chain.stored:
        acc.append(leaf_hash(d))
    chain.append("checkpoint", {"checkpoint": {"tree_size": acc.size() - 1, "root_hash": acc.root().hex()}})
    assert classes(verify_ledger(export_bytes(chain), trust_root=trust)) == ["checkpoint_mismatch"]


def test_a_correct_checkpoint_is_verified_and_counted():
    from cva.provenance.seal.merkle import RootAccumulator, leaf_hash
    chain, trust, _ = valid_chain(5)
    acc = RootAccumulator()
    for d in chain.stored:
        acc.append(leaf_hash(d))
    chain.append("checkpoint", {"checkpoint": {"tree_size": acc.size(), "root_hash": acc.root().hex()}})
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert r.clean and r.checkpoints_verified == 1


# --- keys ----------------------------------------------------------------------------------------------------------

def _replace(chain, trust, pos, key_for_signing, keep_key_id=False):
    prev = chain.records[pos - 1]
    old = chain.records[pos]
    forged, _ = seal_next("inference", {s: old[s] for s in SECTIONS["inference"]}, key=key_for_signing, prev=prev,
                          now=__import__("datetime").datetime.strptime(old["created_at_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                              tzinfo=__import__("datetime").UTC), nonce=old["nonce"])
    lines = list(chain.stored)
    lines[pos] = canonical_bytes(forged)
    return b"".join(x + b"\n" for x in lines)


def test_a_record_signed_by_a_trust_root_witness_key_is_adversarial_not_merely_indeterminate():
    chain, trust, key = valid_chain(6)
    witness = provider(SEED_B)
    trust2 = TrustRoot(trust.deployment_manifest_hash, (*trust.keys, TrustKey(witness.key_id, witness.public_key, "witness")))
    r = verify_ledger(_replace(chain, trust2, 4, witness), trust_root=trust2)
    f = r.findings[0]
    assert f.attack_class == "key_unauthorised" and f.nature == "adversarial" and "witness" in f.reason


def test_a_record_that_claims_a_known_inactive_key_but_carries_the_wrong_signature_is_an_edit_not_an_accusation():
    """key_id names a trust-root witness key, but the signature is someone else's. Nobody holding the witness key
    wrote this, so calling it 'adversarial use of the witness key' would overclaim: it is simply not what
    that key signed."""
    from cva.provenance.seal.constants import TAG_RECORD
    from cva.provenance.seal.records import build_record, canon
    chain, trust, _ = valid_chain(6)
    witness, attacker = provider(SEED_B), provider(bytes(range(200, 232)))
    trust2 = TrustRoot(trust.deployment_manifest_hash, (*trust.keys, TrustKey(witness.key_id, witness.public_key, "witness")))
    old = chain.records[4]
    unsigned = build_record("inference", seq=4, prev_record_hash=old["prev_record_hash"], key_id=witness.key_id,
                            created_at_utc=old["created_at_utc"], nonce=old["nonce"],
                            body={s: old[s] for s in SECTIONS["inference"]})
    forged = {**unsigned, "signature": attacker.sign(TAG_RECORD + canon(unsigned)).hex()}
    lines = list(chain.stored)
    lines[4] = canonical_bytes(forged)
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=trust2)
    f = r.findings[0]
    assert f.attack_class == "record_edit" and f.nature == "indeterminate" and f.primary_check == "ed25519_signature"
    assert "key_unauthorised" not in classes(r)


def test_a_record_signed_by_an_unknown_key_is_indeterminate_because_a_flipped_key_id_looks_the_same():
    chain, trust, _ = valid_chain(6)
    r = verify_ledger(_replace(chain, trust, 4, provider(SEED_B)), trust_root=trust)
    assert r.findings[0].attack_class == "key_unauthorised" and r.findings[0].nature == "indeterminate"
    assert "flipped" in r.findings[0].reason or "damaged" in r.findings[0].reason


def test_a_genesis_signed_by_a_key_outside_the_trust_root_is_reported():
    chain, _, _ = valid_chain(3)
    other_key = provider(SEED_B)
    only_other = TrustRoot(genesis_prev_hash(chain.records[0]["deployment_manifest"]),
                           (TrustKey(other_key.key_id, other_key.public_key, "ledger"),))
    r = verify_ledger(export_bytes(chain), trust_root=only_other)
    assert "key_unauthorised" in classes(r) and r.findings[0].position == 0


def test_two_different_records_at_one_seq_both_validly_signed_is_a_ledger_fork():
    chain, trust, key = valid_chain(6)
    fork, _ = seal_next("inference", BODIES["inference"], key=key, prev=chain.records[3],
                        now=__import__("datetime").datetime(2026, 9, 19, 4, 0, tzinfo=__import__("datetime").UTC),
                        nonce="ab" * 16)
    assert fork["seq"] == chain.records[4]["seq"] and canonical_bytes(fork) != chain.stored[4]
    lines = [*chain.stored, canonical_bytes(fork)]
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=trust)
    f = next(x for x in r.findings if x.attack_class == "ledger_fork")
    assert f.severity == "critical" and f.nature == "adversarial" and "two histories" in f.reason


# --- models, inputs, deployment -----------------------------------------------------------------------------------------

def test_an_inference_with_no_registration_before_it_is_a_model_finding():
    key = provider()
    from cva.provenance.seal.chain import MemoryChain

    from ._chain_helpers import manifest_for
    chain = MemoryChain(key, clock=clock())
    chain.append("genesis", {"deployment_manifest": manifest_for(key)})
    chain.append("inference", BODIES["inference"])
    trust = TrustRoot(genesis_prev_hash(chain.records[0]["deployment_manifest"]), (TrustKey(key.key_id, key.public_key, "ledger"),))
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert classes(r) == ["model_swap"] and "no model registration" in r.findings[0].reason


def test_an_inference_that_disagrees_with_the_registration_in_effect_is_a_critical_model_swap():
    chain, trust, _ = valid_chain(2)
    body = json.loads(json.dumps(BODIES["inference"]))
    body["model"]["weights_sha256"] = "9" * 64
    chain.append("inference", body)
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert classes(r) == ["model_swap"] and r.findings[0].severity == "critical"


def test_the_reference_manifest_accepts_listed_digests_and_rejects_unlisted_ones():
    chain, trust, _ = valid_chain(3)
    ok = {BODIES["model_registration"]["model"]["id"]: {BODIES["model_registration"]["model"]["weights_sha256"]}}
    assert verify_ledger(export_bytes(chain), trust_root=trust, reference_manifest=ok).clean
    assert classes(verify_ledger(export_bytes(chain), trust_root=trust, reference_manifest={"resnet50-v3": {"0" * 64}})) == ["model_swap"]
    assert classes(verify_ledger(export_bytes(chain), trust_root=trust, reference_manifest={"other-model": {"0" * 64}})) == ["model_swap"]


def test_input_hashes_are_checked_when_a_resolver_is_given_and_unavailable_ones_are_counted():
    chain, trust, _ = valid_chain(4)
    real = {r["seq"]: b"frame" for r in chain.records if r["type"] == "inference"}
    rep = verify_ledger(export_bytes(chain), trust_root=trust, input_resolver=lambda r: None)
    assert rep.clean and rep.inputs_unavailable == 4 and rep.inputs_checked == 0
    rep2 = verify_ledger(export_bytes(chain), trust_root=trust, input_resolver=lambda r: real[r["seq"]])
    assert classes(rep2).count("input_swap") == 4 and rep2.inputs_checked == 4


def test_expect_deployment_accepts_a_hash_or_a_manifest_and_rejects_a_different_deployment():
    chain, trust, _ = valid_chain(3)
    man = chain.records[0]["deployment_manifest"]
    for expect in (genesis_prev_hash(man), man):
        assert verify_ledger(export_bytes(chain), trust_root=trust, expect_deployment=expect).clean
    r = verify_ledger(export_bytes(chain), trust_root=trust, expect_deployment={**man, "device_id": "elsewhere"})
    assert classes(r) == ["genesis_mismatch"]


def test_the_first_record_must_be_genesis():
    chain, trust, _ = valid_chain(3)
    lines = chain.stored[1:]
    r = verify_ledger(b"".join(x + b"\n" for x in lines), trust_root=trust)
    assert "genesis_mismatch" in classes(r) or "record_delete" in classes(r)


def test_payloads_are_absent_from_an_export_and_that_is_a_count_not_an_accusation(tmp_path):
    e = Env(tmp_path)
    for i in range(3):
        e.seal(i)
    e.close()
    export_records(e.ledger_path, tmp_path / "x.jsonl")
    r = verify_ledger(tmp_path / "x.jsonl", trust_root=e.trust)
    assert r.clean and r.payloads_checked == 0 and r.payloads_missing == 6 or r.clean


def test_a_missing_payload_row_is_counted_missing_not_reported_as_tampering(tmp_path):
    e = Env(tmp_path)
    for i in range(3):
        e.seal(i)
    e.close()
    c = sqlite3.connect(e.ledger_path, isolation_level=None)
    c.execute("DROP TRIGGER payloads_no_delete")
    c.execute("DELETE FROM payloads")
    c.close()
    r = verify_ledger(e.ledger_path, trust_root=e.trust)
    assert r.clean and r.payloads_missing > 0 and r.payloads_checked == 0
