"""Inclusion proofs (plan §7.10; gate C7): one record in a signed root, from the record + proof + anchor alone."""
from __future__ import annotations

import json

import pytest

from cva.provenance.seal.anchor import logbook_anchor
from cva.provenance.seal.errors import SealError
from cva.provenance.seal.proof import encode_proof, make_inclusion_proof, verify_inclusion_proof
from cva.provenance.seal.verify import export_payloads, load_payloads, verify_ledger

from ._anchor_helpers import AnchorEnv


def build(tmp_path, n):
    e = AnchorEnv(tmp_path)
    for i in range(n):
        e.seal(i)
    a = e.anchor()
    return e, a, e.export()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 9, 12])
def test_every_record_in_the_anchored_tree_has_a_proof_that_verifies_at_odd_and_even_sizes(tmp_path, n):
    e, a, exp = build(tmp_path, n)
    size = a["checkpoint"]["checkpoint"]["tree_size"]
    for seq in range(size):
        p = make_inclusion_proof(exp, seq, a)
        r = verify_inclusion_proof(p, a, e.trust)
        assert r.valid and r.signature_ok is True and r.seq == seq, (n, seq, r.problems)
    assert size == n + 2                                        # genesis + registration + n inferences


def test_a_proof_verifies_with_no_ledger_at_all(tmp_path):
    e, a, exp = build(tmp_path, 8)
    p = make_inclusion_proof(exp, 5, a)
    e.ledger_path.unlink()
    exp.unlink()
    r = verify_inclusion_proof(encode_proof(p), json.dumps(a).encode(), e.trust)
    assert r.valid and r.record_type in ("inference", "model_registration")


def test_a_proof_does_not_verify_at_the_wrong_index_or_with_a_changed_path(tmp_path):
    e, a, exp = build(tmp_path, 9)
    p = make_inclusion_proof(exp, 4, a)
    assert not verify_inclusion_proof({**p, "seq": 5}, a, e.trust).valid
    bad_path = list(p["path"])
    bad_path[0] = ("0" if bad_path[0][0] != "0" else "1") + bad_path[0][1:]
    assert not verify_inclusion_proof({**p, "path": bad_path}, a, e.trust).valid
    assert not verify_inclusion_proof({**p, "path": p["path"][:-1]}, a, e.trust).valid
    assert not verify_inclusion_proof({**p, "path": [*p["path"], "0" * 64]}, a, e.trust).valid


def test_a_proof_for_an_edited_record_fails_even_though_the_path_is_unchanged(tmp_path):
    e, a, exp = build(tmp_path, 9)
    p = make_inclusion_proof(exp, 4, a)
    rec = json.loads(p["record"])
    rec["dims"] = [65, 64]
    from cva.provenance.seal.canonical import canonical_bytes
    r = verify_inclusion_proof({**p, "record": canonical_bytes(rec).decode()}, a, e.trust)
    assert not r.valid and "audit path" in r.problems[0]


def _logbook_for(rows: list[bytes], size: int):
    from cva.provenance.seal.merkle import leaf_hash, mth
    return logbook_anchor(size, mth([leaf_hash(r) for r in rows[:size]]).hex())


def test_a_record_whose_signature_is_bad_is_rejected_even_when_it_really_is_in_the_anchored_tree(tmp_path):
    """Inclusion proves position; the record's own signature is checked on top of it. The anchor here is a logbook
    entry taken over the forged history itself, so the audit path is genuinely valid — only the signature is wrong."""
    e, a, exp = build(tmp_path, 6)
    rows = exp.read_bytes().split(b"\n")[:-1]
    rec = json.loads(rows[3])
    rec["signature"] = "0" * 128
    from cva.provenance.seal.canonical import canonical_bytes
    rows[3] = canonical_bytes(rec)
    forged = tmp_path / "forged.jsonl"
    forged.write_bytes(b"".join(r + b"\n" for r in rows))
    lb = _logbook_for(rows, a["checkpoint"]["checkpoint"]["tree_size"])
    r = verify_inclusion_proof(make_inclusion_proof(forged, 3, lb), lb, e.trust)
    assert not r.valid and r.signature_ok is False and "signature does not verify" in r.problems[0]


def test_a_record_placed_at_an_index_other_than_its_own_seq_is_rejected(tmp_path):
    e, a, exp = build(tmp_path, 6)
    rows = exp.read_bytes().split(b"\n")[:-1]
    rows[2], rows[4] = rows[4], rows[2]                                     # a reordered history, with a genuine path
    swapped = tmp_path / "swapped.jsonl"
    swapped.write_bytes(b"".join(r + b"\n" for r in rows))
    lb = _logbook_for(rows, a["checkpoint"]["checkpoint"]["tree_size"])
    r = verify_inclusion_proof(make_inclusion_proof(swapped, 2, lb), lb, e.trust)
    assert not r.valid and "places it at index" in r.problems[0]


def test_a_proof_against_a_different_anchor_or_an_unsigned_size_is_refused(tmp_path):
    e, a, exp = build(tmp_path, 6)
    p = make_inclusion_proof(exp, 2, a)
    other = logbook_anchor(a["checkpoint"]["checkpoint"]["tree_size"], "0" * 64)
    r = verify_inclusion_proof(p, other, e.trust)
    assert not r.valid and "different tree size or root" in r.problems[0]


def test_a_tampered_anchor_makes_every_proof_against_it_invalid(tmp_path):
    e, a, exp = build(tmp_path, 6)
    p = make_inclusion_proof(exp, 2, a)
    a["checkpoint"]["checkpoint"]["root_hash"] = p["root_hash"] = "1" * 64
    r = verify_inclusion_proof(p, a, e.trust)
    assert not r.valid and not r.anchor_valid


def test_a_proof_works_against_a_logbook_root_without_any_signature(tmp_path):
    e, a, exp = build(tmp_path, 6)
    cp = a["checkpoint"]["checkpoint"]
    lb = logbook_anchor(cp["tree_size"], cp["root_hash"])
    p = make_inclusion_proof(exp, 2, lb)
    assert verify_inclusion_proof(p, lb, e.trust).valid


def test_the_seq_must_be_inside_the_anchored_tree(tmp_path):
    e, a, exp = build(tmp_path, 4)
    with pytest.raises(SealError, match="not inside the anchored tree"):
        make_inclusion_proof(exp, a["checkpoint"]["checkpoint"]["tree_size"], a)
    with pytest.raises(SealError):
        make_inclusion_proof(exp, -1, a)


def test_a_malformed_proof_is_a_result_never_an_exception(tmp_path):
    e, a, _ = build(tmp_path, 4)
    for junk in (b"nope", b"{}", {"v": "x"}, {"v": "cva-seal/1", "seq": "1", "tree_size": 1, "root_hash": "0",
                                              "record": "x", "path": []}):
        r = verify_inclusion_proof(junk, a, e.trust)
        assert not r.valid and r.problems


def test_a_record_signed_by_a_rotated_in_key_is_proven_included_but_its_signature_is_not_checked_here(tmp_path):
    from ._anchor_helpers import NEW_SEED
    from ._chain_helpers import provider
    e = AnchorEnv(tmp_path)
    for i in range(3):
        e.seal(i)
    e.sealer.rotate_key(provider(NEW_SEED))
    for i in range(3, 6):
        e.seal(i)
    e.sealer.close()
    from cva.provenance.seal.anchor import export_anchor
    from cva.provenance.seal.store import SealedLedger
    led = SealedLedger.open(e.ledger_path, key=provider(NEW_SEED), clock=e.clock, rng=e.rng, background_flush=False)
    a = export_anchor(led, tmp_path / "post.json")
    led.close()
    logbook = logbook_anchor(a["checkpoint"]["checkpoint"]["tree_size"], a["checkpoint"]["checkpoint"]["root_hash"])
    from cva.provenance.seal.verify import export_records
    export_records(e.ledger_path, tmp_path / "x.jsonl")
    p = make_inclusion_proof(tmp_path / "x.jsonl", 8, logbook)
    r = verify_inclusion_proof(p, logbook, e.trust)
    assert r.valid and r.signature_ok is None and "rotated-in key" in r.notes[0]


# --- the payload sidecar (an export carries records only) --------------------------------------------------------------

def test_a_payload_sidecar_lets_an_export_be_checked_for_output_payload_tampering(tmp_path):
    e, a, exp = build(tmp_path, 5)
    n = export_payloads(e.ledger_path, tmp_path / "x.payloads.jsonl")
    assert n >= 3
    pl = load_payloads(tmp_path / "x.payloads.jsonl")
    without = verify_ledger(exp, trust_root=e.trust)
    with_ = verify_ledger(exp, trust_root=e.trust, payloads=pl)
    assert without.payloads_checked == 0 and without.payloads_missing > 0
    assert with_.payloads_checked > 0 and with_.clean
    victim = next(k for k, v in pl.items() if b'"cls"' in v)
    pl[victim] = pl[victim] + b" "
    bad = verify_ledger(exp, trust_root=e.trust, payloads=pl)
    got = {f.attack_class for f in bad.findings if f.severity != "info"}
    assert got == {"output_payload_mismatch"}          # (every record here sealed the same output, so several report it)


def test_a_malformed_payload_sidecar_is_rejected_not_half_read(tmp_path):
    from cva.provenance.seal.errors import LedgerUnreadable
    (tmp_path / "p.jsonl").write_bytes(b'{"address":"ab","b64":"AAAA"}\nnot json\n')
    with pytest.raises(LedgerUnreadable, match="line 2"):
        load_payloads(tmp_path / "p.jsonl")
