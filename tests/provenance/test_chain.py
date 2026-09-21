"""Signing, the hash chain, and the in-memory verifier (plan §5.5, §11; gate C3)."""
from __future__ import annotations

import hashlib
import random
import time
from datetime import UTC, datetime

import pytest

from cva.provenance.seal import constants as C
from cva.provenance.seal.chain import (
    MemoryChain,
    check_record,
    link_hash,
    seal_next,
    sign_record,
    verify_chain,
    verify_signature,
)
from cva.provenance.seal.errors import InvalidRecord
from cva.provenance.seal.keys import verify_ed25519
from cva.provenance.seal.records import build_record, canon, key_id_of, record_hash

from ._chain_helpers import (
    SEED_B,
    clock,
    ledger_keys,
    make_chain,
    manifest_for,
    provider,
    rng,
)
from ._fixtures import BODIES, TS, H

KEY = provider()
NOW = datetime(2026, 9, 19, 2, 14, 7, 482913, tzinfo=UTC)
NONCE = "cd" * 16


def unsigned(rtype="checkpoint", seq=5, key=KEY):
    return build_record(rtype, seq=seq, prev_record_hash=H("2"), key_id=key.key_id, created_at_utc=TS,
                        nonce=NONCE, body=BODIES[rtype])


def code(check):
    return check.failure.code if check.failure else None


# --- signing (D1) -----------------------------------------------------------------------------------------

def test_the_signature_is_pure_ed25519_over_the_tag_then_the_canonical_bytes():
    u = unsigned()
    s = sign_record(u, KEY)
    sig = bytes.fromhex(s["signature"])
    assert len(sig) == 64
    assert verify_ed25519(KEY.public_key, C.TAG_RECORD + canon(u), sig)
    assert sig == KEY.sign(C.TAG_RECORD + canon(u))                     # deterministic: same input, same bytes


def test_a_signature_made_without_the_tag_does_not_verify_and_neither_do_other_tags():
    """D1: a signature over one structure must not be presentable as valid over another."""
    u = unsigned()
    untagged = u | {"signature": KEY.sign(canon(u)).hex()}
    assert not verify_signature(untagged, KEY.public_key)
    good = sign_record(u, KEY)
    sig = bytes.fromhex(good["signature"])
    for tag in (C.TAG_CHECKPOINT, C.TAG_COSIGN, C.TAG_ROTATION_POP, C.TAG_WITNESS, b""):
        assert not verify_ed25519(KEY.public_key, tag + canon(u), sig)
    for tag in (C.TAG_CHECKPOINT, C.TAG_COSIGN, C.TAG_ROTATION_POP, C.TAG_WITNESS):
        other = u | {"signature": KEY.sign(tag + canon(u)).hex()}       # signed for another purpose
        assert not verify_signature(other, KEY.public_key)


def test_sign_record_does_not_mutate_its_input_and_the_result_validates():
    u = unsigned()
    before = dict(u)
    s = sign_record(u, KEY)
    assert u == before and "signature" not in u and set(s) == set(u) | {"signature"}


def test_sign_record_refuses_a_record_that_names_another_key():
    other = provider(SEED_B)
    with pytest.raises(ValueError, match="names key"):
        sign_record(unsigned(), other)


def test_sign_record_refuses_to_resign_a_signed_record():
    with pytest.raises(ValueError, match="already signed"):
        sign_record(sign_record(unsigned(), KEY), KEY)


def test_verify_signature_fails_on_any_field_edit_a_wrong_key_or_a_missing_signature():
    s = sign_record(unsigned("inference", seq=9), KEY)
    assert verify_signature(s, KEY.public_key)
    assert not verify_signature(s, provider(SEED_B).public_key)
    assert not verify_signature({k: v for k, v in s.items() if k != "signature"}, KEY.public_key)
    for path, val in [(("seq",), 10), (("nonce",), "ce" * 16), (("created_at_utc",), "2026-09-19T02:14:08.482913Z"),
                      (("input", "dims"), [1, 1]), (("output", "jcs_sha256"), "0" * 64), (("model", "id"), "other")]:
        edited = {**s, path[0]: s[path[0]]}
        node = edited
        for p in path[:-1]:
            node[p] = dict(node[p])
            node = node[p]
        node[path[-1]] = val
        assert not verify_signature(edited, KEY.public_key), path
    assert not verify_signature({**s, "signature": "zz" * 64}, KEY.public_key)
    assert not verify_signature({**s, "signature": s["signature"][:-2]}, KEY.public_key)


def test_signing_the_same_record_under_two_keys_gives_two_different_signatures():
    other = provider(SEED_B)
    a = sign_record(unsigned(), KEY)["signature"]
    b = sign_record(unsigned(key=other), other)["signature"]
    assert a != b


# --- the chain link (S4) ----------------------------------------------------------------------------------

def test_link_is_sha256_of_0x02_then_record_hash_then_the_signature_bytes():
    s = sign_record(unsigned(), KEY)
    expected = hashlib.sha256(b"\x02" + hashlib.sha256(canon(s)).digest() + bytes.fromhex(s["signature"])).hexdigest()
    assert link_hash(s) == expected


def test_the_link_covers_the_signature_not_just_the_fields():
    s = sign_record(unsigned(), KEY)
    other_sig = {**s, "signature": bytes([s["signature"] and int(s["signature"][:2], 16) ^ 1]).hex() + s["signature"][2:]}
    assert record_hash(s) == record_hash(other_sig)                     # same fields ...
    assert link_hash(s) != link_hash(other_sig)                         # ... different link


def test_the_link_is_domain_separated_from_the_plain_record_hash():
    s = sign_record(unsigned(), KEY)
    assert link_hash(s) != record_hash(s).hex()
    assert link_hash(s) != hashlib.sha256(record_hash(s) + bytes.fromhex(s["signature"])).hexdigest()   # no 0x02


def test_link_hash_requires_a_valid_signed_record():
    with pytest.raises(InvalidRecord):
        link_hash(unsigned())


# --- seal_next --------------------------------------------------------------------------------------------

def genesis_body(key=KEY):
    return {"deployment_manifest": manifest_for(key)}


def test_the_first_record_must_be_genesis_and_genesis_can_only_be_first():
    with pytest.raises(ValueError, match="must be genesis"):
        seal_next("inference", BODIES["inference"], key=KEY, prev=None, now=NOW, nonce=NONCE)
    g, _ = seal_next("genesis", genesis_body(), key=KEY, prev=None, now=NOW, nonce=NONCE)
    with pytest.raises(ValueError, match="only be the first"):
        seal_next("genesis", genesis_body(), key=KEY, prev=g, now=NOW, nonce=NONCE)


def test_seal_next_numbers_and_links_records_and_never_mutates_prev():
    g, gb = seal_next("genesis", genesis_body(), key=KEY, prev=None, now=NOW, nonce=NONCE)
    snapshot = dict(g)
    r1, b1 = seal_next("inference", BODIES["inference"], key=KEY, prev=g, now=NOW, nonce="01" * 16)
    assert g == snapshot
    assert (g["seq"], r1["seq"]) == (0, 1)
    assert r1["prev_record_hash"] == link_hash(g)
    assert g["prev_record_hash"] != "0" * 64                             # D5: not the all-zero start
    assert verify_chain([gb, b1], ledger_keys=ledger_keys(KEY)).ok


def test_genesis_must_be_signed_by_the_key_its_manifest_names():
    other = provider(SEED_B)
    with pytest.raises(InvalidRecord, match="key its manifest names"):
        seal_next("genesis", genesis_body(KEY), key=other, prev=None, now=NOW, nonce=NONCE)


# --- a clean chain verifies ------------------------------------------------------------------------------

def test_a_mixed_type_chain_verifies_and_reports_its_length():
    c = make_chain(8, types=("inference", "checkpoint", "degraded_marker", "scan_record", "analyst_event",
                             "model_registration"))
    res = c.verify()
    assert res.ok and res.records_checked == 9 and res.failure is None
    assert {r["type"] for r in c.records} == {"genesis", "inference", "checkpoint", "degraded_marker",
                                              "scan_record", "analyst_event", "model_registration"}


def test_a_thousand_record_chain_verifies_quickly():
    t0 = time.perf_counter()
    c = make_chain(999)
    built = time.perf_counter() - t0
    t1 = time.perf_counter()
    res = c.verify()
    checked = time.perf_counter() - t1
    assert res.ok and res.records_checked == 1000
    assert built < 20 and checked < 20, (built, checked)                 # generous: only guards pathologies


def test_the_stored_bytes_are_one_canonical_line_each():
    for data in make_chain(5).stored:
        assert data.startswith(b"{") and data.endswith(b"}") and b"\n" not in data


def test_records_carry_distinct_nonces_and_strictly_increasing_seq():
    c = make_chain(50)
    assert [r["seq"] for r in c.records] == list(range(51))
    assert len({r["nonce"] for r in c.records}) == 51


def test_an_empty_ledger_never_verifies():
    """'Verified over zero records' would be false assurance."""
    res = verify_chain([], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "empty_ledger"


def test_verifying_with_an_inconsistent_key_map_is_a_programming_error():
    with pytest.raises(ValueError, match="not SHA-256"):
        verify_chain(make_chain(1).stored, ledger_keys={"f" * 64: KEY.public_key})


# --- the gate: flipping any bit of any record makes verification fail --------------------------------------

def flip_bit(data: bytes, byte: int, bit: int) -> bytes:
    b = bytearray(data)
    b[byte] ^= 1 << bit
    return bytes(b)


def _ctx(c: MemoryChain, i: int):
    """(prev record, active key) for position i of a chain whose records 0..i-1 are known good."""
    return (c.records[i - 1] if i else None), c.records[0]["key_id"]


def test_exhaustively_every_bit_of_every_record_in_a_small_chain_is_load_bearing():
    c = make_chain(5, types=("checkpoint", "degraded_marker", "scan_record", "checkpoint", "degraded_marker"))
    keys = ledger_keys(KEY)
    assert c.verify().ok
    total = 0
    for i, data in enumerate(c.stored):
        prev, active = _ctx(c, i)
        assert check_record(data, i, prev, active, keys)[1] is None       # the unmodified record is fine
        for byte in range(len(data)):
            for bit in range(8):
                _, failure = check_record(flip_bit(data, byte, bit), i, prev, active, keys)
                assert failure is not None, f"undetected flip: record {i} byte {byte} bit {bit}"
                assert failure.position == i
                total += 1
    assert total > 20_000                                                  # a genuinely exhaustive sweep


def test_the_exhaustive_sweep_agrees_with_whole_chain_verification():
    """`check_record` is what `verify_chain` loops over — sample flips end to end to prove they agree."""
    c = make_chain(5, types=("checkpoint", "degraded_marker", "scan_record"))
    keys = ledger_keys(KEY)
    r = random.Random(7)
    for _ in range(300):
        i = r.randrange(len(c.stored))
        mutated = list(c.stored)
        mutated[i] = flip_bit(mutated[i], r.randrange(len(mutated[i])), r.randrange(8))
        res = verify_chain(mutated, ledger_keys=keys)
        assert not res.ok and res.failure.position == i


def test_a_sampled_bit_flip_sweep_over_a_thousand_record_chain_finds_nothing_undetected():
    c = make_chain(999, types=("inference", "checkpoint", "scan_record"))
    keys = ledger_keys(KEY)
    r = random.Random(20260919)
    for _ in range(3000):
        i = r.randrange(len(c.stored))
        prev, active = _ctx(c, i)
        data = c.stored[i]
        _, failure = check_record(flip_bit(data, r.randrange(len(data)), r.randrange(8)), i, prev, active, keys)
        assert failure is not None, f"undetected flip in record {i}"


def test_a_change_confined_to_one_records_signature_is_caught_at_that_record_and_at_its_successor():
    """The link covers the signature, so it is double-covered: the record's own check fails, and even a
    verifier that skipped signatures would see the successor's link break."""
    c = make_chain(4)
    keys = ledger_keys(KEY)
    i = 2
    rec = dict(c.records[i])
    rec["signature"] = ("00" if rec["signature"][:2] != "00" else "01") + rec["signature"][2:]
    from cva.provenance.seal.canonical import canonical_bytes
    assert not verify_chain([*c.stored[:i], canonical_bytes(rec), *c.stored[i + 1:]], ledger_keys=keys).ok
    assert link_hash(rec) != c.records[i + 1]["prev_record_hash"]


def test_a_flipped_bit_in_a_hex_field_is_caught_whatever_it_becomes():
    """Hex 'a'->'A' style flips become uppercase (rejected), digit<->letter flips change the value
    (signature fails); non-hex characters fail the schema. Every path is a detection."""
    c = make_chain(1)
    keys = ledger_keys(KEY)
    data = c.stored[1]
    start = data.index(b'"nonce":"') + len(b'"nonce":"')
    for off in range(32):
        for bit in range(8):
            mutated = [c.stored[0], flip_bit(data, start + off, bit)]
            assert not verify_chain(mutated, ledger_keys=keys).ok


# --- tampering: each ends in a specific low-level failure -----------------------------------------------------

def test_deleting_an_interior_record_is_a_sequence_gap():
    c = make_chain(6)
    res = verify_chain([*c.stored[:3], *c.stored[4:]], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_seq" and res.failure.position == 3


def test_deleting_and_renumbering_breaks_the_signature_instead():
    c = make_chain(6)
    rec = dict(c.records[4])
    rec["seq"] = 3
    from cva.provenance.seal.canonical import canonical_bytes
    forged = canonical_bytes(rec)
    res = verify_chain([*c.stored[:3], forged, *c.stored[5:]], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_signature" and res.failure.position == 3


def test_reordering_two_records_is_detected():
    c = make_chain(6)
    s = list(c.stored)
    s[2], s[4] = s[4], s[2]
    res = verify_chain(s, ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_seq" and res.failure.position == 2


def test_replaying_an_old_record_at_the_tip_is_detected():
    c = make_chain(6)
    res = verify_chain([*c.stored, c.stored[3]], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_seq" and res.failure.position == 7


def test_a_record_signed_by_someone_elses_key_is_unauthorised_even_when_perfectly_formed():
    c = make_chain(4)
    attacker = provider(SEED_B)
    forged, data = seal_next("inference", BODIES["inference"], key=attacker, prev=c.records[-1],
                             now=NOW, nonce="ee" * 16)
    assert verify_signature(forged, attacker.public_key)               # a well-formed, validly signed record
    res = verify_chain([*c.stored, data], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "key_unauthorised" and res.failure.position == 5
    both = verify_chain([*c.stored, data], ledger_keys={**ledger_keys(KEY), **ledger_keys(attacker)})
    assert not both.ok and code(both) == "key_unauthorised"           # a known-but-inactive key is no better


def test_editing_a_record_and_re_signing_with_the_attackers_key_is_unauthorised():
    c = make_chain(4)
    attacker = provider(SEED_B)
    edited = {k: v for k, v in c.records[2].items() if k not in ("signature",)}
    edited["key_id"] = attacker.key_id
    forged = sign_record(edited, attacker)
    from cva.provenance.seal.records import stored_bytes
    res = verify_chain([*c.stored[:2], stored_bytes(forged), *c.stored[3:]], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "key_unauthorised" and res.failure.position == 2


def test_editing_a_record_without_the_key_and_patching_the_next_link_breaks_a_signature():
    """T1c: recompute k+1's prev_record_hash and it no longer matches its own signature."""
    c = make_chain(6)
    from cva.provenance.seal.canonical import canonical_bytes
    edited = {**c.records[2], "nonce": "ff" * 16}
    nxt = {**c.records[3], "prev_record_hash": link_hash(edited)}
    res = verify_chain([*c.stored[:2], canonical_bytes(edited), canonical_bytes(nxt), *c.stored[4:]],
                       ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_signature" and res.failure.position == 2


def test_swapping_in_a_signature_from_the_same_key_but_another_record_fails():
    c = make_chain(4)
    from cva.provenance.seal.canonical import canonical_bytes
    swapped = {**c.records[2], "signature": c.records[3]["signature"]}
    res = verify_chain([*c.stored[:2], canonical_bytes(swapped), *c.stored[3:]], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_signature" and res.failure.position == 2


def test_splicing_one_devices_records_under_another_devices_genesis_is_a_broken_chain():
    """D5: same key, different deployment manifests -> different genesis -> the successor's link fails."""
    a = make_chain(5, device_id="jetson-07")
    b = make_chain(0, device_id="jetson-99")
    res = verify_chain([b.stored[0], *a.stored[1:]], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "chain_broken" and res.failure.position == 1


def test_a_second_genesis_or_a_missing_genesis_is_rejected():
    c = make_chain(3)
    res = verify_chain(c.stored[1:], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "bad_seq" and res.failure.position == 0
    other = make_chain(0, device_id="jetson-99")
    res2 = verify_chain([*c.stored[:2], other.stored[0]], ledger_keys=ledger_keys(KEY))
    assert not res2.ok and res2.failure.position == 2


def test_rewriting_the_encoding_of_a_valid_record_is_its_own_finding():
    c = make_chain(2)
    import json
    pretty = json.dumps(json.loads(c.stored[1]), indent=1).encode()
    res = verify_chain([c.stored[0], pretty], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "non_canonical_encoding" and res.failure.position == 1


def test_uppercase_hex_in_a_signed_record_is_malformed_not_a_second_spelling():
    """T14: `deadbeef` and `DEADBEEF` are the same bytes but must not both be valid spellings."""
    c = make_chain(2)
    data = c.stored[1]
    marker = b'"prev_record_hash":"'
    start = data.index(marker) + len(marker)
    original = data[start:start + 64]
    assert original != original.upper()                       # the fixture hash contains hex letters
    upper = data[:start] + original.upper() + data[start + 64:]
    res = verify_chain([c.stored[0], upper], ledger_keys=ledger_keys(KEY))
    assert not res.ok and code(res) == "malformed_record" and res.failure.position == 1


def test_the_first_failure_is_the_one_reported_and_later_damage_is_not_a_second_finding():
    c = make_chain(8)
    s = list(c.stored)
    s[2] = flip_bit(s[2], 40, 0)
    s[5] = flip_bit(s[5], 40, 0)
    res = verify_chain(s, ledger_keys=ledger_keys(KEY))
    assert res.failure.position == 2 and res.records_checked == 2


def test_a_key_rotation_hands_signing_to_the_new_key_and_the_chain_still_verifies():
    from cva.provenance.seal.chain import rotation_body
    c = make_chain(2)
    new = provider(bytes(range(50, 82)))
    c.rotate_key(new)
    c.append("inference", BODIES["inference"])
    c.append("inference", BODIES["inference"])
    assert c.verify().ok
    assert c.records[3]["key_id"] == KEY.key_id and c.records[4]["key_id"] == new.key_id
    assert c.records[3]["rotation"] == rotation_body(KEY, new, 3)["rotation"]


def test_the_outgoing_key_cannot_sign_after_its_own_rotation():
    """effective_seq == seq + 1: the record right after the rotation is the new key's."""
    c = make_chain(2)
    new = provider(bytes(range(50, 82)))
    c.rotate_key(new)
    c._key = KEY                                             # the old key keeps signing: a rotation that did not happen
    c.append("inference", BODIES["inference"])
    res = c.verify()
    assert not res.ok and code(res) == "key_unauthorised" and res.failure.position == 4


def test_a_rotation_whose_incoming_key_never_proved_possession_is_refused():
    c = make_chain(2)
    new, other = provider(bytes(range(50, 82))), provider(bytes(range(60, 92)))
    from cva.provenance.seal.chain import rotation_body
    body = rotation_body(KEY, new, 3)
    body["rotation"]["new_key_pop"] = other.sign(b"anything").hex()       # a PoP by someone else's key
    c.append("key_rotation", body)
    res = c.verify()
    assert not res.ok and code(res) == "bad_rotation_pop"


def test_a_proof_of_possession_cannot_be_replayed_into_a_different_rotation():
    """The PoP binds the outgoing key and the position, so copying it elsewhere fails."""
    from cva.provenance.seal.chain import rotation_body, rotation_pop_valid
    new = provider(bytes(range(50, 82)))
    good = rotation_body(KEY, new, 3)["rotation"]
    rec = {"key_id": KEY.key_id, "rotation": good}
    assert rotation_pop_valid(rec)
    assert not rotation_pop_valid({"key_id": provider(SEED_B).key_id, "rotation": good})
    assert not rotation_pop_valid({"key_id": KEY.key_id, "rotation": {**good, "effective_seq": 9}})


# --- what the chain cannot see (documented limitation, plan §14) -------------------------------------------------

def test_documented_limitation_a_holder_of_the_signing_key_can_rewrite_history_undetectably():
    """Not a bug — the boundary of the arithmetic. Someone holding the key can build a wholly different,
    internally perfect chain, and nothing INSIDE the chain distinguishes it. That is exactly what the
    anchoring ceremony (C7) exists for. Stated here so the limit is a tested fact, not a sentence."""
    real = make_chain(6)
    forged = MemoryChain(KEY, clock=clock(), rng=rng())
    forged.append("genesis", {"deployment_manifest": manifest_for(KEY)})
    for _ in range(3):
        forged.append("inference", BODIES["inference"] | {"output": {**BODIES["inference"]["output"], "jcs_sha256": "9" * 64}})
    assert forged.stored != real.stored
    assert verify_chain(forged.stored, ledger_keys=ledger_keys(KEY)).ok


def test_documented_limitation_deleting_the_last_records_leaves_a_perfectly_valid_chain():
    """D11: tail truncation has no successor to disagree — invisible to the chain, caught only against
    an anchor or an independent count."""
    c = make_chain(9)
    assert verify_chain(c.stored[:5], ledger_keys=ledger_keys(KEY)).ok


def test_key_id_of_matches_the_provider():
    assert key_id_of(KEY.public_key.hex()) == KEY.key_id
