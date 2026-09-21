"""Data-integrity check on the vendored vectors (Module C plan §11.1, gate C0).

These tests prove the VECTOR FILES are correct — that extraction from the RFC/CT sources did not
mangle them — by checking them against the reference libraries we depend on. They are not tests of
our own canonicaliser / signer / Merkle tree: those arrive at C1 / C3 / C2 and will consume the same
files. If a vendored value is wrong, every later "passes the published vectors" claim is hollow, so
this comes first.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

V = Path(__file__).resolve().parents[2] / "spec" / "vectors"


def load(rel: str) -> dict:
    return json.loads((V / rel).read_text())


# --- RFC 8032 §7.1 -------------------------------------------------------------------------

RFC8032 = load("rfc8032.json")["vectors"]


def test_rfc8032_has_the_four_pure_ed25519_vectors_at_the_rfc_lengths():
    assert [(v["name"], len(v["message"]) // 2) for v in RFC8032] == [
        ("TEST 1", 0), ("TEST 2", 1), ("TEST 3", 2), ("TEST 1024", 1023)]


@pytest.mark.parametrize("v", RFC8032, ids=[v["name"] for v in RFC8032])
def test_rfc8032_public_key_and_signature_match(v):
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(v["secret_key"]))
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    assert pub.hex() == v["public_key"]
    # Ed25519 is deterministic, so the RFC signature must be reproduced EXACTLY.
    assert key.sign(bytes.fromhex(v["message"])).hex() == v["signature"]


# --- RFC 8785 ------------------------------------------------------------------------------

NUMBERS = load("rfc8785/numbers.json")["rows"]


def test_rfc8785_number_table_is_complete():
    assert len(NUMBERS) == 26 and sum(r["must_error"] for r in NUMBERS) == 2


@pytest.mark.parametrize("row", NUMBERS, ids=[r["ieee754"] for r in NUMBERS])
def test_rfc8785_number_serialisation(row):
    value = struct.unpack(">d", bytes.fromhex(row["ieee754"]))[0]
    if row["must_error"]:
        with pytest.raises(rfc8785.FloatDomainError):          # NaN / Infinity are not JSON
            rfc8785.dumps(value)
    else:
        assert rfc8785.dumps(value).decode() == row["expected"]


def test_rfc8785_sorting_example_matches_the_rfc_byte_for_byte():
    s = load("rfc8785/sorting.json")
    assert rfc8785.dumps(json.loads(s["input_json"])).hex() == s["expected_utf8_hex"]


# --- transparency-dev/merkle (RFC 6962 tree) -------------------------------------------------

CT = load("merkle_ct.json")


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _mth(leaf_hashes: list[bytes]) -> bytes:
    """Deliberately independent of the product Merkle tree (which does not exist yet): the RFC
    text's recursive definition, written for this data-integrity check only."""
    n = len(leaf_hashes)
    if n == 0:
        return _h(b"")
    if n == 1:
        return leaf_hashes[0]
    k = 1
    while k * 2 < n:
        k *= 2
    return _h(b"\x01" + _mth(leaf_hashes[:k]) + _mth(leaf_hashes[k:]))


def test_ct_roots_reproduce_from_the_leaf_inputs():
    leaves = [_h(b"\x00" + bytes.fromhex(x)) for x in CT["leaf_inputs"]]
    assert CT["empty_root"] == _h(b"").hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert len(CT["root_hashes_by_size"]) == len(leaves) + 1
    for size, expected in enumerate(CT["root_hashes_by_size"]):
        assert _mth(leaves[:size]).hex() == expected, f"tree size {size}"


def test_ct_node_hashes_are_the_complete_subtrees():
    leaves = [_h(b"\x00" + bytes.fromhex(x)) for x in CT["leaf_inputs"]]
    for level, row in enumerate(CT["node_hashes_by_level"]):
        width = 1 << level
        for idx, expected in enumerate(row):
            assert _mth(leaves[idx * width:(idx + 1) * width]).hex() == expected


def test_every_vendored_file_records_its_source_and_hash():
    for rel in ("rfc8032.json", "rfc8785/numbers.json", "rfc8785/sorting.json", "merkle_ct.json"):
        src = load(rel)["source"]
        assert src["url"].startswith("https://") and len(src["sha256"]) == 64, rel
    assert load("merkle_ct.json")["licence"]["spdx"] == "Apache-2.0"
