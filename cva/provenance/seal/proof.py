"""Inclusion proofs (plan §7.10; gate C7): prove ONE record is in a signed root, given only the record, the proof
and an anchor — no ledger, no database. This is one of the four properties a blockchain would have given us
(plan C-1), obtained from an RFC 6962 audit path and a signed checkpoint.

    { "v": "cva-seal/1", "seq": N, "tree_size": T, "root_hash": <hex>,
      "record": "<the stored record, verbatim>", "path": [<hex>, ...] }

`tree_size` and `root_hash` are NOT trusted from the proof: `verify_inclusion_proof` takes them from the ANCHOR
that the verifier already holds (RFC 9162 §2.1.3.2 — a proof binds a leaf to a root, not to a size).
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .anchor import load_anchor, verify_anchor
from .canonical import canonical_bytes, parse_strict
from .chain import verify_signature
from .constants import VERSION
from .errors import AnchorError, InvalidRecord, NonCanonical, SealError
from .keys import TrustRoot
from .merkle import MerkleTree, leaf_hash, verify_inclusion
from .records import validate_record
from .verify import open_source


def make_inclusion_proof(source: Any, seq: int, anchor: Mapping[str, Any] | bytes | str | os.PathLike[str]) -> dict[str, Any]:
    """Build the proof that record `seq` is in the tree the anchor fixes. `source` is a ledger path or export bytes."""
    a = load_anchor(anchor)
    size = int(a["tree_size"]) if a.get("kind") == "logbook" else int(a["checkpoint"]["checkpoint"]["tree_size"])
    if not 0 <= seq < size:
        raise SealError(f"seq {seq} is not inside the anchored tree (records 0..{size - 1})")
    src = open_source(source)
    try:
        rows = [r.data for r in src.rows()]
    finally:
        src.close()
    if len(rows) < size:
        raise SealError(f"the ledger holds {len(rows)} records but the anchor fixes {size}: cannot build the proof")
    tree = MerkleTree.from_leaves([leaf_hash(d) for d in rows[:size]])
    return {"v": VERSION, "seq": seq, "tree_size": size, "root_hash": tree.root(size).hex(),
            "record": rows[seq].decode("ascii"), "path": [h.hex() for h in tree.inclusion_proof(seq, size)]}


def encode_proof(proof: Mapping[str, Any]) -> bytes:
    return canonical_bytes(proof, max_bytes=None) + b"\n"


@dataclass
class ProofResult:
    valid: bool                                   # the record is at `seq` inside the anchored root
    signature_ok: bool | None = None              # None: the signing key is not in the trust root, so not checked
    seq: int | None = None
    record_type: str | None = None
    anchor_valid: bool = False
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def verify_inclusion_proof(proof: Mapping[str, Any] | bytes | str | os.PathLike[str],
                           anchor: Mapping[str, Any] | bytes | str | os.PathLike[str], trust_root: TrustRoot) -> ProofResult:
    """Never raises for a bad proof — a bad proof is a result. `valid` is True only if the anchor verifies, the
    proof's path leads from the record's leaf hash to the anchor's root at the anchor's size, and the record sits
    at the index it claims."""
    res = ProofResult(False)
    try:
        if isinstance(proof, Mapping):
            p = dict(proof)
        else:
            raw = proof if isinstance(proof, (bytes, bytearray)) else open(os.fspath(proof), "rb").read()
            p = parse_strict(bytes(raw).rstrip(b"\n"), require_canonical=False)
        if set(p) != {"v", "seq", "tree_size", "root_hash", "record", "path"} or p["v"] != VERSION:
            raise ValueError("not a cva-seal/1 inclusion proof")
        if isinstance(p["seq"], bool) or not isinstance(p["seq"], int) or not isinstance(p["record"], str) \
                or not isinstance(p["path"], list):
            raise ValueError("malformed proof fields")
        path = [bytes.fromhex(x) for x in p["path"]]
        record_bytes = p["record"].encode("ascii")
    except (OSError, NonCanonical, ValueError, TypeError, KeyError, UnicodeEncodeError) as e:
        res.problems.append(f"the proof is not readable: {e}")
        return res
    res.seq = p["seq"]
    try:
        ar = verify_anchor(anchor, trust_root)
    except AnchorError as e:
        res.problems.append(f"the anchor is not usable: {e}")
        return res
    res.anchor_valid = ar.valid
    if not ar.valid:
        res.problems.append("the anchor does not verify: " + "; ".join(ar.problems))
        return res
    if (p["tree_size"], p["root_hash"]) != (ar.tree_size, ar.root_hash):
        res.problems.append("the proof is about a different tree size or root than the anchor fixes")
        return res
    if not verify_inclusion(leaf_hash(record_bytes), p["seq"], ar.tree_size, path, bytes.fromhex(ar.root_hash)):
        res.problems.append("the audit path does not lead from this record to the anchored root")
        return res
    try:
        rec = parse_strict(record_bytes)
        validate_record(rec, signed=True)
    except (NonCanonical, InvalidRecord) as e:
        res.problems.append(f"the record is in the tree but is not a valid stored record: {e}")
        return res
    if rec["seq"] != p["seq"]:
        res.problems.append(f"the record says seq {rec['seq']} but the proof places it at index {p['seq']}")
        return res
    res.valid, res.record_type = True, rec["type"]
    pub = trust_root.ledger_keys().get(rec["key_id"])
    if pub is None:
        res.notes.append(f"signed by key {rec['key_id'][:12]}…, which is not in the trust root (a rotated-in key): "
                         "inclusion in the anchored root is proven, the record's own signature is not checked here")
    else:
        res.signature_ok = verify_signature(rec, pub)
        if not res.signature_ok:
            res.valid = False
            res.problems.append("the record is in the anchored tree but its signature does not verify")
    return res
