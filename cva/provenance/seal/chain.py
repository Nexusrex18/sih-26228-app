"""Signing, the hash chain, and an in-memory chain (plan §5.5; decisions D1, D7, S4).

    signature(r) = Ed25519.sign( "cva-seal/1 record\\n" ‖ canon(r) )        # D1: pure Ed25519, tagged
    link(prev)   = SHA-256( 0x02 ‖ record_hash(prev) ‖ sig_bytes(prev) )    # S4: the chain covers the signature
    r.prev_record_hash = hex(link(prev))                                    # genesis: see records.genesis_prev_hash

`seal_next` is the one function that turns "the previous record + a body" into the next signed record.
The in-memory chain here and the SQLite ledger (C4) both call it, so there is a single definition of
"what a valid next record is".

`verify_chain` is the C3 precursor of the standalone verifier (`verify.py`, C5): it answers *is this
sequence of stored records an intact, correctly signed, correctly chained history?* and reports the
FIRST failure. It deliberately does not classify failures into the attack taxonomy (record_edit vs
record_replace vs record_reorder ...), does not know about anchors, rotation or nonces, and does not
produce Findings — that is C5's normative decision procedure. Its codes are low-level facts.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .canonical import canonical_bytes, parse_strict
from .constants import DOMAIN_LINK, TAG_RECORD, TAG_ROTATION_POP
from .errors import InvalidRecord, NonCanonical, SealError, SigningFailed
from .keys import KeyProvider, verify_ed25519
from .records import (
    build_record,
    canon,
    genesis_prev_hash,
    key_id_of,
    new_nonce,
    record_hash,
    stored_bytes,
    validate_record,
)


def sign_record(unsigned: Mapping[str, Any], key: KeyProvider) -> dict[str, Any]:
    """Return a signed copy of `unsigned`. Refuses a record that names a different key than `key`, and
    refuses one that is already signed (re-signing would silently discard the existing signature)."""
    if "signature" in unsigned:
        raise ValueError("record is already signed")
    validate_record(unsigned, signed=False)
    if unsigned["key_id"] != key.key_id:
        raise ValueError(f"record names key {unsigned['key_id'][:16]}… but the signing key is {key.key_id[:16]}…")
    try:
        sig = key.sign(TAG_RECORD + canon(unsigned))
    except SealError:
        raise
    except Exception as e:                                   # noqa: BLE001 - any provider failure means "cannot sign"
        raise SigningFailed(f"the signing key could not sign: {type(e).__name__}: {e}") from None
    signed = dict(unsigned)
    signed["signature"] = sig.hex()
    validate_record(signed, signed=True)
    return signed


def verify_signature(record: Mapping[str, Any], public_key: bytes) -> bool:
    """True iff `record["signature"]` is a valid Ed25519 signature over TAG_RECORD ‖ canon(record) under
    `public_key`. A signature made without the tag (or under another record class's tag) does not verify."""
    try:
        sig = bytes.fromhex(record["signature"])
        message = TAG_RECORD + canon(record)
    except (KeyError, ValueError, InvalidRecord, NonCanonical):
        return False
    return verify_ed25519(public_key, message, sig)


def rotation_pop_message(new_key_id: str, effective_seq: int, prev_key_id: str) -> bytes:
    """What the INCOMING key signs to prove it is held by whoever is rotating to it (plan §5.10):
    `"cva-seal/1 rotation-pop\\n" ‖ JCS({new_key_id, effective_seq, prev_key_id})`. Binding the outgoing key
    and the position stops a proof from being replayed into a different rotation."""
    return TAG_ROTATION_POP + canonical_bytes({"new_key_id": new_key_id, "effective_seq": effective_seq,
                                               "prev_key_id": prev_key_id})


def rotation_pop_valid(record: Mapping[str, Any]) -> bool:
    """True iff a `key_rotation` record's proof-of-possession verifies under the key it introduces."""
    try:
        rot = record["rotation"]
        return verify_ed25519(bytes.fromhex(rot["new_public_key"]),
                              rotation_pop_message(rot["new_key_id"], rot["effective_seq"], record["key_id"]),
                              bytes.fromhex(rot["new_key_pop"]))
    except (KeyError, ValueError, TypeError, NonCanonical):
        return False


def rotation_body(outgoing: KeyProvider, incoming: KeyProvider, seq: int) -> dict[str, Any]:
    """The body of a `key_rotation` record that will sit at `seq` (so `effective_seq` = seq + 1)."""
    eff = seq + 1
    pop = incoming.sign(rotation_pop_message(incoming.key_id, eff, outgoing.key_id))
    return {"rotation": {"new_key_id": incoming.key_id, "new_public_key": incoming.public_key.hex(),
                         "effective_seq": eff, "new_key_pop": pop.hex()}}


def link_from(rec_hash: bytes, signature_hex: str) -> str:
    """The one definition of the chain link, from an already-computed record_hash and the signature."""
    return hashlib.sha256(DOMAIN_LINK + rec_hash + bytes.fromhex(signature_hex)).hexdigest()


def link_hash(prev_signed: Mapping[str, Any]) -> str:
    """The value the NEXT record stores as `prev_record_hash`. It covers the previous record's signature,
    not just its fields: an attacker cannot swap in a differently-signed copy of the same fields."""
    validate_record(prev_signed, signed=True)
    return link_from(record_hash(prev_signed), prev_signed["signature"])


def seal_next(rtype: str, body: Mapping[str, Any], *, key: KeyProvider, prev: Mapping[str, Any] | None,
              now: datetime, nonce: str, prev_link: str | None = None) -> tuple[dict[str, Any], bytes]:
    """Build, link, sign and serialise the record that follows `prev` (`None` = this is genesis).

    Returns `(signed_record, stored_bytes)`. Pure: no I/O, so the caller decides where it is stored, and
    the clock and nonce are passed in so tests are reproducible. `prev_link` may carry the already-known
    `link_hash(prev)` so a writer that just produced `prev` need not re-validate and re-hash it.
    """
    if prev is None:
        if rtype != "genesis":
            raise ValueError("the first record of a chain must be genesis")
        seq, prev_hash = 0, genesis_prev_hash(body["deployment_manifest"])
    else:
        if rtype == "genesis":
            raise ValueError("genesis can only be the first record")
        seq, prev_hash = prev["seq"] + 1, prev_link or link_hash(prev)
    unsigned = build_record(rtype, seq=seq, prev_record_hash=prev_hash, key_id=key.key_id,
                            created_at_utc=now, nonce=nonce, body=body)
    signed = sign_record(unsigned, key)
    return signed, stored_bytes(signed)


class MemoryChain:
    """An in-memory append-only chain: the reference for the SQLite ledger and the fixture for tests."""

    def __init__(self, key: KeyProvider, *, clock: Callable[[], datetime] | None = None,
                 rng: Callable[[int], bytes] | None = None) -> None:
        self._key = key
        self._keys: dict[str, bytes] = {key.key_id: key.public_key}
        self._clock = clock or (lambda: datetime.now(UTC))
        self._rng = rng
        self.records: list[dict[str, Any]] = []
        self.stored: list[bytes] = []

    def append(self, rtype: str, body: Mapping[str, Any]) -> bytes:
        prev = self.records[-1] if self.records else None
        nonce = new_nonce(self._rng) if self._rng else new_nonce()
        signed, data = seal_next(rtype, body, key=self._key, prev=prev, now=self._clock(), nonce=nonce)
        self.records.append(signed)
        self.stored.append(data)
        return data

    def checkpoint_now(self) -> bytes:
        """Append a signed checkpoint over the records so far (the SQLite ledger does this on its cadence)."""
        from .merkle import leaf_hash, mth
        n = len(self.stored)
        root = mth([leaf_hash(d) for d in self.stored]).hex()
        return self.append("checkpoint", {"checkpoint": {"tree_size": n, "root_hash": root}})

    def anchor_now(self, *, medium: str = "write_once", label: str = "") -> bytes:
        """Checkpoint, then record that an anchor was exported for it."""
        self.checkpoint_now()
        cp = self.records[-1]["checkpoint"]
        return self.append("anchor_event", {"anchor": {"checkpoint_seq": self.records[-1]["seq"],
                                                       "tree_size": cp["tree_size"], "root_hash": cp["root_hash"],
                                                       "cosigner_key_ids": [], "medium": medium, "label": label}})

    def rotate_key(self, new_key: KeyProvider) -> bytes:
        """Append a `key_rotation` signed by the OUTGOING key, with the incoming key's proof-of-possession;
        later records are signed by `new_key`."""
        self._keys[new_key.key_id] = new_key.public_key
        data = self.append("key_rotation", rotation_body(self._key, new_key, len(self.records)))
        self._key = new_key
        return data

    def verify(self) -> ChainCheck:
        return verify_chain(self.stored, ledger_keys=dict(self._keys))


@dataclass(frozen=True)
class ChainFailure:
    position: int          # index in the supplied sequence (NOT trusted to equal the record's own seq)
    code: str
    detail: str


@dataclass(frozen=True)
class ChainCheck:
    ok: bool
    records_checked: int
    failure: ChainFailure | None = None


def check_record(data: bytes, pos: int, prev: Mapping[str, Any] | None, active_key: str | None,
                 ledger_keys: Mapping[str, bytes]) -> tuple[dict[str, Any] | None, ChainFailure | None]:
    """Check ONE stored record in the context of the previous (already verified) record and the key that
    is active. Returns `(record, None)` on success or `(None, failure)`. `verify_chain` is this in a loop;
    it is public so the C5 verifier and the bit-flip sweeps can reuse the exact same definition.

    Order — parse, schema, position, key, signature, link, type rules — is normative: it decides which
    failure names the cause when one physical change trips several checks.
    """
    def fail(code: str, detail: str) -> tuple[None, ChainFailure]:
        return None, ChainFailure(pos, code, detail)

    try:
        rec = parse_strict(data)
    except NonCanonical as e:
        return fail(e.code, e.detail)
    try:
        validate_record(rec, signed=True)
    except InvalidRecord as e:
        return fail("malformed_record", str(e))
    if rec["seq"] != pos:
        return fail("bad_seq", f"record says seq {rec['seq']} at position {pos}")
    if (rec["type"] == "genesis") != (pos == 0):
        return fail("bad_seq" if pos else "genesis_mismatch", "genesis must be the first record and only the first")
    key = rec["key_id"] if pos == 0 else active_key
    if rec["key_id"] != key or rec["key_id"] not in ledger_keys:
        return fail("key_unauthorised", f"signed by {rec['key_id'][:16]}…, not an authorised ledger key")
    if not verify_signature(rec, ledger_keys[rec["key_id"]]):
        return fail("bad_signature", "Ed25519 signature does not verify over the record bytes")
    if prev is not None and rec["prev_record_hash"] != link_hash(prev):
        return fail("chain_broken", "prev_record_hash does not match the previous record's link")
    if rec["type"] == "key_rotation" and not rotation_pop_valid(rec):
        return fail("bad_rotation_pop", "the incoming key's proof-of-possession does not verify")
    return rec, None


def verify_chain(stored: Sequence[bytes], *, ledger_keys: Mapping[str, bytes]) -> ChainCheck:
    """Check that `stored` (each item exactly the stored bytes of one record, oldest first) is an intact,
    correctly signed, correctly chained history under `ledger_keys` (key_id -> raw public key), and
    report the FIRST failure. Codes:
      malformed_record · non_canonical_encoding · bad_seq · key_unauthorised · bad_signature ·
      chain_broken · genesis_mismatch · empty_ledger · bad_rotation_pop
    """
    for kid, pub in ledger_keys.items():
        if key_id_of(pub.hex()) != kid:
            raise ValueError(f"ledger_keys entry {kid[:16]}… is not SHA-256 of its public key")
    if not stored:                # "verified" over zero records would be false assurance
        return ChainCheck(False, 0, ChainFailure(0, "empty_ledger", "no records — a ledger must begin with a genesis record"))
    prev: dict[str, Any] | None = None
    active_key: str | None = None
    for pos, data in enumerate(stored):
        rec, failure = check_record(data, pos, prev, active_key, ledger_keys)
        if failure is not None:
            return ChainCheck(False, pos, failure)
        assert rec is not None
        if pos == 0:
            active_key = rec["key_id"]
        elif rec["type"] == "key_rotation":       # signed by the outgoing key, PoP-checked: the new key takes over
            rot = rec["rotation"]
            ledger_keys = {**ledger_keys, rot["new_key_id"]: bytes.fromhex(rot["new_public_key"])}
            active_key = rot["new_key_id"]
        prev = rec
    return ChainCheck(True, len(stored))
