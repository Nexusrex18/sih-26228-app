"""Typed records with a common header (plan §5.3/§5.4/§5.7, decision D2), schemas IN CODE.

The schema lives here, not in a separate JSON Schema file: a second definition of the structure that
is actually hashed could silently drift from it (the same reasoning that keeps the ledger record out
of Backend's schema list). Rules of the house:

  * exact key sets — an unknown key is an error, a missing key is an error ("no dynamic keys")
  * hashes / keys / signatures / nonces are LOWERCASE hex of fixed length; uppercase is rejected,
    because a second valid spelling of the same bytes is a malleability channel
  * optional fields are OMITTED when absent, never set to null; `null` appears exactly once, in
    `input.phash` (plan §5.4), and is paired with a declared `phash_omitted_reason`

This module builds and validates. It does NOT sign, link or store — those need keys and a tip and
arrive at C3/C4 — so `signature` is only validated here (128 lowercase hex), never produced.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from .canonical import canonical_bytes
from .constants import DOMAIN_MANIFEST, INT_MAX, TIMESTAMP_LEN, VERSION
from .errors import InvalidRecord

RECORD_TYPES = ("genesis", "model_registration", "inference", "checkpoint", "anchor_event",
                "key_rotation", "degraded_marker", "scan_record", "analyst_event")

HEADER_FIELDS = ("v", "type", "seq", "prev_record_hash", "key_id", "created_at_utc", "nonce")

# The body sections each type carries, in addition to the header (plan §5.3/§5.7).
SECTIONS: dict[str, tuple[str, ...]] = {
    "genesis": ("deployment_manifest",),
    "model_registration": ("model", "config"),
    "inference": ("input", "model", "config", "output"),
    "checkpoint": ("checkpoint",),
    "anchor_event": ("anchor",),
    "key_rotation": ("rotation",),
    "degraded_marker": ("gap",),
    "scan_record": ("scan",),
    "analyst_event": ("analyst",),
}

SOURCE_KINDS = ("encoded_file", "model_input_tensor")
PHASH_OMITTED_REASONS = ("not_computed", "sampled_out", "unsupported_source")
ANCHOR_MEDIA = ("write_once", "cosign", "logbook")
TARGET_TYPES = ("sample", "contributor", "batch", "model", "record", "dataset")
DISPOSITIONS = ("accept", "review", "quarantine")
ANALYST_ACTIONS = ("assign", "acknowledge", "override", "approve", "quarantine", "release")
# Closed enum (Module E plan §7 D-E7/D-E8) — an unknown code is rejected, never stored. Kept in ONE
# place here; Module E's ledgerd validator must import or mirror it, not redefine it.
REASON_CODES = ("quality_issue", "known_benign", "insufficient_evidence", "accepted_risk",
                "superseded_by_rescan", "other_with_justification",
                "known_restore", "test_data", "superseded_ledger", "false_positive_confirmed")

_HEX = {n: re.compile(rf"[0-9a-f]{{{n}}}") for n in (16, 32, 40, 64, 128)}
_REF = re.compile(r"sha256:[0-9a-f]{64}")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_PRINTABLE = re.compile(r"[\x20-\x7e]+")
_TOKEN = re.compile(r"[a-z0-9_]+")
_ID = re.compile(r"[A-Za-z0-9._:-]+")


# --- field checkers ----------------------------------------------------------------------------

def _fail(path: str, msg: str) -> InvalidRecord:
    return InvalidRecord(path, msg)


def _hex(v: Any, n: int, path: str) -> str:
    if not isinstance(v, str) or not _HEX[n].fullmatch(v):
        raise _fail(path, f"expected {n} lowercase hex characters, got {_show(v)}")
    return v


def _int(v: Any, path: str, lo: int = 0, hi: int = INT_MAX) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise _fail(path, f"expected an integer in [{lo}, {hi}], got {_show(v)}")
    return v


def _text(v: Any, path: str, lo: int = 1, hi: int = 128) -> str:
    if not isinstance(v, str) or not (lo <= len(v) <= hi) or (v and not _PRINTABLE.fullmatch(v)):
        raise _fail(path, f"expected printable ASCII of length {lo}..{hi}, got {_show(v)}")
    return v


def _token(v: Any, path: str, hi: int = 32) -> str:
    if not isinstance(v, str) or not (1 <= len(v) <= hi) or not _TOKEN.fullmatch(v):
        raise _fail(path, f"expected [a-z0-9_]{{1,{hi}}}, got {_show(v)}")
    return v


def _ident(v: Any, path: str, hi: int = 64) -> str:
    if not isinstance(v, str) or not (1 <= len(v) <= hi) or not _ID.fullmatch(v):
        raise _fail(path, f"expected an identifier [A-Za-z0-9._:-]{{1,{hi}}}, got {_show(v)}")
    return v


def _enum(v: Any, allowed: tuple[str, ...], path: str) -> str:
    if v not in allowed:
        raise _fail(path, f"expected one of {list(allowed)}, got {_show(v)}")
    return v  # type: ignore[no-any-return]


def _ref(v: Any, path: str) -> str:
    if not isinstance(v, str) or not _REF.fullmatch(v):
        raise _fail(path, f"expected 'sha256:<64 lowercase hex>', got {_show(v)}")
    return v


def _ts(v: Any, path: str) -> str:
    if not isinstance(v, str) or len(v) != TIMESTAMP_LEN or not _TS.fullmatch(v):
        raise _fail(path, f"expected YYYY-MM-DDTHH:MM:SS.ffffffZ (27 chars), got {_show(v)}")
    try:
        datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)   # validity check only
    except ValueError:
        raise _fail(path, f"not a real calendar time: {v}") from None
    return v


def _keys(obj: Any, required: tuple[str, ...], path: str, optional: tuple[str, ...] = ()) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise _fail(path, "expected an object")
    missing = [k for k in required if k not in obj]
    unknown = [k for k in obj if k not in required and k not in optional]
    if missing:
        raise _fail(path, f"missing key(s) {missing}")
    if unknown:
        raise _fail(path, f"unknown key(s) {sorted(unknown)}")
    return obj


def _show(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 40 else s[:37] + "..."


def key_id_of(public_key_hex: str) -> str:
    """key_id = SHA-256(raw 32-byte Ed25519 public key), lowercase hex (plan §5.3)."""
    return hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()


# --- section validators ------------------------------------------------------------------------

def _v_model(o: Any, p: str) -> None:
    o = _keys(o, ("id", "weights_sha256", "format", "arch_hash"), p)
    _text(o["id"], f"{p}.id")
    _hex(o["weights_sha256"], 64, f"{p}.weights_sha256")
    _token(o["format"], f"{p}.format")
    _hex(o["arch_hash"], 64, f"{p}.arch_hash")


def _v_config(o: Any, p: str) -> None:
    o = _keys(o, ("preprocess_hash", "preprocess_ref", "postprocess_hash", "runtime",
                  "version_pins_hash", "code_commit"), p)
    _hex(o["preprocess_hash"], 64, f"{p}.preprocess_hash")
    _ref(o["preprocess_ref"], f"{p}.preprocess_ref")
    _hex(o["postprocess_hash"], 64, f"{p}.postprocess_hash")
    _text(o["runtime"], f"{p}.runtime", hi=200)
    _hex(o["version_pins_hash"], 64, f"{p}.version_pins_hash")
    _hex(o["code_commit"], 40, f"{p}.code_commit")


def _v_input(o: Any, p: str) -> None:
    o = _keys(o, ("sha256", "source_kind", "phash", "dims"), p, optional=("phash_omitted_reason",))
    _hex(o["sha256"], 64, f"{p}.sha256")
    _enum(o["source_kind"], SOURCE_KINDS, f"{p}.source_kind")
    dims = o["dims"]
    if not isinstance(dims, list) or len(dims) != 2:
        raise _fail(f"{p}.dims", f"expected [width, height], got {_show(dims)}")
    for i, d in enumerate(dims):
        _int(d, f"{p}.dims[{i}]", lo=1)
    ph = o["phash"]
    if ph is None:
        # D10: a pHash is optional and caller-supplied; omitting it must be DECLARED, not silent.
        if "phash_omitted_reason" not in o:
            raise _fail(f"{p}.phash_omitted_reason", "required when phash is null")
        _enum(o["phash_omitted_reason"], PHASH_OMITTED_REASONS, f"{p}.phash_omitted_reason")
    else:
        if "phash_omitted_reason" in o:
            raise _fail(f"{p}.phash_omitted_reason", "must be absent when phash is present")
        ph = _keys(ph, ("algo", "value"), f"{p}.phash")
        _enum(ph["algo"], ("phash64-v1",), f"{p}.phash.algo")
        _hex(ph["value"], 16, f"{p}.phash.value")


def _v_output(o: Any, p: str) -> None:
    o = _keys(o, ("jcs_sha256", "decision_sha256", "raw_jcs_sha256", "payload_ref"), p)
    for k in ("jcs_sha256", "decision_sha256", "raw_jcs_sha256"):
        _hex(o[k], 64, f"{p}.{k}")
    _ref(o["payload_ref"], f"{p}.payload_ref")


def _v_manifest(o: Any, p: str) -> None:
    o = _keys(o, ("device_id", "key_id", "profile_hash", "unit", "checkpoint_every", "spec"), p)
    _text(o["device_id"], f"{p}.device_id")
    _hex(o["key_id"], 64, f"{p}.key_id")
    _hex(o["profile_hash"], 64, f"{p}.profile_hash")
    _text(o["unit"], f"{p}.unit")
    _int(o["checkpoint_every"], f"{p}.checkpoint_every", lo=1)
    if o["spec"] != VERSION:
        raise _fail(f"{p}.spec", f"expected {VERSION!r}, got {_show(o['spec'])}")


def _v_checkpoint(o: Any, p: str) -> None:
    o = _keys(o, ("tree_size", "root_hash"), p)
    _int(o["tree_size"], f"{p}.tree_size")
    _hex(o["root_hash"], 64, f"{p}.root_hash")


def _v_anchor(o: Any, p: str) -> None:
    o = _keys(o, ("checkpoint_seq", "tree_size", "root_hash", "cosigner_key_ids", "medium", "label"), p)
    _int(o["checkpoint_seq"], f"{p}.checkpoint_seq")
    _int(o["tree_size"], f"{p}.tree_size")
    _hex(o["root_hash"], 64, f"{p}.root_hash")
    ids = o["cosigner_key_ids"]
    if not isinstance(ids, list):
        raise _fail(f"{p}.cosigner_key_ids", "expected an array")
    for i, k in enumerate(ids):
        _hex(k, 64, f"{p}.cosigner_key_ids[{i}]")
    if len(set(ids)) != len(ids):
        raise _fail(f"{p}.cosigner_key_ids", "duplicate key ids")
    _enum(o["medium"], ANCHOR_MEDIA, f"{p}.medium")
    _text(o["label"], f"{p}.label", lo=0, hi=64)


def _v_rotation(o: Any, p: str) -> None:
    o = _keys(o, ("new_key_id", "new_public_key", "effective_seq", "new_key_pop"), p)
    _hex(o["new_key_id"], 64, f"{p}.new_key_id")
    _hex(o["new_public_key"], 64, f"{p}.new_public_key")
    _int(o["effective_seq"], f"{p}.effective_seq", lo=1)
    _hex(o["new_key_pop"], 128, f"{p}.new_key_pop")
    if key_id_of(o["new_public_key"]) != o["new_key_id"]:
        raise _fail(f"{p}.new_key_id", "is not SHA-256 of new_public_key")


def _v_gap(o: Any, p: str) -> None:
    o = _keys(o, ("first_unsealed_utc", "last_unsealed_utc", "reported_count", "reason", "spill_sha256"), p)
    a = _ts(o["first_unsealed_utc"], f"{p}.first_unsealed_utc")
    b = _ts(o["last_unsealed_utc"], f"{p}.last_unsealed_utc")
    if a > b:                                    # fixed-width UTC strings order lexicographically
        raise _fail(p, "first_unsealed_utc is after last_unsealed_utc")
    _int(o["reported_count"], f"{p}.reported_count")
    _token(o["reason"], f"{p}.reason", hi=64)
    if o["spill_sha256"] != "unavailable":       # plan §8: declared, not hidden, when the spill failed
        _hex(o["spill_sha256"], 64, f"{p}.spill_sha256")


def _v_scan(o: Any, p: str) -> None:
    o = _keys(o, ("scan_id", "report_sha256", "profile_hash", "code_commit", "finding_counts"), p)
    _ident(o["scan_id"], f"{p}.scan_id")
    _hex(o["report_sha256"], 64, f"{p}.report_sha256")
    _hex(o["profile_hash"], 64, f"{p}.profile_hash")
    _hex(o["code_commit"], 40, f"{p}.code_commit")
    fc = o["finding_counts"]
    if not isinstance(fc, Mapping):
        raise _fail(f"{p}.finding_counts", "expected an object")
    for k, v in fc.items():
        _token(k, f"{p}.finding_counts key", hi=64)
        _int(v, f"{p}.finding_counts.{k}")


def _v_analyst(o: Any, p: str) -> None:
    o = _keys(o, ("actor_id", "role", "action", "scan_id", "target_type", "target_ref",
                  "finding_id", "justification", "request_id"), p,
              optional=("new_disposition", "reason_code", "assignee", "refs_seq", "expected_prev_seq"))
    _ident(o["actor_id"], f"{p}.actor_id")
    _token(o["role"], f"{p}.role")
    action = _enum(o["action"], ANALYST_ACTIONS, f"{p}.action")
    _ident(o["scan_id"], f"{p}.scan_id")
    _enum(o["target_type"], TARGET_TYPES, f"{p}.target_type")
    _text(o["target_ref"], f"{p}.target_ref", hi=200)
    _ident(o["finding_id"], f"{p}.finding_id", hi=128)
    _text(o["justification"], f"{p}.justification", lo=0, hi=8192)   # already percent-encoded
    _hex(o["request_id"], 32, f"{p}.request_id")
    if "new_disposition" in o:
        _enum(o["new_disposition"], DISPOSITIONS, f"{p}.new_disposition")
    if "reason_code" in o:
        _enum(o["reason_code"], REASON_CODES, f"{p}.reason_code")
    if "assignee" in o:
        _ident(o["assignee"], f"{p}.assignee")
    for k in ("refs_seq", "expected_prev_seq"):
        if k in o:
            _int(o[k], f"{p}.{k}")
    # Per-action requirements. `override` is where analyst decisions can go wrong, so it is strict:
    # what it changes (new_disposition), why (reason_code), a non-empty justification (D14 / C9 gate,
    # "an override without a recorded justification is how assurance systems fail"), and the stale-write
    # guard. `approve` names the exact event it blesses; `assign` names who.
    if action == "override":
        for k in ("new_disposition", "reason_code", "expected_prev_seq"):
            if k not in o:
                raise _fail(f"{p}.{k}", "required for action 'override'")
        if not o["justification"].strip():
            raise _fail(f"{p}.justification", "must be non-empty for action 'override'")
    elif action == "approve" and "refs_seq" not in o:
        raise _fail(f"{p}.refs_seq", "required for action 'approve'")
    elif action == "assign" and "assignee" not in o:
        raise _fail(f"{p}.assignee", "required for action 'assign'")


_SECTION_VALIDATORS: dict[str, Callable[[Any, str], None]] = {
    "deployment_manifest": _v_manifest, "model": _v_model, "config": _v_config, "input": _v_input,
    "output": _v_output, "checkpoint": _v_checkpoint, "anchor": _v_anchor, "rotation": _v_rotation,
    "gap": _v_gap, "scan": _v_scan, "analyst": _v_analyst,
}


def validate_section(name: str, obj: Any) -> None:
    """Validate one body section (`model`, `config`, `input`, ...) against its in-code schema."""
    if name not in _SECTION_VALIDATORS:
        raise _fail("$", f"unknown section {name!r}")
    _SECTION_VALIDATORS[name](obj, f"$.{name}")


# --- construction primitives ---------------------------------------------------------------------

def format_utc(dt: datetime) -> str:
    """`YYYY-MM-DDTHH:MM:SS.ffffffZ` — always UTC, always 6 fractional digits (27 chars)."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime: pass an aware one so 'UTC' is never a guess")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_nonce(rng: Callable[[int], bytes] = os.urandom) -> str:
    """16 bytes from a CSPRNG as 32 lowercase hex. Never derived, never counter-based (§5.3)."""
    b = rng(16)
    if len(b) != 16:
        raise ValueError("rng must return exactly 16 bytes")
    return b.hex()


def genesis_prev_hash(deployment_manifest: Mapping[str, Any]) -> str:
    """D5: `prev_record_hash` of the genesis record = SHA-256(0x03 ‖ JCS(deployment_manifest)).

    Committing to deployment identity (instead of 32 zero bytes) means a valid chain from device A
    cannot be presented as device B's history: the verifier recomputes this from the manifest.
    """
    _v_manifest(deployment_manifest, "deployment_manifest")
    return hashlib.sha256(DOMAIN_MANIFEST + canonical_bytes(deployment_manifest)).hexdigest()


def validate_record(record: Mapping[str, Any], *, signed: bool = True) -> None:
    """Raise InvalidRecord unless `record` is a well-formed cva-seal/1 record of its declared type.

    Checks the header, the exact section set for the type, every section's schema, and the
    cross-field rules that hold regardless of chain position (genesis identity, rotation arithmetic).
    Chain-position rules (seq continuity, link, active key) belong to the verifier, not here.
    """
    if not isinstance(record, Mapping):
        raise _fail("$", "expected an object")
    rtype = record.get("type")
    if rtype not in RECORD_TYPES:
        raise _fail("$.type", f"expected one of {list(RECORD_TYPES)}, got {_show(rtype)}")
    sections = SECTIONS[rtype]
    required = HEADER_FIELDS + sections + (("signature",) if signed else ())
    _keys(record, required, "$")
    if record["v"] != VERSION:
        raise _fail("$.v", f"expected {VERSION!r}, got {_show(record['v'])}")
    _int(record["seq"], "$.seq")
    _hex(record["prev_record_hash"], 64, "$.prev_record_hash")
    _hex(record["key_id"], 64, "$.key_id")
    _ts(record["created_at_utc"], "$.created_at_utc")
    _hex(record["nonce"], 32, "$.nonce")
    if signed:
        _hex(record["signature"], 128, "$.signature")
    for name in sections:
        _SECTION_VALIDATORS[name](record[name], f"$.{name}")

    if rtype == "genesis":
        man = record["deployment_manifest"]
        if record["seq"] != 0:
            raise _fail("$.seq", "genesis must be seq 0")
        if record["prev_record_hash"] != genesis_prev_hash(man):
            raise _fail("$.prev_record_hash", "does not commit to the deployment manifest (D5)")
        if record["key_id"] != man["key_id"]:
            raise _fail("$.key_id", "genesis must be signed by the key its manifest names")
    else:
        if record["seq"] < 1:
            raise _fail("$.seq", "seq 0 is reserved for genesis")
    if rtype == "key_rotation" and record["rotation"]["effective_seq"] != record["seq"] + 1:
        raise _fail("$.rotation.effective_seq", "must equal seq + 1 (plan §5.10)")
    if rtype == "anchor_event" and record["anchor"]["checkpoint_seq"] >= record["seq"]:
        raise _fail("$.anchor.checkpoint_seq", "an anchor cannot precede its checkpoint")


def build_record(rtype: str, *, seq: int, prev_record_hash: str, key_id: str,
                 created_at_utc: str | datetime, nonce: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble and validate an UNSIGNED record. `body` maps section name -> section object and must
    contain exactly the sections `SECTIONS[rtype]` names. The result is a deep copy: mutating the
    caller's `body` afterwards cannot change what gets signed."""
    if rtype not in RECORD_TYPES:
        raise _fail("$.type", f"unknown record type {_show(rtype)}")
    if set(body) != set(SECTIONS[rtype]):
        raise _fail("$", f"{rtype} needs exactly sections {list(SECTIONS[rtype])}, got {sorted(body)}")
    ts = format_utc(created_at_utc) if isinstance(created_at_utc, datetime) else created_at_utc
    record: dict[str, Any] = {"v": VERSION, "type": rtype, "seq": seq,
                              "prev_record_hash": prev_record_hash, "key_id": key_id,
                              "created_at_utc": ts, "nonce": nonce}
    record.update(copy.deepcopy(dict(body)))
    validate_record(record, signed=False)
    return record


def canon(record: Mapping[str, Any]) -> bytes:
    """Canonical bytes of the record WITHOUT its signature — what gets signed (behind TAG_RECORD)
    and what `record_hash` covers (plan §5.5). Accepts a signed record and drops the signature."""
    unsigned = {k: v for k, v in record.items() if k != "signature"}
    validate_record(unsigned, signed=False)
    return canonical_bytes(unsigned)


def stored_bytes(record: Mapping[str, Any]) -> bytes:
    """Canonical bytes of the SIGNED record: exactly what is stored, and the Merkle leaf data (D7).
    A stored record is its canonical bytes and nothing else (§5.1 rule 7)."""
    validate_record(record, signed=True)
    return canonical_bytes(record)


def record_hash(record: Mapping[str, Any]) -> bytes:
    """SHA-256(canon(record)) — plain, no prefix: the input always starts with 0x7B ('{'), so it
    cannot collide with any 0x00–0x03-prefixed hash use (plan §5.5)."""
    return hashlib.sha256(canon(record)).digest()
