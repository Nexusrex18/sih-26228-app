"""The anchoring ceremony (plan §5.9, §7.10; gate C7).

An ANCHOR is a small, self-verifying artefact that lets someone who holds NONE of the ledger check it against
a root the key holder committed to at a point in time:

    { "v": "cva-seal/1",
      "checkpoint":    <the full signed checkpoint record>,
      "cosignatures":  [ {"key_id", "sig"} ],          # a second party: Ed25519(witness, TAG_COSIGN ‖ stored checkpoint)
      "attestations":  [ {"v","kind":"witness_statement","root_hash","tree_size","observed_at_utc","key_id","sig"} ] }

Three media carry the same fact (`Module-C`): the FILE on write-once media, a COSIGNATURE by a second party,
and a physical LOGBOOK (`print_anchor`: the tree size and root in 4-character groups, with a checksum a
transcription error cannot survive). An anchor fixes what the ledger's first `tree_size` records were: any
later rewrite of them — by anyone, INCLUDING the holder of the signing key — no longer hashes to the anchored
root. It does not, and cannot, cover records written after it: that unwitnessed window is printed in every
report (plan §14, decision D11).

Nothing here needs the ledger or a network; `verify_anchor` needs only the trust root.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .canonical import canonical_bytes, parse_strict
from .chain import verify_signature
from .constants import TAG_COSIGN, TAG_WITNESS, VERSION
from .errors import AnchorError, InvalidRecord, NonCanonical
from .keys import KeyProvider, TrustRoot, verify_ed25519
from .records import format_utc, stored_bytes, validate_record

if TYPE_CHECKING:
    from .store import SealedLedger

MEDIA = ("write_once", "cosign", "logbook")
_HEX = set("0123456789abcdef")
_PRINT_TAG = f"{VERSION} anchor-print\n".encode("ascii")
_ATTESTATION_FIELDS = {"v", "kind", "root_hash", "tree_size", "observed_at_utc", "key_id", "sig"}


def _is_hex(v: object, n: int) -> bool:
    return isinstance(v, str) and len(v) == n and set(v) <= _HEX


# --- building ---------------------------------------------------------------------------------------

def build_anchor(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """An anchor around a signed checkpoint record, with no cosignatures or attestations yet."""
    if checkpoint.get("type") != "checkpoint":
        raise AnchorError(f"an anchor is built around a checkpoint record, not {checkpoint.get('type')!r}")
    try:
        validate_record(checkpoint, signed=True)
    except InvalidRecord as e:
        raise AnchorError(f"the checkpoint is not a valid signed record: {e}") from None
    return {"v": VERSION, "checkpoint": dict(checkpoint), "cosignatures": [], "attestations": []}


def _cosign_message(checkpoint: Mapping[str, Any]) -> bytes:
    return TAG_COSIGN + stored_bytes(checkpoint)


def cosign(anchor: Mapping[str, Any], witness_key: KeyProvider) -> dict[str, Any]:
    """Add a witness's cosignature over the checkpoint. A second cosignature by the same key replaces the first."""
    out = _copy(anchor)
    sig = witness_key.sign(_cosign_message(out["checkpoint"]))
    out["cosignatures"] = [c for c in out["cosignatures"] if c["key_id"] != witness_key.key_id]
    out["cosignatures"].append({"key_id": witness_key.key_id, "sig": sig.hex()})
    return out


def _statement(anchor: Mapping[str, Any], key_id: str, observed_at_utc: str) -> dict[str, Any]:
    cp = anchor["checkpoint"]["checkpoint"]
    return {"v": VERSION, "kind": "witness_statement", "root_hash": cp["root_hash"], "tree_size": cp["tree_size"],
            "observed_at_utc": observed_at_utc, "key_id": key_id}


def attest(anchor: Mapping[str, Any], boundary_key: KeyProvider, observed_at: datetime | str) -> dict[str, Any]:
    """Add a witness statement: the boundary device, on its OWN clock, saw this root at `observed_at`. That is the
    only thing in the system that bounds absolute time — `created_at_utc` is the untrusted host clock."""
    ts = format_utc(observed_at) if isinstance(observed_at, datetime) else observed_at
    st = _statement(anchor, boundary_key.key_id, ts)
    sig = boundary_key.sign(TAG_WITNESS + canonical_bytes(st))
    out = _copy(anchor)
    out["attestations"] = [a for a in out["attestations"] if a["key_id"] != boundary_key.key_id]
    out["attestations"].append({**st, "sig": sig.hex()})
    return out


def _copy(anchor: Mapping[str, Any]) -> dict[str, Any]:
    return {"v": anchor["v"], "checkpoint": dict(anchor["checkpoint"]),
            "cosignatures": [dict(c) for c in anchor["cosignatures"]],
            "attestations": [dict(a) for a in anchor["attestations"]]}


# --- (de)serialisation ------------------------------------------------------------------------------

def encode_anchor(anchor: Mapping[str, Any]) -> bytes:
    """Canonical JSON plus a newline: byte-for-byte reproducible, so an anchor can be hashed and compared."""
    return canonical_bytes(anchor, max_bytes=None) + b"\n"


def parse_anchor(data: bytes) -> dict[str, Any]:
    """Strict parse of an anchor file. Raises `AnchorError` when it is not an anchor at all."""
    try:
        obj = parse_strict(data.rstrip(b"\n") if data.endswith(b"\n") else data, require_canonical=False)
    except NonCanonical as e:
        raise AnchorError(f"not a valid anchor file: {e.detail}") from None
    if obj.get("v") == VERSION and obj.get("kind") == "logbook":
        return _logbook_from_obj(obj)
    if set(obj) != {"v", "checkpoint", "cosignatures", "attestations"}:
        raise AnchorError(f"an anchor has exactly v, checkpoint, cosignatures, attestations; got {sorted(obj)}")
    if obj["v"] != VERSION:
        raise AnchorError(f"unsupported anchor version {obj['v']!r}")
    if not isinstance(obj["cosignatures"], list) or not isinstance(obj["attestations"], list):
        raise AnchorError("cosignatures and attestations must be arrays")
    for c in obj["cosignatures"]:
        if not isinstance(c, dict) or set(c) != {"key_id", "sig"} or not _is_hex(c["key_id"], 64) \
                or not _is_hex(c["sig"], 128):
            raise AnchorError("each cosignature is exactly {key_id: 64 hex, sig: 128 hex}")
    for a in obj["attestations"]:
        if not isinstance(a, dict) or set(a) != _ATTESTATION_FIELDS:
            raise AnchorError(f"each attestation has exactly {sorted(_ATTESTATION_FIELDS)}")
    if not isinstance(obj["checkpoint"], dict):
        raise AnchorError("checkpoint must be a record object")
    return obj


def load_anchor(source: Any) -> dict[str, Any]:
    """An anchor from a dict, bytes, or a file path."""
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, (bytes, bytearray)):
        return parse_anchor(bytes(source))
    try:
        with open(os.fspath(source), "rb") as f:
            return parse_anchor(f.read())
    except OSError as e:
        raise AnchorError(f"{source}: {e.strerror}") from None


# --- the logbook medium -----------------------------------------------------------------------------

def print_checksum(tree_size: int, root_hash: str) -> str:
    """Eight hex characters: a transcription check, NOT a signature. It exists so that a wrong digit copied into
    a paper logbook is caught when it is typed back in, not discovered years later as a false `ledger_fork`."""
    return hashlib.sha256(_PRINT_TAG + f"{tree_size}\n{root_hash}".encode("ascii")).hexdigest()[:8]


def _groups(h: str, n: int = 4) -> str:
    return " ".join(h[i:i + n] for i in range(0, len(h), n))


def print_anchor(anchor: Mapping[str, Any]) -> str:
    """The block to copy into a physical logbook: what was fixed, in groups a person can read out and copy."""
    if anchor.get("kind") == "logbook":
        tree_size, root, key_id = int(anchor["tree_size"]), str(anchor["root_hash"]), ""
    else:
        cp = anchor["checkpoint"]
        tree_size, root, key_id = cp["checkpoint"]["tree_size"], cp["checkpoint"]["root_hash"], cp["key_id"]
    lines = ["CVA-SEAL/1 LEDGER ANCHOR", f"tree size : {tree_size}",
             "root hash : " + _groups(root[:32]), "            " + _groups(root[32:]),
             f"checksum  : {_groups(print_checksum(tree_size, root))}"]
    if key_id:
        lines.append(f"ledger key: {key_id[:8]}…{key_id[-4:]}")
    lines += ["", "recorded by: ______________   witnessed by: ______________   date: __________"]
    return "\n".join(lines) + "\n"


def logbook_anchor(tree_size: int, root_hash: str, checksum: str | None = None) -> dict[str, Any]:
    """An anchor typed back in from a logbook: just the tree size and root (spaces allowed). With a `checksum`
    the transcription is checked; a mismatch raises, because guessing which digit was mistyped is not our call."""
    root = root_hash.replace(" ", "").lower()
    if isinstance(tree_size, bool) or not isinstance(tree_size, int) or tree_size < 0:
        raise AnchorError("tree_size must be a non-negative integer")
    if not _is_hex(root, 64):
        raise AnchorError("root_hash must be 64 hex characters (spaces are ignored)")
    if checksum is not None and checksum.replace(" ", "").lower() != print_checksum(tree_size, root):
        raise AnchorError("the checksum does not match the transcribed tree size and root: one of them was mistyped")
    return {"v": VERSION, "kind": "logbook", "tree_size": tree_size, "root_hash": root}


def _logbook_from_obj(obj: Mapping[str, Any]) -> dict[str, Any]:
    if set(obj) != {"v", "kind", "tree_size", "root_hash"}:
        raise AnchorError("a logbook anchor has exactly v, kind, tree_size, root_hash")
    return logbook_anchor(obj["tree_size"], obj["root_hash"])


# --- verifying --------------------------------------------------------------------------------------

@dataclass
class AnchorResult:
    """What `verify_anchor` found. `valid` means EVERY element present verified — one bad cosignature makes the
    whole artefact suspect, because an anchor someone has edited should not be quietly half-trusted."""

    valid: bool
    medium: str                                      # "file" | "logbook"
    tree_size: int = -1
    root_hash: str = ""
    checkpoint_seq: int | None = None
    checkpoint_bytes: bytes | None = None           # the exact leaf this anchor claims sits at index tree_size
    ledger_key_id: str = ""
    cosigners: list[str] = field(default_factory=list)              # witness keys whose cosignature verified
    attestations: list[dict[str, str]] = field(default_factory=list)  # verified {key_id, observed_at_utc}
    time_bound_utc: str | None = None                # the earliest verified observed_at: the tree existed by then
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def records_covered(self) -> int:
        """How many leading records the anchor fixes: the tree, plus the checkpoint itself if it carries one."""
        return self.tree_size + (1 if self.checkpoint_bytes is not None else 0)


def verify_anchor(anchor: Mapping[str, Any] | bytes | str | os.PathLike[str], trust_root: TrustRoot, *,
                  extra_ledger_keys: Mapping[str, bytes] | None = None) -> AnchorResult:
    """Verify an anchor against the trust root ALONE (plus, optionally, ledger keys introduced by a verified
    rotation chain). Never raises for a bad signature — that is a result — only for input that is not an anchor."""
    a = load_anchor(anchor)
    if a.get("kind") == "logbook":
        a = _logbook_from_obj(a)
        return AnchorResult(True, "logbook", int(a["tree_size"]), str(a["root_hash"]),
                            notes=["a logbook entry carries no signature: it fixes the root only for whoever "
                                   "trusts the book's custody"])
    a = parse_anchor(canonical_bytes(a, max_bytes=None)) if isinstance(anchor, Mapping) else a
    res = AnchorResult(False, "file")
    cp = a["checkpoint"]
    try:
        validate_record(cp, signed=True)
    except InvalidRecord as e:
        res.problems.append(f"the checkpoint is not a valid signed record: {e}")
        return res
    if cp["type"] != "checkpoint":
        res.problems.append(f"the anchored record is a {cp['type']}, not a checkpoint")
        return res
    body = cp["checkpoint"]
    res.tree_size, res.root_hash, res.checkpoint_seq = body["tree_size"], body["root_hash"], cp["seq"]
    res.ledger_key_id = cp["key_id"]
    res.checkpoint_bytes = stored_bytes(cp)
    if cp["seq"] != body["tree_size"]:
        # the checkpoint record sits AFTER the records it summarises, at index == tree_size
        res.problems.append(f"the checkpoint has seq {cp['seq']} but claims tree size {body['tree_size']}: "
                            "a checkpoint sits at index tree_size")
    keys = dict(trust_root.ledger_keys())
    keys.update(extra_ledger_keys or {})
    pub = keys.get(cp["key_id"])
    if pub is None:
        res.problems.append(f"the checkpoint is signed by key {cp['key_id'][:12]}…, which is not a ledger key in "
                            "the trust root (a rotated-in key is only known once the ledger's rotation chain is "
                            "verified: verify the anchor together with the ledger)")
    elif not verify_signature(cp, pub):
        res.problems.append("the checkpoint's signature does not verify under its ledger key")
    witness = {k.key_id: k.public_key for k in trust_root.by_role("witness")}
    for c in a["cosignatures"]:
        wpub = witness.get(c["key_id"])
        if wpub is None:
            res.problems.append(f"cosignature by {c['key_id'][:12]}…, which is not a witness key in the trust root")
        elif not verify_ed25519(wpub, _cosign_message(cp), bytes.fromhex(c["sig"])):
            res.problems.append(f"cosignature by {c['key_id'][:12]}… does not verify")
        elif c["key_id"] not in res.cosigners:
            res.cosigners.append(c["key_id"])
    boundary = {k.key_id: k.public_key for k in trust_root.by_role("boundary")}
    for at in a["attestations"]:
        why = _check_attestation(at, body, boundary)
        if why:
            res.problems.append(why)
        else:
            res.attestations.append({"key_id": at["key_id"], "observed_at_utc": at["observed_at_utc"]})
    if res.attestations:
        res.time_bound_utc = min(x["observed_at_utc"] for x in res.attestations)
        if cp["created_at_utc"] > res.time_bound_utc:
            res.notes.append(f"the checkpoint's host-clock time {cp['created_at_utc']} is later than the witness's "
                             f"{res.time_bound_utc}: the host clock ran ahead (informational — it is untrusted)")
    res.valid = not res.problems
    return res


def _check_attestation(at: Mapping[str, Any], body: Mapping[str, Any], boundary: Mapping[str, bytes]) -> str | None:
    who = f"witness statement by {str(at.get('key_id'))[:12]}…"
    if at.get("kind") != "witness_statement" or at.get("v") != VERSION:
        return f"{who}: not a {VERSION} witness_statement"
    if at["root_hash"] != body["root_hash"] or at["tree_size"] != body["tree_size"]:
        return f"{who} is about a different root or tree size than the anchored checkpoint"
    if not _is_hex(at["key_id"], 64) or not _is_hex(at["sig"], 128):
        return f"{who} is malformed"
    ts = at["observed_at_utc"]
    try:
        if not (isinstance(ts, str) and len(ts) == 27):
            raise ValueError
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        return f"{who} has an invalid observed_at_utc"
    bpub = boundary.get(at["key_id"])
    if bpub is None:
        return f"{who}: not a boundary key in the trust root"
    st = {k: v for k, v in at.items() if k != "sig"}
    try:
        msg = TAG_WITNESS + canonical_bytes(st)
    except (NonCanonical, TypeError):
        return f"{who} is not canonicalisable"
    if not verify_ed25519(bpub, msg, bytes.fromhex(at["sig"])):
        return f"{who} does not verify"
    return None


# --- the ledger side: making an anchor and recording that it was made ---------------------------------

def prepare_anchor(ledger: SealedLedger) -> dict[str, Any]:
    """The anchor for the ledger's CURRENT state. Reuses the newest checkpoint if nothing but `anchor_event`s
    follows it; otherwise appends a fresh checkpoint first, so the anchor covers everything sealed so far."""
    cp = ledger.latest_checkpoint()
    if cp is None or any(t != "anchor_event" for t in ledger.records_after(cp["seq"])):
        ledger.checkpoint_now()
        ledger.flush()
        cp = ledger.latest_checkpoint()
    assert cp is not None
    return build_anchor(cp)


def record_anchor(ledger: SealedLedger, anchor: Mapping[str, Any], *, medium: str = "write_once", label: str = "",
                  cosigner_key_ids: tuple[str, ...] = ()) -> int:
    """Append the `anchor_event` that says this anchor was exported. It closes the nonce window (D9) and names the
    expected cosigners. Returns its seq. Call it AFTER the anchor file is safely written: an event for an anchor
    that was never produced would be a claim with nothing behind it."""
    if medium not in MEDIA:
        raise AnchorError(f"medium must be one of {MEDIA}")
    body = anchor["checkpoint"]["checkpoint"]
    written = ledger.append_typed("anchor_event", {"anchor": {
        "checkpoint_seq": anchor["checkpoint"]["seq"], "tree_size": body["tree_size"], "root_hash": body["root_hash"],
        "cosigner_key_ids": sorted(cosigner_key_ids), "medium": medium, "label": label}})
    ledger.flush()
    return written[-1].seq


def write_anchor(anchor: Mapping[str, Any], out_path: str | os.PathLike[str]) -> None:
    """Write an anchor durably, never overwriting: replacing an anchor silently would defeat its purpose."""
    try:
        fd = os.open(os.fspath(out_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise AnchorError(f"{out_path} already exists; refusing to overwrite an anchor") from None
    try:
        os.write(fd, encode_anchor(anchor))
        os.fsync(fd)
    finally:
        os.close(fd)


def export_anchor(ledger: SealedLedger, out_path: str | os.PathLike[str], *, medium: str = "write_once",
                  label: str = "", cosigner_key_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    """Prepare an anchor, write it durably to `out_path` (never overwriting), then record the `anchor_event`."""
    anchor = prepare_anchor(ledger)
    write_anchor(anchor, out_path)
    record_anchor(ledger, anchor, medium=medium, label=label, cosigner_key_ids=cosigner_key_ids)
    return anchor
