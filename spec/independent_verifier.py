"""An independent verifier for cva-seal/1, written from `spec/cva-seal-spec-v1.md` ONLY (plan §11.9, gate C8).

It imports nothing from the `cva` package: only the standard library and `cryptography` (for Ed25519 — the one
primitive nobody should hand-roll). JSON parsing, canonicalisation, Merkle trees, the schema checks, the key state
machine, the anchor checks and the folding rules are all re-implemented here from the specification's text. If this
file and the reference disagree on an artefact, one of them or the spec is wrong — and finding out which is the point.

Usage:  python spec/independent_verifier.py LEDGER.jsonl --trust trust_root.json [--anchor A.json ...]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

VERSION = "cva-seal/1"
TAG_RECORD = b"cva-seal/1 record\n"
TAG_COSIGN = b"cva-seal/1 cosign\n"
TAG_WITNESS = b"cva-seal/1 witness\n"
TAG_POP = b"cva-seal/1 rotation-pop\n"
INT_MAX = 2**53 - 1

SEVERITY = {
    "record_edit": ("critical", "indeterminate"), "record_replay": ("high", "indeterminate"),
    "record_delete": ("high", "indeterminate"), "record_reorder": ("high", "indeterminate"),
    "input_swap": ("critical", "indeterminate"), "model_swap": ("critical", "indeterminate"),
    "output_payload_mismatch": ("high", "indeterminate"), "tail_truncation": ("high", "indeterminate"),
    "ledger_fork": ("critical", "adversarial"), "key_unauthorised": ("high", "indeterminate"),
    "chain_broken": ("high", "indeterminate"), "nonce_reuse": ("high", "indeterminate"),
    "genesis_mismatch": ("high", "indeterminate"), "checkpoint_mismatch": ("high", "indeterminate"),
    "ledger_incomplete": ("high", "indeterminate"), "non_canonical_encoding": ("high", "indeterminate"),
    "malformed_record": ("high", "indeterminate"), "anchor_invalid": ("medium", "indeterminate"),
    "degraded_gap": ("low", "quality"), "clock_regression": ("info", "quality"),
}
TAINTING = {"record_edit", "record_delete", "record_reorder", "record_replay", "ledger_fork", "malformed_record",
            "non_canonical_encoding", "key_unauthorised"}


# ---- §2 the JSON profile and canonical bytes ----------------------------------------------------------------

class Bad(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return "null" if v is None else "other"


def _profile(v: Any, depth: int = 0) -> None:
    k = _kind(v)
    if k in ("object", "array") and depth + 1 > 8:
        raise Bad("malformed_record", "nesting deeper than 8")
    if k == "object":
        for key, val in v.items():
            if not re.fullmatch(r"[a-z0-9_]+", key):
                raise Bad("malformed_record", f"bad key {key!r}")
            _profile(val, depth + 1)
    elif k == "array":
        kinds = {_kind(x) for x in v}
        if "null" in kinds or len(kinds) > 1:
            raise Bad("malformed_record", "array is not homogeneous / holds null")
        for x in v:
            _profile(x, depth + 1)
    elif k == "str":
        if not re.fullmatch(r"[\x20-\x7e]*", v):
            raise Bad("malformed_record", "string outside printable ASCII")
    elif k == "int":
        if abs(v) > INT_MAX:
            raise Bad("malformed_record", "integer out of range")
    elif k == "other":
        raise Bad("malformed_record", "unsupported value")


def canon(v: Any) -> bytes:
    if v is None:
        return b"null"
    if v is True:
        return b"true"
    if v is False:
        return b"false"
    if isinstance(v, int):
        return str(v).encode()
    if isinstance(v, str):
        return b'"' + v.replace("\\", "\\\\").replace('"', '\\"').encode("ascii") + b'"'
    if isinstance(v, list):
        return b"[" + b",".join(canon(x) for x in v) + b"]"
    return b"{" + b",".join(canon(k) + b":" + canon(v[k]) for k in sorted(v, key=lambda s: s.encode())) + b"}"


def strict_parse(data: bytes) -> dict[str, Any]:
    if len(data) > 65536:
        raise Bad("malformed_record", "over 65536 bytes")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise Bad("malformed_record", "non-ASCII byte") from None

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [k for k, _ in items]
        if len(set(keys)) != len(keys):
            raise Bad("malformed_record", "duplicate key")
        return dict(items)

    def refuse(s: str) -> Any:
        raise Bad("malformed_record", f"float or constant {s!r}")
    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except Bad:
        raise
    except (ValueError, RecursionError):
        raise Bad("malformed_record", "not JSON") from None
    if not isinstance(obj, dict):
        raise Bad("malformed_record", "top level is not an object")
    _profile(obj)
    if canon(obj) != data:
        raise Bad("non_canonical_encoding", "valid but not canonical")
    return obj


# ---- §4 schemas ------------------------------------------------------------------------------------------------

def _hex(v: Any, n: int) -> bool:
    return isinstance(v, str) and re.fullmatch(rf"[0-9a-f]{{{n}}}", v) is not None


def _isint(v: Any, lo: int = 0) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= INT_MAX


def _text(v: Any, lo: int, hi: int) -> bool:
    return isinstance(v, str) and lo <= len(v) <= hi and re.fullmatch(r"[\x20-\x7e]*", v) is not None


def _tok(v: Any, n: int) -> bool:
    return isinstance(v, str) and re.fullmatch(rf"[a-z0-9_]{{1,{n}}}", v) is not None


def _ident(v: Any, n: int) -> bool:
    return isinstance(v, str) and re.fullmatch(rf"[A-Za-z0-9._:-]{{1,{n}}}", v) is not None


def _ref(v: Any) -> bool:
    return isinstance(v, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", v) is not None


def _ts(v: Any) -> bool:
    if not isinstance(v, str) or len(v) != 27 or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", v):
        return False
    try:
        datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return False
    return True


def _exact(o: Any, req: tuple[str, ...], opt: tuple[str, ...] = ()) -> bool:
    return isinstance(o, dict) and all(k in o for k in req) and all(k in req or k in opt for k in o)


def _s_manifest(o: Any) -> bool:
    return (_exact(o, ("device_id", "key_id", "profile_hash", "unit", "checkpoint_every", "spec")) and _text(o["device_id"], 1, 128)
            and _hex(o["key_id"], 64) and _hex(o["profile_hash"], 64) and _text(o["unit"], 1, 128)
            and _isint(o["checkpoint_every"], 1) and o["spec"] == VERSION)


def _s_model(o: Any) -> bool:
    return (_exact(o, ("id", "weights_sha256", "format", "arch_hash")) and _text(o["id"], 1, 128)
            and _hex(o["weights_sha256"], 64) and _tok(o["format"], 32) and _hex(o["arch_hash"], 64))


def _s_config(o: Any) -> bool:
    return (_exact(o, ("preprocess_hash", "preprocess_ref", "postprocess_hash", "runtime", "version_pins_hash", "code_commit"))
            and _hex(o["preprocess_hash"], 64) and _ref(o["preprocess_ref"]) and _hex(o["postprocess_hash"], 64)
            and _text(o["runtime"], 1, 200) and _hex(o["version_pins_hash"], 64) and _hex(o["code_commit"], 40))


def _s_input(o: Any) -> bool:
    if not _exact(o, ("sha256", "source_kind", "phash", "dims"), ("phash_omitted_reason",)):
        return False
    if not (_hex(o["sha256"], 64) and o["source_kind"] in ("encoded_file", "model_input_tensor")):
        return False
    d = o["dims"]
    if not (isinstance(d, list) and len(d) == 2 and all(_isint(x, 1) for x in d)):
        return False
    if o["phash"] is None:
        return o.get("phash_omitted_reason") in ("not_computed", "sampled_out", "unsupported_source")
    ph = o["phash"]
    return ("phash_omitted_reason" not in o and _exact(ph, ("algo", "value")) and ph["algo"] == "phash64-v1"
            and _hex(ph["value"], 16))


def _s_output(o: Any) -> bool:
    return (_exact(o, ("jcs_sha256", "decision_sha256", "raw_jcs_sha256", "payload_ref"))
            and all(_hex(o[k], 64) for k in ("jcs_sha256", "decision_sha256", "raw_jcs_sha256")) and _ref(o["payload_ref"]))


def _s_checkpoint(o: Any) -> bool:
    return _exact(o, ("tree_size", "root_hash")) and _isint(o["tree_size"]) and _hex(o["root_hash"], 64)


def _s_anchor(o: Any) -> bool:
    if not _exact(o, ("checkpoint_seq", "tree_size", "root_hash", "cosigner_key_ids", "medium", "label")):
        return False
    ids = o["cosigner_key_ids"]
    return (_isint(o["checkpoint_seq"]) and _isint(o["tree_size"]) and _hex(o["root_hash"], 64) and isinstance(ids, list)
            and all(_hex(k, 64) for k in ids) and len(set(ids)) == len(ids)
            and o["medium"] in ("write_once", "cosign", "logbook") and _text(o["label"], 0, 64))


def _s_rotation(o: Any) -> bool:
    return (_exact(o, ("new_key_id", "new_public_key", "effective_seq", "new_key_pop")) and _hex(o["new_key_id"], 64)
            and _hex(o["new_public_key"], 64) and _isint(o["effective_seq"], 1) and _hex(o["new_key_pop"], 128)
            and hashlib.sha256(bytes.fromhex(o["new_public_key"])).hexdigest() == o["new_key_id"])


def _s_gap(o: Any) -> bool:
    return (_exact(o, ("first_unsealed_utc", "last_unsealed_utc", "reported_count", "reason", "spill_sha256"))
            and _ts(o["first_unsealed_utc"]) and _ts(o["last_unsealed_utc"]) and o["first_unsealed_utc"] <= o["last_unsealed_utc"]
            and _isint(o["reported_count"]) and _tok(o["reason"], 64) and (_hex(o["spill_sha256"], 64) or o["spill_sha256"] == "unavailable"))


def _s_scan(o: Any) -> bool:
    return (_exact(o, ("scan_id", "report_sha256", "profile_hash", "code_commit", "finding_counts")) and _ident(o["scan_id"], 64)
            and _hex(o["report_sha256"], 64) and _hex(o["profile_hash"], 64) and _hex(o["code_commit"], 40)
            and isinstance(o["finding_counts"], dict) and all(_tok(k, 64) and _isint(v) for k, v in o["finding_counts"].items()))


REASONS = ("quality_issue", "known_benign", "insufficient_evidence", "accepted_risk", "superseded_by_rescan",
           "other_with_justification", "known_restore", "test_data", "superseded_ledger", "false_positive_confirmed")


def _s_analyst(o: Any) -> bool:
    if not _exact(o, ("actor_id", "role", "action", "scan_id", "target_type", "target_ref", "finding_id", "justification",
                      "request_id"), ("new_disposition", "reason_code", "assignee", "refs_seq", "expected_prev_seq")):
        return False
    ok = (_ident(o["actor_id"], 64) and _tok(o["role"], 32)
          and o["action"] in ("assign", "acknowledge", "override", "approve", "quarantine", "release")
          and _ident(o["scan_id"], 64) and o["target_type"] in ("sample", "contributor", "batch", "model", "record", "dataset")
          and _text(o["target_ref"], 1, 200) and _ident(o["finding_id"], 128) and _text(o["justification"], 0, 8192)
          and _hex(o["request_id"], 32))
    if not ok:
        return False
    if "new_disposition" in o and o["new_disposition"] not in ("accept", "review", "quarantine"):
        return False
    if "reason_code" in o and o["reason_code"] not in REASONS:
        return False
    if "assignee" in o and not _ident(o["assignee"], 64):
        return False
    if any(k in o and not _isint(o[k]) for k in ("refs_seq", "expected_prev_seq")):
        return False
    a = o["action"]
    if a == "override":
        return all(k in o for k in ("new_disposition", "reason_code", "expected_prev_seq")) and o["justification"].strip() != ""
    if a == "approve":
        return "refs_seq" in o
    if a == "assign":
        return "assignee" in o
    return True


SECTIONS = {"genesis": {"deployment_manifest": _s_manifest}, "model_registration": {"model": _s_model, "config": _s_config},
            "inference": {"input": _s_input, "model": _s_model, "config": _s_config, "output": _s_output},
            "checkpoint": {"checkpoint": _s_checkpoint}, "anchor_event": {"anchor": _s_anchor},
            "key_rotation": {"rotation": _s_rotation}, "degraded_marker": {"gap": _s_gap},
            "scan_record": {"scan": _s_scan}, "analyst_event": {"analyst": _s_analyst}}
HEADER = ("v", "type", "seq", "prev_record_hash", "key_id", "created_at_utc", "nonce")


def _manifest_hash(m: Mapping[str, Any]) -> str:
    return hashlib.sha256(b"\x03" + canon(dict(m))).hexdigest()


def validate(r: dict[str, Any]) -> None:
    t = r.get("type")
    if t not in SECTIONS:
        raise Bad("malformed_record", "unknown type")
    if not _exact(r, HEADER + tuple(SECTIONS[t]) + ("signature",)):
        raise Bad("malformed_record", "wrong key set")
    if not (r["v"] == VERSION and _isint(r["seq"]) and _hex(r["prev_record_hash"], 64) and _hex(r["key_id"], 64)
            and _ts(r["created_at_utc"]) and _hex(r["nonce"], 32) and _hex(r["signature"], 128)):
        raise Bad("malformed_record", "bad header field")
    for name, check in SECTIONS[t].items():
        if not check(r[name]):
            raise Bad("malformed_record", f"bad section {name}")
    if t == "genesis":
        m = r["deployment_manifest"]
        if r["seq"] != 0 or r["prev_record_hash"] != _manifest_hash(m) or r["key_id"] != m["key_id"]:
            raise Bad("malformed_record", "genesis rules")
    elif r["seq"] < 1:
        raise Bad("malformed_record", "seq 0 is genesis only")
    if t == "key_rotation" and r["rotation"]["effective_seq"] != r["seq"] + 1:
        raise Bad("malformed_record", "effective_seq")
    if t == "anchor_event" and r["anchor"]["checkpoint_seq"] >= r["seq"]:
        raise Bad("malformed_record", "anchor precedes its checkpoint")


# ---- §6 signing and §8 Merkle ------------------------------------------------------------------------------------

def sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def ed_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
        return True
    except (InvalidSignature, ValueError):
        return False


def unsigned_canon(r: Mapping[str, Any]) -> bytes:
    return canon({k: v for k, v in r.items() if k != "signature"})


def link_of(r: Mapping[str, Any]) -> str:
    return sha(b"\x02" + sha(unsigned_canon(r)) + bytes.fromhex(r["signature"])).hex()


def leaf(data: bytes) -> bytes:
    return sha(b"\x00" + data)


def mth(leaves: list[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return sha(b"")
    if n == 1:
        return leaves[0]
    k = 1
    while k * 2 < n:
        k *= 2
    return sha(b"\x01" + mth(leaves[:k]) + mth(leaves[k:]))


def verify_inclusion(leaf_h: bytes, index: int, size: int, path: list[bytes], root: bytes) -> bool:
    """RFC 9162 §2.1.3.2."""
    if index >= size or index < 0:
        return False
    fn, sn, r = index, size - 1, leaf_h
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = sha(b"\x01" + p + r)
            if not fn & 1:
                while fn & 1 == 0 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = sha(b"\x01" + r + p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


# ---- §10 the trust root and §11 anchors ---------------------------------------------------------------------------

@dataclass
class Trust:
    manifest_hash: str
    keys: dict[str, tuple[bytes, str]]        # key_id -> (public key, role)


def parse_trust(obj: Mapping[str, Any]) -> Trust:
    keys: dict[str, tuple[bytes, str]] = {}
    for k in obj["keys"]:
        pub = bytes.fromhex(k["public_key"])
        assert hashlib.sha256(pub).hexdigest() == k["key_id"] and k["role"] in ("ledger", "witness", "boundary")
        assert k["key_id"] not in keys
        keys[k["key_id"]] = (pub, k["role"])
    assert any(role == "ledger" for _, role in keys.values())
    return Trust(obj["deployment_manifest_hash"], keys)


@dataclass
class AnchorInfo:
    valid: bool
    t: int = -1
    root: str = ""
    checkpoint_bytes: bytes | None = None
    problems: list[str] = field(default_factory=list)
    time_bound: str | None = None
    logbook: bool = False


def _print_checksum(size: int, root: str) -> str:
    return sha(b"cva-seal/1 anchor-print\n" + f"{size}\n{root}".encode()).hex()[:8]


def check_anchor(a: Any, trust: Trust, extra_ledger: Mapping[str, bytes]) -> AnchorInfo:
    if not isinstance(a, dict):
        return AnchorInfo(False, problems=["not an object"])
    if a.get("kind") == "logbook":
        if set(a) != {"v", "kind", "tree_size", "root_hash"} or not _isint(a["tree_size"]) or not _hex(a["root_hash"], 64):
            return AnchorInfo(False, problems=["malformed logbook anchor"])
        return AnchorInfo(True, a["tree_size"], a["root_hash"], None, logbook=True)
    if set(a) != {"v", "checkpoint", "cosignatures", "attestations"} or a["v"] != VERSION:
        return AnchorInfo(False, problems=["not an anchor"])
    cp = a["checkpoint"]
    info = AnchorInfo(False)
    try:
        validate(cp)
    except (Bad, KeyError, TypeError, AttributeError):
        info.problems.append("checkpoint is not a valid record")
        return info
    if cp["type"] != "checkpoint":
        info.problems.append("not a checkpoint")
        return info
    body = cp["checkpoint"]
    info.t, info.root = body["tree_size"], body["root_hash"]
    stored = canon(cp)
    info.checkpoint_bytes = stored
    if cp["seq"] != body["tree_size"]:
        info.problems.append("checkpoint index != tree_size")
    ledger_keys = {kid: pub for kid, (pub, role) in trust.keys.items() if role == "ledger"} | dict(extra_ledger)
    pub = ledger_keys.get(cp["key_id"])
    if pub is None or not ed_verify(pub, TAG_RECORD + unsigned_canon(cp), bytes.fromhex(cp["signature"])):
        info.problems.append("checkpoint signature / key")
    for c in a["cosignatures"]:
        if not (isinstance(c, dict) and set(c) == {"key_id", "sig"} and _hex(c["key_id"], 64) and _hex(c["sig"], 128)):
            info.problems.append("malformed cosignature")
            continue
        w = trust.keys.get(c["key_id"])
        if w is None or w[1] != "witness" or not ed_verify(w[0], TAG_COSIGN + stored, bytes.fromhex(c["sig"])):
            info.problems.append("cosignature")
    times = []
    for at in a["attestations"]:
        ok = (isinstance(at, dict) and set(at) == {"v", "kind", "root_hash", "tree_size", "observed_at_utc", "key_id", "sig"}
              and at["v"] == VERSION and at["kind"] == "witness_statement" and at["root_hash"] == body["root_hash"]
              and at["tree_size"] == body["tree_size"] and _ts(at["observed_at_utc"]) and _hex(at["key_id"], 64)
              and _hex(at["sig"], 128))
        if ok:
            b = trust.keys.get(at["key_id"])
            statement = {k: v for k, v in at.items() if k != "sig"}
            ok = b is not None and b[1] == "boundary" and ed_verify(b[0], TAG_WITNESS + canon(statement), bytes.fromhex(at["sig"]))
        if ok:
            times.append(at["observed_at_utc"])
        else:
            info.problems.append("attestation")
    info.time_bound = min(times) if times else None
    info.valid = not info.problems
    return info


# ---- §13 the procedure --------------------------------------------------------------------------------------------

@dataclass
class Finding:
    cls: str
    seq: int
    position: int
    severity: str
    nature: str
    primary: bool = True
    folded: int = 0
    check: str = ""


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    records: int = 0
    anchors_verified: int = 0
    anchored_records: int = 0
    unwitnessed: int = 0
    time_bound: str | None = None
    rotations: int = 0
    payloads_missing: int = 0

    @property
    def above_info(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "info"]

    @property
    def classes(self) -> list[str]:
        return [f.cls for f in self.above_info]


def verify(export: bytes, trust_root: Mapping[str, Any], *, anchors: Iterable[Any] = (),
           expect_deployment_hash: str | None = None, expected_count: int | None = None,
           reference_manifest: Mapping[str, Iterable[str]] | None = None, payloads: Mapping[str, bytes] | None = None,
           input_resolver: Callable[[Mapping[str, Any]], bytes | None] | None = None) -> Report:
    trust = parse_trust(trust_root)
    rep = Report()
    lines: list[bytes] = []
    unterminated = False
    if export:
        parts = export.split(b"\n")
        unterminated = parts[-1] != b""
        lines = parts if unterminated else parts[:-1]
    n = len(lines)
    rep.records = n
    if n == 0:
        rep.findings.append(Finding("genesis_mismatch", 0, 0, *SEVERITY["genesis_mismatch"], primary=False))
        return rep

    parsed: list[dict[str, Any] | None] = []
    errors: list[Bad | None] = []
    seq_positions: dict[int, list[int]] = {}
    for p, line in enumerate(lines):
        try:
            if unterminated and p == n - 1:
                raise Bad("malformed_record", "unterminated last line")
            r = strict_parse(line)
            validate(r)
            parsed.append(r)
            errors.append(None)
            seq_positions.setdefault(r["seq"], []).append(p)
        except Bad as e:
            parsed.append(None)
            errors.append(e)

    known: dict[str, tuple[bytes, str]] = dict(trust.keys)
    rotated_in: dict[str, bytes] = {}
    retired: set[str] = set()
    manifest = None if reference_manifest is None else {k: set(v) for k, v in reference_manifest.items()}

    st: dict[str, Any] = {"shift": 0, "shift_before": 0, "pending": {}, "prev_link": None, "active": None,
                          "rejected": {}, "nonces": set(), "verified_cp": {}, "last_primary": None, "last_pos": -10,
                          "taint": None, "regs": {}, "last_created": None}
    sig_ok_records: list[tuple[int, dict[str, Any]]] = []
    acc: list[bytes] = []

    def emit(cls: str, seq: int, pos: int, *, primary: bool = True, sev: str | None = None, nature: str | None = None,
             check: str = "") -> Finding:
        s, nat = SEVERITY[cls]
        f = Finding(cls, seq, pos, sev or s, nature or nat, primary, check=check)
        rep.findings.append(f)
        if primary:
            st["last_primary"], st["last_pos"] = f, pos
            if cls in TAINTING and st["taint"] is None:
                st["taint"] = f
        return f

    def fold(owner: Finding | None) -> None:
        if owner is not None:
            owner.folded += 1

    def sig_under(r: dict[str, Any], key_id: str | None) -> bool:
        e = known.get(key_id or "")
        return e is not None and ed_verify(e[0], TAG_RECORD + unsigned_canon(r), bytes.fromhex(r["signature"]))

    anchor_sizes: set[int] = set()
    loaded: list[tuple[Any, bool]] = []
    for raw in anchors:
        a = raw
        if isinstance(raw, (bytes, str)):
            try:
                a = json.loads(raw)
            except ValueError:
                loaded.append((None, False))
                continue
        try:
            size = a["tree_size"] if a.get("kind") == "logbook" else a["checkpoint"]["checkpoint"]["tree_size"]
            if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
                anchor_sizes.add(size)
        except (KeyError, TypeError, AttributeError):
            pass
        loaded.append((a, True))
    prefix_root: dict[int, str] = {}

    for p in range(n):
        if p in anchor_sizes:
            prefix_root[p] = mth(acc).hex()
        r, err = parsed[p], errors[p]
        try:
            if r is None:
                assert err is not None
                emit(err.code, p, p, check="parse")
                st["prev_link"] = None
                continue
            # 3 SEQ
            expected = p + st["shift"]
            suppress, replay_copy = False, False
            if r["seq"] != expected:
                pending = st["pending"]
                if r["seq"] in pending:
                    owner = pending.pop(r["seq"])
                    fold(owner)
                    st["last_primary"], st["last_pos"] = owner, p
                    suppress = True
                    if not pending:
                        st["shift"] = st["shift_before"]
                elif r["seq"] > expected:
                    missing = list(range(expected, r["seq"]))
                    if all(any(q > p for q in seq_positions.get(m, [])) for m in missing):
                        f = emit("record_reorder", r["seq"], p, check="seq_position")
                        st["shift_before"] = st["shift"]
                        for m in missing:
                            pending[m] = f
                        st["shift"] = r["seq"] - p
                    else:
                        emit("record_delete", missing[0], p, check="seq_position")
                        st["shift"] = r["seq"] - p
                        suppress = True
                else:
                    earlier = [q for q in seq_positions.get(r["seq"], []) if q < p]
                    if earlier:
                        same = lines[earlier[0]] == lines[p]
                        if same:
                            emit("record_replay", r["seq"], p, check="seq_position")
                        elif st["active"] is not None and sig_under(r, st["active"]):
                            emit("ledger_fork", r["seq"], p, nature="adversarial", check="ed25519_signature")
                        else:
                            emit("record_reorder", r["seq"], p, check="seq_position")
                        replay_copy = same
                        st["shift"] -= 1
                        suppress = True
                    else:
                        emit("record_reorder", r["seq"], p, check="seq_position")
                        suppress = True
            if p == 0 and r["type"] != "genesis":
                emit("genesis_mismatch", r["seq"], p, check="genesis_binding")
            # 4-5 KEY / SIGNATURE
            sig_ok = False
            if p == 0 and st["active"] is None:
                st["active"] = r["key_id"]
            if p == 0 and (r["key_id"] not in trust.keys or trust.keys[r["key_id"]][1] != "ledger"):
                emit("key_unauthorised", r["seq"], p, check="key_authorised")
            elif r["key_id"] in st["rejected"]:
                fold(st["rejected"][r["key_id"]])
                st["last_primary"], st["last_pos"] = st["rejected"][r["key_id"]], p
            elif r["key_id"] != st["active"]:
                if r["key_id"] in known and sig_under(r, r["key_id"]):
                    f = emit("key_unauthorised", r["seq"], p, nature="adversarial", check="key_authorised")
                elif r["key_id"] in known:
                    f = emit("record_edit", r["seq"], p, check="ed25519_signature")
                else:
                    f = emit("key_unauthorised", r["seq"], p, check="key_authorised")
                if r["type"] == "key_rotation" and f.cls == "key_unauthorised":
                    st["rejected"][r["rotation"]["new_key_id"]] = f
            else:
                sig_ok = sig_under(r, st["active"])
                lp = st["last_primary"]
                if not sig_ok and lp is not None and lp.cls == "record_edit" and st["last_pos"] == p - 1:
                    fold(lp)
                    st["last_pos"] = p
                elif not sig_ok:
                    emit("record_edit", r["seq"], p, check="ed25519_signature")
            # 6 LINK
            if p == 0:
                if r["prev_record_hash"] != trust.manifest_hash:
                    emit("genesis_mismatch", r["seq"], p, check="genesis_binding")
                if expect_deployment_hash is not None and r["prev_record_hash"] != expect_deployment_hash:
                    emit("genesis_mismatch", r["seq"], p, primary=False, check="genesis_binding")
            elif st["prev_link"] is None or r["prev_record_hash"] != st["prev_link"]:
                if suppress or (st["last_primary"] is not None and st["last_pos"] in (p, p - 1)):
                    fold(st["last_primary"])
                else:
                    emit("chain_broken", r["seq"], p, check="hash_link")
            if sig_ok:
                sig_ok_records.append((p, r))
                # 7 NONCE
                if r["nonce"] in st["nonces"] and not replay_copy:
                    emit("nonce_reuse", r["seq"], p, check="nonce_unique")
                st["nonces"].add(r["nonce"])
                # 8 TYPE RULES
                t = r["type"]
                if t == "checkpoint":
                    cp = r["checkpoint"]
                    if cp["tree_size"] == len(acc) and cp["root_hash"] == mth(acc).hex():
                        st["verified_cp"][r["seq"]] = (cp["tree_size"], cp["root_hash"])
                    elif st["taint"] is not None:
                        fold(st["taint"])
                    else:
                        emit("checkpoint_mismatch", r["seq"], p, check="checkpoint_root")
                elif t == "anchor_event":
                    a = r["anchor"]
                    if st["verified_cp"].get(a["checkpoint_seq"]) != (a["tree_size"], a["root_hash"]):
                        if st["taint"] is not None:
                            fold(st["taint"])
                        else:
                            emit("checkpoint_mismatch", r["seq"], p, primary=False, check="anchor_event")
                    st["nonces"] = {r["nonce"]}
                elif t == "degraded_marker":
                    emit("degraded_gap", r["seq"], p, primary=False, check="declared_gap")
                elif t == "model_registration":
                    m = r["model"]
                    old = st["regs"].get(m["id"])
                    st["regs"][m["id"]] = m
                    if manifest is not None:
                        allowed = manifest.get(m["id"])
                        if allowed is None or m["weights_sha256"] not in allowed:
                            emit("model_swap", r["seq"], p, check="model_registration")
                    elif old is not None and old["weights_sha256"] != m["weights_sha256"]:
                        emit("model_swap", r["seq"], p, primary=False, sev="info", check="model_registration")
                elif t == "inference":
                    m = r["model"]
                    reg = st["regs"].get(m["id"])
                    if reg is None or (m["weights_sha256"], m["arch_hash"], m["format"]) != (
                            reg["weights_sha256"], reg["arch_hash"], reg["format"]):
                        emit("model_swap", r["seq"], p, check="model_registration")
                elif t == "key_rotation":
                    rot = r["rotation"]
                    pop = TAG_POP + canon({"new_key_id": rot["new_key_id"], "effective_seq": rot["effective_seq"],
                                           "prev_key_id": r["key_id"]})
                    if not ed_verify(bytes.fromhex(rot["new_public_key"]), pop, bytes.fromhex(rot["new_key_pop"])):
                        f = emit("key_unauthorised", r["seq"], p, check="rotation_pop")
                        st["rejected"][rot["new_key_id"]] = f
                    else:
                        known[rot["new_key_id"]] = (bytes.fromhex(rot["new_public_key"]), "ledger")
                        rotated_in[rot["new_key_id"]] = bytes.fromhex(rot["new_public_key"])
                        retired.add(st["active"])
                        st["active"] = rot["new_key_id"]
                        rep.rotations += 1
            # 9 TIME
            if st["last_created"] is not None and r["created_at_utc"] < st["last_created"]:
                emit("clock_regression", r["seq"], p, primary=False, check="clock_order")
            st["last_created"] = r["created_at_utc"]
            st["prev_link"] = link_of(r)
        finally:
            acc.append(leaf(lines[p]))
    if n in anchor_sizes:
        prefix_root[n] = mth(acc).hex()

    # anchors
    best = -1

    def _explained(taint: Finding | None, limit: int) -> bool:
        return taint is not None and taint.position < limit
    last_anchor_event = None
    for _p, r in sig_ok_records:
        if r["type"] == "anchor_event":
            last_anchor_event = r["seq"]
    for a, parsable in loaded:
        if not parsable or a is None:
            emit("anchor_invalid", n, n, primary=False)
            continue
        info = check_anchor(a, trust, rotated_in)
        if not info.valid:
            emit("anchor_invalid", n, n, primary=False)
            continue
        t = info.t
        covered = t if info.logbook else t + 1
        if n < covered:
            emit("tail_truncation", n, n, primary=False, check="anchor_size")
            continue
        taint = st["taint"]
        if prefix_root.get(t) != info.root:
            if _explained(taint, t):
                fold(taint)
            else:
                emit("ledger_fork", n, n, primary=False, nature="adversarial", check="anchor_root")
            continue
        if info.checkpoint_bytes is not None and lines[t] != info.checkpoint_bytes:
            if _explained(taint, t + 1):
                fold(taint)
            else:
                emit("ledger_fork", n, n, primary=False, nature="adversarial", check="anchor_checkpoint")
            continue
        rep.anchors_verified += 1
        if covered > best:
            best, rep.anchored_records, rep.time_bound = covered, covered, info.time_bound
    rep.unwitnessed = (n - rep.anchored_records) if rep.anchors_verified else (
        n if last_anchor_event is None else max(0, n - 1 - last_anchor_event))
    if expected_count is not None:
        got = sum(1 for _, r in [(p, x) for p, x in enumerate(parsed) if x is not None] if r["type"] == "inference")
        if got != expected_count:
            emit("ledger_incomplete", n, n, primary=False, check="count_reconcile")
    for p, r in sig_ok_records:
        if r["type"] != "inference":
            continue
        if input_resolver is not None:
            data = input_resolver(r)
            if data is not None and hashlib.sha256(data).hexdigest() != r["input"]["sha256"]:
                emit("input_swap", r["seq"], p, primary=False, check="input_hash")
        out = r["output"]
        if out["payload_ref"] != "sha256:" + out["jcs_sha256"]:
            emit("output_payload_mismatch", r["seq"], p, primary=False, check="payload_hash")
        for addr in dict.fromkeys((out["jcs_sha256"], out["raw_jcs_sha256"])):
            data = None if payloads is None else payloads.get(addr)
            if data is None:
                rep.payloads_missing += 1
            elif hashlib.sha256(data).hexdigest() != addr:
                emit("output_payload_mismatch", r["seq"], p, primary=False, check="payload_hash")
    return rep


# ---- quantisation (§3) — for the conformance vectors ----------------------------------------------------------------

def _floor(v: float) -> int:
    i = int(v)
    return i - 1 if v < 0 and i != v else i


def q_conf(p: float) -> int:
    return max(0, min(1_000_000, _floor(float(p) * 1_000_000.0 + 0.5)))


def q_box64(x: float) -> int:
    return _floor(float(x) * 64.0 + 0.5)


def q_px(x: float) -> int:
    return _floor(float(x) + 0.5)


def output_objects(raw: Mapping[str, Any], filtered: Mapping[str, Any] | None) -> tuple[bytes, bytes, bytes]:
    """The three canonical output objects of §5: (raw, fine, coarse)."""
    fil = raw if filtered is None else filtered

    def fine(o: Mapping[str, Any], with_filter: bool) -> dict[str, Any]:
        if o["task"] == "classify":
            out: dict[str, Any] = {"task": "classify", "top": [{"cls": t["cls"], "conf_e6": q_conf(t["conf"])} for t in o["top"]]}
        else:
            out = {"task": "detect", "detections": [{"cls": d["cls"], "conf_e6": q_conf(d["conf"]),
                                                     "box_q64": [q_box64(x) for x in d["box"]]} for d in o["detections"]]}
        if with_filter and "filter" in o:
            out["filter"] = {"conf_thr_e6": q_conf(o["filter"]["conf_thr"]), "nms_iou_e6": q_conf(o["filter"]["nms_iou"])}
        return out

    def coarse(o: Mapping[str, Any]) -> dict[str, Any]:
        if o["task"] == "classify":
            out: dict[str, Any] = {"task": "classify", "top": [{"cls": t["cls"]} for t in o["top"]]}
        else:
            dets = sorted(({"cls": d["cls"], "box_px": [q_px(x) for x in d["box"]]} for d in o["detections"]),
                          key=lambda d: (d["cls"], d["box_px"]))
            out = {"task": "detect", "detections": dets}
        if "filter" in o:
            out["filter"] = {"conf_thr_e6": q_conf(o["filter"]["conf_thr"]), "nms_iou_e6": q_conf(o["filter"]["nms_iou"])}
        return out
    return canon(fine(raw, False)), canon(fine(fil, True)), canon(coarse(fil))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ledger")
    ap.add_argument("--trust", required=True)
    ap.add_argument("--anchor", action="append", default=[])
    ap.add_argument("--payloads")
    a = ap.parse_args()
    pl = None
    if a.payloads:
        pl = {}
        for line in open(a.payloads, "rb"):
            o = json.loads(line)
            pl[o["address"]] = base64.b64decode(o["b64"], validate=True)
    rep = verify(open(a.ledger, "rb").read(), json.load(open(a.trust)),
                 anchors=[json.load(open(x)) for x in a.anchor], payloads=pl)
    for f in rep.above_info:
        print(f"[{f.severity}] {f.cls} @ {f.seq}" + (f"  (+{f.folded} folded)" if f.folded else ""))
    declared_only = bool(rep.above_info) and all(f.cls == "degraded_gap" for f in rep.above_info)
    print(f"{'CLEAN' if not rep.above_info else ('INTACT, WITH DECLARED GAPS' if declared_only else 'PROBLEMS')}: {rep.records} records, {rep.anchors_verified} anchor(s) verified, "
          f"{rep.unwitnessed} in the unwitnessed window" + ("" if rep.anchors_verified else
                                                          " (NO anchor: tail truncation and a key-holder rewrite are not excluded)"))
    return 0 if not rep.above_info else 2


if __name__ == "__main__":
    sys.exit(main())
