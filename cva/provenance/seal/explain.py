"""`cva-seal explain` and `cva-seal selftest` (plan §7.11): a human-readable account of one record, and a
self-check an operator can run on the box before trusting anything else it says.

`explain` is for the person at the screen (demo step 4: hand-edit a record, verification points at it): it says what
the record is, whether it verifies, how it is chained, whether an anchor covers it, and what the verifier found about
it — in words, with the hashes an auditor would want to copy.
"""
from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, parse_strict
from .chain import link_from, verify_signature
from .errors import NonCanonical, SealError
from .keys import EnvKeyProvider, TrustKey, TrustRoot, load_trust_root, verify_ed25519
from .merkle import mth
from .records import record_hash, validate_record
from .verify import export_records, open_source, verify_ledger


def explain_record(source: Any, trust_root: TrustRoot | str | os.PathLike[str], seq: int, *,
                   anchors: Iterable[Any] = ()) -> str:
    tr = trust_root if isinstance(trust_root, TrustRoot) else load_trust_root(trust_root)
    anchors = list(anchors)
    rep = verify_ledger(source, trust_root=tr, anchors=anchors)
    src = open_source(source)
    try:
        rows = [r.data for r in src.rows()]
    finally:
        src.close()
    lines: list[str] = []
    here = [f for f in rep.findings if f.seq == seq and f.severity != "info"]
    if not 0 <= seq < len(rows):
        lines.append(f"No record with seq {seq}: the ledger holds {len(rows)} record(s) (seq 0…{len(rows) - 1}).")
        return "\n".join(lines + _findings(here))
    try:
        rec = parse_strict(rows[seq])
        validate_record(rec, signed=True)
    except (NonCanonical, SealError) as e:
        return "\n".join([f"Record at position {seq} is not a valid record: {e}."] + _findings(here))
    lines.append(f"Record seq {rec['seq']} — {rec['type']}, written {rec['created_at_utc']} (host clock, untrusted).")
    lines.append(f"  signed by key {rec['key_id'][:16]}… ; record hash {record_hash(rec).hex()[:24]}…")
    pub = next((k.public_key for k in tr.keys if k.key_id == rec["key_id"]), None)
    if pub is None:
        lines.append("  signature: not checked here — the key is not in the trust root (it may have been introduced "
                     "by a key rotation; `verify` follows the rotation chain).")
    else:
        ok = verify_signature(rec, pub)
        lines.append(f"  signature: {'VERIFIES' if ok else 'DOES NOT VERIFY — the stored bytes differ from what was signed'}")
    if seq > 0:
        try:
            prev = parse_strict(rows[seq - 1])
            want = link_from(record_hash(prev), prev["signature"])
            lines.append(f"  chain: links to seq {seq - 1} — "
                         f"{'matches' if rec['prev_record_hash'] == want else 'DOES NOT MATCH the previous record'}")
        except (NonCanonical, SealError, KeyError):
            lines.append(f"  chain: the previous record (seq {seq - 1}) is itself unreadable")
    if seq + 1 < len(rows):
        lines.append(f"  successor: seq {seq + 1} commits to this record's signature, so changing this record "
                     "breaks that link")
    else:
        lines.append("  successor: none — this is the last record, so nothing in the chain would notice if it (and "
                     "those before it) were removed; only an anchor can")
    if rec["type"] == "inference":
        m, i, o = rec["model"], rec["input"], rec["output"]
        lines += [f"  model: {m['id']} ({m['format']}), weights {m['weights_sha256'][:16]}…",
                  f"  input: {i['dims'][0]}x{i['dims'][1]}, {i['source_kind']}, sha256 {i['sha256'][:16]}…",
                  f"  output: decision hash {o['decision_sha256'][:16]}…, fine hash {o['jcs_sha256'][:16]}…, "
                  f"raw hash {o['raw_jcs_sha256'][:16]}…",
                  f"  runtime: {rec['config']['runtime']}"]
        exists, data = src_payload(source, o["jcs_sha256"])
        if exists:
            good = data is not None and hashlib.sha256(data).hexdigest() == o["jcs_sha256"]
            lines.append(f"  stored output: {'matches its committed hash' if good else 'DOES NOT match its committed hash'}")
        else:
            lines.append("  stored output: not available in this source (an export carries records only)")
    elif rec["type"] == "key_rotation":
        lines.append(f"  rotates signing to key {rec['rotation']['new_key_id'][:16]}… from seq {rec['rotation']['effective_seq']}")
    elif rec["type"] == "degraded_marker":
        g = rec["gap"]
        lines.append(f"  declares {g['reported_count']} unsealed inference(s) between {g['first_unsealed_utc']} and {g['last_unsealed_utc']} "
                     f"({g['reason']})")
    elif rec["type"] in ("checkpoint", "anchor_event"):
        body = rec.get("checkpoint") or rec["anchor"]
        lines.append(f"  covers the first {body['tree_size']} records, root {body['root_hash'][:16]}…")
    covered = rep.anchored_records
    if rep.anchors_verified:
        lines.append(f"  anchor: {'COVERED — this record matches an anchor file (see the report for whether anyone cosigned or attested it)' if seq < covered else 'NOT covered — it was written after the newest verified anchor'}"
                     + (f"; existed by {rep.attested_not_after} (witness clock)" if seq < covered and rep.attested_not_after else ""))
    else:
        lines.append("  anchor: none supplied — nothing outside the ledger vouches for this record's presence")
    lines += _findings(here)
    if not here:
        lines.append("  findings: none for this record.")
    return "\n".join(lines)


def _findings(fs: list[Any]) -> list[str]:
    out = []
    for f in fs:
        out.append(f"  FINDING [{f.severity}] {f.attack_class}: {f.reason}")
    return out


def src_payload(source: Any, address: str) -> tuple[bool, bytes | None]:
    s = open_source(source)
    try:
        return s.payload(address)
    finally:
        s.close()


# --- selftest --------------------------------------------------------------------------------------------------

_RFC8032_1 = ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
              "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
              "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")


def selftest(records: int = 1000) -> list[tuple[str, bool, str]]:
    """Run the known-answer checks and a round trip. Returns (name, passed, detail) rows; all must pass."""
    from .anchor import export_anchor
    from .keys import generate_keypair
    from .sealer import Sealer
    from .store import SealedLedger
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    check("SHA-256 of the empty string", hashlib.sha256(b"").hexdigest() ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    check("empty Merkle tree root = SHA-256('')", mth([]).hex() == hashlib.sha256(b"").hexdigest())
    seed, pub, sig = (bytes.fromhex(x) for x in _RFC8032_1)
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    check("Ed25519 matches RFC 8032 §7.1 test 1 (public key)", sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw) == pub)
    check("Ed25519 matches RFC 8032 §7.1 test 1 (signature)", sk.sign(b"") == sig and verify_ed25519(pub, b"", sig))
    check("canonicalisation orders keys and forbids floats", canonical_bytes({"b": 1, "a": {"y": [1, 2], "x": True}}) ==
          b'{"a":{"x":true,"y":[1,2]},"b":1}')
    try:
        canonical_bytes({"a": 1.5})
        check("a float is refused", False)
    except NonCanonical:
        check("a float is refused", True)
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        generate_keypair(dd / "k")
        key = EnvKeyProvider("K", environ={"K": base64.b64encode((dd / "k").read_bytes()).decode()})
        led = SealedLedger.init_ledger(dd / "l.db", key, {"device_id": "selftest", "unit": "selftest",
                                                          "profile_hash": "0" * 64, "checkpoint_every": 100},
                                       durability="group_commit")
        trust = TrustRoot(_genesis(led), (TrustKey(key.key_id, key.public_key, "ledger"),))
        led.close()
        with Sealer.open(dd / "l.db", key=key, trust_root=trust) as s:
            m = s.register_model(id="selftest", weights_sha256="a" * 64, arch_hash="b" * 64, format="onnx")
            c = s.register_config(preprocess_spec={"a": 1}, postprocess_spec={"b": 2}, runtime="selftest",
                                  version_pins_hash="c" * 64, code_commit="0" * 40)
            for i in range(records):
                s.seal(i.to_bytes(4, "big") * 64, m, c, output={"task": "classify", "top": [{"cls": i % 10, "conf": 0.9}]},
                       dims=(8, 8))
        rep = verify_ledger(dd / "l.db", trust_root=trust)
        check(f"{records}-record round trip verifies clean", rep.clean and rep.records_checked > records,
              f"{rep.records_checked} records, {rep.checkpoints_verified} checkpoints recomputed")
        export_records(dd / "l.db", dd / "x.jsonl")
        check("the export verifies the same as the database", verify_ledger(dd / "x.jsonl", trust_root=trust).clean)
        with SealedLedger.open(dd / "l.db", key=key, background_flush=False) as led2:
            anchor = export_anchor(led2, dd / "a.json")
        export_records(dd / "l.db", dd / "x2.jsonl")
        rep2 = verify_ledger(dd / "x2.jsonl", trust_root=trust, anchors=[anchor])
        check("an anchor verifies against the ledger", rep2.clean and rep2.anchors_verified == 1)
        raw = bytearray((dd / "x2.jsonl").read_bytes())
        mid = len(raw) // 2
        raw[mid] ^= 0x01
        check("a single flipped bit in the middle is detected", not verify_ledger(bytes(raw), trust_root=trust).clean)
        lines = (dd / "x2.jsonl").read_bytes().split(b"\n")[:-1]
        cut = b"".join(x + b"\n" for x in lines[:-5])
        check("tail truncation is invisible without an anchor (stated, not hidden)",
              verify_ledger(cut, trust_root=trust).clean)
        check("...and caught with one", "tail_truncation" in verify_ledger(cut, trust_root=trust, anchors=[anchor]).classes())
    return results


def _genesis(led: Any) -> str:
    from .records import genesis_prev_hash
    return genesis_prev_hash(led.deployment_manifest)
