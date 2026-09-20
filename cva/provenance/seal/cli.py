"""`cva-seal` — the operator's command line for the seal (plan §7.11).

Implemented so far: `keygen`, `init`, `bench-durability` (C4); `verify`, `export` (C5); `rotate`, `trust add`,
`anchor export|cosign|attest|verify|print`, `proof inclusion|verify` (C7). `selftest`, `explain` (C8).

Exit codes: 0 = done / nothing above `info` found; 1 = the command could not run; 2 = it ran and found problems
(a verify with findings above `info`, an anchor or proof that does not verify).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .anchor import (
    attest,
    cosign,
    export_anchor,
    load_anchor,
    logbook_anchor,
    print_anchor,
    verify_anchor,
    write_anchor,
)
from .bench import measure_durability
from .errors import SealError
from .explain import explain_record, selftest
from .keys import (
    TRUST_ROLES,
    FileKeyProvider,
    TrustKey,
    TrustRoot,
    generate_keypair,
    load_trust_root,
)
from .proof import encode_proof, make_inclusion_proof, verify_inclusion_proof
from .records import genesis_prev_hash, key_id_of
from .store import DURABILITY_MODES, SealedLedger
from .verify import export_payloads, export_records, load_payloads, verify_ledger


def _keygen(a: argparse.Namespace) -> int:
    kid = generate_keypair(a.out)
    pub = FileKeyProvider(a.out).public_key.hex()
    print(json.dumps({"key_file": a.out, "key_id": kid, "public_key": pub, "custody": "file",
                      "note": "back this file up and restrict access; it is the deployment's trust anchor"}, indent=2))
    return 0


def _init(a: argparse.Namespace) -> int:
    key = FileKeyProvider(a.key)
    manifest = {"device_id": a.device_id, "unit": a.unit,
                "profile_hash": a.profile_hash or hashlib.sha256(b"").hexdigest(),
                "checkpoint_every": a.checkpoint_every}
    led = SealedLedger.init_ledger(a.ledger, key, manifest, durability=a.durability,
                                   group_n=a.group_n, group_ms=a.group_ms)
    tr = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    if a.trust_out:
        Path(a.trust_out).write_bytes(tr.to_bytes() + b"\n")
    print(json.dumps({"ledger": a.ledger, "key_id": key.key_id, "deployment_manifest_hash": tr.deployment_manifest_hash,
                      "durability": a.durability,
                      "loss_window": "0 records" if a.durability == "per_record"
                      else f"up to {a.group_n} records / {a.group_ms} ms",
                      "trust_root": a.trust_out}, indent=2))
    return 0


def _bench(a: argparse.Namespace) -> int:
    r = measure_durability(a.dir, n=a.n, input_bytes=a.input_bytes, budget_ms=a.budget_ms,
                           group_n=a.group_n, group_ms=a.group_ms)
    print(json.dumps(r, indent=2))
    print(f"\nfilesystem: {r['filesystem']}   recommendation: {r['recommendation']}\n{r['reason']}", file=sys.stderr)
    return 0 if r["fsync_reliable"] else 2


def _dump(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _verify(a: argparse.Namespace) -> int:
    anchors: list[object] = [Path(x) for x in a.anchor or []]
    for spec in a.logbook or []:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise SealError("--logbook is SIZE:ROOT[:CHECKSUM]")
        if not parts[0].isdigit():
            raise SealError("--logbook SIZE must be an integer")
        anchors.append(logbook_anchor(int(parts[0]), parts[1], parts[2] if len(parts) == 3 else None))
    payloads = load_payloads(a.payloads) if a.payloads else None
    r = verify_ledger(a.records, trust_root=a.trust, anchors=anchors, payloads=payloads,
                      expected_count=a.expected_count)
    summary = {"source": r.source_kind, "records_checked": r.records_checked,
               "checkpoints_verified": r.checkpoints_verified, "key_rotations": r.rotations,
               "anchors_verified": r.anchors_verified, "anchors_unusable": r.anchors_invalid,
               "records_fixed_by_anchor": r.anchored_records, "sealed_not_after_utc": r.attested_not_after,
               "records_in_unwitnessed_window": r.unwitnessed_records, "declared_gaps": r.declared_gaps,
               "durability": r.durability, "clean": r.clean,
               "findings": [{"class": f.attack_class, "seq": f.seq, "severity": f.severity, "nature": f.nature,
                             "reason": f.reason} for f in r.findings if f.severity != "info"]}
    if a.json:
        _dump(summary)
    else:
        only_declared = not r.clean and all(f.attack_class == "degraded_gap" for f in r.findings if f.severity != "info")
        label = "CLEAN" if r.clean else ("INTACT, WITH DECLARED GAPS" if only_declared else "PROBLEMS FOUND")
        print(f"{label}: {r.records_checked} record(s), {r.checkpoints_verified} "
              f"checkpoint(s) recomputed, {r.rotations} key rotation(s), {r.anchors_verified} anchor(s) verified"
              + (f", {r.anchors_invalid} unusable" if r.anchors_invalid else ""))
        if r.anchors_verified:
            custody = (f"{r.witnessed_anchors} cosigned/attested" if r.witnessed_anchors else
                       "NONE cosigned or attested — custody of the anchor file is not established")
            print(f"  anchor custody: {custody}")
            print(f"  anchored: the first {r.anchored_records} records match an anchor file"
                  + (f"; they existed by {r.attested_not_after} (witness clock)" if r.attested_not_after else ""))
        print(f"  {r.unwitnessed_records} record(s) lie in the unwitnessed window — "
              + ("after the newest verified anchor." if r.anchors_verified else
                 "NO external anchor was supplied, so tail truncation and a wholesale rewrite by the key holder "
                 "cannot be excluded."))
        for f in summary["findings"]:  # type: ignore[attr-defined]
            print(f"  [{f['severity']}] {f['class']} @ {f['seq']}: {f['reason']}")
    return 0 if r.clean else 2


def _export(a: argparse.Namespace) -> int:
    n = export_records(a.ledger, a.out)
    out: dict[str, object] = {"records": n, "out": a.out}
    if a.payloads_out:
        out["payloads"] = export_payloads(a.ledger, a.payloads_out)
        out["payloads_out"] = a.payloads_out
    _dump(out)
    return 0


def _rotate(a: argparse.Namespace) -> int:
    old, new = FileKeyProvider(a.key), FileKeyProvider(a.new_key)
    with SealedLedger.open(a.ledger, key=old) as led:
        w = led.rotate_key(new)
    _dump({"rotated_at_seq": w.seq, "outgoing_key_id": old.key_id, "incoming_key_id": new.key_id,
           "note": "the trust root needs no change: the incoming key is vouched for by this signed rotation record"})
    return 0


def _trust_add(a: argparse.Namespace) -> int:
    tr = load_trust_root(a.trust)
    pub = bytes.fromhex(a.public_key)
    if len(pub) != 32:
        raise SealError("--public-key must be 64 hex characters")
    kid = key_id_of(a.public_key)
    if any(k.key_id == kid for k in tr.keys):
        raise SealError(f"key {kid[:16]}… is already in the trust root")
    out = TrustRoot(tr.deployment_manifest_hash, (*tr.keys, TrustKey(kid, pub, a.role)))
    Path(a.out or a.trust).write_bytes(out.to_bytes() + b"\n")
    _dump({"trust_root": a.out or a.trust, "added": {"key_id": kid, "role": a.role}})
    return 0


def _anchor_export(a: argparse.Namespace) -> int:
    key = FileKeyProvider(a.key)
    with SealedLedger.open(a.ledger, key=key) as led:
        anchor = export_anchor(led, a.out, medium=a.medium, label=a.label, cosigner_key_ids=tuple(a.cosigner or ()))
    cp = anchor["checkpoint"]["checkpoint"]
    _dump({"anchor": a.out, "tree_size": cp["tree_size"], "root_hash": cp["root_hash"], "medium": a.medium,
           "note": "an anchor_event was appended to the ledger; copy the anchor to media the key holder cannot rewrite"})
    return 0


def _anchor_cosign(a: argparse.Namespace) -> int:
    w = FileKeyProvider(a.witness_key)
    write_anchor(cosign(load_anchor(a.anchor), w), a.out)
    _dump({"cosigned_by": w.key_id, "out": a.out})
    return 0


def _anchor_attest(a: argparse.Namespace) -> int:
    b = FileKeyProvider(a.boundary_key)
    at = a.observed_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    write_anchor(attest(load_anchor(a.anchor), b, at), a.out)
    _dump({"attested_by": b.key_id, "observed_at_utc": at, "out": a.out})
    return 0


def _anchor_verify(a: argparse.Namespace) -> int:
    r = verify_anchor(a.anchor, load_trust_root(a.trust))
    _dump({"valid": r.valid, "medium": r.medium, "tree_size": r.tree_size, "root_hash": r.root_hash,
           "cosigners": r.cosigners, "attestations": r.attestations, "sealed_not_after_utc": r.time_bound_utc,
           "problems": r.problems, "notes": r.notes})
    return 0 if r.valid else 2


def _anchor_print(a: argparse.Namespace) -> int:
    print(print_anchor(load_anchor(a.anchor)), end="")
    return 0


def _proof_inclusion(a: argparse.Namespace) -> int:
    proof = make_inclusion_proof(a.records, a.seq, a.anchor)
    Path(a.out).write_bytes(encode_proof(proof))
    _dump({"proof": a.out, "seq": a.seq, "tree_size": proof["tree_size"], "path_length": len(proof["path"])})
    return 0


def _proof_verify(a: argparse.Namespace) -> int:
    r = verify_inclusion_proof(Path(a.proof), Path(a.anchor), load_trust_root(a.trust))
    _dump({"valid": r.valid, "seq": r.seq, "record_type": r.record_type, "signature_ok": r.signature_ok,
           "problems": r.problems, "notes": r.notes})
    return 0 if r.valid else 2


def _explain(a: argparse.Namespace) -> int:
    anchors: list[object] = [Path(x) for x in a.anchor or []]
    print(explain_record(a.records, a.trust, a.seq, anchors=anchors))
    return 0


def _selftest(a: argparse.Namespace) -> int:
    rows = selftest(a.records)
    for name, ok, detail in rows:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    bad = [r for r in rows if not r[1]]
    print("selftest: " + ("all checks passed" if not bad else f"{len(bad)} check(s) FAILED — do not trust this installation"))
    return 0 if not bad else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cva-seal", description="Inference provenance seal (Module C)")
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("keygen", help="create a NEW signing key (never done implicitly)")
    k.add_argument("--out", required=True, help="path for the new key file (mode 0600; never overwritten)")
    k.set_defaults(fn=_keygen)

    i = sub.add_parser("init", help="create a ledger: write genesis, optionally emit the trust root")
    i.add_argument("--ledger", required=True)
    i.add_argument("--key", required=True, help="the signing key file (from `keygen`)")
    i.add_argument("--device-id", required=True)
    i.add_argument("--unit", required=True)
    i.add_argument("--profile-hash", help="64-hex hash of the deployment profile (default: SHA-256 of empty)")
    i.add_argument("--checkpoint-every", type=int, default=1000)
    i.add_argument("--durability", choices=DURABILITY_MODES, default="per_record",
                   help="decide with `bench-durability`; stored in the ledger and printed in every verify report")
    i.add_argument("--group-n", type=int, default=100)
    i.add_argument("--group-ms", type=int, default=50)
    i.add_argument("--trust-out", help="write trust_root.json here")
    i.set_defaults(fn=_init)

    b = sub.add_parser("bench-durability", help="measure per_record vs group_commit on the TARGET storage")
    b.add_argument("--dir", required=True, help="a directory on the real storage (NOT tmpfs)")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--input-bytes", type=int, default=2_000_000)
    b.add_argument("--budget-ms", type=float, default=5.0)
    b.add_argument("--group-n", type=int, default=100)
    b.add_argument("--group-ms", type=int, default=50)
    b.set_defaults(fn=_bench)

    v = sub.add_parser("verify", help="verify a ledger (SQLite file or JSONL export) — no live database needed")
    v.add_argument("--records", required=True, help="the ledger .db, or a `cva-seal export` file")
    v.add_argument("--trust", required=True, help="trust_root.json")
    v.add_argument("--anchor", action="append", help="an anchor file (repeatable)")
    v.add_argument("--logbook", action="append", help="SIZE:ROOT[:CHECKSUM] copied from a physical logbook")
    v.add_argument("--payloads", help="payload sidecar from `export --payloads-out`")
    v.add_argument("--expected-count", type=int, help="the operator's independent count of sealed inferences")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=_verify)

    e = sub.add_parser("export", help="write the strict JSONL export (each line = the stored record, verbatim)")
    e.add_argument("--ledger", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--payloads-out", help="also write the payload sidecar")
    e.set_defaults(fn=_export)

    r = sub.add_parser("rotate", help="rotate the signing key (outgoing-signed, incoming proves possession)")
    r.add_argument("--ledger", required=True)
    r.add_argument("--key", required=True, help="the CURRENT signing key file")
    r.add_argument("--new-key", required=True, help="the incoming key file (from `keygen`)")
    r.set_defaults(fn=_rotate)

    t = sub.add_parser("trust", help="edit a trust root").add_subparsers(dest="trust_cmd", required=True)
    ta = t.add_parser("add", help="add a witness/boundary/ledger public key to a trust root")
    ta.add_argument("--trust", required=True)
    ta.add_argument("--public-key", required=True, help="64 hex characters")
    ta.add_argument("--role", required=True, choices=TRUST_ROLES)
    ta.add_argument("--out", help="write here instead of rewriting --trust in place")
    ta.set_defaults(fn=_trust_add)

    an = sub.add_parser("anchor", help="the anchoring ceremony").add_subparsers(dest="anchor_cmd", required=True)
    ae = an.add_parser("export", help="write a signed-checkpoint anchor file and append an anchor_event")
    ae.add_argument("--ledger", required=True)
    ae.add_argument("--key", required=True)
    ae.add_argument("--out", required=True)
    ae.add_argument("--medium", choices=("write_once", "cosign", "logbook"), default="write_once")
    ae.add_argument("--label", default="")
    ae.add_argument("--cosigner", action="append", help="key id of an expected cosigner (repeatable)")
    ae.set_defaults(fn=_anchor_export)
    ac = an.add_parser("cosign", help="a second party countersigns an anchor")
    ac.add_argument("--anchor", required=True)
    ac.add_argument("--witness-key", required=True)
    ac.add_argument("--out", required=True)
    ac.set_defaults(fn=_anchor_cosign)
    at = an.add_parser("attest", help="the boundary device attests it saw this root (bounds absolute time)")
    at.add_argument("--anchor", required=True)
    at.add_argument("--boundary-key", required=True)
    at.add_argument("--observed-at", help="UTC, YYYY-MM-DDTHH:MM:SS.ffffffZ (default: now)")
    at.add_argument("--out", required=True)
    at.set_defaults(fn=_anchor_attest)
    av = an.add_parser("verify", help="verify an anchor against a trust root")
    av.add_argument("--anchor", required=True)
    av.add_argument("--trust", required=True)
    av.set_defaults(fn=_anchor_verify)
    ap = an.add_parser("print", help="a block to copy into a physical logbook")
    ap.add_argument("--anchor", required=True)
    ap.set_defaults(fn=_anchor_print)

    pr = sub.add_parser("proof", help="inclusion proofs").add_subparsers(dest="proof_cmd", required=True)
    pi = pr.add_parser("inclusion", help="prove one record is in an anchored root")
    pi.add_argument("--records", required=True)
    pi.add_argument("--seq", type=int, required=True)
    pi.add_argument("--anchor", required=True)
    pi.add_argument("--out", required=True)
    pi.set_defaults(fn=_proof_inclusion)
    pv = pr.add_parser("verify", help="verify an inclusion proof from the proof + anchor + trust root alone")
    pv.add_argument("--proof", required=True)
    pv.add_argument("--anchor", required=True)
    pv.add_argument("--trust", required=True)
    pv.set_defaults(fn=_proof_verify)

    x = sub.add_parser("explain", help="a human-readable account of one record")
    x.add_argument("--records", required=True)
    x.add_argument("--trust", required=True)
    x.add_argument("--seq", type=int, required=True)
    x.add_argument("--anchor", action="append")
    x.set_defaults(fn=_explain)

    st = sub.add_parser("selftest", help="known-answer checks plus a round trip; run this before trusting an install")
    st.add_argument("--records", type=int, default=1000)
    st.set_defaults(fn=_selftest)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except SealError as e:
        print(f"cva-seal: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
