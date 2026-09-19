"""`cva-seal` — the operator's command line for the seal (plan §7.11).

Implemented so far (gate C4): `keygen`, `init`, `bench-durability`. `verify`, `export`, `anchor`, `proof`,
`selftest` and `explain` arrive with the gates that build what they call (C5, C7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .bench import measure_durability
from .errors import SealError
from .keys import FileKeyProvider, TrustKey, TrustRoot, generate_keypair
from .records import genesis_prev_hash
from .store import DURABILITY_MODES, SealedLedger


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
