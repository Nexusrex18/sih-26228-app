#!/usr/bin/env python3
"""Bring the whole dashboard up locally: fixtures, a signed ledger, ledgerd and cva-web.

One command for the demo and for looking at the thing while building it. Everything lands
in a scratch directory that can be deleted.

    python scripts/dev_dashboard.py --out .scratch/dev

Then open http://127.0.0.1:8713/app/ and sign in as `Tan.00` / `12345678`.
`b.rao` (same password) is the approver: lowering a disposition needs a second person.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO_PASSWORD = "12345678"
ACCOUNTS = [
    ("Tan.00", "analyst"),
    ("b.rao", "approver"),
    ("v.iyer", "viewer"),
    ("root.admin", "admin"),
]


def build(out: Path, fresh: bool) -> tuple[Path, Path, Path, Path]:
    if fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    from cva.provenance.seal import FileKeyProvider, SealedLedger, TrustRoot, generate_keypair
    from cva.provenance.seal.keys import TrustKey
    from cva.provenance.seal.records import genesis_prev_hash
    from cva.web import fixtures

    reports = out / "reports"
    if not reports.exists():
        fixtures.write(reports)
        print(f"  fixtures   {reports}")

    key_path = out / "ledger.key"
    ledger_path = out / "audit.db"
    trust_path = out / "trust_root.json"
    if not key_path.exists():
        generate_keypair(key_path)
    key = FileKeyProvider(key_path)
    if not ledger_path.exists():
        led = SealedLedger.init_ledger(
            ledger_path, key,
            {"device_id": "dev-box", "unit": "demo", "profile_hash": "0" * 64,
             "checkpoint_every": 1000})
        trust = TrustRoot(genesis_prev_hash(led.deployment_manifest),
                          (TrustKey(key.key_id, key.public_key, "ledger"),))
        led.close()
        trust_path.write_bytes(trust.to_bytes() + b"\n")
        print(f"  ledger     {ledger_path}  (key {key.key_id[:16]}…)")
    return reports, key_path, ledger_path, trust_path


def seal_the_fixtures(ledger_path: Path, key_path: Path, reports: Path) -> None:
    """Write a `scan_record` for each fixture so the sealed badge has something to show."""
    import hashlib
    import json

    from cva.provenance.seal import FileKeyProvider, SealedLedger

    led = SealedLedger.open(ledger_path, key=FileKeyProvider(key_path))
    try:
        already = {
            (r.get("scan") or {}).get("scan_id")
            for r in led.records()
            if r.get("type") == "scan_record"
        }
        for scan_dir in sorted(reports.iterdir()):
            report = scan_dir / "report.json"
            if not report.is_file() or scan_dir.name in already:
                continue
            data = json.loads(report.read_text())
            counts: dict[str, int] = {}
            for f in data.get("findings", []):
                counts[f["severity"]] = counts.get(f["severity"], 0) + 1
            led.append_typed("scan_record", {"scan": {
                "scan_id": scan_dir.name,
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "profile_hash": data["produced_by"]["profile_hash"],
                "code_commit": data["produced_by"]["code_commit"],
                "finding_counts": counts,
            }})
            print(f"  sealed     {scan_dir.name}")
    finally:
        led.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".scratch/dev")
    ap.add_argument("--port", type=int, default=8713)
    ap.add_argument("--fresh", action="store_true", help="delete and rebuild everything")
    ap.add_argument("--no-seal", action="store_true",
                    help="leave the fixtures unsealed, to see the 'not sealed' badge")
    a = ap.parse_args(argv)

    out = (ROOT / a.out).resolve()
    print(f"building into {out}")
    reports, key_path, ledger_path, trust_path = build(out, a.fresh)
    if not a.no_seal:
        seal_the_fixtures(ledger_path, key_path, reports)

    from cva.ledgerd.policy import development_policy
    from cva.ledgerd.server import serve
    from cva.web.app import create_app
    from cva.web.config import WebConfig

    sock = out / "ledgerd.sock"
    server = serve(sock, ledger_path, key_path, development_policy())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"  ledgerd    {sock}")

    cfg = WebConfig(reports_dir=reports, index_db=out / "index.db",
                    accounts_db=out / "accounts.db", ledgerd_socket=sock,
                    trust_root=trust_path, ledger_path=ledger_path,
                    bind_port=a.port)
    app = create_app(cfg)

    store = app.extensions["cva_accounts"]
    for actor, role in ACCOUNTS:
        if store.get(actor) is None:
            store.create(actor, DEMO_PASSWORD, role)
            print(f"  account    {actor} ({role})")

    spa = ROOT / "frontend" / "out"
    print()
    print(f"  dashboard  http://127.0.0.1:{a.port}/app/"
          + ("" if spa.is_dir() else "   [NOT BUILT — run `npm --prefix frontend run build`]"))
    print(f"  sign in    Tan.00 / {DEMO_PASSWORD}   (b.rao is the approver)")
    print()

    from waitress import serve as wserve
    try:
        wserve(app, host="127.0.0.1", port=a.port, threads=8)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
