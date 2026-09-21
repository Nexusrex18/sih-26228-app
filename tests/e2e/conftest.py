"""The whole system, as separate processes, set up the way SETUP.md says.

Nothing here is in-process and nothing is mocked:

  * the key and the ledger are made by `cva-seal keygen` / `init`, as an operator would;
  * `cva-ledgerd` runs as its own process, holding the key, on a real Unix socket;
  * the fixture reports are sealed THROUGH that socket, the way `cva scan
    --audit-ledger-socket` seals a scan — the scanner never touches the key;
  * accounts are made by `python -m cva.web.accounts`, password on stdin;
  * `cva-web` runs as its own process under waitress, configured only by `CVA_WEB_*`
    environment variables, on a real TCP port, serving the real `frontend/out` export.

Tests then drive it over HTTP with `browser.Browser`.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "correct-horse-battery"
ACCOUNTS = (("a.sharma", "analyst"), ("b.rao", "approver"), ("v.iyer", "viewer"),
            ("root.admin", "admin"))

pytest.importorskip("cva.provenance.seal", reason="the seal SDK needs cryptography")
pytest.importorskip("waitress", reason="cva-web's server")


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CVA_")}
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def _run(*argv: str, cwd: Path, stdin: str = "", env: dict[str, str] | None = None,
         timeout: int = 600) -> subprocess.CompletedProcess[str]:
    r = subprocess.run([sys.executable, *argv], cwd=cwd, input=stdin, text=True,
                       capture_output=True, timeout=timeout, env=env or _env())
    assert r.returncode == 0, f"{argv}: exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    return r


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for(predicate, what: str, proc: subprocess.Popen[bytes], timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise AssertionError(f"{what}: the process exited ({proc.returncode})\n{out}")
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"{what}: not ready after {timeout:.0f}s")


def seal_through_socket(sock: Path, report: Path) -> int:
    """Append a scan_record for `report` through ledgerd, the way a scan does."""
    from cva.web.workflow.ledgerd_client import LedgerdAuditLedger

    data = json.loads(report.read_text())
    counts: dict[str, int] = {}
    for f in data.get("findings", []):
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    return int(LedgerdAuditLedger(sock).append({"type": "scan_record", "scan": {
        "scan_id": report.parent.name,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "profile_hash": data["produced_by"]["profile_hash"],
        "code_commit": str(data["produced_by"]["code_commit"]),
        "finding_counts": counts,
    }}))


@dataclass
class Stack:
    base: str
    root: Path
    reports: Path
    ledger: Path
    trust_root: Path
    socket: Path
    web_log: Path
    ledgerd_log: Path


@pytest.fixture(scope="module")
def stack(tmp_path_factory) -> Stack:
    if not (ROOT / "frontend" / "out" / "index.html").is_file():
        pytest.skip("frontend/out is not built: npm --prefix frontend run build")

    root = tmp_path_factory.mktemp("e2e")
    (root / "ledger").mkdir()
    (root / "keys").mkdir()
    (root / "index").mkdir()
    reports = root / "reports"
    seal_cli = ("-m", "cva.provenance.seal.cli")

    # SETUP.md steps 4-5: the key, explicitly; the ledger and its trust root.
    _run(*seal_cli, "keygen", "--out", "keys/signing.key", cwd=root)
    _run(*seal_cli, "init", "--ledger", "ledger/audit.db", "--key", "keys/signing.key",
         "--device-id", "e2e-host", "--unit", "e2e", "--trust-out",
         "ledger/trust_root.json", cwd=root)

    from cva.web import fixtures
    fixtures.write(reports)

    # SETUP.md step 6: ledgerd, its own process, holding the key.
    sock = root / "run" / "ledgerd.sock"
    sock.parent.mkdir()
    ledgerd_log = root / "ledgerd.log"
    ledgerd = subprocess.Popen(
        [sys.executable, "-m", "cva.ledgerd.server", "--socket", str(sock),
         "--ledger", str(root / "ledger" / "audit.db"),
         "--key", str(root / "keys" / "signing.key")],
        cwd=root, env=_env(), stdout=ledgerd_log.open("wb"), stderr=subprocess.STDOUT)
    web = None
    try:
        _wait_for(sock.exists, "cva-ledgerd socket", ledgerd)

        # Seal every fixture report through the socket (the O2 path), so the dashboard
        # has sealed scans to show. The two unreadable fixtures are left alone.
        for scan_dir in sorted(reports.iterdir()):
            report = scan_dir / "report.json"
            if report.is_file():
                try:
                    json.loads(report.read_text())
                except ValueError:
                    continue
                seal_through_socket(sock, report)

        # SETUP.md step 7: accounts, password on stdin.
        for actor, role in ACCOUNTS:
            _run("-m", "cva.web.accounts", "--db", str(root / "index" / "accounts.db"),
                 "create", actor, "--role", role, "--password-stdin", cwd=root,
                 stdin=PASSWORD + "\n")

        # cva-web, its own process, configured only through the environment.
        port = _free_port()
        web_log = root / "web.log"
        web = subprocess.Popen(
            [sys.executable, "-m", "cva.web.app"], cwd=root, stdout=web_log.open("wb"),
            stderr=subprocess.STDOUT,
            env=_env({"CVA_WEB_BIND_PORT": str(port),
                      "CVA_WEB_REPORTS_DIR": str(reports),
                      "CVA_WEB_INDEX_DB": str(root / "index" / "index.db"),
                      "CVA_WEB_ACCOUNTS_DB": str(root / "index" / "accounts.db"),
                      "CVA_WEB_LEDGERD_SOCKET": str(sock),
                      "CVA_WEB_LEDGER_PATH": str(root / "ledger" / "audit.db"),
                      "CVA_WEB_TRUST_ROOT": str(root / "ledger" / "trust_root.json"),
                      "CVA_WEB_RATE_LIMIT_PER_MINUTE": "100000"}))
        base = f"http://127.0.0.1:{port}"

        def up() -> bool:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return True
            except OSError:
                return False
        _wait_for(up, "cva-web port", web)

        yield Stack(base=base, root=root, reports=reports,
                    ledger=root / "ledger" / "audit.db",
                    trust_root=root / "ledger" / "trust_root.json", socket=sock,
                    web_log=web_log, ledgerd_log=ledgerd_log)
    finally:
        for proc in (web, ledgerd):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


@pytest.fixture
def browser_for(stack):
    """A fresh browser per actor, each with its own cookie jar."""
    from .browser import Browser

    def make(actor: str | None = None) -> Browser:
        b = Browser(stack.base)
        if actor:
            r = b.sign_in(actor, PASSWORD)
            assert r.status in (302, 303), (
                f"sign-in as {actor} returned {r.status}: {r.text[:400]}")
        return b
    return make
