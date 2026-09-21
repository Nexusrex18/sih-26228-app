"""The bridge to `cva-seal verify` (plan §7.4, S11).

A FIXED argv list, `shell=False`, validated path arguments, a timeout, and captured output
that is displayed escaped and never interpreted. Nothing here is built by string formatting,
and no part of a request ever reaches this function.

The web process verifies this way rather than by importing the verifier because it must not
hold a ledger handle at all (D-E3, CI invariant 5). A subprocess boundary is also what an
auditor can reproduce by hand: the command in `docs/VERIFICATION-PROCEDURE.md` is the one
this runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: exit 0 clean, 2 findings, 1 could not verify at all (`cva-seal`'s own contract).
EXIT_CLEAN = 0
EXIT_UNREADABLE = 1
EXIT_FINDINGS = 2


@dataclass(frozen=True)
class VerifyState:
    state: str                      # OK | FAILED | UNAVAILABLE
    detail: str
    checked_at: float = 0.0
    records_checked: int = 0
    anchors_verified: int = 0
    anchors_in_chain: int = 0
    unwitnessed_records: int = 0
    declared_gaps: int = 0
    durability: str = ""
    loss_window: str = ""
    worst: str = "info"
    findings: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    command: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.state == "OK"

    @property
    def summary(self) -> str:
        if self.state == "UNAVAILABLE":
            return self.detail
        if self.ok:
            return f"{self.records_checked} records verified, no finding above info"
        classes = sorted({str(f.get("attack_class")) for f in self.findings})
        return f"{len(self.findings)} finding(s): {', '.join(classes)}"

    @property
    def checked_at_label(self) -> str:
        if not self.checked_at:
            return "never"
        # Host clock, untrusted, and labelled as such wherever it is shown (§8).
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.checked_at)) + " UTC"

    @property
    def command_line(self) -> str:
        """Shown to the analyst so they can run it themselves. Display only."""
        return " ".join(self.command)


def _executable() -> list[str]:
    """Prefer the installed console script; fall back to the module under this interpreter.

    Both forms are a fixed argv. `shutil.which` is used rather than a shell lookup so that a
    PATH entry containing a space or a shell metacharacter cannot change what runs.
    """
    found = shutil.which("cva-seal")
    if found:
        return [found]
    return [sys.executable, "-m", "cva.provenance.seal.cli"]


def _env() -> dict[str, str]:
    """The child's environment, with THIS `cva` package importable whatever the cwd.

    The `-m` fallback resolves `cva` from `sys.path`, which for a child process starts at its
    working directory. Run from anywhere but the repo root (a systemd unit, a container
    entrypoint, a test runner started one directory up) and the verifier failed with "No
    module named 'cva'" — the dashboard then reported verification UNAVAILABLE for a reason
    that had nothing to do with the ledger. The directory is derived from this file, never
    from the request, so the argv stays fixed.
    """
    root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else os.pathsep.join((root, existing))
    return env


def _unavailable(detail: str) -> VerifyState:
    return VerifyState(state="UNAVAILABLE", detail=detail, checked_at=time.time())


def verify(ledger_path: Path | None, trust_root: Path | None, *,
           timeout_s: int = 300) -> VerifyState:
    """Run the verifier. Never raises: an unavailable verification is a STATE, not an error.

    `UNAVAILABLE` and `FAILED` are kept apart everywhere, because they mean opposite things
    to an analyst: one says "we did not check", the other says "we checked and it is wrong".
    """
    if ledger_path is None or trust_root is None:
        missing = [n for n, v in (("ledger_path", ledger_path),
                                  ("trust_root", trust_root)) if v is None]
        return _unavailable(
            f"verification is not configured: {', '.join(missing)} is not set. Until it is, "
            "the dashboard cannot say whether the recorded decisions are the ones that were "
            "signed, and it will not imply that it can.")
    ledger_path, trust_root = Path(ledger_path), Path(trust_root)
    for name, p in (("ledger", ledger_path), ("trust root", trust_root)):
        if not p.is_file():
            return _unavailable(f"the {name} file {p} does not exist")

    argv = [*_executable(), "verify", "--ledger", str(ledger_path),
            "--trust-root", str(trust_root), "--json"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              shell=False, check=False, env=_env())
    except subprocess.TimeoutExpired:
        return _unavailable(f"cva-seal verify did not finish within {timeout_s} s")
    except OSError as e:
        return _unavailable(f"cva-seal verify could not be started: {e}")

    if proc.returncode == EXIT_UNREADABLE:
        return VerifyState(state="UNAVAILABLE",
                           detail=(proc.stderr or "the ledger could not be read").strip()[:500],
                           checked_at=time.time(), command=tuple(argv))
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _unavailable(
            "cva-seal verify returned output this dashboard could not parse. The raw output "
            f"begins: {proc.stdout[:200]!r}")
    if not isinstance(payload, dict):
        return _unavailable("cva-seal verify returned a non-object result")

    findings = tuple(payload.get("findings") or ())
    clean = bool(payload.get("ok")) and proc.returncode == EXIT_CLEAN
    return VerifyState(
        state="OK" if clean else "FAILED",
        detail="" if clean else f"{len(findings)} finding(s) above info",
        checked_at=time.time(),
        records_checked=int(payload.get("records_checked") or 0),
        anchors_verified=int(payload.get("anchors_verified") or 0),
        anchors_in_chain=int(payload.get("anchors_in_chain") or 0),
        unwitnessed_records=int(payload.get("unwitnessed_records") or 0),
        declared_gaps=int(payload.get("declared_gaps") or 0),
        durability=str(payload.get("durability") or ""),
        loss_window=str(payload.get("loss_window") or ""),
        worst=str(payload.get("worst") or "info"),
        findings=findings,
        limitations=tuple(payload.get("limitations") or ()),
        command=tuple(argv))


def export(ledger_path: Path, out_path: Path, *, timeout_s: int = 300) -> tuple[bool, str]:
    """`cva-seal export` — the artefact a third party verifies without our database."""
    argv = [*_executable(), "export", "--ledger", str(Path(ledger_path)),
            "--out", str(Path(out_path))]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              shell=False, check=False, env=_env())
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)
    return proc.returncode == 0, (proc.stdout or proc.stderr).strip()[:1000]


__all__ = ["VerifyState", "export", "verify"]
