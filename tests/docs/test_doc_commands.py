"""Every command in the docs is either executed here or checked against the real CLI.

Plan test 10.10: a setup document whose commands have drifted from the code is worse than
none, because it is followed exactly at the moment the operator has least context. Two
layers:

  * **static**: every `cva …`, `cva-seal …`, `cva-ledgerd …` and accounts command in a
    fenced shell block names a subcommand and flags that exist in the parser that will run
    it;
  * **executed**: the host path of SETUP.md — keygen, init, accounts, export, verify — is
    run for real, in order, in a temporary directory.

The docker commands are covered by `tests/boundaries/test_image_files.py` (consistency)
rather than executed: building the image needs a registry pull.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "docs" / n for n in ("SETUP.md", "VERIFICATION-PROCEDURE.md",
                                    "operator-manual.md", "THREAT-MODEL.md")]
DOCS += [ROOT / "docker" / "hardening.md", ROOT / "demo" / "script.md"]

#: program -> the source file whose argparse defines it.
SOURCES = {
    "cva": ROOT / "cva" / "cli.py",
    "cva-seal": ROOT / "cva" / "provenance" / "seal" / "cli.py",
    "cva-ledgerd": ROOT / "cva" / "ledgerd" / "server.py",
    "cva-web": ROOT / "cva" / "web" / "app.py",
    "accounts": ROOT / "cva" / "web" / "accounts.py",
}


def _shell_commands() -> list[tuple[str, str]]:
    """`(doc name, one logical command)` from every ```sh block, continuations joined."""
    out = []
    for doc in DOCS:
        if not doc.exists():
            continue
        for block in re.findall(r"```sh\n(.*?)```", doc.read_text(), re.S):
            joined = re.sub(r"\\\n\s*", " ", block)
            for line in joined.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append((doc.name, line))
    return out


def _program_and_args(line: str) -> tuple[str, list[str]] | None:
    try:
        words = shlex.split(line)
    except ValueError:
        return None
    while words and re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", words[0]):
        words = words[1:]                                  # leading VAR=value assignments
    if not words:
        return None
    if words[:3] == ["python", "-m", "cva.web.accounts"]:
        return "accounts", words[3:]
    if words[0] in SOURCES:
        return words[0], words[1:]
    return None


def _defined(program: str) -> tuple[set[str], set[str]]:
    src = SOURCES[program].read_text()
    subs = set(re.findall(r"add_parser\(\s*\"([a-z][a-z-]*)\"", src))
    flags = set(re.findall(r"\"(--[a-z][a-z0-9-]*)\"", src))
    return subs, flags


DOC_COMMANDS = [(doc, line, parsed) for doc, line in _shell_commands()
                if (parsed := _program_and_args(line)) is not None]


def test_the_docs_contain_commands_to_check():
    assert len(DOC_COMMANDS) >= 10, "the parser found almost nothing; the check is not running"


@pytest.mark.parametrize("doc,line,parsed", DOC_COMMANDS,
                         ids=[f"{d}:{p[0]}:{i}" for i, (d, _l, p) in enumerate(DOC_COMMANDS)])
def test_every_documented_command_uses_a_real_subcommand_and_real_flags(doc, line, parsed):
    program, args = parsed
    subs, flags = _defined(program)
    positional = [a for a in args if not a.startswith("-")]
    if subs and positional:
        assert positional[0] in subs, f"{doc}: `{positional[0]}` is not a {program} subcommand"
    for a in args:
        if a.startswith("--"):
            flag = a.split("=", 1)[0]
            assert flag in flags, f"{doc}: {program} has no flag {flag} — `{line}`"


def test_licenses_md_matches_what_ships():
    """LICENSES.md is generated, never typed. A wheel added without regenerating it would
    ship a dependency the licence file does not mention."""
    if not (ROOT / "wheelhouse").is_dir():
        pytest.skip("no wheelhouse/ in this checkout; LICENSES.md is generated from it")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_licenses.py"),
                        "--check"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


# --- executed: SETUP.md's host path, for real -----------------------------------------------

pytest.importorskip("cva.provenance.seal", reason="the seal SDK needs cryptography")


def _run(*argv: str, cwd: Path, stdin: str = "") -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(cwd),
           "CVA_WEB_ACCOUNTS_DB": str(cwd / "index" / "accounts.db")}
    return subprocess.run([sys.executable, *argv], cwd=cwd, env=env, input=stdin,
                          capture_output=True, text=True, timeout=300)


def test_setup_md_host_path_runs_end_to_end(tmp_path: Path):
    seal = ("-m", "cva.provenance.seal.cli")
    (tmp_path / "ledger").mkdir()
    (tmp_path / "keys").mkdir()

    # Step 4: the key, explicitly.
    r = _run(*seal, "keygen", "--out", "keys/signing.key", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "keys" / "signing.key").stat().st_mode & 0o077 == 0, "key is not 0600"

    # Step 5: the ledger and trust root.
    r = _run(*seal, "init", "--ledger", "ledger/audit.db", "--key", "keys/signing.key",
             "--device-id", "analyst-host-01", "--unit", "acceptance-cell",
             "--trust-out", "ledger/trust_root.json", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "ledger" / "trust_root.json").is_file()

    # Step 7: accounts, with the password on stdin, never in argv.
    for actor, role in (("root.admin", "admin"), ("a.sharma", "analyst"),
                        ("b.rao", "approver")):
        r = _run("-m", "cva.web.accounts", "create", actor, "--role", role,
                 "--password-stdin", cwd=tmp_path, stdin="correct-horse-battery\n")
        assert r.returncode == 0, r.stderr
    r = _run("-m", "cva.web.accounts", "list", cwd=tmp_path)
    assert "b.rao\tapprover\tactive" in r.stdout, r.stdout

    # Step 11: export, then verify the export — the artefact a third party checks.
    r = _run(*seal, "export", "--ledger", "ledger/audit.db",
             "--out", "ledger/audit.export.jsonl", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    r = _run(*seal, "verify", "--ledger", "ledger/audit.export.jsonl",
             "--trust-root", "ledger/trust_root.json", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run(*seal, "verify", "--ledger", "ledger/audit.export.jsonl",
             "--trust-root", "ledger/trust_root.json", "--json", cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.lstrip().startswith("{"), r.stdout[:200]


def test_verification_procedure_exit_codes_are_the_ones_it_documents(tmp_path: Path):
    """`2` means findings, `1` means could not read. Editing one byte must give 2."""
    seal = ("-m", "cva.provenance.seal.cli")
    (tmp_path / "ledger").mkdir()
    _run(*seal, "keygen", "--out", "k", cwd=tmp_path)
    _run(*seal, "init", "--ledger", "ledger/a.db", "--key", "k", "--device-id", "d",
         "--unit", "u", "--trust-out", "ledger/t.json", cwd=tmp_path)
    _run(*seal, "export", "--ledger", "ledger/a.db", "--out", "e.jsonl", cwd=tmp_path)

    unreadable = _run(*seal, "verify", "--ledger", "missing.jsonl",
                      "--trust-root", "ledger/t.json", cwd=tmp_path)
    assert unreadable.returncode == 1

    export = tmp_path / "e.jsonl"
    text = export.read_text()
    edited = text.replace('"unit":"u"', '"unit":"v"', 1)
    assert edited != text, "the fixture edit did not apply; the test would prove nothing"
    export.write_text(edited)
    tampered = _run(*seal, "verify", "--ledger", "e.jsonl", "--trust-root", "ledger/t.json",
                    cwd=tmp_path)
    assert tampered.returncode == 2, tampered.stdout + tampered.stderr
