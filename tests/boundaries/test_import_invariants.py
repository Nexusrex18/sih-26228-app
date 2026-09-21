"""The four CI import-boundary invariants. Path-based, which is why the repo tree is fixed.

These are the checks ADR-008 wrote to prevent a specific silent failure: if `core/` starts
importing `detectors/`, the plug-in architecture is dead and NOTHING FAILS — the code still
works, so the next detector gets added by editing core. Same for `provenance/`: the field
SDK must stay installable without torch.

backend_plan.md §6.5 fixes all four, and V2 requires each one to be shown to FAIL when
deliberately violated — an invariant that cannot fail is enforcing nothing. The checks are
therefore written as reusable functions (`scan_imports`, `scan_allowlist`) and exercised
against a synthetic violating tree in `test_invariants_fail_when_violated.py`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Invariant 4's allowlist. An ALLOWLIST, not a denylist, per §6.5: invariant 1 stops the
#: seal importing our ML packages but says nothing about it quietly growing a `numpy`
#: import "just for the pHash". A denylist cannot catch the dependency nobody thought of.
SEAL_ALLOWED_THIRD_PARTY = {"cryptography", "rfc8785"}


def module_imports(path: Path) -> set[str]:
    """Every absolute module name imported by one source file."""
    mods: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def _sources(pkg: Path):
    for f in sorted(pkg.rglob("*.py")):
        if "__pycache__" not in f.parts:
            yield f


def scan_imports(pkg: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Return every `file imports module` pair violating the denylist."""
    bad = []
    for f in _sources(pkg):
        for m in sorted(module_imports(f)):
            if any(m == b or m.startswith(b + ".") for b in forbidden):
                bad.append(f"{f} imports {m}")
    return bad


def scan_allowlist(pkg: Path, own_prefix: str, allowed: set[str]) -> list[str]:
    """Return every third-party import outside the allowlist.

    Standard-library modules are permitted implicitly: the budget Mode C is defending is
    the *install* size, and the stdlib costs nothing to install.
    """
    import sys

    stdlib = sys.stdlib_module_names
    bad = []
    for f in _sources(pkg):
        for m in sorted(module_imports(f)):
            top = m.split(".")[0]
            if top in stdlib or top in allowed:
                continue
            if m == own_prefix or m.startswith(own_prefix + "."):
                continue
            bad.append(f"{f} imports {m}")
    return bad


def _require(pkg: str) -> Path:
    p = ROOT / "cva" / pkg
    if not p.exists():
        pytest.skip(f"cva/{pkg} not present")
    return p


# --- invariant 1 -----------------------------------------------------------
def test_invariant_1_provenance_never_imports_the_ml_stack():
    """Mode C: the seal ships as a standalone ~3-dependency SDK. An operator asked to
    install a multi-gigabyte ML stack in order to record one detection will decline — and
    then Module C does not exist in production, however well it works in the demo."""
    bad = scan_imports(_require("provenance"),
                       ("cva.detectors", "cva.features", "cva.risk"))
    assert not bad, "CI invariant 1: provenance/ ↛ detectors/ | features/ | risk/\n  " + \
                    "\n  ".join(bad)


# --- invariant 2 -----------------------------------------------------------
def test_invariant_2_core_never_imports_detectors():
    bad = scan_imports(_require("core"), ("cva.detectors",))
    assert not bad, (
        "CI invariant 2: core/ must never import detectors/. If it does, the plug-in "
        "architecture is already dead and no test fails — registration is pushed from "
        "the plug-in packages, never pulled from core.\n  " + "\n  ".join(bad))


# --- invariant 3 -----------------------------------------------------------
def test_invariant_3_detectors_never_import_remediation():
    bad = scan_imports(_require("detectors"), ("cva.remediation",))
    assert not bad, (
        "CI invariant 3: detectors/ must never import remediation/. This is what makes "
        "ADR-004's absolute 'a baseline check never retrains' constraint a build failure "
        "rather than a code-review habit.\n  " + "\n  ".join(bad))


# --- invariant 4 -----------------------------------------------------------
def test_invariant_4_provenance_seal_is_an_allowlist():
    bad = scan_allowlist(_require("provenance/seal"), "cva.provenance.seal",
                         SEAL_ALLOWED_THIRD_PARTY)
    assert not bad, (
        "CI invariant 4: provenance/seal/ may import only the stdlib, "
        f"{sorted(SEAL_ALLOWED_THIRD_PARTY)} and itself. A denylist cannot catch the "
        "dependency nobody thought of; this allowlist can.\n  " + "\n  ".join(bad))


# --- invariant 5 -----------------------------------------------------------
def test_invariant_5_the_web_process_never_reaches_the_ledger_or_the_key():
    """Module E D-E3, the privilege-separation boundary, made structural.

    `cva-web` is the most exposed process on the host: it parses strings that came out of the
    audited material, from a supplier who is an adversary by premise. It holds no signing key
    and no ledger handle — it ASKS `cva-ledgerd` over a Unix socket and shells out to
    `cva-seal verify`. An `import cva.provenance.seal` anywhere under `cva/web/` would put
    `SealedLedger.append` one attribute lookup away from that parser, and nothing would fail.
    This is what fails.
    """
    bad = scan_imports(_require("web"), ("cva.provenance",))
    assert not bad, (
        "CI invariant 5: cva/web/ must never import cva/provenance/. The web process "
        "requests typed appends through cva-ledgerd, which validates them independently and "
        "owns the key. A compromised web process must be able to spam requests and nothing "
        "more.\n  " + "\n  ".join(bad))


def test_invariant_5b_the_key_holding_daemon_never_imports_the_web_framework():
    """The mirror image. `cva/ledgerd/` imports two PURE modules from `cva.web.workflow`
    (the fold and the enums) so the two processes cannot disagree about what the ledger says.
    That is only acceptable while importing them drags in nothing: the process holding the
    signing key must not acquire Flask's, Werkzeug's and Jinja2's attack surface as a side
    effect of sharing a function."""
    bad = scan_imports(_require("ledgerd"),
                       ("flask", "werkzeug", "jinja2", "waitress", "cva.web.app",
                        "cva.web.security", "cva.web.views", "cva.web.auth"))
    assert not bad, (
        "CI invariant 5b: cva/ledgerd/ may import cva.web.workflow.{fold,events,states} and "
        "nothing else from the web app.\n  " + "\n  ".join(bad))


# --- structural traps ------------------------------------------------------
def test_remediation_is_top_level_not_under_detectors():
    """A remediation package nested inside detectors/ makes invariant 3 pass VACUOUSLY,
    which is worse than failing: ADR-004 would be enforced by nothing while showing green."""
    assert (ROOT / "cva" / "remediation").is_dir()
    assert not list((ROOT / "cva" / "detectors").rglob("remediation"))


def test_detectors_live_where_the_invariants_look():
    """§6.4's blocker: three plans proposed three different trees, and the invariants are
    PATH-BASED. Against a tree where detectors live at services/module_a/detectors/, the
    check passes vacuously while the architecture it protects quietly dies."""
    assert (ROOT / "cva" / "detectors" / "data").is_dir()
    assert (ROOT / "cva" / "detectors" / "model").is_dir()
    assert (ROOT / "cva" / "detectors" / "drift").is_dir()
    assert not list(ROOT.glob("services"))


def test_attacklab_is_at_repo_root_not_inside_the_scanner_package():
    """Under cva/ it ships training-time dependencies inside the air-gapped image."""
    assert (ROOT / "attacklab").is_dir()
    assert not (ROOT / "cva" / "attacklab").exists()


def test_scanner_code_never_imports_the_attack_lab():
    """attacklab/ trains and converts models and generates poisoned data — Mode A, dev machine
    only. If a plug-in under detectors/ or core/ imports it, the air-gapped scanner image drags
    in training-time dependencies and a synthetic-data generator becomes load-bearing."""
    for pkg in ("detectors", "core"):
        bad = scan_imports(_require(pkg), ("attacklab",))
        assert not bad, (
            f"{pkg}/ must never import attacklab/ (the attack lab is not part of the "
            "scanner).\n  " + "\n  ".join(bad))
