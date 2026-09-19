"""The CI import-boundary invariants. Path-based, which is why the repo tree is fixed.

These are the checks ADR-008 wrote to prevent a specific silent failure: if `core/` starts
importing `detectors/`, the plug-in architecture is dead and NOTHING FAILS — the code still
works, so the next detector gets added by editing core. Same for `provenance/`: the field
SDK must stay installable without torch.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _imports(pkg: Path) -> dict[Path, set[str]]:
    out = {}
    for f in pkg.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        mods = set()
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module)
        out[f] = mods
    return out


def _assert_no_import(pkg: str, forbidden: tuple[str, ...], why: str):
    p = ROOT / "cva" / pkg
    if not p.exists():
        pytest.skip(f"{pkg}/ not present")
    bad = []
    for f, mods in _imports(p).items():
        for m in mods:
            if any(m == b or m.startswith(b + ".") for b in forbidden):
                bad.append(f"{f.relative_to(ROOT)} imports {m}")
    assert not bad, why + "\n  " + "\n  ".join(bad)


def test_invariant_2_core_never_imports_detectors():
    _assert_no_import(
        "core", ("cva.detectors",),
        "CI invariant 2: core/ must never import detectors/. If it does, the plug-in "
        "architecture is already dead and no test fails — registration is pushed from "
        "the plug-in packages, never pulled from core.")


def test_invariant_3_detectors_never_import_remediation():
    _assert_no_import(
        "detectors", ("cva.remediation",),
        "CI invariant 3: detectors/ must never import remediation/. This is what keeps "
        "ADR-004's absolute 'a baseline check never retrains' constraint enforced.")


def test_invariant_4_provenance_seal_is_standalone():
    p = ROOT / "cva" / "provenance"
    if not p.exists():
        pytest.skip("provenance/ is Crypto's, not built here")
    _assert_no_import(
        "provenance/seal", ("torch", "onnx", "onnxruntime", "cva.detectors",
                            "cva.features", "cva.risk"),
        "CI invariant 4: the field SDK must install without the ML stack.")


def test_remediation_is_top_level_not_under_detectors():
    """A remediation package nested inside detectors/ makes invariant 3 pass VACUOUSLY,
    which is worse than failing: ADR-004 would be enforced by nothing while showing green."""
    assert (ROOT / "cva" / "remediation").is_dir()
    assert not list((ROOT / "cva" / "detectors").rglob("remediation"))


def test_attacklab_is_at_repo_root_not_inside_the_scanner_package():
    """Under cva/ it ships training-time dependencies inside the air-gapped image."""
    assert (ROOT / "attacklab").is_dir()
    assert not (ROOT / "cva" / "attacklab").exists()
