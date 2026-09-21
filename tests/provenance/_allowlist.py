"""AST import-allowlist checker (Module C plan §2.3, CI invariant 4) plus invariant 1.

An ALLOWlist, not a denylist: a denylist cannot catch the dependency nobody thought of.

`check_seal_dir` returns a list of human-readable violations (empty = clean). It is a plain function
over a directory so the companion test can point it at a temp copy with a planted forbidden import
and prove the check can actually fail.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SEAL_ALLOWED_TOP = set(sys.stdlib_module_names) | {"cryptography", "rfc8785"}
SEAL_PACKAGE = ("cva", "provenance", "seal")
# Invariant 1 (backend_plan §6.5): provenance/ as a whole must not reach into the ML packages.
INVARIANT1_FORBIDDEN = {("cva", "detectors"), ("cva", "features"), ("cva", "risk")}
DYNAMIC_IMPORT_CALLS = {"__import__", "import_module"}


def _imports(tree: ast.AST):
    """Yield (node, absolute-or-None module dotted name, relative level) for every import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node, a.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node, node.module or "", node.level


def _dynamic_import_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in DYNAMIC_IMPORT_CALLS:
                yield node


def _relative_target(file: Path, root: Path, level: int, module: str) -> tuple[str, ...] | None:
    """Resolve a relative import to an absolute dotted tuple rooted at the seal package.
    Returns None if it climbs out of the seal package (a violation)."""
    pkg_parts = list(file.relative_to(root).parent.parts)          # package path inside seal/
    up = level - 1
    if up > len(pkg_parts):
        return None
    base = pkg_parts[: len(pkg_parts) - up]
    return SEAL_PACKAGE + tuple(base) + tuple(p for p in module.split(".") if p)


def check_seal_dir(root: Path) -> list[str]:
    """Every .py under `root` (the seal/ directory) may import only stdlib, cryptography, rfc8785,
    and the seal package itself. Dynamic imports are forbidden outright: they defeat static checks."""
    bad: list[str] = []
    for f in sorted(root.rglob("*.py")):
        tree = ast.parse(f.read_text(), filename=str(f))
        rel = f.relative_to(root)
        for node, mod, level in _imports(tree):
            if level > 0:
                target = _relative_target(f, root, level, mod)
                if target is None:
                    bad.append(f"{rel}:{node.lineno}: relative import climbs out of seal/")
                continue
            top = mod.split(".")[0]
            if top == "cva":
                if tuple(mod.split("."))[: len(SEAL_PACKAGE)] != SEAL_PACKAGE:
                    bad.append(f"{rel}:{node.lineno}: imports {mod!r} — outside the seal package")
            elif top not in SEAL_ALLOWED_TOP:
                bad.append(f"{rel}:{node.lineno}: imports {mod!r} — not stdlib/cryptography/rfc8785")
        for node in _dynamic_import_calls(tree):
            bad.append(f"{rel}:{node.lineno}: dynamic import — cannot be checked statically")
    return bad


def check_invariant1(provenance_dir: Path) -> list[str]:
    """provenance/ (seal + checks + extras) imports nothing from detectors/, features/, risk/."""
    bad: list[str] = []
    for f in sorted(provenance_dir.rglob("*.py")):
        tree = ast.parse(f.read_text(), filename=str(f))
        for node, mod, level in _imports(tree):
            if level > 0:
                continue
            parts = tuple(mod.split("."))
            if any(parts[: len(p)] == p for p in INVARIANT1_FORBIDDEN):
                bad.append(f"{f.relative_to(provenance_dir)}:{node.lineno}: imports {mod!r}")
    return bad
