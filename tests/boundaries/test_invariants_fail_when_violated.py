"""V2 — each invariant must ACTUALLY FAIL when deliberately violated.

backend_plan.md §11 V2: *"an invariant that cannot fail is not enforcing anything."* The
failure mode this guards is real and quiet: a check written against the wrong path, or a
denylist whose prefixes never match, shows green forever while the architecture it claims
to protect dies. So the checks are run here against a synthetic tree that violates each
one, and the assertion is that they report the violation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.boundaries.test_import_invariants import (
    SEAL_ALLOWED_THIRD_PARTY,
    scan_allowlist,
    scan_imports,
)


@pytest.fixture()
def violating_tree(tmp_path: Path) -> Path:
    """A miniature cva/ whose every package breaks its invariant."""
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "orchestrator.py").write_text(
        "from cva.detectors.model.registry import REGISTRY\n")

    (tmp_path / "provenance").mkdir()
    (tmp_path / "provenance" / "checks.py").write_text("import cva.features.embeddings\n")

    (tmp_path / "provenance" / "seal").mkdir()
    (tmp_path / "provenance" / "seal" / "record.py").write_text(
        "import numpy as np\nimport hashlib\n")

    (tmp_path / "detectors").mkdir()
    (tmp_path / "detectors" / "greedy.py").write_text(
        "from cva.remediation.data_clean import clean\n")
    return tmp_path


def test_invariant_1_fails_when_provenance_imports_features(violating_tree: Path):
    bad = scan_imports(violating_tree / "provenance",
                       ("cva.detectors", "cva.features", "cva.risk"))
    assert any("cva.features.embeddings" in b for b in bad)


def test_invariant_2_fails_when_core_imports_detectors(violating_tree: Path):
    bad = scan_imports(violating_tree / "core", ("cva.detectors",))
    assert any("cva.detectors.model.registry" in b for b in bad)


def test_invariant_3_fails_when_a_detector_imports_remediation(violating_tree: Path):
    bad = scan_imports(violating_tree / "detectors", ("cva.remediation",))
    assert any("cva.remediation.data_clean" in b for b in bad)


def test_invariant_4_fails_on_a_dependency_no_denylist_would_have_named(
        violating_tree: Path):
    """The allowlist's whole point: `numpy` is not on anyone's denylist, and it is exactly
    the import that would arrive "just for the pHash" and blow Mode C's budget."""
    bad = scan_allowlist(violating_tree / "provenance" / "seal",
                         "cva.provenance.seal", SEAL_ALLOWED_THIRD_PARTY)
    assert any("numpy" in b for b in bad)
    # ...and the stdlib import in the same file is NOT reported: the budget defended is
    # install size, and hashlib costs nothing to install.
    assert not any("hashlib" in b for b in bad)


def test_the_allowlist_admits_what_mode_c_actually_needs(tmp_path: Path):
    (tmp_path / "seal").mkdir()
    (tmp_path / "seal" / "sign.py").write_text(
        "import sqlite3\nimport rfc8785\nfrom cryptography.hazmat.primitives import "
        "serialization\nfrom cva.provenance.seal.record import Record\n")
    assert not scan_allowlist(tmp_path / "seal", "cva.provenance.seal",
                              SEAL_ALLOWED_THIRD_PARTY)
