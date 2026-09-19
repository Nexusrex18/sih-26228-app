"""Invariant 4 (Module C plan §2.3) and invariant 1 (backend_plan §6.5).

The allowlist test is only worth having if it can FAIL. The companion tests plant each kind of
forbidden import in a temp copy of seal/ and assert the checker catches it — same discipline as
backend_plan V2. If a planted violation ever passes, the invariant is decorative.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ._allowlist import check_invariant1, check_seal_dir

PROV = Path(__file__).resolve().parents[2] / "cva" / "provenance"
SEAL = PROV / "seal"


def test_seal_package_passes_the_allowlist():
    assert check_seal_dir(SEAL) == []


def test_provenance_passes_invariant_1():
    assert check_invariant1(PROV) == []


def test_seal_has_files_so_the_allowlist_is_not_vacuous():
    assert list(SEAL.rglob("*.py")), "seal/ is empty — the allowlist would pass trivially"


# --- the checker must fail when violated ---------------------------------------------------

PLANTS = {
    "third-party (numpy)": "import numpy\n",
    "third-party from-import (PIL)": "from PIL import Image\n",
    "ML stack (torch)": "import torch\n",
    "onnxruntime": "import onnxruntime as ort\n",
    "core/ (invariant 1 spirit)": "from cva.core.finding import Finding\n",
    "detectors/": "import cva.detectors.data\n",
    "relative import out of seal/": "from .. import checks\n",
    "import hidden in a function": "def f():\n    import numpy\n",
    "import under TYPE_CHECKING": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy\n",
    "dynamic __import__": "m = __import__('numpy')\n",
    "dynamic importlib": "import importlib\nm = importlib.import_module('numpy')\n",
}


@pytest.mark.parametrize("label,source", list(PLANTS.items()), ids=list(PLANTS))
def test_allowlist_fails_when_a_forbidden_import_is_planted(tmp_path, label, source):
    copy = tmp_path / "seal"
    shutil.copytree(SEAL, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "planted.py").write_text(source)
    assert check_seal_dir(copy), f"allowlist did NOT catch: {label}"


@pytest.mark.parametrize("source", [
    "import hashlib, json, sqlite3, os, struct\n",
    "from cryptography.hazmat.primitives.asymmetric import ed25519\n",
    "import rfc8785\n",
    "from cva.provenance.seal import errors\n",          # itself, absolute
    "from . import errors\n",                            # itself, relative
])
def test_allowlist_accepts_the_permitted_imports(tmp_path, source):
    copy = tmp_path / "seal"
    shutil.copytree(SEAL, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "ok.py").write_text(source)
    assert check_seal_dir(copy) == []


def test_invariant_1_fails_when_provenance_imports_the_ml_packages(tmp_path):
    copy = tmp_path / "provenance"
    shutil.copytree(PROV, copy, ignore=shutil.ignore_patterns("__pycache__"))
    for i, bad in enumerate(["import cva.detectors.data", "from cva.features import cache",
                             "from cva.risk import rollup"]):
        (copy / "checks" / f"planted{i}.py").write_text(bad + "\n")
    assert len(check_invariant1(copy)) == 3
