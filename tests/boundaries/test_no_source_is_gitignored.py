"""No source file may be matched by `.gitignore`.

An unanchored `reports/` rule, meant for scan output, also matched the package
`cva/web/reports/` — the report loader, the index and the seal check. They were never
committed. Every test passed on the machine that had them, and every clone failed to
import `cva.web`. A test run from a working copy cannot see that on its own, so this asks
git directly.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_TREES = ("cva", "tests", "scripts", "attacklab", "schemas", "spec", "docs", "demo",
                "docker", "profiles", "frontend/src")
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".css", ".html", ".json", ".yaml", ".yml", ".md",
                   ".sh", ".mk", ".txt", ".toml", ".j2", ".svg"}


def test_no_source_file_is_ignored_by_git():
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        [git, "-C", str(ROOT), "ls-files", "--others", "--ignored", "--exclude-standard",
         "--", *SOURCE_TREES],
        capture_output=True, text=True, check=True).stdout.splitlines()
    ignored = [p for p in out
               if "__pycache__" not in p and Path(p).suffix in SOURCE_SUFFIXES]
    assert not ignored, (
        "these source files are matched by .gitignore and would never be committed:\n  "
        + "\n  ".join(ignored))
