"""The reviewed half of the coverage statement (`cva/report/standing.py`, plan §7.8).

What is being protected here is a property, not a file format: **a shorter coverage
statement must never read as a cleaner one.** Every path through the loader — file absent,
file corrupt, file empty — has to end with the report saying which half is missing.
"""
from __future__ import annotations

import pytest

from cva.report import standing

pytest.importorskip("yaml", reason="the reviewed file is YAML")


def test_the_shipped_file_parses_and_carries_its_review_metadata():
    s = standing.load()
    assert s.found, f"{standing.DEFAULT_PATH} is the reviewed file every report prints"
    assert s.invariant, "the Consolidated/13 invariant leads the standing half"
    assert len(s.limitations) >= 10
    assert s.reviewed_at, "an unreviewed standing statement is a fact about the statement"
    assert len(set(s.ids)) == len(s.ids), "ids are how a limitation is referred to; unique"


def test_every_entry_is_one_flat_paragraph():
    """The Jinja view, the React view and `coverage.md` each print a limitation as a
    paragraph. The hard wrapping the file needs to stay reviewable must not survive."""
    s = standing.load()
    for text in (s.invariant, *s.limitations):
        assert "\n" not in text
        assert "  " not in text
        assert text == text.strip()


def test_the_shipped_file_covers_every_module_and_the_cross_cutting_residual():
    s = standing.load()
    joined = " ".join(s.limitations).lower()
    # The four named in plan §7.8 as the minimum content.
    assert "signing key" in joined                  # key custody
    assert "sensor" in joined                       # the sensor-to-SDK gap
    assert "anchoring" in joined or "anchor" in joined
    assert "synthetic contributor assignment" in joined


def test_an_absent_file_is_declared_and_never_silent(tmp_path):
    s = standing.load(tmp_path / "nothing-here.yaml")
    assert s.found is False
    assert s.lines() == [standing.MISSING]


def test_a_corrupt_file_names_the_error_rather_than_printing_nothing(tmp_path):
    p = tmp_path / "coverage-standing.yaml"
    p.write_text("limitations: [this: is, not: valid\n")
    s = standing.load(p)
    assert s.found is False
    line = s.lines()[0]
    assert "could not be read" in line
    assert s.error and s.error.split(":")[0] in line


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    p = tmp_path / "coverage-standing.yaml"
    p.write_text("- just\n- a\n- list\n")
    s = standing.load(p)
    assert s.found is False
    assert "mapping" in s.lines()[0]


def test_an_unreviewed_file_says_nobody_signed_it(tmp_path):
    p = tmp_path / "coverage-standing.yaml"
    p.write_text('limitations:\n  - id: x\n    text: "A residual nobody has signed."\n')
    s = standing.load(p)
    assert s.found is True
    assert s.limitations == ("A residual nobody has signed.",)
    review = s.lines()[-1]
    assert "nobody recorded" in review and "no date recorded" in review


def test_an_entry_with_no_text_is_dropped_rather_than_printed_empty(tmp_path):
    p = tmp_path / "coverage-standing.yaml"
    p.write_text('limitations:\n  - id: empty\n    text: ""\n  - id: real\n    text: "Kept."\n')
    s = standing.load(p)
    assert s.limitations == ("Kept.",)


def test_the_env_var_overrides_the_default_path(tmp_path, monkeypatch):
    p = tmp_path / "elsewhere.yaml"
    p.write_text('reviewed_at: "2026-01-01"\nlimitations: []\n')
    monkeypatch.setenv(standing.ENV_VAR, str(p))
    assert standing.resolve_path() == p
    assert standing.load().reviewed_at == "2026-01-01"
