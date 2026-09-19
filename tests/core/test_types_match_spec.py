"""V4 — `core/types.py` and `Architecture/Plugin-Interfaces.md` agree field-for-field.

This is the test that makes the freeze real. §5.11 puts the freeze in force from the
`core/types.py` commit, and R1 is that the freeze slips — types changing after four seats
have built against them. Keeping code and spec in lockstep by TEST rather than by
discipline is what stops the spec quietly becoming fiction while every other test passes.

The vault lives outside the repo, so the test SKIPS when it is absent rather than failing:
a teammate cloning only the app repo has not broken anything. It fails loudly when the
vault IS present and disagrees, which is the case that matters.
"""
from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from cva.core.capability import Availability, Capability
from cva.core.types import (
    ContributorSource,
    Disposition,
    Evidence,
    ExclusionReason,
    Finding,
    Nature,
    Sample,
    Severity,
)

VAULT = Path(__file__).resolve().parents[2].parent / "sih26228-notes"
SPEC = VAULT / "Architecture" / "Plugin-Interfaces.md"


def _spec_text() -> str:
    if not SPEC.exists():
        pytest.skip(f"vault not present at {SPEC}; nothing to compare against")
    return SPEC.read_text()


def _block(text: str, header: str) -> str:
    """The fenced block immediately following a bolded type name."""
    i = text.index(header)
    start = text.index("```", i) + 3
    return text[start:text.index("```", start)]


def _field_names(block: str) -> set[str]:
    # `[a-z_0-9]+`, not `[a-z_]+`: the latter captures `content_sha` out of
    # `content_sha256` and then fails to see the colon, so the field silently vanishes
    # from the spec side of every comparison below. A parser that drops fields makes this
    # test claim agreement it never checked.
    return set(re.findall(r"^\s{2}([a-z_][a-z_0-9]*)\s*:", block, re.M))


def test_finding_fields_match_the_spec():
    spec = _field_names(_block(_spec_text(), "**`Finding`**"))
    code = {f.name for f in fields(Finding)}
    missing = spec - code
    assert not missing, (
        f"Plugin-Interfaces.md declares fields core/types.py does not have: {sorted(missing)}. "
        "The spec is precedence 1 — the code is what is wrong here.")

    # The additions are the §5 amendments, decided on Backend's authority and landed WITH
    # the spec edit in the same commit. Anything else appearing here is drift: a field
    # added to the code and never written into the contract four other seats read.
    amendments = {"nature", "exclusion_reason", "availability"}
    assert code - spec <= amendments, (
        f"undeclared fields on Finding: {sorted(code - spec - amendments)}")


def test_sample_fields_match_the_spec():
    spec = _field_names(_block(_spec_text(), "**`Sample`**"))
    code = {f.name for f in fields(Sample)}
    assert not spec - code, f"spec fields absent from Sample: {sorted(spec - code)}"
    assert code - spec <= {"contributor_source"}, (
        f"undeclared fields on Sample: {sorted(code - spec - {'contributor_source'})}")


def test_the_five_amendments_are_written_into_the_spec_not_only_the_code():
    """§5.9/§5.10/§5.5/§5.2/§5.13 land WITH the types commit, amending the spec in the same
    change. That is what makes the decision visible rather than tacit — and what stops
    another seat building against a contract this seat has already moved."""
    text = _spec_text()
    for token in ("nature", "exclusion_reason", "contributor_source",
                  "AuditLedger", "InferenceLedgerSource", "Remediator",
                  "SUSPECT_INPUTS"):
        assert token in text, (
            f"{token!r} is in core/types.py but not in Plugin-Interfaces.md. The amendment "
            "must land in the same commit as the code, or the other four seats are "
            "building against a contract that has already moved.")


def test_capability_enum_matches_the_spec():
    text = _spec_text()
    for cap in Capability:
        if cap is Capability.SUSPECT_INPUTS:
            continue          # the Backend amendment, asserted above
        assert cap.value in text, f"{cap.value} is not in Plugin-Interfaces.md"


def test_availability_is_frozen_at_exactly_four_states():
    """§5.5: the temptation is a fifth state for budget exclusion. Every consumer — the
    renderer, the coverage generator, the dashboard Network is already mocking — switches
    on exactly these four."""
    assert [a.value for a in Availability] == ["OK", "DEGRADED", "UNAVAILABLE", "ERROR"]


def test_severity_has_no_warning_member():
    """§9.5: Module C uses `warning` for degraded_gap, and it is not in the frozen enum.
    Crypto maps it to `low`."""
    assert "warning" not in {s.value for s in Severity}
    assert [s.value for s in Severity] == ["info", "low", "medium", "high", "critical"]


def test_the_enums_the_amendments_introduced():
    assert {n.value for n in Nature} == {"adversarial", "quality", "indeterminate"}
    assert {e.value for e in ExclusionReason} == {"capability", "budget"}
    assert {c.value for c in ContributorSource} == {
        "sidecar", "directory", "format_field", "exif_cluster", "none"}
    assert {d.value for d in Disposition} == {"accept", "review", "quarantine"}


def test_evidence_is_kind_path_caption():
    names = {f.name for f in fields(Evidence)}
    assert {"kind", "path", "caption"} <= names
    assert names - {"kind", "path", "caption"} == {"data"}


def test_there_is_no_extension_dict_on_finding():
    """§7.1, stated explicitly: an extension dict would become an untyped side-channel
    that Module E cannot render and the schema cannot validate. Per-detector diagnostics
    travel as Evidence(kind='json') instead — including Module C's primary_check and
    cascade_suppressed."""
    forbidden = {"extra", "extensions", "meta", "extra_fields", "payload"}
    assert not forbidden & {f.name for f in fields(Finding)}
