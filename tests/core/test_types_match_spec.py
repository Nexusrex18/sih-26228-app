"""V4 — `core/types.py` and `Architecture/Plugin-Interfaces.md` agree field-for-field.

This is the test that makes the freeze real. §5.11 puts the freeze in force from the
`core/types.py` commit, and R1 is that the freeze slips — types changing after four seats
have built against them. Keeping code and spec in lockstep by TEST rather than by
discipline is what stops the spec quietly becoming fiction while every other test passes.

**The contract is VENDORED into this repo** — `spec/frozen/Plugin-Interfaces.md` (item 25).
It used to be read from a sibling vault clone outside the repo, and skipped when that clone
was absent. B1's definition of done is "the contract agrees field-for-field, ASSERTED BY A
TEST", and a skip is not an assertion: on a machine without the vault — every CI runner, and
any teammate who cloned only the app — the gate proved nothing while reading green. A test
that reports agreement it never checked is the failure shape this whole system exists to
prevent, one level up.

The vendored copy is now the authority the test compares against, so it always runs. When a
vault clone IS present, a separate test compares the two and fails if the vendored copy has
drifted from it — the freeze is only real if the copy cannot go stale unnoticed.
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

REPO = Path(__file__).resolve().parents[2]
#: The frozen contract, in-repo. Present on every checkout, so these tests never skip.
SPEC = REPO / "spec" / "frozen" / "Plugin-Interfaces.md"
#: The sibling vault, when someone has it. Used ONLY to detect drift in the vendored copy.
VAULT_SPEC = REPO.parent / "sih26228-notes" / "Architecture" / "Plugin-Interfaces.md"


def _spec_text() -> str:
    # Deliberately NOT a skip. The vendored contract ships with the repo, so its absence is
    # a broken checkout, not a missing optional dependency — and B1 is asserted, not hoped.
    assert SPEC.exists(), (
        f"the frozen contract is missing from the repo at {SPEC}. It is vendored precisely "
        "so this gate cannot pass by skipping; restore it rather than relaxing this test.")
    return SPEC.read_text()


def test_the_vendored_contract_has_not_drifted_from_the_vault():
    """The one test that may legitimately skip: it needs the vault, and its subject is
    whether the in-repo copy is stale. Everything else reads the vendored copy and runs
    everywhere."""
    if not VAULT_SPEC.exists():
        pytest.skip(f"no vault clone at {VAULT_SPEC}; nothing to compare the copy against")
    assert SPEC.read_text() == VAULT_SPEC.read_text(), (
        f"{SPEC} has drifted from {VAULT_SPEC}. The vendored copy is what every other test "
        "in this file asserts against, so a stale copy means the freeze is being checked "
        "against the wrong contract. Re-copy it and re-run.")


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
