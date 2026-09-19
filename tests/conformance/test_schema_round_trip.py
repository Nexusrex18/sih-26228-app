"""Gate B1's second clause, paid here: every frozen type round-trips through its schema.

The three schemas were written at B1 and nothing validated an instance against them, which
is the shape of failure they exist to prevent — a schema nobody validates against is
documentation that happens to be machine-readable, and it drifts from the code the first
time a field moves.

The round trip specifically, not just validation: `Finding.to_dict()` is what the renderer
writes and what Module E reads back, so the thing that must satisfy the schema is the
SERIALISED form, not the dataclass. A test that validated a hand-built dict would assert
the schema agrees with the test author.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cva.core.capability import Availability
from cva.core.types import (
    ContributorSource,
    Disposition,
    Evidence,
    Finding,
    Nature,
    Severity,
    unavailable_finding,
)

SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def _validator(name: str):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / name).read_text())
    # Draft202012 explicitly, never `validators.validator_for`: the draft is what decides
    # whether `unevaluatedProperties` is honoured at all, and letting the file choose means
    # a typo'd $schema silently downgrades every constraint below it to a no-op.
    return jsonschema.Draft202012Validator(schema)


def _finding_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / "report.schema.json").read_text())
    sub = dict(schema["$defs"]["finding"])
    sub["$defs"] = schema["$defs"]          # keep sibling $refs resolvable
    return jsonschema.Draft202012Validator(sub)


def _full_finding() -> Finding:
    return Finding(
        detector_id="data.near_duplicate",
        detector_version="1.0.0",
        target_type="contributor",
        target_ref="contrib_B",
        severity=Severity.HIGH,
        confidence=0.91,
        reason="214 samples from Contributor B share a 6x6 high-frequency anomaly.",
        attack_class="near_duplicate_flooding",
        scan_id="s-2026-09-19-0001",
        produced_by="abc1234/profile:def5678",
        score_raw=0.87,
        threshold=0.5,
        evidence=[Evidence("contact_sheet", "5 examples", path="a" * 64)],
        access_assumptions=["dataset images decodable"],
        limitations=["pHash only; embeddings unavailable"],
        disposition=Disposition.QUARANTINE,
        disposition_rule="D3",
        nature=Nature.QUALITY,
        availability=Availability.OK,
    )


def test_a_fully_populated_finding_validates_against_the_report_schema():
    _finding_validator().validate(_full_finding().to_dict())


def test_an_unavailable_finding_validates_too():
    """The state that is NOT a skip has to serialise like any other finding, or the
    renderer cannot print all four states at equal prominence."""
    from cva.core.capability import Capability

    f = unavailable_finding(
        "model.strip", "1.0.0", "model-1", "no suspect inputs supplied",
        [Capability.SUSPECT_INPUTS], "backdoor_trigger")
    _finding_validator().validate(f.to_dict())


def test_the_round_trip_is_lossless_for_every_enum_on_the_finding():
    """json.dumps → json.loads must give back exactly what `to_dict` produced. The enums
    are the risk: `Severity` is a `str` Enum, so a bare `json.dumps` of the dataclass would
    emit the member and compare equal to the string in Python while being a different
    value to every other language reading the report."""
    d = _full_finding().to_dict()
    back = json.loads(json.dumps(d))
    assert back == d
    # severity/disposition/nature are lowercase; Availability is UPPERCASE and frozen that
    # way ("OK"/"DEGRADED"/"UNAVAILABLE"/"ERROR"). Asserting one convention across all four
    # would be asserting a tidiness the contract does not have.
    for key in ("severity", "disposition", "nature"):
        assert isinstance(back[key], str) and back[key] == back[key].lower()
    assert back["availability"] in ("OK", "DEGRADED", "UNAVAILABLE", "ERROR")
    assert back["exclusion_reason"] is None


def test_the_schema_actually_rejects_a_bad_finding():
    """The assertion that makes the three above mean something. A schema with a typo in a
    `$ref` validates everything, so a suite that only ever feeds it correct instances
    cannot tell a working schema from an inert one."""
    jsonschema = pytest.importorskip("jsonschema")
    v = _finding_validator()

    missing = _full_finding().to_dict()
    del missing["attack_class"]
    with pytest.raises(jsonschema.ValidationError):
        v.validate(missing)

    bad_enum = _full_finding().to_dict()
    bad_enum["severity"] = "warning"        # §9.5 — deliberately not a member
    with pytest.raises(jsonschema.ValidationError):
        v.validate(bad_enum)

    out_of_range = _full_finding().to_dict()
    out_of_range["confidence"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        v.validate(out_of_range)


def test_unknown_keys_are_rejected_rather_than_ignored():
    """`unevaluatedProperties: false`, not `additionalProperties: false` — the latter does
    not compose under `allOf`, where it is evaluated against the local schema alone and
    every inherited field reads as unknown. An extension dict smuggled onto a Finding is
    the thing this stops: an untyped side-channel Module E cannot render."""
    jsonschema = pytest.importorskip("jsonschema")
    d = _full_finding().to_dict()
    d["extra_hint"] = {"anything": 1}
    with pytest.raises(jsonschema.ValidationError):
        _finding_validator().validate(d)


@pytest.mark.parametrize("name", ["report.schema.json", "profile.schema.json",
                                  "scenario.schema.json"])
def test_every_schema_is_itself_valid_and_versioned(name):
    """A versioned `$id` is what makes a report re-readable next month; §5.6 requires it on
    all three, and a schema that is not itself valid fails open rather than closed."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / name).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert "$id" in schema and "/1.0.0/" in schema["$id"]
    _validator(name)


def test_the_contributor_source_enum_is_in_the_schema():
    """§5.10's amendment has to reach the serialised form, or the report still cannot say
    which precedence tier resolved a contributor."""
    schema = json.loads((SCHEMAS / "report.schema.json").read_text())
    declared = set(schema["$defs"]["contributorSource"]["enum"])
    # `None` is in the schema enum because the field is nullable — a Sample loaded before
    # resolution has not yet been through the precedence at all, which is distinct from
    # having been through it and come out with "none".
    assert {c.value for c in ContributorSource} == declared - {None}
