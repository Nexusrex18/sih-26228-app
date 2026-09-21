"""The enums live in one place by TEST, not by import.

`cva.web` may not import `cva.provenance` (D-E3, CI invariant 5), so `web/workflow/events.py`
restates Module C's closed sets. That is only safe if a drift is a red build: a reason code an
analyst can pick and the ledger will not store is a bug discovered at the worst moment.
"""
from __future__ import annotations

import pytest

from cva.web.workflow import events as E

records = pytest.importorskip("cva.provenance.seal.records",
                              reason="the seal SDK needs `cryptography` and `rfc8785`")


def test_analyst_actions_match():
    assert E.ANALYST_ACTIONS == records.ANALYST_ACTIONS


def test_reason_codes_match():
    assert sorted(E.REASON_CODES) == sorted(records.REASON_CODES)
    assert set(E.GENERAL_REASON_CODES) | set(E.PROV_REASON_CODES) == set(records.REASON_CODES)
    assert not set(E.GENERAL_REASON_CODES) & set(E.PROV_REASON_CODES), \
        "D-E8 splits the enum; a code in both halves would make the prov.* restriction a no-op"


def test_target_types_match():
    assert E.TARGET_TYPES == records.TARGET_TYPES


def test_dispositions_match():
    assert E.DISPOSITIONS == records.DISPOSITIONS
    assert set(E.DISPOSITION_RANK) == set(records.DISPOSITIONS)


def test_every_disposition_the_schema_allows_has_a_rank():
    import json
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas"
                         / "report.schema.json").read_text())
    assert set(schema["$defs"]["disposition"]["enum"]) == set(E.DISPOSITION_RANK)


def test_a_built_event_passes_module_cs_own_validator():
    """The strongest form of the check: the thing we build is the thing they accept."""
    req = E.build(actor_id="a.sharma", role="analyst", action="override",
                  scan_id="s-2026-09-19-0001", target_type="sample",
                  target_ref="img_04471.jpg",
                  finding_id=E.synthetic_finding_id("sample", "img_04471.jpg"),
                  new_disposition="review", reason_code="quality_issue",
                  justification="Duplicate cluster confirmed as a scanner re-export.",
                  expected_prev_seq=0)
    records.validate_section("analyst", dict(req.body))


@pytest.mark.parametrize("action", E.ANALYST_ACTIONS)
def test_every_action_this_ui_offers_builds_a_valid_section(action):
    kw = {"actor_id": "a.sharma", "role": "approver", "action": action,
          "scan_id": "s-2026-09-19-0001", "target_type": "contributor",
          "target_ref": "survey_team_north",
          "finding_id": E.synthetic_finding_id("contributor", "survey_team_north")}
    if action == "override":
        kw |= {"new_disposition": "review", "reason_code": "quality_issue",
               "expected_prev_seq": 0,
               "justification": "The neighbouring batch explains this cluster in full."}
    elif action == "approve":
        kw["refs_seq"] = 7
    elif action == "assign":
        kw["assignee"] = "b.rao"
    elif action == "release":
        kw["justification"] = "A signed manifest for the affected batch was produced."
    records.validate_section("analyst", dict(E.build(**kw).body))
