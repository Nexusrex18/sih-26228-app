"""Gate B3 — the gate sentence from backend_plan.md §8, turned into tests.

    `cva scan` runs end to end with ZERO detectors registered, prints a correct plan, and
    appends via NullAuditLedger; a deliberately-raising stub plug-in resolves ERROR, not
    DEGRADED; the access-assumptions block matches §7.4's shape.

WRITTEN BUT NEVER RUN — the session that authored it could not execute pytest. Treat every
assertion here as a claim, not a result.
"""
from __future__ import annotations

import pytest

from cva.core.access_block import ROWS, render_access_block
from cva.core.capability import Availability, Capability, CapabilitySet
from cva.core.ledger import NullAuditLedger
from cva.core.orchestrator import (
    UnknownProfile,
    build_plan,
    plan_text,
    resolve_profile,
    scan,
)
from cva.core.runcontext import RunContext
from cva.core.types import Finding, Severity

EMPTY: tuple[dict, ...] = ({}, {})


class FakeModel:
    model_id = "fake-001"
    fmt = "onnx"
    opset = 17

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                             ((Capability.MODEL_WEIGHTS, "probed: no weights readable"),))


class RaisingCheck:
    id = "stub.raises"
    version = "0.0.1"
    requires: frozenset = frozenset()
    optional: frozenset = frozenset()
    attack_classes = frozenset({"tool.error"})

    def check(self, model, ctx):
        raise RuntimeError("deliberate")


class OkCheck(RaisingCheck):
    id = "stub.ok"
    attack_classes = frozenset({"model_substitution"})

    def check(self, model, ctx):
        return [Finding(detector_id=self.id, detector_version=self.version,
                        target_type="model", target_ref=model.model_id,
                        severity=Severity.INFO, confidence=0.5, reason="ran",
                        attack_class="model_substitution")]


class _Needs(OkCheck):
    id = "stub.needs"
    requires = frozenset({Capability.MODEL_GRADIENTS})


def test_scan_runs_end_to_end_with_zero_detectors_registered():
    ledger = NullAuditLedger()
    ctx = RunContext(audit_ledger=ledger, code_commit="cafe123")
    res = scan(FakeModel(), ctx, "deep", registries=EMPTY)

    assert res.plan == []
    assert res.findings == []
    assert res.verdict == "ACCEPT"
    assert "no checks registered" in plan_text(res)
    # ...and it appended. UNAVAILABLE is not a skip; neither is an empty scan.
    assert len(ledger.records_seen) == 1
    assert ledger.records_seen[0]["scan"]["scan_id"] == res.scan_id
    assert ledger.records_seen[0]["type"] == "scan_record"
    assert res.ledger_seq == "unsealed-1"


def test_zero_detector_scan_says_it_assessed_nothing():
    res = scan(FakeModel(), RunContext(), "deep", registries=EMPTY)
    assert "assessed nothing" in res.access_assumptions


def test_a_raising_plugin_resolves_ERROR_never_DEGRADED():
    res = scan(FakeModel(), RunContext(), "deep",
               registries=({"stub.raises": RaisingCheck}, {}))
    (f,) = res.findings
    assert f.availability is Availability.ERROR
    assert f.availability is not Availability.DEGRADED
    assert f.attack_class == "tool.error"
    assert "RuntimeError" in f.reason
    # The ERROR finding must not move the verdict: it is a defect in us, not in the model.
    assert res.verdict == "ACCEPT"


def test_an_ERROR_does_not_contaminate_a_sibling_check():
    res = scan(FakeModel(), RunContext(), "deep",
               registries=({"stub.raises": RaisingCheck, "stub.ok": OkCheck}, {}))
    by_id = {f.detector_id: f for f in res.findings}
    assert by_id["stub.raises"].availability is Availability.ERROR
    assert by_id["stub.ok"].availability is Availability.OK


def test_the_orchestrator_stamps_scan_id_and_produced_by():
    res = scan(FakeModel(), RunContext(code_commit="cafe123"), "deep",
               registries=({"stub.ok": OkCheck}, {}))
    (f,) = res.findings
    assert f.scan_id == res.scan_id
    assert f.produced_by == f"cafe123/profile:{res.profile_hash[:12]}"


def test_the_plan_is_emitted_before_any_check_runs():
    """V7's real assertion is ORDERING, not content."""
    events: list[str] = []

    class Recording(OkCheck):
        def check(self, model, ctx):
            events.append("ran")
            return []

    scan(FakeModel(), RunContext(), "deep",
         registries=({"stub.ok": Recording}, {}),
         plan_sink=lambda text: events.append("planned"))
    assert events == ["planned", "ran"]


def test_dry_run_returns_after_planning():
    events: list[str] = []

    class Recording(OkCheck):
        def check(self, model, ctx):
            events.append("ran")
            return []

    res = scan(FakeModel(), RunContext(), "deep", dry_run=True,
               registries=({"stub.ok": Recording}, {}))
    assert events == []
    assert res.verdict == "DRY-RUN"
    assert len(res.plan) == 1
    assert res.findings == []


def test_an_unknown_profile_is_a_load_error_not_a_silent_default():
    with pytest.raises(UnknownProfile):
        resolve_profile("stirct")
    with pytest.raises(UnknownProfile):
        resolve_profile("baseline", budget_tier="deeep")


def test_budget_exclusion_carries_its_reason_and_is_not_a_fifth_state():
    prof = resolve_profile("triage")
    prof["estimated_cost"] = {"stub.ok": "~40 min"}
    rows = build_plan(CapabilitySet(), prof, "triage",
                      registries=({"stub.ok": OkCheck}, {}))
    (row,) = rows
    assert row.resolution.state is Availability.UNAVAILABLE
    assert row.resolution.exclusion_reason == "budget"
    assert row.resolution.estimated_cost == "~40 min"
    assert "~40 min" in row.resolution.reason


def test_the_blackbox_policy_says_not_asked_to_rather_than_could_not():
    prof = resolve_profile("blackbox")
    rows = build_plan(
        CapabilitySet(frozenset(Capability)), prof, "blackbox",
        registries=({"model.weight_digest": OkCheck, "stub.ok": OkCheck}, {}))
    by_id = {r.check_id: r for r in rows}
    assert by_id["model.weight_digest"].resolution.state is Availability.UNAVAILABLE
    assert "not asked to run" in by_id["model.weight_digest"].resolution.reason


# --- §7.4's access-assumptions block ---------------------------------------

def test_the_access_block_matches_section_7_4_shape():
    caps = CapabilitySet(
        frozenset({Capability.MODEL_WEIGHTS}),
        ((Capability.MODEL_GRADIENTS, "onnxruntime-training not bundled"),
         (Capability.REFERENCE_MANIFEST, "no manifest supplied")))
    text = render_access_block(caps, model_fmt="onnx",
                               model_detail="ONNX (opset read from model)",
                               details={Capability.MODEL_WEIGHTS: "sha256 4f2a…"},
                               plan=[])
    lines = text.splitlines()
    assert lines[0] == "Access assumptions for this scan"
    assert "Model format ............... ONNX (opset read from model)" in lines[1]
    assert any("Weights" in ln and "AVAILABLE (sha256 4f2a…)" in ln for ln in lines)
    assert any("Gradients" in ln and
               "NOT AVAILABLE — onnxruntime-training not bundled" in ln for ln in lines)
    assert any("Registered fingerprint" in ln and
               "NOT AVAILABLE — no manifest supplied" in ln for ln in lines)
    assert lines[-1].strip().startswith("Consequence:")


def test_an_absent_capability_still_occupies_a_line():
    """A block that silently shortens reads as a clean scan."""
    text = render_access_block(CapabilitySet(), plan=[])
    for label, _cap in ROWS:
        assert any(ln.strip().startswith(label) for ln in text.splitlines()), label


def test_the_consequence_line_separates_capability_from_budget():
    caps = CapabilitySet()
    prof = resolve_profile("triage")
    # `model.weight_digest` is IN triage's checks list, so it is the one that reaches
    # capability resolution; `stub.ok` is not, so it is the one budget excludes. The
    # first draft had these the other way round and asserted 0 capability / 1 budget.
    rows = build_plan(caps, prof, "triage",
                      registries=({"model.weight_digest": _Needs,
                                   "stub.ok": OkCheck}, {}))
    line = render_access_block(caps, plan=rows).splitlines()[-1]
    assert "1 check(s) UNAVAILABLE (capability)" in line
    assert "1 UNAVAILABLE (budget)" in line
