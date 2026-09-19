"""The disposition table (backend_plan.md §7.9) — V16 / V16b / V16c.

These tests drive the table through `decide()` / `apply_dispositions()` and assert BEHAVIOUR:
the disposition a finding routes to, and (for the named D-rules) which rule fired. The method
ceiling rows (C1..C3) are asserted by outcome only, because their ids and descriptions are
profile data that may be re-cut.

Selector map: `-k benign_conversion` (V16: D2 + defer_digest_to_fingerprint),
`-k prov_floors` (V16b: D1 scoping), `-k info_not_reviewed` (V16c: D5/D7 floor).
"""
from __future__ import annotations

import pytest

from cva.core.capability import Availability
from cva.core.taxonomy import TAXONOMY
from cva.core.types import Disposition, Finding, Severity
from cva.risk.disposition import (
    DEFAULT_RULES,
    apply_dispositions,
    contributor_disposition,
    decide,
    default_policy,
    defer_digest_to_fingerprint,
)

POLICY = default_policy()
SEV = {s.value: s for s in Severity}


def mk(detector_id: str, attack_class: str, severity: str, confidence: float,
       availability: Availability = Availability.OK, target_type: str = "model") -> Finding:
    return Finding(
        detector_id=detector_id, detector_version="0.0.1",
        target_type=target_type,  # type: ignore[arg-type]
        target_ref="t", severity=SEV[severity], confidence=confidence,
        reason="r", attack_class=attack_class, availability=availability)


def route(f: Finding) -> tuple[Disposition, str]:
    return decide(POLICY, f)


# --------------------------------------------------------------------------------------
# sanity: the attack classes this file relies on are real, in the modules it assumes
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,module", [
    ("benign_conversion", "B"), ("model_substitution", "B"),
    ("boundary_flip", "C"), ("degraded_gap", "C"), ("clock_regression", "C"),
    ("record_edit", "C"), ("distribution_shift", "D"), ("semantic_shift", "D"),
    ("suspicious_manipulation", "D"), ("activation_anomaly", "B"),
    ("model_anomalous", "B"), ("backdoor_trigger", "B"),
])
def test_attack_classes_used_here_are_in_the_taxonomy(name: str, module: str) -> None:
    assert TAXONOMY[name].module == module


def test_default_policy_is_a_copy_of_default_rules() -> None:
    p = default_policy()
    assert p["rules"] == DEFAULT_RULES
    p["rules"][0]["disposition"] = "accept"
    p["rules"].append({"id": "X"})
    assert DEFAULT_RULES[0]["disposition"] == "quarantine"     # a mutated copy leaves the source
    assert default_policy()["rules"] == DEFAULT_RULES


def test_default_policy_first_rule_is_scoped_d1_and_last_is_the_accept_catch_all() -> None:
    rules = DEFAULT_RULES
    assert rules[0]["id"] == "D1" and rules[0]["detector_prefix"] == "prov."
    assert rules[-1]["id"] == "D7" and rules[-1]["disposition"] == "accept"
    ids = [r["id"] for r in rules]
    d2, d3 = ids.index("D2"), ids.index("D3")
    assert d2 < d3                                            # D2 above D3 is load-bearing
    assert ids.index("D6") < d3                               # D6 above D3 too


# --------------------------------------------------------------------------------------
# V16 — D2: benign_conversion routes to review, never quarantine
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
@pytest.mark.parametrize("confidence", [0.5, 0.95, 1.0])
def test_benign_conversion_is_review_never_quarantine(severity: str, confidence: float) -> None:
    disp, rule = route(mk("model.fingerprint", "benign_conversion", severity, confidence))
    assert disp == Disposition.REVIEW
    assert rule == "D2"


def test_benign_conversion_critical_full_confidence_is_review_not_quarantine() -> None:
    f = mk("model.fingerprint", "benign_conversion", "critical", 1.0)
    apply_dispositions([f], POLICY)
    assert f.disposition == Disposition.REVIEW
    assert f.disposition_rule == "D2"


def test_benign_conversion_sits_above_d3_a_genuine_substitution_does_not() -> None:
    """Same detector, severity and confidence — only the attack class differs."""
    benign = route(mk("model.fingerprint", "benign_conversion", "critical", 1.0))
    genuine = route(mk("model.fingerprint", "model_substitution", "critical", 1.0))
    assert benign == (Disposition.REVIEW, "D2")
    assert genuine == (Disposition.QUARANTINE, "D3")


def test_benign_conversion_defer_digest_to_fingerprint_lowers_digest_to_info() -> None:
    fingerprint = mk("model.fingerprint", "benign_conversion", "medium", 0.9)
    digest = mk("model.weight_digest", "model_substitution", "critical", 1.0)
    reason_before = digest.reason
    defer_digest_to_fingerprint([fingerprint, digest])
    assert digest.severity == Severity.INFO
    assert digest.reason.startswith("Digest comparison recorded")
    assert digest.reason.endswith(reason_before)
    assert any("model.fingerprint" in lim for lim in digest.limitations)
    # the fingerprint finding itself is untouched
    assert fingerprint.severity == Severity.MEDIUM


def test_benign_conversion_defer_digest_to_fingerprint_routes_digest_to_accept() -> None:
    fingerprint = mk("model.fingerprint", "benign_conversion", "medium", 0.9)
    digest = mk("model.weight_digest", "model_substitution", "critical", 1.0)
    findings = [fingerprint, digest]
    defer_digest_to_fingerprint(findings)
    apply_dispositions(findings, POLICY)
    assert digest.disposition == Disposition.ACCEPT
    assert fingerprint.disposition == Disposition.REVIEW
    assert not any(f.disposition == Disposition.QUARANTINE for f in findings)


def test_benign_conversion_defer_digest_without_fingerprint_stays_critical_quarantine() -> None:
    digest = mk("model.weight_digest", "model_substitution", "critical", 1.0)
    defer_digest_to_fingerprint([digest])
    assert digest.severity == Severity.CRITICAL
    assert digest.limitations == []
    apply_dispositions([digest], POLICY)
    assert digest.disposition == Disposition.QUARANTINE
    assert digest.disposition_rule == "D3"


@pytest.mark.parametrize("fp_state", [Availability.UNAVAILABLE, Availability.ERROR])
def test_benign_conversion_defer_digest_ignores_a_fingerprint_that_did_not_run(
        fp_state: Availability) -> None:
    fingerprint = mk("model.fingerprint", "not_assessed", "info", 0.0, availability=fp_state)
    digest = mk("model.weight_digest", "model_substitution", "critical", 1.0)
    defer_digest_to_fingerprint([fingerprint, digest])
    assert digest.severity == Severity.CRITICAL
    apply_dispositions([digest], POLICY)
    assert digest.disposition == Disposition.QUARANTINE


def test_benign_conversion_defer_digest_counts_a_degraded_fingerprint_as_having_run() -> None:
    fingerprint = mk("model.fingerprint", "benign_conversion", "low", 0.7,
                     availability=Availability.DEGRADED)
    digest = mk("model.weight_digest", "model_substitution", "critical", 1.0)
    defer_digest_to_fingerprint([fingerprint, digest])
    assert digest.severity == Severity.INFO


def test_benign_conversion_defer_digest_only_touches_the_digests_substitution_finding() -> None:
    fingerprint = mk("model.fingerprint", "benign_conversion", "medium", 0.9)
    other_digest_class = mk("model.weight_digest", "model_anomalous", "critical", 1.0)
    other_detector = mk("model.other", "model_substitution", "critical", 1.0)
    digest_not_run = mk("model.weight_digest", "model_substitution", "critical", 1.0,
                        availability=Availability.UNAVAILABLE)
    defer_digest_to_fingerprint([fingerprint, other_digest_class, other_detector, digest_not_run])
    assert other_digest_class.severity == Severity.CRITICAL
    assert other_detector.severity == Severity.CRITICAL
    assert digest_not_run.severity == Severity.CRITICAL


def test_benign_conversion_defer_digest_is_idempotent_on_an_already_info_digest() -> None:
    fingerprint = mk("model.fingerprint", "benign_conversion", "medium", 0.9)
    digest = mk("model.weight_digest", "model_substitution", "critical", 1.0)
    defer_digest_to_fingerprint([fingerprint, digest])
    once = (digest.severity, digest.reason, list(digest.limitations))
    defer_digest_to_fingerprint([fingerprint, digest])
    assert (digest.severity, digest.reason, list(digest.limitations)) == once


# --------------------------------------------------------------------------------------
# V16b — D1 is scoped to prov.* and floored at severity >= high
# --------------------------------------------------------------------------------------
def test_prov_floors_boundary_flip_medium_full_confidence_is_review() -> None:
    disp, rule = route(mk("prov.verify", "boundary_flip", "medium", 1.0, target_type="record"))
    assert disp == Disposition.REVIEW
    assert rule != "D1"


def test_prov_floors_degraded_gap_low_is_review() -> None:
    disp, _ = route(mk("prov.verify", "degraded_gap", "low", 1.0, target_type="record"))
    assert disp == Disposition.REVIEW


def test_prov_floors_clock_regression_info_is_accept() -> None:
    disp, rule = route(mk("prov.verify", "clock_regression", "info", 1.0, target_type="record"))
    assert disp == Disposition.ACCEPT
    assert rule == "D7"


@pytest.mark.parametrize("attack_class,severity", [
    ("boundary_flip", "medium"), ("boundary_flip", "low"), ("boundary_flip", "info"),
    ("degraded_gap", "low"), ("degraded_gap", "info"),
    ("clock_regression", "info"), ("clock_regression", "low"),
])
def test_prov_floors_none_of_the_operational_provenance_classes_reach_quarantine(
        attack_class: str, severity: str) -> None:
    disp, rule = route(mk("prov.verify", attack_class, severity, 1.0, target_type="record"))
    assert disp != Disposition.QUARANTINE
    assert rule != "D1"


@pytest.mark.parametrize("severity", ["high", "critical"])
def test_prov_floors_record_edit_at_high_or_critical_is_quarantine_via_d1(severity: str) -> None:
    disp, rule = route(mk("prov.chain", "record_edit", severity, 1.0, target_type="record"))
    assert disp == Disposition.QUARANTINE
    assert rule == "D1"


@pytest.mark.parametrize("severity", ["high", "critical"])
def test_prov_floors_d1_fires_regardless_of_confidence(severity: str) -> None:
    """D1 has no confidence floor: prov.* bypasses calibration and is arithmetic."""
    disp, rule = route(mk("prov.chain", "record_edit", severity, 0.0, target_type="record"))
    assert (disp, rule) == (Disposition.QUARANTINE, "D1")


def test_prov_floors_record_edit_below_high_is_not_d1() -> None:
    disp, rule = route(mk("prov.chain", "record_edit", "medium", 1.0, target_type="record"))
    assert rule != "D1"
    assert disp == Disposition.REVIEW


def test_prov_floors_d1_is_scoped_to_prov_a_model_weight_digest_critical_is_not_d1() -> None:
    disp, rule = route(mk("model.weight_digest", "model_substitution", "critical", 1.0))
    assert rule != "D1"
    assert (disp, rule) == (Disposition.QUARANTINE, "D3")


def test_prov_floors_d1_prefix_is_a_prefix_not_a_substring() -> None:
    disp, rule = route(mk("model.prov.thing", "model_substitution", "critical", 1.0))
    assert rule != "D1"


# --------------------------------------------------------------------------------------
# V16c — D5/D7: an info finding is never reviewed on confidence alone
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("detector,attack_class", [
    ("model.weight_digest", "model_substitution"),
    ("dataset.dup", "near_duplicate_flooding"),
    ("prov.verify", "clock_regression"),
])
def test_info_not_reviewed_at_confidence_099_is_accept_d7(detector: str,
                                                          attack_class: str) -> None:
    assert route(mk(detector, attack_class, "info", 0.99)) == (Disposition.ACCEPT, "D7")


def test_info_not_reviewed_high_confidence_is_accept() -> None:
    f = mk("dataset.dup", "near_duplicate_flooding", "info", 0.99)
    apply_dispositions([f], POLICY)
    assert f.disposition == Disposition.ACCEPT
    assert f.disposition_rule == "D7"


def test_info_not_reviewed_but_the_same_confidence_at_low_severity_is_review_d5() -> None:
    disp, rule = route(mk("dataset.dup", "near_duplicate_flooding", "low", 0.99))
    assert (disp, rule) == (Disposition.REVIEW, "D5")


# --------------------------------------------------------------------------------------
# D3 / D5 boundaries
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("severity", ["high", "critical"])
def test_d3_confidence_at_or_above_090_and_severity_at_least_high_is_quarantine(
        severity: str) -> None:
    for conf in (0.9, 0.95, 1.0):
        assert route(mk("dataset.dup", "label_flipping", severity, conf)) == (
            Disposition.QUARANTINE, "D3")


def test_d3_confidence_089_falls_to_review_via_d5() -> None:
    assert route(mk("dataset.dup", "label_flipping", "critical", 0.89)) == (
        Disposition.REVIEW, "D5")


def test_d3_needs_severity_high_medium_at_full_confidence_is_review() -> None:
    assert route(mk("dataset.dup", "label_flipping", "medium", 1.0)) == (
        Disposition.REVIEW, "D5")


def test_d5_confidence_060_at_low_severity_is_review_and_059_is_accept() -> None:
    assert route(mk("dataset.dup", "label_flipping", "low", 0.6)) == (Disposition.REVIEW, "D5")
    assert route(mk("dataset.dup", "label_flipping", "low", 0.59)) == (Disposition.ACCEPT, "D7")


def test_d3_and_d5_are_not_fooled_by_confidence_alone_at_zero() -> None:
    assert route(mk("dataset.dup", "label_flipping", "critical", 0.0)) == (
        Disposition.ACCEPT, "D7")


# --------------------------------------------------------------------------------------
# D6 — Module D attack classes are capped at review
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("attack_class", ["distribution_shift", "semantic_shift",
                                          "suspicious_manipulation"])
@pytest.mark.parametrize("severity,confidence", [
    ("high", 0.95), ("critical", 1.0), ("high", 0.9), ("medium", 0.7)])
def test_d6_module_d_classes_are_review_never_quarantine(attack_class: str, severity: str,
                                                        confidence: float) -> None:
    disp, rule = route(mk("drift.psi", attack_class, severity, confidence, target_type="dataset"))
    assert disp == Disposition.REVIEW
    assert rule == "D6"


def test_d6_a_non_module_d_class_at_the_same_strength_is_quarantine() -> None:
    assert route(mk("drift.psi", "label_flipping", "high", 0.95, target_type="dataset")) == (
        Disposition.QUARANTINE, "D3")


def test_d6_module_d_below_its_confidence_floor_is_never_quarantine() -> None:
    disp, _ = route(mk("drift.psi", "distribution_shift", "high", 0.5, target_type="dataset"))
    assert disp != Disposition.QUARANTINE


def test_d6_unknown_attack_class_does_not_match_a_module_rule_and_never_raises() -> None:
    disp, rule = route(mk("x.y", "not_in_taxonomy", "high", 0.95))
    assert (disp, rule) == (Disposition.QUARANTINE, "D3")     # D6 skipped, D3 applies


# --------------------------------------------------------------------------------------
# method ceilings (C1..C3): behaviour through decide()
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
def test_ceiling_intrinsic_probes_high_confidence_is_review_never_quarantine(
        severity: str) -> None:
    disp, _ = route(mk("model.intrinsic_probes", "backdoor_trigger", severity, 0.99))
    assert disp == Disposition.REVIEW


def test_ceiling_intrinsic_probes_info_falls_through_to_accept() -> None:
    disp, rule = route(mk("model.intrinsic_probes", "backdoor_trigger", "info", 0.99))
    assert disp == Disposition.ACCEPT
    assert rule == "D7"


def test_ceiling_intrinsic_probes_low_confidence_flagged_outlier_is_still_review() -> None:
    """The ceiling has no confidence floor: a flagged outlier at 0.3 must not be accepted."""
    disp, _ = route(mk("model.intrinsic_probes", "backdoor_trigger", "medium", 0.3))
    assert disp == Disposition.REVIEW


def test_ceiling_intrinsic_probes_is_a_prefix_match_on_the_detector_id() -> None:
    disp, _ = route(mk("model.intrinsic_probes.extra", "backdoor_trigger", "critical", 0.99))
    assert disp == Disposition.REVIEW
    other, _ = route(mk("model.neural_cleanse", "backdoor_trigger", "critical", 0.99))
    assert other == Disposition.QUARANTINE                     # a different detector is uncapped


@pytest.mark.parametrize("attack_class", ["activation_anomaly", "model_anomalous"])
def test_ceiling_activation_and_anomalous_classes_are_review_never_quarantine(
        attack_class: str) -> None:
    for severity in ("high", "critical"):
        disp, _ = route(mk("model.weight_stats", attack_class, severity, 0.99))
        assert disp == Disposition.REVIEW


@pytest.mark.parametrize("attack_class", ["activation_anomaly", "model_anomalous"])
def test_ceiling_activation_and_anomalous_classes_info_falls_through_to_accept(
        attack_class: str) -> None:
    disp, rule = route(mk("model.weight_stats", attack_class, "info", 0.99))
    assert (disp, rule) == (Disposition.ACCEPT, "D7")


def test_ceiling_a_cap_row_lowers_a_quarantine_row_to_review() -> None:
    policy = {"rules": [{"id": "T", "disposition": "quarantine", "cap": "review"},
                        {"id": "D7", "disposition": "accept"}]}
    f = mk("x.y", "label_flipping", "critical", 1.0)
    assert decide(policy, f) == (Disposition.REVIEW, "T")


def test_ceiling_a_cap_never_raises_a_lower_disposition() -> None:
    policy = {"rules": [{"id": "T", "disposition": "accept", "cap": "review"}]}
    f = mk("x.y", "label_flipping", "critical", 1.0)
    assert decide(policy, f) == (Disposition.ACCEPT, "T")


def test_decide_with_no_matching_rule_defaults_to_accept_d7() -> None:
    policy = {"rules": [{"id": "T", "disposition": "quarantine", "min_severity": "critical"}]}
    assert decide(policy, mk("x.y", "label_flipping", "low", 1.0)) == (Disposition.ACCEPT, "D7")


def test_decide_first_match_wins() -> None:
    policy = {"rules": [{"id": "first", "disposition": "review"},
                        {"id": "second", "disposition": "quarantine"}]}
    assert decide(policy, mk("x.y", "label_flipping", "critical", 1.0)) == (
        Disposition.REVIEW, "first")


def test_a_contributor_rule_never_matches_a_finding() -> None:
    policy = {"rules": [{"id": "D4", "min_posterior": 0.0, "disposition": "quarantine"},
                        {"id": "D7", "disposition": "accept"}]}
    assert decide(policy, mk("x.y", "label_flipping", "critical", 1.0)) == (
        Disposition.ACCEPT, "D7")


# --------------------------------------------------------------------------------------
# apply_dispositions: only findings that ran are routed
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("state", [Availability.UNAVAILABLE, Availability.ERROR])
def test_apply_dispositions_leaves_findings_that_did_not_run_untouched(
        state: Availability) -> None:
    f = mk("prov.chain", "record_edit", "critical", 1.0, availability=state, target_type="record")
    f.disposition = Disposition.REVIEW
    f.disposition_rule = "orchestrator"
    apply_dispositions([f], POLICY)
    assert f.disposition == Disposition.REVIEW
    assert f.disposition_rule == "orchestrator"


@pytest.mark.parametrize("state", [Availability.OK, Availability.DEGRADED])
def test_apply_dispositions_routes_findings_that_ran(state: Availability) -> None:
    f = mk("prov.chain", "record_edit", "critical", 1.0, availability=state, target_type="record")
    f.disposition = Disposition.ACCEPT
    apply_dispositions([f], POLICY)
    assert f.disposition == Disposition.QUARANTINE
    assert f.disposition_rule == "D1"


def test_apply_dispositions_ignores_a_detector_set_disposition() -> None:
    """The engine owns `disposition`: a detector claiming `accept` cannot dodge D1."""
    f = mk("prov.chain", "record_edit", "critical", 1.0, target_type="record")
    f.disposition = Disposition.ACCEPT
    apply_dispositions([f], POLICY)
    assert f.disposition == Disposition.QUARANTINE


def test_apply_dispositions_mixed_batch_gets_the_right_rule_each() -> None:
    fs = [
        mk("prov.chain", "record_edit", "critical", 1.0, target_type="record"),
        mk("model.fingerprint", "benign_conversion", "critical", 1.0),
        mk("drift.psi", "distribution_shift", "high", 0.95, target_type="dataset"),
        mk("dataset.dup", "label_flipping", "high", 0.95, target_type="batch"),
        mk("dataset.dup", "label_flipping", "medium", 0.7, target_type="batch"),
        mk("prov.verify", "clock_regression", "info", 1.0, target_type="record"),
    ]
    apply_dispositions(fs, POLICY)
    assert [f.disposition_rule for f in fs] == ["D1", "D2", "D6", "D3", "D5", "D7"]
    assert [f.disposition for f in fs] == [
        Disposition.QUARANTINE, Disposition.REVIEW, Disposition.REVIEW,
        Disposition.QUARANTINE, Disposition.REVIEW, Disposition.ACCEPT]


# --------------------------------------------------------------------------------------
# D4 — contributor_disposition
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("posterior", [0.25, 0.3, 0.9, 1.0])
def test_d4_posterior_at_or_above_floor_with_ci_excluding_cohort_is_quarantine(
        posterior: float) -> None:
    assert contributor_disposition(POLICY, posterior, True) == "quarantine"


@pytest.mark.parametrize("posterior", [0.25, 0.5, 0.99, 1.0])
def test_d4_never_quarantines_on_the_point_estimate_alone(posterior: float) -> None:
    assert contributor_disposition(POLICY, posterior, False) != "quarantine"
    assert contributor_disposition(POLICY, posterior, False) == "accept"


def test_d4_below_floor_with_ci_excluding_cohort_is_review() -> None:
    assert contributor_disposition(POLICY, 0.24, True) == "review"
    assert contributor_disposition(POLICY, 0.0, True) == "review"


def test_d4_below_floor_with_ci_not_excluding_cohort_is_accept() -> None:
    assert contributor_disposition(POLICY, 0.24, False) == "accept"
    assert contributor_disposition(POLICY, 0.0, False) == "accept"


def test_d4_the_policy_decides_whether_a_ci_is_required() -> None:
    lenient = {"rules": [{"id": "D4", "min_posterior": 0.25, "disposition": "quarantine"}]}
    assert contributor_disposition(lenient, 0.3, False) == "quarantine"


def test_d4_a_policy_without_a_contributor_rule_never_quarantines() -> None:
    policy = {"rules": [{"id": "D7", "disposition": "accept"}]}
    assert contributor_disposition(policy, 1.0, True) == "review"
    assert contributor_disposition(policy, 1.0, False) == "accept"
