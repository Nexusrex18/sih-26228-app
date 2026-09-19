"""The risk engine: everything between "detectors returned findings" and "the report".

Order matters. The digest defers to the fingerprint first (so D2 is reachable), then
calibration (so D3/D5 read the calibrated confidence), then dispositions, and only then the
contributor aggregation, because "flagged" means "routed to review or quarantine".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cva.core.capability import Availability
from cva.core.types import Disposition, Finding
from cva.risk.calibration import CalibrationSet, apply_calibration
from cva.risk.contributor import assess_groups_detailed
from cva.risk.disposition import (
    apply_dispositions,
    default_policy,
    defer_digest_to_fingerprint,
)

_RAN = (Availability.OK, Availability.DEGRADED)


@dataclass
class RiskOutcome:
    contributor_risk: list[dict[str, Any]] = field(default_factory=list)
    permutation_test: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    # B7's absolute rate against a known-clean reference dataset; None unless one was usable.
    contributor_baseline: dict[str, Any] | None = None
    # Why it is None when a reference WAS supplied; None when none was supplied.
    contributor_baseline_unavailable: str | None = None

    @property
    def quarantined_contributors(self) -> list[str]:
        return [f"{r['group_key']}:{r['group_value']}" for r in self.contributor_risk
                if r["disposition"] == "quarantine"]


def flagged_sample_ids(findings: list[Finding]) -> set[str]:
    """THE definition of "flagged", used for the cohort and for the reference dataset alike: a
    sample-level finding that ran and whose disposition is not accept."""
    return {f.target_ref for f in findings
            if f.target_type == "sample" and f.availability in _RAN
            and f.disposition != Disposition.ACCEPT}


def _route(findings: list[Finding], prof: dict[str, Any],
           calibration: CalibrationSet | None) -> dict[str, Any]:
    policy: dict[str, Any] = prof.get("disposition") or default_policy()
    prefixes = tuple((prof.get("calibration") or {}).get("exclude_detector_prefixes") or ("prov.",))
    apply_calibration(findings, calibration, prefixes)
    apply_dispositions(findings, policy)
    return policy


def reference_flag_counts(findings: list[Finding], dataset: Any, prof: dict[str, Any],
                          calibration: CalibrationSet | None = None) -> dict[str, int]:
    """Flag count of the REFERENCE dataset, through the same calibration, policy and flag
    definition as the cohort. The findings are consumed here and never reach the report."""
    _route(findings, prof, calibration)
    ids = {s.sample_id for s in dataset.samples}
    return {"reference_n": len(ids), "reference_flagged": len(flagged_sample_ids(findings) & ids)}


def _reference_ceiling(n: int, flagged: int) -> float:
    """Upper end of the 95% Jeffreys interval on the reference's flag rate. A reference that
    flagged 0 of 60 images does not show a true rate of zero, so the cohort is compared to what
    the reference could plausibly be producing, not to its point estimate."""
    from scipy.stats import beta as beta_dist
    return float(beta_dist.ppf(0.975, flagged + 0.5, n - flagged + 0.5))


def assess(findings: list[Finding], dataset: Any, prof: dict[str, Any], seed: int,
           calibration: CalibrationSet | None = None,
           reference: dict[str, int] | None = None,
           reference_unavailable: str | None = None) -> RiskOutcome:
    defer_digest_to_fingerprint(findings)
    policy = _route(findings, prof, calibration)

    out = RiskOutcome(calibration=calibration.summary() if calibration else None,
                      contributor_baseline_unavailable=reference_unavailable)
    if dataset is None or not getattr(dataset, "samples", None):
        return out
    ceiling: float | None = None
    if reference is not None and reference["reference_n"] > 0:
        ceiling = _reference_ceiling(reference["reference_n"], reference["reference_flagged"])
    got = assess_groups_detailed(dataset.samples, flagged_sample_ids(findings), policy, seed,
                                 reference_ceiling=ceiling)
    out.contributor_risk, out.permutation_test = got.rows, got.permutation_test
    if reference is not None and ceiling is not None:
        if not got.cohort_rates:
            out.contributor_baseline_unavailable = (
                "no grouping key (contributor, batch, source) carries data in the scanned "
                "dataset, so there is no cohort rate to compare with the reference")
            return out
        cohort_rate = got.cohort_rates[next(iter(got.cohort_rates))]
        n, k = reference["reference_n"], reference["reference_flagged"]
        out.contributor_baseline = {
            "reference_n": n, "reference_flagged": k, "reference_rate": round(k / n, 6),
            "reference_rate_ci_high": round(ceiling, 6),
            "cohort_rate": round(cohort_rate, 6),
            "cohort_exceeds_reference": bool(cohort_rate > ceiling),
        }
    return out
