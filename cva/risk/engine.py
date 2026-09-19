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
from cva.risk.contributor import assess_groups
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

    @property
    def quarantined_contributors(self) -> list[str]:
        return [f"{r['group_key']}:{r['group_value']}" for r in self.contributor_risk
                if r["disposition"] == "quarantine"]


def assess(findings: list[Finding], dataset: Any, prof: dict[str, Any], seed: int,
           calibration: CalibrationSet | None = None) -> RiskOutcome:
    policy = prof.get("disposition") or default_policy()
    prefixes = tuple((prof.get("calibration") or {}).get("exclude_detector_prefixes") or ("prov.",))

    defer_digest_to_fingerprint(findings)
    apply_calibration(findings, calibration, prefixes)
    apply_dispositions(findings, policy)

    out = RiskOutcome(calibration=calibration.summary() if calibration else None)
    if dataset is None or not getattr(dataset, "samples", None):
        return out
    flagged = {f.target_ref for f in findings
               if f.target_type == "sample" and f.availability in _RAN
               and f.disposition != Disposition.ACCEPT}
    out.contributor_risk, out.permutation_test = assess_groups(
        dataset.samples, flagged, policy, seed)
    return out
