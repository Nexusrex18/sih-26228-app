"""`prov.ledger_verify` — the scan-side wrapper around the standalone verifier (plan §3.2, §3.3, §13).

This package MAY import `core/`, which is exactly why it is separate from `seal/`: the standalone verifier
returns plain dataclasses, and this module turns them into Backend's frozen `Finding`.

Rules it enforces, each a lesson from the plan:

  * `prov.*` findings BYPASS CALIBRATION: `confidence = score_raw = threshold = 1.0`. A hash mismatch is
    arithmetic, not a belief, and a calibrated probability on a certainty would be a lie in the opposite
    direction (and would corrupt the isotonic fit for every other detector). Backend's fit excludes every
    `prov.*` detector_id; this module makes sure nothing here needs excluding by anything but that id.
  * Severity is load-bearing for disposition. Backend's rule D1 fires on `prov.*` with severity >= high, so
    every class that must quarantine arrives `high`/`critical`; `boundary_flip` (medium) and `degraded_gap`
    (low) route to `review`, and `clock_regression` (info) can never change a disposition.
  * The `disposition` set here is PROVISIONAL. The risk engine owns the final value; `disposition_rule` says so.
  * Every scan gets ONE `ledger_verified` info finding, including when nothing failed (open item O9): a report
    with no provenance section is indistinguishable from a scan where provenance was never checked. It reports
    the unwitnessed window — the records after the last anchor, which still trust the key holder.
  * A ledger that cannot be opened at all is not a tamper finding: nothing could be assessed, so the check
    reports `UNAVAILABLE` with the reason instead of guessing.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from cva.core.capability import Availability, Capability, CapabilitySet, Resolution
from cva.core.types import Disposition, Evidence, ExclusionReason, Finding, Nature, Severity

from ..seal.errors import LedgerUnreadable
from ..seal.keys import TrustRoot
from ..seal.verify import CLASS_PROFILE, VerifyFinding, VerifyReport, verify_ledger

DETECTOR_ID = "prov.ledger_verify"
VERSION = "1"

LIMITATIONS = (
    "Detects tampering AFTER a record was sealed by our pipeline; it cannot attest that an input was genuine "
    "before it entered (Modules A and D).",
    "Tail truncation and selective logging are invisible in-band; they need an independent record count "
    "(expected_count) or an external anchor.",
    "A holder of the signing key from the start can forge a wholly consistent history; anchoring and an HSM "
    "are the mitigations, and both are deployment decisions.",
    "created_at is the untrusted host clock: never used for ordering.",
)


class LedgerVerify:
    """Registrable plug-in shape (`core.registry.Registrable`): `id`, `version`, `requires`, `optional`,
    `attack_classes`. It is deliberately NOT registered in `core.registry.REGISTRY` here: that registry holds
    `ModelCheck`s, which the orchestrator calls with a model. Wiring this into the scan is Backend's call —
    see `checks/registry.py`."""

    id = DETECTOR_ID
    version = VERSION
    requires = frozenset({Capability.INFERENCE_LEDGER})
    optional = frozenset({Capability.REFERENCE_MANIFEST})      # absent -> DEGRADED: in-chain registrations only
    attack_classes = frozenset(CLASS_PROFILE) | {"ledger_verified"}

    def resolve(self, caps: CapabilitySet) -> Resolution:
        """The three-state machine, exactly as the orchestrator resolves every plug-in."""
        return caps.resolve(set(self.requires), set(self.optional))

    def verify(self, source: Any, trust_root: TrustRoot | str | os.PathLike[str], *, scan_id: str = "",
               produced_by: str = "", reference_manifest: Mapping[str, Iterable[str]] | None = None,
               input_resolver: Callable[[Mapping[str, Any]], bytes | None] | None = None,
               expected_count: int | None = None,
               expect_deployment: str | Mapping[str, Any] | None = None,
               anchors: Iterable[Any] | None = None) -> list[Finding]:
        """Verify a ledger and return findings: one per problem, then the always-present summary."""
        degraded = reference_manifest is None
        try:
            report = verify_ledger(source, trust_root=trust_root, reference_manifest=reference_manifest,
                                   input_resolver=input_resolver, expected_count=expected_count,
                                   expect_deployment=expect_deployment, anchors=anchors)
        except LedgerUnreadable as e:
            return [self._not_assessed(str(e), scan_id, produced_by)]
        out = [self._map(f, report, scan_id, produced_by) for f in report.findings]
        out.append(self._summary(report, degraded, scan_id, produced_by, len(out)))
        return out

    # -- mapping ---------------------------------------------------------------------------------------

    def _access(self, report: VerifyReport) -> list[str]:
        return [f"INFERENCE_LEDGER read from a {report.source_kind} source",
                "trust root supplied out of band (not self-signed)",
                (f"{report.anchors_verified} external anchor(s) verified; {report.unwitnessed_records} record(s) sit "
                 "after the newest one, in the unwitnessed window" if report.anchors_verified else
                 f"no external anchor verified: {report.unwitnessed_records} record(s) sit in the unwitnessed window")]

    def _map(self, f: VerifyFinding, report: VerifyReport, scan_id: str, produced_by: str) -> Finding:
        sev = Severity(f.severity)
        disposition = (Disposition.QUARANTINE if sev in (Severity.HIGH, Severity.CRITICAL)
                       else Disposition.REVIEW if sev in (Severity.MEDIUM, Severity.LOW) else Disposition.ACCEPT)
        evidence = [Evidence(kind=kind, caption=caption, data=data) for kind, caption, data in f.evidence]   # type: ignore[arg-type]
        return Finding(
            detector_id=DETECTOR_ID, detector_version=VERSION, target_type="record", target_ref=f.target_ref,
            severity=sev, confidence=1.0, reason=f.reason, attack_class=f.attack_class, scan_id=scan_id,
            produced_by=produced_by, score_raw=1.0, threshold=1.0, evidence=evidence,
            access_assumptions=self._access(report), limitations=list(LIMITATIONS),
            disposition=disposition,
            disposition_rule="prov.provisional — the risk engine owns the final disposition",
            nature=Nature(f.nature), availability=Availability.OK)

    def _summary(self, report: VerifyReport, degraded: bool, scan_id: str, produced_by: str, n_findings: int) -> Finding:
        """O9: printed whether or not anything failed."""
        problems = [f for f in report.findings if f.severity != "info"]
        stats = {
            "records_verified": report.records_checked, "checkpoints_verified": report.checkpoints_verified,
            "anchors_verified": report.anchors_verified, "anchors_recorded_in_chain": report.anchors_in_chain,
            "anchors_unusable": report.anchors_invalid, "records_fixed_by_anchor": report.anchored_records,
            "sealed_not_after_utc": report.attested_not_after, "key_rotations": report.rotations,
            "declared_degraded_intervals": report.declared_gaps,
            "records_after_last_anchor": report.unwitnessed_records,
            "unwitnessed_window": ("records after the newest verified EXTERNAL anchor" if report.anchors_verified
                                   else "no external anchor was verified: records after the last in-chain anchor "
                                        "event, which the key holder wrote itself"),
            "payloads_checked": report.payloads_checked, "payloads_missing": report.payloads_missing,
            "durability": report.durability or "not recorded in this source",
            "loss_window": report.loss_window or "not recorded in this source",
            "custody": "not recorded in the ledger — stated by the deployment, not provable from it",
            "findings_above_info": len(problems), "verifier_limitations": list(report.limitations),
        }
        head = ("Ledger verified" if not problems else f"Ledger checked, {len(problems)} problem(s) found")
        reason = (f"{head}: {report.records_checked} record(s) checked, {report.checkpoints_verified} Merkle "
                  f"checkpoint(s) recomputed, {report.declared_gaps} declared degraded interval(s), "
                  f"{report.anchors_verified} external anchor(s) verified. {report.unwitnessed_records} record(s) "
                  "lie after the last anchor — that window still trusts the key holder. Durability: "
                  f"{stats['durability']} (loss window {stats['loss_window']}).")
        limits = list(LIMITATIONS)
        if degraded:
            limits.append("No REFERENCE_MANIFEST: model changes are visible only as in-chain re-registrations, "
                          "reported as information — the ledger alone cannot tell a reload from a swap.")
        limits += [f"Not verified: {x}" for x in report.limitations]
        return Finding(
            detector_id=DETECTOR_ID, detector_version=VERSION, target_type="record", target_ref="ledger",
            severity=Severity.INFO, confidence=1.0, reason=reason, attack_class="ledger_verified", scan_id=scan_id,
            produced_by=produced_by, score_raw=1.0, threshold=1.0,
            evidence=[Evidence(kind="table", caption="what was checked", data=stats)],
            access_assumptions=self._access(report), limitations=limits, disposition=Disposition.ACCEPT,
            disposition_rule="prov.summary — information only", nature=Nature.INDETERMINATE,
            availability=Availability.DEGRADED if degraded else Availability.OK)

    def _not_assessed(self, why: str, scan_id: str, produced_by: str) -> Finding:
        return Finding(
            detector_id=DETECTOR_ID, detector_version=VERSION, target_type="record", target_ref="ledger",
            severity=Severity.MEDIUM, confidence=1.0,
            reason=f"The inference ledger could not be opened or read, so nothing was assessed: {why}. This is not "
                   "evidence of tampering and not evidence of integrity — the assessment was not performed.",
            attack_class="not_assessed", scan_id=scan_id, produced_by=produced_by, score_raw=1.0, threshold=1.0,
            evidence=[Evidence(kind="json", caption="why the ledger was not assessed", data={"error": why})],
            access_assumptions=["INFERENCE_LEDGER was supplied but could not be read"],
            limitations=["This attack class family (record tampering) was NOT assessed in this scan."],
            disposition=Disposition.REVIEW, disposition_rule="capability.unavailable", nature=Nature.INDETERMINATE,
            exclusion_reason=ExclusionReason.CAPABILITY, availability=Availability.UNAVAILABLE)
