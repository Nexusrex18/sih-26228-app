"""Drift orchestration using the frozen Dataset and DriftTest protocols.

Concrete checks are supplied by the entrypoint; core never imports detectors. No new
plug-in protocol or parallel registry is introduced.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cva.core.capability import Availability, Capability, CapabilitySet, Resolution
from cva.core.interfaces import DriftTest
from cva.core.orchestrator import PlanRow, ScanResult
from cva.core.scanid import validate_scan_id
from cva.core.taxonomy import TAXONOMY
from cva.core.types import (
    Dataset,
    ExclusionReason,
    Finding,
    Nature,
    Severity,
    unavailable_finding,
)
from cva.risk.drift import apply_policy


@dataclass
class DriftScanResult(ScanResult):
    asset_kind: str = 'dataset'
    relevant_capabilities: tuple[Capability, ...] = ()
    drift_summary: dict[str, Any] = field(default_factory=dict)
    report_dir: str = ''


def scan_drift(reference: Dataset | None, incoming: Dataset, *,
               checks: Sequence[DriftTest], scan_id: str, min_samples: int = 20,
               reference_dist: Any = None, incoming_dist: Any = None) -> DriftScanResult:
    validate_scan_id(scan_id)
    caps = incoming.capabilities()
    notes = list(caps.notes)
    have = set(caps.caps)
    if reference is not None and Capability.DATASET_IMAGES in reference.capabilities():
        have.add(Capability.REFERENCE_CLEAN_SET)
    else:
        notes.append((Capability.REFERENCE_CLEAN_SET,'No usable declared reference distribution supplied'))
    caps = CapabilitySet(frozenset(have),tuple(notes))
    target = hashlib.sha256('␟'.join(s.content_sha256 for s in incoming.samples).encode()).hexdigest()[:20]
    overlap = bool(reference is not None and
                   {s.content_sha256 for s in reference.samples} &
                   {s.content_sha256 for s in incoming.samples})
    plan: list[PlanRow] = []
    findings: list[Finding] = []
    timings: dict[str,float] = {}
    relevant: set[Capability] = set()
    seen: set[str] = set()
    # Resolve declared capabilities for ALL checks before executing any.
    for check in checks:
        if check.id in seen or not check.attack_classes or not set(check.attack_classes) <= TAXONOMY.keys():
            raise ValueError(f'Duplicate check or undeclared taxonomy: {check.id}')
        seen.add(check.id)
        relevant.update(check.requires | check.optional)
        resolution = caps.resolve(set(check.requires),set(check.optional))
        if resolution.runnable and overlap:
            resolution = Resolution(Availability.UNAVAILABLE,
                'Reference and incoming share image content; independent two-sample assessment is invalid',
                exclusion_reason='capability')
        elif resolution.runnable and (reference is None or min(len(reference.samples),len(incoming.samples)) < min_samples):
            resolution = Resolution(Availability.UNAVAILABLE,
                f'Need at least {min_samples} samples in each batch', exclusion_reason='capability')
        plan.append(PlanRow(check.id,resolution,set(check.attack_classes)))
    for check,row in zip(checks,plan,strict=True):
        start = time.perf_counter()
        if not row.resolution.runnable:
            got = [unavailable_finding(check.id,check.version,target,row.resolution.reason,
                    row.resolution.missing,ac,target_type='batch') for ac in sorted(check.attack_classes)]
        else:
            try:
                assert reference is not None
                got = check.assess(reference,incoming,reference_dist,incoming_dist)
                if not got:
                    raise RuntimeError('Drift check returned no assessment')
                for f in got:
                    if f.attack_class not in check.attack_classes:
                        raise ValueError(f'Undeclared emitted attack class: {f.attack_class}')
                    if f.availability == Availability.OK and row.resolution.state == Availability.DEGRADED:
                        f.availability = Availability.DEGRADED
                        f.limitations.append(row.resolution.reason)
                states = {f.availability for f in got}
                state = next((s for s in (Availability.ERROR,Availability.UNAVAILABLE,Availability.DEGRADED)
                              if s in states),Availability.OK)
                if state != Availability.OK:
                    row.resolution = Resolution(state,got[0].reason, exclusion_reason="capability" if state == Availability.UNAVAILABLE else None)
            except Exception as exc:
                reason = f'Drift check failed: {type(exc).__name__}: {exc}'
                got = [Finding(check.id,check.version,'batch',target,Severity.MEDIUM,0.,reason,
                               'tool.error',availability=Availability.ERROR,nature=Nature.INDETERMINATE)]
                row.resolution = Resolution(Availability.ERROR,reason)
        for f in got:
            f.scan_id = scan_id
            if f.availability == Availability.UNAVAILABLE:
                f.exclusion_reason = ExclusionReason.CAPABILITY
            f.access_assumptions.append('Reference representativeness and independent sampling are assumed.')
        findings.extend(got)
        timings[check.id] = round(time.perf_counter()-start,3)
    verdict = apply_policy(findings)
    return DriftScanResult(scan_id,target,'image-dataset',caps,plan,findings,timings,verdict,
                           relevant_capabilities=tuple(sorted(relevant,key=lambda c:c.value)))
