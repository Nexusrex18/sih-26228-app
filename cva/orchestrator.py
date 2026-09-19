"""Scan orchestration: assemble the CapabilitySet, resolve every check BEFORE running
anything, print the plan, then run.

Two properties fall out of resolving first. Coverage is known at minute zero rather than
at the end of a long scan. And a check that raises is caught as ERROR — a bug in our code —
never folded into DEGRADED, because collapsing them lets defects hide inside what looks
like an honest coverage gap.
"""
from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cva.core.capability import (Availability, Capability, CapabilitySet, Resolution)
from cva.core.finding import Evidence, Finding, Severity, Disposition, Nature
from cva.core.model import ModelBattery
from cva.detectors.base import REGISTRY, CheckContext
import cva.detectors.model.registry  # noqa: F401 — explicit registration, ADR-008


@dataclass
class RunContext:
    probes_x: np.ndarray | None = None
    probes_y: np.ndarray | None = None
    suspect_x: np.ndarray | None = None
    battery: ModelBattery | None = None
    profile: dict = field(default_factory=dict)
    out_dir: Path | None = None
    seed: int = 0

    def capabilities(self) -> CapabilitySet:
        caps, notes = set(), []
        if self.probes_x is not None and len(self.probes_x):
            caps.add(Capability.REFERENCE_CLEAN_SET)
        else:
            notes.append((Capability.REFERENCE_CLEAN_SET, "no clean probe set supplied"))
        if self.suspect_x is not None and len(self.suspect_x):
            caps.add(Capability.SUSPECT_INPUTS)
        else:
            notes.append((Capability.SUSPECT_INPUTS,
                          "no suspect input set supplied — only known-clean probes"))
        if self.battery and self.battery.models:
            caps.add(Capability.REFERENCE_MODEL_BATTERY)
        else:
            notes.append((Capability.REFERENCE_MODEL_BATTERY,
                          "no reference model battery supplied — organisers provide none, "
                          "so this is the expected case"))
        if self.battery and self.battery.manifest:
            caps.add(Capability.REFERENCE_MANIFEST)
        else:
            notes.append((Capability.REFERENCE_MANIFEST,
                          "no manifest registered for this model"))
        return CapabilitySet(frozenset(caps), tuple(notes))


@dataclass
class PlanRow:
    check_id: str
    resolution: Resolution
    attack_classes: set


@dataclass
class ScanResult:
    scan_id: str
    model_id: str
    model_fmt: str
    capabilities: CapabilitySet
    plan: list[PlanRow]
    findings: list[Finding]
    timings: dict
    verdict: str


PROFILES = {
    "triage":   {"checks": {"model.weight_digest", "model.graph_structure"}},
    "standard": {"checks": {"model.weight_digest", "model.graph_structure",
                            "model.behavioural_fingerprint", "model.anomalous",
                            "model.intrinsic_probes", "model.weight_statistics",
                            "model.activation_statistics"}},
    "deep":     {"checks": None, "nc_top_k": 3},      # None = everything registered
    "forensic": {"checks": None, "nc_top_k": 999, "nc_steps": 200, "nes_steps": 120},
}


def scan(model, ctx: RunContext, profile_name: str = "deep") -> ScanResult:
    scan_id = uuid.uuid4().hex[:12]
    prof = dict(PROFILES.get(profile_name, PROFILES["deep"]))
    prof.update(ctx.profile)
    enabled = prof.get("checks")

    caps = CapabilitySet.union(model.capabilities(), ctx.capabilities())

    # --- resolve everything first -----------------------------------------
    plan: list[PlanRow] = []
    for cid, cls in sorted(REGISTRY.items()):
        inst = cls()
        if enabled is not None and cid not in enabled:
            plan.append(PlanRow(cid, Resolution(
                Availability.UNAVAILABLE,
                f"excluded by budget profile '{profile_name}'"), set(inst.attack_classes)))
            continue
        plan.append(PlanRow(cid, caps.resolve(set(inst.requires), set(inst.optional)),
                            set(inst.attack_classes)))

    # --- intrinsic ranking feeds Neural Cleanse's top-K -------------------
    findings: list[Finding] = []
    timings: dict[str, float] = {}
    ordered = sorted(plan, key=lambda r: 0 if r.check_id == "model.intrinsic_probes" else 1)

    for row in ordered:
        if not row.resolution.runnable:
            findings.append(_plan_finding(row, scan_id, model.model_id, profile_name))
            continue
        inst = REGISTRY[row.check_id]()
        cctx = CheckContext(ctx.probes_x, ctx.probes_y, ctx.suspect_x, ctx.battery,
                            prof, scan_id, ctx.out_dir, ctx.seed)
        t0 = time.time()
        try:
            got = inst.check(model, cctx)
            for f in got:
                if f.availability == Availability.OK and \
                        row.resolution.state == Availability.DEGRADED:
                    f.availability = Availability.DEGRADED
                    f.limitations.append(f"Ran DEGRADED: {row.resolution.reason}")
            findings.extend(got)
            if row.check_id == "model.intrinsic_probes":
                for f in got:
                    if f.produced_by.startswith("ranking="):
                        import json as _json
                        prof["nc_class_order"] = _json.loads(
                            f.produced_by.split("=", 1)[1])
        except Exception as exc:                       # ERROR, never DEGRADED
            findings.append(Finding(
                detector_id=row.check_id, detector_version="?", scan_id=scan_id,
                target_type="model", target_ref=model.model_id,
                severity=Severity.MEDIUM, confidence=0.0,
                reason=f"Check raised {type(exc).__name__}: {exc}. This is a defect in the "
                       "assurance tool, not a property of the model.",
                attack_class="tool.error",
                evidence=[Evidence("json", "traceback",
                                   data=traceback.format_exc().splitlines()[-6:])],
                limitations=["This attack class was NOT assessed — the check failed."],
                disposition=Disposition.REVIEW, disposition_rule="tool.error",
                availability=Availability.ERROR))
        timings[row.check_id] = round(time.time() - t0, 2)

    return ScanResult(scan_id, model.model_id, model.fmt, caps, plan, findings, timings,
                      _verdict(findings))


def _plan_finding(row: PlanRow, scan_id: str, model_id: str, profile: str) -> Finding:
    return Finding(
        detector_id=row.check_id, detector_version="-", scan_id=scan_id,
        target_type="model", target_ref=model_id,
        severity=Severity.INFO, confidence=0.0,
        reason=f"Not performed: {row.resolution.reason}",
        attack_class=sorted(row.attack_classes)[0] if row.attack_classes else "unknown",
        evidence=[Evidence("json", "missing capabilities",
                           data=[str(m) for m in row.resolution.missing])],
        limitations=[f"Attack classes NOT assessed: {', '.join(sorted(row.attack_classes))}"],
        disposition=Disposition.REVIEW, disposition_rule="capability.unavailable",
        availability=row.resolution.state)


def _verdict(findings: list[Finding]) -> str:
    real = [f for f in findings if f.availability in (Availability.OK, Availability.DEGRADED)]
    if any(f.disposition == Disposition.QUARANTINE for f in real):
        return "QUARANTINE"
    if any(f.disposition == Disposition.REVIEW and f.severity.rank >= Severity.MEDIUM.rank
           for f in real):
        return "REVIEW"
    return "ACCEPT"
