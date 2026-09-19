"""Scan orchestration: assemble the CapabilitySet, resolve every check BEFORE running
anything, print the plan, then run.

Two properties fall out of resolving first. Coverage is known at minute zero rather than
at the end of a long scan. And a check that raises is caught as ERROR — a bug in our code —
never folded into DEGRADED, because collapsing them lets defects hide inside what looks
like an honest coverage gap.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cva.core.access_block import render_access_block
from cva.core.capability import Availability, CapabilitySet, Resolution, budget_excluded
from cva.core.context import CheckContext
from cva.core.ledger import scan_record
from cva.core.profile import POLICIES, PROFILES, TIERS
from cva.core.registry import DETECTOR_REGISTRY, REGISTRY
from cva.core.runcontext import RunContext
from cva.core.scanid import new_scan_id, selftest_scan_id
from cva.core.types import Disposition, Evidence, Finding, Severity


@dataclass
class PlanRow:
    check_id: str
    resolution: Resolution
    attack_classes: set[str]


@dataclass
class ScanResult:
    scan_id: str
    model_id: str
    model_fmt: str
    capabilities: CapabilitySet
    plan: list[PlanRow]
    findings: list[Finding]
    timings: dict[str, float]
    verdict: str
    profile_name: str = "deep"
    profile_hash: str = ""
    code_commit: str = "unknown"
    access_assumptions: str = ""
    ledger_seq: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)


def profile_hash_of(prof: dict[str, Any]) -> str:
    from cva.core.quantise import canonical_bytes
    return hashlib.sha256(canonical_bytes(_jsonable(prof))).hexdigest()


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (set, frozenset)):
        return sorted(str(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


Registries = tuple[dict[str, Any], ...]


def default_registries() -> Registries:
    """Resolved at CALL time, never bound as a default argument.

    A tuple default would capture the two dict OBJECTS at import, so a test that rebinds
    `orchestrator.REGISTRY` would still plan against the real one — the zero-detector gate
    would silently pass while resolving every Module B check.
    """
    return (REGISTRY, DETECTOR_REGISTRY)


def build_plan(caps: CapabilitySet, prof: dict[str, Any], profile_name: str,
               registries: Registries | None = None) -> list[PlanRow]:
    """Resolve every registered check BEFORE anything runs, so coverage is known at
    minute zero. Budget exclusion is a SECOND axis, not a fifth Availability state."""
    registries = default_registries() if registries is None else registries
    enabled = prof.get("checks")
    disabled = set(prof.get("disabled_checks") or ())
    rows: list[PlanRow] = []
    for reg in registries:
        for cid, cls in sorted(reg.items()):
            inst: Any = cls()
            classes = set(inst.attack_classes)
            if cid in disabled:
                rows.append(PlanRow(cid, Resolution(
                    Availability.UNAVAILABLE,
                    f"disabled by policy '{profile_name}' — not asked to run, as distinct "
                    f"from could not run", (), exclusion_reason="budget"), classes))
            else:
                res = caps.resolve(set(inst.requires), set(inst.optional))
                # Capability BEFORE budget, deliberately. A check that is both
                # tier-excluded and missing a required capability reported
                # exclusion_reason="budget" before this change, which tells the operator
                # "raise the tier and this runs" — and it does not. The honest fact is
                # that it could not have run at any tier. Budget exclusion is only
                # reported for checks that would otherwise have been runnable.
                if res.state is Availability.UNAVAILABLE:
                    rows.append(PlanRow(cid, res, classes))
                elif enabled is not None and cid not in enabled:
                    rows.append(PlanRow(cid, budget_excluded(
                        prof.get("budget_tier", profile_name),
                        prof.get("estimated_cost", {}).get(cid)), classes))
                else:
                    rows.append(PlanRow(cid, res, classes))
    return sorted(rows, key=lambda r: r.check_id)


def plan_text(result_or_plan: Any) -> str:
    """V7's artefact: the plan, printed BEFORE any work starts."""
    plan = getattr(result_or_plan, "plan", result_or_plan)
    lines = ["Plan"]
    for row in plan:
        lines.append(f"  {row.check_id:34} {row.resolution.state.value:12} "
                     f"{row.resolution.reason}")
    if not plan:
        lines.append("  (no checks registered)")
    return "\n".join(lines)


class UnknownProfile(KeyError):
    """An unknown --profile is a LOAD ERROR, never a silent fall back to the default.

    `PROFILES.get(name, PROFILES["deep"])` is precisely the false assurance this system
    exists to prevent: an operator who typos `--profile stirct` believes they tightened
    something, and gets `deep` with no signal at all. Same rule the profile schema states
    for unknown KEYS (`unevaluatedProperties: false`); this is it for unknown NAMES.
    """


def resolve_profile(profile_name: str, budget_tier: str | None = None) -> dict[str, Any]:
    """Merge the two axes: tier first, policy on top, explicit --budget-tier wins."""
    if profile_name not in PROFILES:
        raise UnknownProfile(
            f"unknown profile {profile_name!r}; known: {', '.join(sorted(PROFILES))}")
    policy = POLICIES.get(profile_name, {})
    tier_name = budget_tier or policy.get("budget_tier") or profile_name
    if tier_name not in TIERS:
        raise UnknownProfile(
            f"unknown budget tier {tier_name!r}; known: {', '.join(sorted(TIERS))}")
    prof: dict[str, Any] = dict(TIERS[tier_name])
    prof.update(policy)
    prof["budget_tier"] = tier_name
    if policy.get("disabled_checks") and prof.get("checks") is None:
        prof["checks"] = None      # resolved against the registry in build_plan
    return prof


def scan(model: Any, ctx: RunContext, profile_name: str = "deep",
         dry_run: bool = False, registries: Registries | None = None,
         plan_sink: Callable[[str], None] | None = None,
         budget_tier: str | None = None) -> ScanResult:
    prof: dict[str, Any] = resolve_profile(profile_name, budget_tier)
    prof.update(ctx.profile)

    # Keyed on the PROFILE'S OWN FLAG, not on `profile_name == "selftest"`. The string
    # compare works today only because `selftest` happens to be the one profile that sets
    # it; profile.schema.json makes `scan_id_from_seed` a property of the POLICY axis, so
    # any policy may set it and B6 would have had to unwind the compare.
    scan_id = (selftest_scan_id(ctx.seed) if prof.get("scan_id_from_seed")
               else new_scan_id())

    model_caps = model.capabilities() if model is not None else CapabilitySet()
    caps = CapabilitySet.union(model_caps, ctx.capabilities())
    plan = build_plan(caps, prof, profile_name, registries)
    phash = profile_hash_of(prof)
    access = render_access_block(
        caps,
        model_fmt=getattr(model, "fmt", None),
        model_detail=_model_detail(model),
        plan=plan)

    # Coverage at minute zero: the plan goes out BEFORE the first check runs, not after
    # the scan that may take an hour. V7 asserts the ordering, not merely the content.
    if plan_sink is not None:
        plan_sink(access + "\n\n" + plan_text(plan))

    if dry_run:
        return ScanResult(scan_id, getattr(model, "model_id", "-"),
                          getattr(model, "fmt", "-"), caps, plan, [], {}, "DRY-RUN",
                          profile_name, phash, ctx.code_commit, access)

    model_id = getattr(model, "model_id", "-")
    check_registry = (registries or default_registries())[0]

    # --- intrinsic ranking feeds Neural Cleanse's top-K -------------------
    findings: list[Finding] = []
    timings: dict[str, float] = {}
    ordered = sorted(plan, key=lambda r: 0 if r.check_id == "model.intrinsic_probes" else 1)

    for row in ordered:
        if not row.resolution.runnable:
            findings.append(_plan_finding(row, scan_id, model_id, profile_name))
            continue
        if row.check_id not in check_registry:
            # A runnable DETECTOR_REGISTRY row: Module A's dataset path runs it, not this
            # loop. B7 obligation — `coverage_of()` counts a runnable row as ASSESSED, so
            # the moment Module A's registry is imported the report claims dataset
            # coverage nothing produced. Invisible at B3 (zero detectors); must be closed
            # when the coverage generator lands.
            continue
        inst = check_registry[row.check_id]()
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
                        prof["nc_class_order"] = json.loads(
                            f.produced_by.split("=", 1)[1])
        except Exception as exc:                       # ERROR, never DEGRADED
            findings.append(Finding(
                detector_id=row.check_id, detector_version="?", scan_id=scan_id,
                target_type="model", target_ref=model_id,
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

    for f in findings:
        if not f.scan_id:
            f.scan_id = scan_id
        if not f.produced_by or f.produced_by.startswith("ranking="):
            f.produced_by = f"{ctx.code_commit}/profile:{phash[:12]}"

    result = ScanResult(scan_id, model_id, getattr(model, "fmt", "-"), caps, plan,
                        findings, timings,
                        _verdict(findings), profile_name, phash, ctx.code_commit, access)
    result.ledger_seq = append_scan_record(result, ctx)
    return result


def _model_detail(model: Any) -> str | None:
    if model is None:
        return None
    opset = getattr(model, "opset", None)
    return f"{model.fmt} (opset {opset})" if opset else model.fmt


def append_scan_record(result: ScanResult, ctx: RunContext,
                       report_sha256: str = "") -> str | None:
    """The orchestrator's last step. NullAuditLedger is the default, and the report then
    says honestly that the scan record was not sealed."""
    counts: dict[str, int] = {}
    for f in result.findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    rec = scan_record(result.scan_id, report_sha256, result.profile_hash,
                      result.code_commit, counts)
    try:
        return ctx.audit_ledger.append(rec)
    except Exception:
        return None


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
