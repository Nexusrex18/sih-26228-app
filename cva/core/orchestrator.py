"""Scan orchestration: assemble the CapabilitySet, resolve every check BEFORE running
anything, print the plan, then run.

Two properties fall out of resolving first. Coverage is known at minute zero rather than
at the end of a long scan. And a check that raises is caught as ERROR — a bug in our code —
never folded into DEGRADED, because collapsing them lets defects hide inside what looks
like an honest coverage gap.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from cva.core.access_block import render_access_block
from cva.core.capability import Availability, CapabilitySet, Resolution, budget_excluded
from cva.core.context import CheckContext
from cva.core.ledger import scan_record
from cva.core.profile import CALIBRATION, EMBED_IMG_PER_S, POLICIES, PROFILES, TIERS
from cva.core.registry import DETECTOR_REGISTRY, REGISTRY
from cva.core.runcontext import RunContext
from cva.core.scanid import (
    SELFTEST_EPOCH,
    EvidenceStore,
    materialise_evidence,
    new_scan_id,
    selftest_scan_id,
)
from cva.core.types import Disposition, Evidence, Finding, Severity
from cva.risk.disposition import default_policy
from cva.risk.engine import RiskOutcome, assess, reference_flag_counts


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
    # What the report needs and only the orchestrator knows (B5).
    created_at_utc: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    target: dict[str, Any] = field(default_factory=dict)
    # Filled by the risk engine (B7); empty means "not computed", never "nothing found".
    contributor_risk: list[dict[str, Any]] = field(default_factory=list)
    permutation_test: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    # B7's absolute rate against a reference dataset. None with no reference supplied, and
    # None with `contributor_baseline_unavailable` saying why when one was supplied but unusable.
    contributor_baseline: dict[str, Any] | None = None
    contributor_baseline_unavailable: str | None = None


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _created_at(prof: dict[str, Any]) -> str:
    """The pinned instant under a profile that pins the clock (V10), else the wall clock."""
    if prof.get("pin_clock"):
        return SELFTEST_EPOCH.isoformat()
    return datetime.now(UTC).isoformat()


def _estimated_costs(ctx: RunContext, registries: Registries | None) -> dict[str, str]:
    """What a budget exclusion costs to lift, at THIS dataset's size.

    The dominant cost of a `data.*` check is the shared embedding pass, so that is what is
    priced: n images at the measured CPU throughput. It is an estimate of the pass, shared
    across the data.* checks, not a per-check timing, and the string says so.

    A model check's cost depends on this model's inference speed and size, which is not
    measured anywhere, so it gets an honest "not estimated" string and never an invented
    number. Every id gets SOME string, so a budget exclusion is never printed without one.

    These strings embed the dataset size, so they go to `build_plan(costs=...)` and NEVER
    into the profile: the profile hash must be a function of the profile alone.
    """
    if ctx.dataset is None:
        data_cost = "not estimated: no dataset was supplied to size the embedding pass"
    else:
        n = len(ctx.dataset.samples)
        secs = n / EMBED_IMG_PER_S
        span = f"{secs:.0f} s" if secs < 120 else f"{secs / 60:.1f} min"
        data_cost = (f"~{span} for the shared embedding pass over {n} images "
                     f"(CPU, {EMBED_IMG_PER_S} img/s)")
    model_cost = "not estimated: depends on this model's inference speed and size"
    regs = default_registries() if registries is None else registries
    return {cid: data_cost if cid.startswith("data.") else model_cost
            for reg in regs for cid in reg}


def _target_of(model: Any, ctx: RunContext) -> dict[str, Any]:
    """What was scanned. An absent fact is an absent key, never an empty string."""
    t: dict[str, Any] = {}
    if model is not None:
        t["model_id"] = getattr(model, "model_id", "-")
        t["model_format"] = getattr(model, "fmt", "-")
        opset = getattr(model, "opset", None)
        if isinstance(opset, dict):           # ONNX: {domain: version}; the default domain
            opset = opset.get("ai.onnx", opset.get(""))
        if isinstance(opset, int):
            t["model_opset"] = opset
        try:
            digest = model.weight_digest()
        except Exception:
            digest = None
        # A frozen TorchScript archive returns "unavailable:frozen": omit it, never emit it.
        if isinstance(digest, str) and _HEX64.match(digest):
            t["model_sha256"] = digest
        # The structure digest (a method on the file loaders; a query-only adapter has none).
        try:
            arch = getattr(model, "arch_hash", None)
            arch = arch() if callable(arch) else arch
        except Exception:
            arch = None
        if isinstance(arch, str) and _HEX64.match(arch):
            t["arch_hash"] = arch
        # The DECLARED preprocessing spec (`cva.loaders.preprocess`), attached as plain
        # attributes by the loader layer. Absent when none was declared: never an empty string.
        pre_hash = getattr(model, "preprocess_hash", None)
        pre_ref = getattr(model, "preprocess_ref", None)
        if isinstance(pre_hash, str) and _HEX64.match(pre_hash):
            t["preprocess_hash"] = pre_hash
        if isinstance(pre_ref, str) and pre_ref:
            t["preprocess_ref"] = pre_ref
    if ctx.dataset is not None:
        t["n_samples"] = len(ctx.dataset.samples)
        t["n_categories"] = len(ctx.dataset.categories)
        # The loader's own name for the layout. `dataset_path` is deliberately NOT recorded:
        # selftest builds its fixtures under a different temp dir each run, and a path would
        # make V10's two-run diff fail on a correct run.
        samples = ctx.dataset.samples
        fmt = (getattr(samples[0], "source_meta", None) or {}).get("format") if samples else None
        if isinstance(fmt, str) and fmt:
            t["dataset_format"] = fmt
    return t


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
               registries: Registries | None = None,
               model_present: bool = True,
               costs: dict[str, str] | None = None) -> list[PlanRow]:
    """Resolve every registered check BEFORE anything runs, so coverage is known at
    minute zero. Budget exclusion is a SECOND axis, not a fifth Availability state.

    `costs` maps check_id -> the estimated cost printed beside a budget exclusion. It is a
    parameter, not a profile key, because the strings embed the dataset size and a profile
    that carried them would hash differently per dataset. `None` keeps the older behaviour
    of reading `prof["estimated_cost"]`.

    `model_present=False` (a dataset-only scan) makes every MODEL check unavailable. Some
    declare no required capability at all — `model.weight_digest` says "file access only" —
    so capability resolution alone would call them runnable and they would run against
    nothing. The first registry is the model checks; the second is the data detectors.
    """
    registries = default_registries() if registries is None else registries
    enabled = prof.get("checks")
    except_checks = set(prof.get("except_checks") or ())
    disabled = set(prof.get("disabled_checks") or ())
    cost_of: dict[str, str] = costs if costs is not None else (prof.get("estimated_cost") or {})
    rows: list[PlanRow] = []
    for reg_index, reg in enumerate(registries):
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
                elif reg_index == 0 and not model_present:
                    rows.append(PlanRow(cid, Resolution(
                        Availability.UNAVAILABLE,
                        "no model was supplied to this scan, so a model check has nothing "
                        "to inspect", (), exclusion_reason="capability"), classes))
                elif (enabled is not None and cid not in enabled) or cid in except_checks:
                    rows.append(PlanRow(cid, budget_excluded(
                        prof.get("budget_tier", profile_name),
                        cost_of.get(cid)), classes))
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


class InvalidProfile(ValueError):
    """A resolved profile that does not validate against profile.schema.json.

    Unknown KEYS are a load error for the same reason unknown NAMES are: a typo'd threshold
    silently runs at the default and the operator believes they tightened something.
    """


class UnknownCheckId(KeyError):
    """A profile names a check id that no registry holds.

    An id that matches nothing does not fail: it just disables (or enables) nothing, which is
    how the blackbox policy came to name `model.weight_statistics` and go on running the
    white-box checks it claimed to disable.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


_PROFILE_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "profile.schema.json"


@lru_cache(maxsize=1)
def _profile_validator() -> Any:
    # importlib, not `from jsonschema import ...`: no stubs are installed, and a
    # `type: ignore` would itself error (warn_unused_ignores) on a machine that has them.
    jsonschema = importlib.import_module("jsonschema")
    return jsonschema.Draft202012Validator(json.loads(_PROFILE_SCHEMA_PATH.read_text()))


def validate_profile_schema(prof: dict[str, Any], profile_name: str) -> None:
    """Validate against schemas/profile.schema.json; raise `InvalidProfile` on any error.

    The Python tables carry no `schema_version`, `name` or `thresholds` (the schema's file
    format requires all three), so a validation-only envelope supplies them. It is NOT put
    into the returned profile, so the profile hash does not change. Sets become sorted lists
    (`_jsonable`), which is what the schema's `array` means.
    """
    instance = {"schema_version": "1.0.0", "name": profile_name, "thresholds": {},
                **_jsonable(prof)}
    errors = sorted(_profile_validator().iter_errors(instance),
                    key=lambda e: [str(p) for p in e.absolute_path])
    if errors:
        shown = "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors[:5])
        more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        raise InvalidProfile(
            f"profile {profile_name!r} does not validate against profile.schema.json: "
            f"{shown}{more}")


def validate_profile_ids(prof: dict[str, Any], registries: Registries | None = None) -> None:
    """Every id in `checks`, `disabled_checks` and `except_checks` must be registered.

    Called from the ENTRYPOINT (`execute_scan`), where every registry is imported, and never
    from `scan()` or `build_plan`: the zero-detector gate passes empty registries with a real
    profile and that has to keep working.
    """
    regs = default_registries() if registries is None else registries
    known = {cid for reg in regs for cid in reg}
    bad = {key: sorted(set(prof.get(key) or ()) - known)
           for key in ("checks", "disabled_checks", "except_checks")}
    bad = {key: ids for key, ids in bad.items() if ids}
    if bad:
        detail = "; ".join(f"{key}: {', '.join(ids)}" for key, ids in bad.items())
        raise UnknownCheckId(
            f"profile names check ids that are in no registry ({detail}); "
            f"registered: {', '.join(sorted(known)) or '(none)'}")


def resolve_profile(profile_name: str, budget_tier: str | None = None) -> dict[str, Any]:
    """Merge the two axes: tier first, policy on top, explicit --budget-tier wins.

    The result is validated against profile.schema.json BEFORE anything is injected at run
    time. `scan()` later merges `ctx.profile` into it — a per-run input, not profile content,
    and not validated here. Nothing else may write to the returned dict once `scan()` has
    hashed it: the intrinsic-probe ranking used to be written back here and that made
    `ScanResult.profile` disagree with `ScanResult.profile_hash`. Per-run derived data now
    goes to `CheckContext.run_state`, which nothing hashes.
    """
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
    # The disposition table and calibration policy live in the profile (plan §7.9), so an
    # organisation changes them without touching code. A caller's own block wins.
    prof.setdefault("disposition", default_policy())
    prof.setdefault("calibration", dict(CALIBRATION))
    if policy.get("disabled_checks") and prof.get("checks") is None:
        prof["checks"] = None      # resolved against the registry in build_plan
    validate_profile_schema(prof, profile_name)
    return prof


def scan(model: Any, ctx: RunContext, profile_name: str = "deep",
         dry_run: bool = False, registries: Registries | None = None,
         plan_sink: Callable[[str], None] | None = None,
         budget_tier: str | None = None) -> ScanResult:
    prof: dict[str, Any] = resolve_profile(profile_name, budget_tier)
    prof.update(ctx.profile)
    # Not `prof["estimated_cost"]`: the strings embed the dataset size, and anything in
    # `prof` is hashed, so the same profile would get a different profile_hash per dataset.
    costs = _estimated_costs(ctx, registries)

    # Keyed on the PROFILE'S OWN FLAG, not on `profile_name == "selftest"`. The string
    # compare works today only because `selftest` happens to be the one profile that sets
    # it; profile.schema.json makes `scan_id_from_seed` a property of the POLICY axis, so
    # any policy may set it and B6 would have had to unwind the compare.
    scan_id = (selftest_scan_id(ctx.seed) if prof.get("scan_id_from_seed")
               else new_scan_id())

    model_caps = model.capabilities() if model is not None else CapabilitySet()
    caps = CapabilitySet.union(model_caps, ctx.capabilities())
    plan = build_plan(caps, prof, profile_name, registries, model_present=model is not None,
                      costs=costs)
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
                          profile_name, phash, ctx.code_commit, access,
                          created_at_utc=_created_at(prof), profile=prof, seed=ctx.seed,
                          target=_target_of(model, ctx))

    model_id = getattr(model, "model_id", "-")
    check_registry, detector_registry = (registries or default_registries())[:2]

    # --- intrinsic ranking feeds Neural Cleanse's top-K -------------------
    findings: list[Finding] = []
    timings: dict[str, float] = {}
    ordered = sorted(plan, key=lambda r: 0 if r.check_id == "model.intrinsic_probes" else 1)

    embeddings: Any = None
    embedding_gap: str | None = None
    embeddings_built = False
    # Per-run derived data, threaded to the checks that consume it. Deliberately NOT `prof`:
    # `phash` is taken above, before the first check runs, and a check writing back into the
    # hashed dict made `profile_hash_of(result.profile) != result.profile_hash` — the seal
    # then attested a profile hash that did not match the profile shipped beside it (item 13).
    run_state: dict[str, Any] = {}
    # The reference dataset's own index, built ONCE on the same lazy trigger as the scan's
    # own (the first runnable detector row) and reused by the reference pass below, so a
    # scan with no data detectors still never loads a backbone.
    reference_embeddings: Any = None
    reference_embedding_gap: str | None = None

    for row in ordered:
        if not row.resolution.runnable:
            findings.append(_plan_finding(row, scan_id, model_id, profile_name))
            continue
        is_detector = row.check_id not in check_registry
        if is_detector and row.check_id not in detector_registry:
            continue
        inst = (detector_registry if is_detector else check_registry)[row.check_id]()
        if is_detector and not embeddings_built:
            embeddings, embedding_gap = build_embeddings(ctx, prof)
            if ctx.reference_dataset is not None:
                reference_embeddings, reference_embedding_gap = build_embeddings(
                    replace(ctx, dataset=ctx.reference_dataset), prof)
            embeddings_built = True
        cctx = CheckContext(ctx.probes_x, ctx.probes_y, ctx.suspect_x, ctx.battery,
                            prof, scan_id, ctx.out_dir, ctx.seed, run_state=run_state,
                            reference_embeddings=reference_embeddings)
        t0 = time.time()
        findings.extend(_run_check(row, inst, is_detector, ctx.dataset, embeddings,
                                   embedding_gap, model, cctx, scan_id, model_id))
        timings[row.check_id] = round(time.time() - t0, 2)

    for f in findings:
        if not f.scan_id:
            f.scan_id = scan_id
        if not f.produced_by or f.produced_by.startswith("ranking="):
            f.produced_by = f"{ctx.code_commit}/profile:{phash[:12]}"

    # Idempotent: only entries with a payload and no path are written. The report then
    # records hashes that exist on disk instead of transport payloads.
    if ctx.out_dir is not None:
        materialise_evidence(findings, EvidenceStore(ctx.out_dir))

    # B7's absolute rate: the same data detectors over a second, known-clean dataset, of which
    # only the flag count survives. Its findings never enter `findings` or the coverage.
    reference: dict[str, int] | None = None
    reference_gap: str | None = None
    if ctx.reference_dataset is not None:
        reference, reference_gap = _reference_flags(
            model, ctx, prof, plan, check_registry, detector_registry, scan_id, model_id,
            embedding_gap, reference_embeddings, reference_embedding_gap, run_state)
    risk: RiskOutcome = assess(findings, ctx.dataset, prof, ctx.seed, ctx.calibration,
                               reference, reference_gap)
    result = ScanResult(scan_id, model_id, getattr(model, "fmt", "-"), caps, plan,
                        findings, timings,
                        _verdict(findings, risk), profile_name, phash, ctx.code_commit, access,
                        created_at_utc=_created_at(prof), profile=prof, seed=ctx.seed,
                        target=_target_of(model, ctx), contributor_risk=risk.contributor_risk,
                        permutation_test=risk.permutation_test, calibration=risk.calibration,
                        contributor_baseline=risk.contributor_baseline,
                        contributor_baseline_unavailable=risk.contributor_baseline_unavailable)
    # Sealing is NOT done here: the scan record binds the sha256 of report.json, which does
    # not exist yet. The caller writes the report, then calls `seal_report` (plan §7.9).
    return result


def _run_check(row: PlanRow, inst: Any, is_detector: bool, dataset: Any, embeddings: Any,
               embedding_gap: str | None, model: Any, cctx: CheckContext,
               scan_id: str, model_id: str) -> list[Finding]:
    """Run ONE resolved check and return its findings; a check that raises becomes an ERROR
    finding, never DEGRADED. Shared by the scan proper and the reference-dataset pass so both
    treat DEGRADED rows and a missing embedding index identically."""
    got: list[Finding] = []
    try:
        got = (inst.detect(dataset, embeddings, model, cctx) if is_detector
               else inst.check(model, cctx))
        if is_detector and embedding_gap is not None:
            for f in got:
                f.limitations.append(
                    f"No embedding index was available for this scan: {embedding_gap}")
        for f in got:
            if f.availability == Availability.OK and \
                    row.resolution.state == Availability.DEGRADED:
                f.availability = Availability.DEGRADED
                f.limitations.append(f"Ran DEGRADED: {row.resolution.reason}")
        if row.check_id == "model.intrinsic_probes":
            for f in got:
                if f.produced_by.startswith("ranking="):
                    # `cctx.run_state`, never `prof`: see the run_state comment in `scan()`.
                    cctx.run_state["nc_class_order"] = json.loads(
                        f.produced_by.split("=", 1)[1])
        return got
    except Exception as exc:                           # ERROR, never DEGRADED
        return [*got, Finding(
            detector_id=row.check_id, detector_version="?", scan_id=scan_id,
            target_type="dataset" if is_detector else "model",
            target_ref="dataset" if is_detector else model_id,
            severity=Severity.MEDIUM, confidence=0.0,
            reason=f"Check raised {type(exc).__name__}: {exc}. This is a defect in the "
                   "assurance tool, not a property of the model.",
            attack_class="tool.error",
            evidence=[Evidence("json", "traceback",
                               data=traceback.format_exc().splitlines()[-6:])],
            limitations=["This attack class was NOT assessed — the check failed."],
            disposition=Disposition.REVIEW, disposition_rule="tool.error",
            availability=Availability.ERROR)]


def _reference_flags(model: Any, ctx: RunContext, prof: dict[str, Any], plan: list[PlanRow],
                     check_registry: dict[str, Any], detector_registry: dict[str, Any],
                     scan_id: str, model_id: str, cohort_gap: str | None,
                     ref_embeddings: Any, ref_gap: str | None,
                     run_state: dict[str, Any]) -> tuple[dict[str, int] | None, str | None]:
    """Run the scan's own runnable `data.*` detectors over `ctx.reference_dataset` and return
    `({"reference_n", "reference_flagged"}, None)`, or `(None, why)`.

    Same profile, its own embedding index, the same `embedding_max_images` ceiling. Only the
    flag count leaves this function: the reference findings are never reported and never count
    toward coverage. A rate is comparable only when both datasets got the same treatment, so a
    reference over the ceiling, an embedding index that could not be built, a detector that
    raised, or a scanned dataset that had no index all give `None` and a reason, never a rate
    computed from a run that quietly did less than the scan did.
    """
    ref = ctx.reference_dataset
    if not getattr(ref, "samples", None):
        return None, "the reference dataset has no samples"
    rows = [r for r in plan if r.resolution.runnable
            and r.check_id not in check_registry and r.check_id in detector_registry]
    if not rows:
        return None, ("no data.* check was runnable in this scan, so there was nothing to "
                      "run over the reference dataset")
    if cohort_gap is not None:
        return None, (f"the scanned dataset had no embedding index ({cohort_gap}), so its flag "
                      "rate is not comparable with a reference run that has one")
    # `ref_embeddings is None` with no gap is a legitimate state — a scan whose detectors
    # need no index — so only a stated gap disqualifies the reference. The index itself is
    # built once, in `scan()`'s loop, on the same lazy trigger as the cohort's.
    if ref_gap is not None:
        return None, f"the reference dataset was not embedded: {ref_gap}"
    cctx = CheckContext(ctx.probes_x, ctx.probes_y, ctx.suspect_x, ctx.battery,
                        prof, scan_id, ctx.out_dir, ctx.seed, run_state=dict(run_state),
                        reference_embeddings=ref_embeddings)
    found: list[Finding] = []
    for row in rows:
        got = _run_check(row, detector_registry[row.check_id](), True, ref, ref_embeddings,
                         None, model, cctx, scan_id, model_id)
        if any(f.availability == Availability.ERROR for f in got):
            return None, f"{row.check_id} raised while scanning the reference dataset"
        found.extend(got)
    return reference_flag_counts(found, ref, prof, ctx.calibration), None


def build_embeddings(ctx: RunContext, prof: dict[str, Any]) -> tuple[Any, str | None]:
    """Build the scan's embedding index, or return `(None, why)`.

    Called LAZILY — on the first runnable detector row, never at scan start. The P0 gate
    runs with zero detectors registered, and loading an 88 MB backbone to feed nothing
    would put a minute and a torch import on the one path that has no use for either.

    A backbone that will not load is not fatal: the detectors already treat
    `embeddings is None` as `not_performed`, which is the honest state. What they cannot
    know is WHY, so the reason travels back with the None.
    """
    if ctx.dataset is None:
        return None, "no dataset supplied to this scan"
    ceiling = prof.get("embedding_max_images")
    n = len(ctx.dataset.samples)
    if ceiling is not None and n > ceiling:
        return None, (f"budget tier '{prof.get('budget_tier')}' embeds at most {ceiling} "
                      f"images and this dataset has {n}")
    try:
        from cva.features.cache import FeatureCache
        from cva.features.embed import build_extractor, embed_dataset
    except Exception as exc:
        return None, f"cva.features unimportable ({type(exc).__name__}: {exc})"
    try:
        extractor = build_extractor(prof.get("extractor"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    cache = FeatureCache(enabled=bool(prof.get("feature_cache", True)))
    return embed_dataset(ctx.dataset, extractor, cache), None


def _model_detail(model: Any) -> str | None:
    if model is None:
        return None
    opset = getattr(model, "opset", None)
    if isinstance(opset, dict):               # ONNX: {domain: version}; show the default domain
        opset = opset.get("ai.onnx", opset.get(""))
    return f"{model.fmt} (opset {opset})" if opset else model.fmt


def seal_report(result: ScanResult, ctx: RunContext, report_path: Path) -> str | None:
    """The orchestrator's last step (plan §7.9): seal a report that is ALREADY WRITTEN.

    Order matters and is the whole point. The file is fsynced, then hashed from the bytes
    that are on disk, then the scan record binds that digest and `ledger_seq` is set on the
    in-memory result. The returned seq goes to the log and to the HTML only: `report.json`
    was built before it existed and is never rewritten, because rewriting it would leave the
    ledger holding a digest of a file that no longer exists.

    With `NullAuditLedger` the "seq" is its own `unsealed-N` counter (the report says the
    record is not sealed); `None` is returned only when the ledger raised.
    """
    with open(report_path, "rb") as fh:
        data = fh.read()
        os.fsync(fh.fileno())
    result.ledger_seq = append_scan_record(result, ctx, hashlib.sha256(data).hexdigest())
    return result.ledger_seq


def append_scan_record(result: ScanResult, ctx: RunContext,
                       report_sha256: str = "") -> str | None:
    """Append the scan record. NullAuditLedger is the default, and the report then says
    honestly that the scan record was not sealed."""
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


def _verdict(findings: list[Finding], risk: RiskOutcome | None = None) -> str:
    """The verdict counts what RAN, plus every check that CRASHED.

    `UNAVAILABLE` stays out on purpose: a declared gap is carried by the coverage statement,
    and folding it in would make every dataset scan REVIEW forever (the model-side checks are
    unavailable by construction). `ERROR` is not a declared gap — it is a defect in this tool,
    the plan's "never a silent skip" — so it forces REVIEW. The risk engine never routes an
    ERROR finding (`apply_dispositions` skips anything that did not run), so the availability
    is tested here directly rather than trusting a disposition nothing set.
    """
    real = [f for f in findings if f.availability in (Availability.OK, Availability.DEGRADED)]
    # D4: a contributor whose posterior clears the cohort is a QUARANTINE of the contribution.
    if any(f.disposition == Disposition.QUARANTINE for f in real) or (
            risk is not None and risk.quarantined_contributors):
        return "QUARANTINE"
    if any(f.availability == Availability.ERROR for f in findings):
        return "REVIEW"
    if any(f.disposition == Disposition.REVIEW and f.severity.rank >= Severity.MEDIUM.rank
           for f in real):
        return "REVIEW"
    return "ACCEPT"
