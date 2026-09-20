"""The machine-readable half of the report — one of the four named PS deliverables.

Kept separate from the HTML renderer on purpose: the JSON is the contract other seats and
any third party consume (`schemas/report.schema.json`), and it must not be coupled to how a
page happens to look.

Two rules shape what is and is not in here. Nothing wall-clock-dependent may appear outside
the declared volatile paths, or V9/V10's diff fails on a correct run — which is why per-check
timings are absent from this file. And an absent fact is an absent key, never `""`.
"""
from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from cva.core import determinism
from cva.core.access_block import consequence
from cva.core.capability import Capability
from cva.core.scanid import VOLATILE_PATHS
from cva.report.coverage import STANDING_LIMITATIONS

SCHEMA_VERSION = "1.0.0"
_VERDICTS = ("ACCEPT", "REVIEW", "QUARANTINE")
_ENV_PACKAGES = ("torch", "numpy", "onnxruntime")
_NO_COMMAND = "unrecorded: this report was built outside the `cva scan` command"


def build(result, command: str | None = None) -> dict:
    prof = getattr(result, "profile", None) or {}
    target = dict(getattr(result, "target", None) or {})
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scan_id": result.scan_id,
        "created_at_utc": getattr(result, "created_at_utc", "")
        or datetime.now(UTC).isoformat(),
        "produced_by": _produced_by(result, prof),
    }
    if target:
        report["target"] = target
    if result.verdict in _VERDICTS:          # never "DRY-RUN": the schema enum is closed
        report["verdict"] = result.verdict
    report.update({
        "access_assumptions": access_assumptions_of(result),
        "plan": [_plan_row(r) for r in sorted(result.plan, key=lambda x: x.check_id)],
        "findings": [_finding(f) for f in result.findings],
        # Empty / null means "not computed", never "nothing found".
        "contributor_risk": list(getattr(result, "contributor_risk", None) or []),
        "permutation_test": getattr(result, "permutation_test", None),
        # Null when no reference dataset was supplied, or when it was supplied but unusable
        # (then `contributor_baseline_unavailable` says why).
        "contributor_baseline": getattr(result, "contributor_baseline", None),
        "provenance_summary": provenance_summary_of(result),
        "drift_summary": None,
        "calibration": getattr(result, "calibration", None),
        "coverage": {**coverage_of(result),
                     "standing_limitations": standing_limitations(
                         target, getattr(result, "embedding_gap", None))},
        "reproduction": reproduction_of(result, command),
    })
    unavailable = getattr(result, "contributor_baseline_unavailable", None)
    if unavailable:
        report["contributor_baseline_unavailable"] = unavailable
    return report


def access_assumptions_of(result) -> dict[str, Any]:
    """Every Capability, present or absent — a shortened list reads as a cleaner scan."""
    present = [c.value for c in Capability if c in result.capabilities]
    absent = [{"capability": c.value,
               "reason": result.capabilities.note_for(c) or "not available for this model"}
              for c in Capability if c not in result.capabilities]
    return {"capabilities_present": present, "capabilities_absent": absent,
            "consequence": consequence(result.plan)}


def provenance_summary_of(result) -> dict[str, Any] | None:
    """The always-present provenance section (plan §7.9): state only what is TRUE.

    No INFERENCE_LEDGER means nothing was verified and no anchor was checked, so 0 and 0
    are facts. With a ledger supplied this build still computes nothing, so those counts,
    the unwitnessed window, custody type and durability window stay absent (unknown, not
    zero). `ledger_state` is `not_sealed` only when no signing key exists; with a key the
    scan record is appended AFTER this file is hashed, so its state (and seq) is not knowable
    here and must not be written into report.json. Returns None when nothing is known.
    """
    out: dict[str, Any] = {}
    if Capability.INFERENCE_LEDGER not in result.capabilities:
        out["records_verified"] = 0
        out["anchors_checked"] = 0
    elif not any(r.check_id.startswith("prov.") for r in getattr(result, "plan", ()) or ()):
        # An inference ledger WAS supplied — the capability is present — and yet no prov.*
        # row exists, because `PROV_CHECKS` is not registered with `core.registry` and
        # `LedgerVerify` has no `check(model, ctx)` for the orchestrator to call (item 7,
        # Module C's half). Without this the report would show the capability satisfied and
        # simply have no provenance rows, which reads as "nothing to report". It is not:
        # nothing was checked.
        #
        # The counts stay ABSENT, not zero. `records_verified: 0` beside a supplied ledger
        # reads as "the ledger holds nothing", which is a claim about the artefact; the
        # truth is that this build verified nothing, which is a claim about us. Unknown is
        # an absent key here, exactly as it is everywhere else in this file.
        out["not_assessed_reason"] = (
            "An inference ledger was supplied and INFERENCE_LEDGER resolved, but no prov.* "
            "check is registered with the scanner, so NOTHING in this ledger was verified "
            "and all of Module C's attack classes are unassessed in this scan. This is a "
            "wiring gap on the scan side, not a property of the ledger.")
    if Capability.SIGNING_KEY not in result.capabilities:
        out["ledger_state"] = "not_sealed"
    return out or None


def reproduction_of(result, command: str | None = None) -> dict[str, Any]:
    prof = getattr(result, "profile", None) or {}
    pinned = bool(prof.get("pin_clock"))
    armed = determinism.armed()
    notes = [
        "created_at_utc is pinned to a fixed instant by this profile." if pinned else
        "created_at_utc is the wall clock; it and scan_id differ between runs by design "
        "and are listed in volatile_paths.",
        "Per-check wall-clock timings are not recorded in this file: they differ on every "
        "run and would make the reproducibility diff fail on a correct scan.",
        "Global RNG seeds, deterministic torch algorithms and single-threaded ONNX Runtime "
        "were pinned for this run." if armed else
        "Only the scan seed was recorded: global RNG seeds and thread counts were NOT pinned "
        "for this run (run under a profile that pins the clock, e.g. selftest).",
        "The embedding extractor (DINOv2 vs the ResNet-18 fallback) depends on which "
        "weights are present on the machine and is recorded in each detector finding's "
        "access_assumptions; it is expected to differ between machines with different "
        "weights.",
    ]
    return {"command": command or _NO_COMMAND,
            "volatile_paths": list(VOLATILE_PATHS),
            "seeds": determinism.recorded_seeds(getattr(result, "seed", 0)),
            "env": _env(),
            "determinism_notes": notes}


def standing_limitations(target: dict[str, Any],
                         embedding_gap: str | None = None) -> list[str]:
    from cva.loaders.safety import S3_SANDBOX_LIMITATION  # lazy: keeps the report import light
    out = [*STANDING_LIMITATIONS, str(S3_SANDBOX_LIMITATION)]
    if embedding_gap:
        # Item 20. Without an embedding index, `data.label_consistency` and
        # `data.systematic_mislabel` return `not_performed` and `data.near_dup` silently
        # degrades to pHash-only. Each finding already said so individually; the coverage
        # statement is where a reader looks to find out what this scan could not do, and
        # three detectors failing for one shared reason belongs there as one sentence.
        out.append(
            f"No embedding index was built for this scan ({embedding_gap}). Without one, "
            "data.label_consistency and data.systematic_mislabel cannot run at all and "
            "data.near_dup falls back to perceptual hashing only, which finds near-exact "
            "copies but not semantic near-duplicates. Those detectors are also never "
            "benchmarked on such a machine, so no calibrator is fitted for them.")
    if "model_format" in target and "model_sha256" not in target:
        fmt = str(target["model_format"])
        if fmt == "torchscript":
            out.append(
                "No weight digest could be computed for this model artefact (a frozen "
                "TorchScript archive exposes none): weight-substitution detection is "
                "degraded and no model_sha256 is recorded.")
        elif fmt in ("callable", "http", "subprocess"):
            out.append(
                "This model is query-only (black-box) and exposes no weights, so no weight "
                "digest exists and no model_sha256 is recorded: substitution detection "
                "needs a registered manifest plus the behavioural fingerprint.")
        else:
            out.append(
                "No weight digest could be computed for this model artefact, so no "
                "model_sha256 is recorded and weight-substitution detection is degraded.")
    if "model_format" in target and "preprocess_hash" not in target:
        out.append(
            "No preprocessing spec was declared for this model, so prov.recompute cannot be "
            "performed for it, and scan-time inputs were scaled to [0,1] only — no "
            "mean/std normalisation was applied, because guessing one would be a fact about "
            "the model that nobody declared.")
    # The POSITIVE case — a declared spec, applied at scan time — is not a limitation and is
    # not asserted here. `target.preprocess_hash` is the report's statement that a spec was
    # declared and used; the schema's description of that field says what its presence means.
    return out


def _produced_by(result, prof: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "code_commit": result.code_commit,
        "profile_hash": result.profile_hash,
        "profile_name": result.profile_name,
        "budget_tier": prof.get("budget_tier", result.profile_name),
    }
    try:
        out["tool_version"] = metadata.version("cva")
    except metadata.PackageNotFoundError:
        pass
    return out


def _plan_row(r) -> dict[str, Any]:
    res = r.resolution
    return {"check_id": r.check_id, "state": res.state.value, "reason": res.reason,
            "attack_classes": sorted(r.attack_classes),
            "missing": [str(m) for m in res.missing], "mode": res.mode,
            "exclusion_reason": res.exclusion_reason,
            "estimated_cost": res.estimated_cost}


def _finding(f) -> dict[str, Any]:
    d = f.to_dict()
    # `data` is transport, `path` is the record: once the payload is in the store, the
    # report carries the hash and not a second copy of the bytes.
    d["evidence"] = [
        {"kind": e.kind, "caption": e.caption, "path": e.path,
         **({"data": e.data} if e.path is None and e.data is not None else {})}
        for e in f.evidence]
    return d


def _env() -> dict[str, str]:
    env = {"python": platform.python_version()}
    if determinism.hashseed() is not None:
        env["PYTHONHASHSEED"] = str(determinism.hashseed())
    for name in _ENV_PACKAGES:       # read metadata; never import torch just to print a version
        try:
            env[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return env


def verdict_line(result) -> str:
    """The verdict with its coverage qualifier attached — the ONE string every surface prints.

    A bare `ACCEPT` is the sentence this whole system exists to avoid: a scan with 13 of 19
    checks unrun printed exactly the same word as a scan where everything ran, and the reader
    has no way to tell them apart. `report.json` was never bare (`coverage` is required by the
    schema); the CLI headline was. So the qualifier lives here, beside the coverage generator
    it is derived from, and the CLI, the HTML and the coverage markdown all call this.

    Deliberately NOT "REVIEW on any UNAVAILABLE": on a dataset scan the model-side checks are
    unavailable by construction, so that rule makes every dataset scan REVIEW forever. The
    honest move is to keep the verdict and state what it was computed over.
    """
    plan = list(getattr(result, "plan", None) or [])
    errored = {f.detector_id for f in getattr(result, "findings", None) or []
               if getattr(f.availability, "value", f.availability) == "ERROR"}
    ran = sum(1 for r in plan if r.resolution.runnable and r.check_id not in errored)
    crashed = sum(1 for r in plan if r.resolution.runnable and r.check_id in errored)
    if not plan:
        return f"{result.verdict} — no check was registered, so nothing was assessed"
    parts = [f"{ran}/{len(plan)} checks ran"]
    if crashed:
        parts.append(f"{crashed} crashed")
    not_run = len(plan) - ran - crashed
    if not_run:
        parts.append(f"{not_run} not assessed")
    return f"{result.verdict} — {', '.join(parts)} (see coverage.md)"


def coverage_of(result) -> dict:
    """GENERATED, never written by hand — and it counts `attack` classes only."""
    from cva.core.taxonomy import CLAIMABLE

    # A check that was planned to run and then RAISED did not assess anything: counting it
    # as assessed would overstate coverage in exactly the case where the tool itself failed.
    errored = {f.detector_id for f in result.findings if f.availability.value == "ERROR"}
    assessed: dict[str, list[str]] = {}
    not_assessed: dict[str, list[str]] = {}
    for row in result.plan:
        ok = row.resolution.runnable and row.check_id not in errored
        tgt = assessed if ok else not_assessed
        for ac in row.attack_classes:
            tgt.setdefault(ac, []).append(row.check_id)

    attack_only = CLAIMABLE
    return {
        "counts_only_kind": "attack",
        "assessed": {k: v for k, v in sorted(assessed.items()) if k in attack_only},
        "not_assessed": {k: v for k, v in sorted(not_assessed.items()) if k in attack_only},
        "operational_reports": sorted(
            (set(assessed) | set(not_assessed)) - attack_only),
        "never_covered": sorted(attack_only - set(assessed) - set(not_assessed)),
    }


def _default(o: Any) -> Any:
    item = getattr(o, "item", None)      # numpy scalars
    return item() if callable(item) else str(o)


def write(result, path: Path, command: str | None = None) -> Path:
    path.write_text(json.dumps(build(result, command), indent=2, default=_default))
    return path
