"""The machine-readable half of the report — one of the four named PS deliverables.

Kept separate from the HTML renderer on purpose: the JSON is the contract other seats and
any third party consume, and it must not be coupled to how a page happens to look.
"""
from __future__ import annotations

import json
from pathlib import Path

from cva.core.capability import Capability


def build(result) -> dict:
    return {
        "scan_id": result.scan_id,
        "model": {"id": result.model_id, "format": result.model_fmt},
        "verdict": result.verdict,
        "access_assumptions": {
            c.value: {
                "available": c in result.capabilities,
                "why_absent": result.capabilities.note_for(c),
            }
            for c in Capability
            if c.name.startswith(("MODEL_", "REFERENCE_", "SUSPECT_"))
        },
        "plan": [
            {"check": r.check_id, "state": r.resolution.state.value,
             "reason": r.resolution.reason,
             "missing": [str(m) for m in r.resolution.missing],
             "attack_classes": sorted(r.attack_classes),
             "seconds": result.timings.get(r.check_id)}
            for r in sorted(result.plan, key=lambda x: x.check_id)
        ],
        "findings": [f.to_dict() for f in result.findings],
        "coverage": coverage_of(result),
    }


def coverage_of(result) -> dict:
    """GENERATED, never written by hand — and it counts `attack` classes only."""
    from cva.core.types import ATTACK_CLASSES

    assessed: dict[str, list[str]] = {}
    not_assessed: dict[str, list[str]] = {}
    for row in result.plan:
        tgt = assessed if row.resolution.runnable else not_assessed
        for ac in row.attack_classes:
            tgt.setdefault(ac, []).append(row.check_id)

    attack_only = {k for k, v in ATTACK_CLASSES.items() if v["kind"] == "attack"}
    return {
        "counts_only_kind": "attack",
        "assessed": {k: v for k, v in sorted(assessed.items()) if k in attack_only},
        "not_assessed": {k: v for k, v in sorted(not_assessed.items()) if k in attack_only},
        "operational_reports": sorted(
            (set(assessed) | set(not_assessed)) - attack_only),
        "never_covered": sorted(attack_only - set(assessed) - set(not_assessed)),
    }


def write(result, path: Path) -> Path:
    path.write_text(json.dumps(build(result), indent=2))
    return path
