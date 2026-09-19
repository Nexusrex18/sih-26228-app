"""Budget profiles. No constants in code — every threshold comes from here or a YAML
profile, so a tuning change never requires a code change.

Two orthogonal axes in one table (§5.6): the four BUDGET TIERS say how much work to do,
the four POLICIES say what posture to take. `--profile <policy> --budget-tier <tier>`
selects one of each; a bare `--profile <tier>` still works and is what the existing
Module B call sites pass.

Budget is the SECOND axis on Availability. A check excluded by tier reports UNAVAILABLE
with exclusion_reason="budget", distinct from a capability exclusion: the report must say
whether it could not run or was not asked to.
"""
from __future__ import annotations

from typing import Any

TIERS: dict[str, dict[str, Any]] = {
    "triage":   {"checks": {"model.weight_digest", "model.graph_structure"}},
    "standard": {"checks": {"model.weight_digest", "model.graph_structure",
                            "model.behavioural_fingerprint", "model.anomalous",
                            "model.intrinsic_probes", "model.weight_statistics",
                            "model.activation_statistics"}},
    "deep":     {"checks": None, "nc_top_k": 3},      # None = everything registered
    "forensic": {"checks": None, "nc_top_k": 999, "nc_steps": 200, "nes_steps": 120},
}

#: Policy axis. `budget_tier` is the default tier; --budget-tier overrides it.
POLICIES: dict[str, dict[str, Any]] = {
    "baseline": {"budget_tier": "standard"},
    "strict":   {"budget_tier": "deep"},
    # Not "could not" but "was not asked to": white-box checks are disabled up front.
    "blackbox": {"budget_tier": "standard",
                 "disabled_checks": ["model.weight_digest", "model.weight_statistics",
                                     "model.neural_cleanse", "model.activation_statistics"]},
    "selftest": {"budget_tier": "deep", "nc_top_k": 1, "nc_steps": 10, "nes_steps": 10,
                 "scan_id_from_seed": True, "pin_clock": True, "egress_guard": True},
}

#: Legacy/back-compat: a bare tier name is a valid --profile.
PROFILES: dict[str, dict[str, Any]] = {**TIERS, **POLICIES}
