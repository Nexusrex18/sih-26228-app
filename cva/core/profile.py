"""Budget profiles. No constants in code — every threshold comes from here or a YAML
profile, so a tuning change never requires a code change.

Budget is the SECOND axis on Availability. A check excluded by profile reports
UNAVAILABLE with exclusion_reason="budget", distinct from a capability exclusion: the
report must say whether it could not run or was not asked to.
"""
from __future__ import annotations

from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "triage":   {"checks": {"model.weight_digest", "model.graph_structure"}},
    "standard": {"checks": {"model.weight_digest", "model.graph_structure",
                            "model.behavioural_fingerprint", "model.anomalous",
                            "model.intrinsic_probes", "model.weight_statistics",
                            "model.activation_statistics"}},
    "deep":     {"checks": None, "nc_top_k": 3},      # None = everything registered
    "forensic": {"checks": None, "nc_top_k": 999, "nc_steps": 200, "nes_steps": 120},
}


