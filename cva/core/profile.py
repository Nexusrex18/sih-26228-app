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

# `embedding_max_images` = measured DINOv2 CPU throughput (16.84 img/s, 224 px, batch 16,
# 6 threads; `python -m cva.features.throughput`) x the tier's embedding time budget
# (5 min / 30 min / 2 h), rounded down. triage is 0 so it never loads the backbone.
#
# `evidence` caps what the HTML renderer inlines (report.json always carries everything):
# images per finding, findings rendered, and total bytes. A single-file report that a
# reviewer cannot open is worse than a truncated one that says it truncated.
TIERS: dict[str, dict[str, Any]] = {
    "triage":   {"checks": {"model.weight_digest", "model.graph_structure"},
                 "embedding_max_images": 0,
                 "evidence": {"max_images_per_finding": 2, "max_findings_rendered": 50,
                              "max_report_bytes": 5_000_000}},
    "standard": {"checks": {"model.weight_digest", "model.graph_structure",
                            "model.behavioural_fingerprint", "model.anomalous",
                            "model.intrinsic_probes", "model.weight_statistics",
                            "model.activation_statistics",
                            "data.annotation_geometry", "data.duplicate_label_conflict",
                            "data.label_consistency", "data.metadata_anomaly",
                            "data.near_dup", "data.negative_space", "data.ood",
                            "data.systematic_mislabel", "data.trigger_artifact"},
                 "embedding_max_images": 5000,
                 "evidence": {"max_images_per_finding": 4, "max_findings_rendered": 200,
                              "max_report_bytes": 15_000_000}},
    "deep":     {"checks": None, "nc_top_k": 3,       # None = everything registered
                 "embedding_max_images": 30000,
                 "evidence": {"max_images_per_finding": 6, "max_findings_rendered": 400,
                              "max_report_bytes": 22_000_000}},
    "forensic": {"checks": None, "nc_top_k": 999, "nc_steps": 200, "nes_steps": 120,
                 "embedding_max_images": 120000,
                 "evidence": {"max_images_per_finding": 8, "max_findings_rendered": 800,
                              "max_report_bytes": 24_000_000}},
}

#: Policy axis. `budget_tier` is the default tier; --budget-tier overrides it.
POLICIES: dict[str, dict[str, Any]] = {
    "baseline": {"budget_tier": "standard"},
    "strict":   {"budget_tier": "deep"},
    # Not "could not" but "was not asked to": white-box checks are disabled up front.
    "blackbox": {"budget_tier": "standard",
                 "disabled_checks": ["model.weight_digest", "model.weight_statistics",
                                     "model.neural_cleanse", "model.activation_statistics"]},
    # standard, not triage: triage never loads the backbone, and that load is where an
    # egress is most likely to hide (plan §5.6). nc_*/nes_* only bite if --budget-tier deep.
    "selftest": {"budget_tier": "standard", "nc_top_k": 1, "nc_steps": 10, "nes_steps": 10,
                 "scan_id_from_seed": True, "pin_clock": True, "egress_guard": True},
}

#: Legacy/back-compat: a bare tier name is a valid --profile.
PROFILES: dict[str, dict[str, Any]] = {**TIERS, **POLICIES}
