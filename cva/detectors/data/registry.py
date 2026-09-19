"""Explicit registration for Module A (ADR-008 rejects dynamic discovery).

Importing this module is what puts Module A's detectors into ``DETECTOR_REGISTRY`` — the core
registry, shared with ``model.data_consistency``. ``cva/core/`` must never import it (invariant 2).
"""
from cva.core.registry import DETECTOR_REGISTRY

from . import (annotation_geometry, duplicate_label_conflict, label_flip, metadata_anomaly,  # noqa: F401
               near_duplicate, negative_space, systematic_mislabel, trigger_ood)

__all__ = ["DETECTOR_REGISTRY", "MODULE_A_DETECTORS", "registry_rows"]

MODULE_A_DETECTORS = [
    "data.near_dup", "data.label_consistency", "data.trigger_artifact", "data.ood",
    "data.metadata_anomaly", "data.negative_space", "data.annotation_geometry",
    "data.systematic_mislabel", "data.duplicate_label_conflict",
]


def registry_rows() -> list[dict]:
    """Machine-readable ``requires`` / ``optional`` rows (backend_plan.md §9.4 form)."""
    return [{"id": cid, "requires": sorted(str(c) for c in DETECTOR_REGISTRY[cid].requires),
             "optional": sorted(str(c) for c in DETECTOR_REGISTRY[cid].optional),
             "attack_classes": sorted(DETECTOR_REGISTRY[cid].attack_classes),
             "version": DETECTOR_REGISTRY[cid].version}
            for cid in sorted(MODULE_A_DETECTORS)]
