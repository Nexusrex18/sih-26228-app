"""Module A's registry rows, and the coverage view generated from them.

Ids and ``requires``/``optional`` here are what Backend's coverage generator walks: a mismatch
means a detector silently produces no coverage row. ``registry_rows()`` is the machine-readable
form of ``backend_plan.md`` §9.4's Module A table, including the four rows that table does not yet
carry (``negative_space``, ``annotation_geometry``, ``systematic_mislabel``,
``duplicate_label_conflict``) — Backend needs to add them.
"""
from __future__ import annotations

from .base import DATA_REGISTRY
from . import (annotation_geometry, duplicate_label_conflict, label_flip, metadata_anomaly,  # noqa: F401
               near_duplicate, negative_space, systematic_mislabel, trigger_ood)
from .taxonomy import MODULE_A_ATTACK_CLASSES

EXPECTED_IDS = {
    "data.near_dup", "data.label_consistency", "data.trigger_artifact", "data.ood",
    "data.metadata_anomaly", "data.negative_space", "data.annotation_geometry",
    "data.systematic_mislabel", "data.duplicate_label_conflict",
}


def registry_rows() -> list[dict]:
    return [{"id": cid, "requires": sorted(str(c) for c in cls.requires),
             "optional": sorted(str(c) for c in cls.optional),
             "attack_classes": sorted(cls.attack_classes), "version": cls.version}
            for cid, cls in sorted(DATA_REGISTRY.items())]


def covered_attack_classes() -> set[str]:
    return {a for cls in DATA_REGISTRY.values() for a in cls.attack_classes}


def uncovered_attack_classes() -> set[str]:
    """Module A taxonomy entries no registered detector claims — the generator must say so."""
    return set(MODULE_A_ATTACK_CLASSES) - covered_attack_classes()
