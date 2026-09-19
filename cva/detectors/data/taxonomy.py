"""Module A's ``attack_class`` strings — exactly ``Plan/Module-A-Data-Integrity-Plan.md`` §5 and
``backend_plan.md`` §9.5. Backend owns the single list (``core/taxonomy.py``); this is the slice
Module A declares so it can be merged in. ``kind`` follows §9.5: only ``attack`` entries count
toward generated coverage."""
from __future__ import annotations

NEAR_DUPLICATE_FLOODING = "near_duplicate_flooding"
LABEL_FLIPPING = "label_flipping"
SYSTEMATIC_MISLABELLING = "systematic_mislabelling"
TRIGGER_INJECTION = "trigger_injection"
OUT_OF_DISTRIBUTION = "out_of_distribution"
NEGATIVE_SPACE_POISONING = "negative_space_poisoning"
ANNOTATION_GEOMETRY_TAMPER = "annotation_geometry_tamper"
DUPLICATE_LABEL_CONFLICT = "duplicate_label_conflict"
SCRIPT_GENERATED_BATCH = "script_generated_batch"

# name -> kind. All Module A classes are attacks the system claims to detect.
MODULE_A_ATTACK_CLASSES: dict[str, str] = {
    NEAR_DUPLICATE_FLOODING: "attack",
    LABEL_FLIPPING: "attack",
    SYSTEMATIC_MISLABELLING: "attack",
    TRIGGER_INJECTION: "attack",
    OUT_OF_DISTRIBUTION: "attack",
    NEGATIVE_SPACE_POISONING: "attack",
    ANNOTATION_GEOMETRY_TAMPER: "attack",
    DUPLICATE_LABEL_CONFLICT: "attack",
    SCRIPT_GENERATED_BATCH: "attack",
}
