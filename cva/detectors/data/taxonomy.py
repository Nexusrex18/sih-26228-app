"""Module A's ``attack_class`` names — the strings a detector puts in ``Finding.attack_class``.

The DEFINITIONS (kind / default nature / description) live in ``cva.core.types.ATTACK_CLASSES``, the
single flat registry the coverage generator reads (``cva/report/coverage.py``,
``report_json.coverage_of``). An attack class missing from there is silently dropped from the
coverage statement and raises ``KeyError`` when a description is looked up — so
``tests/detectors/data/test_module_a_contract.py`` asserts every name below is defined there.
"""
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
