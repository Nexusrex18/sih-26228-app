"""The invariants that fail SILENTLY if left to review: registry hygiene, capability declarations
matching backend_plan.md §9.4, one Finding type, and Module B untouched."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import cva.detectors.data as data
from cva.core.finding import Finding
from cva.detectors.data.registry import (EXPECTED_IDS, covered_attack_classes, registry_rows,
                                         uncovered_attack_classes)
from cva.detectors.data.taxonomy import MODULE_A_ATTACK_CLASSES

ROOT = Path(__file__).resolve().parents[3]


def test_all_nine_ids_registered_and_only_those():
    assert set(data.DATA_REGISTRY) == EXPECTED_IDS


def test_module_b_registry_is_untouched():
    import cva.detectors.base as b
    importlib.import_module("cva.detectors.model")
    assert not any(k.startswith("data.") for k in b.REGISTRY)
    assert all(k.startswith("model.") for k in b.REGISTRY)


def test_rows_match_backend_plan_9_4():
    rows = {r["id"]: r for r in registry_rows()}
    assert rows["data.near_dup"]["requires"] == ["DATASET_IMAGES"]
    assert rows["data.label_consistency"]["requires"] == ["DATASET_IMAGES", "DATASET_LABELS"]
    assert rows["data.label_consistency"]["optional"] == []          # settled: no degraded mode
    assert rows["data.trigger_artifact"]["requires"] == ["DATASET_IMAGES"]
    assert rows["data.trigger_artifact"]["optional"] == ["MODEL_ACTIVATIONS", "MODEL_PREDICT"]
    assert rows["data.ood"]["requires"] == ["DATASET_IMAGES", "REFERENCE_CLEAN_SET"]
    assert rows["data.metadata_anomaly"]["requires"] == ["DATASET_IMAGES"]


def test_every_taxonomy_class_is_covered_and_every_claim_is_in_the_taxonomy():
    assert uncovered_attack_classes() == set()
    assert covered_attack_classes() <= set(MODULE_A_ATTACK_CLASSES)
    assert all(k == "attack" for k in MODULE_A_ATTACK_CLASSES.values())


def test_detectors_declare_only_frozen_capabilities():
    from cva.core.capability import Capability
    for cls in data.DATA_REGISTRY.values():
        assert cls.requires | cls.optional <= set(Capability)
        assert Capability.SUSPECT_INPUTS not in cls.requires | cls.optional   # a Module B addition


def test_finding_is_the_one_shared_type(clean):
    from cva.detectors.data.metadata_anomaly import MetadataAnomaly
    from attacklab.script_batch_attack import script_generate_batch
    import tempfile
    ds, _ = script_generate_batch(clean, tempfile.mkdtemp(), 1, "A")
    f = MetadataAnomaly().detect(ds, None, None)[0]
    assert isinstance(f, Finding)


def test_detectors_do_not_import_attacklab_or_remediation():
    """Ratified layout (backend_plan.md §6.5): scanner code never imports the attack lab, and
    detectors never import remediation/."""
    for py in (ROOT / "cva" / "detectors" / "data").glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not n.startswith(("attacklab", "cva.remediation", "remediation")), (py.name, n)


def test_core_never_imports_module_a():
    for py in (ROOT / "cva" / "core").glob("*.py"):
        assert "detectors" not in [n.module for n in ast.walk(ast.parse(py.read_text()))
                                   if isinstance(n, ast.ImportFrom) and n.module], py.name
