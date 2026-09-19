"""Module A against the contracts core actually owns: the shared detector registry, the ONE flat
attack-class taxonomy, the coverage generator, zero-argument construction. Nothing here compares a
Module A structure with itself — every expectation is checked against core or an independent
literal copied from the plan."""
from __future__ import annotations

import types

import pytest

import cva.detectors.data as data
from cva.core.capability import Availability, Capability, Resolution
from cva.core.orchestrator import PlanRow
from cva.core.registry import DETECTOR_REGISTRY, REGISTRY
from cva.core.taxonomy import TAXONOMY
from cva.core.types import Finding
from cva.detectors.data.registry import MODULE_A_DETECTORS, registry_rows

# Independent literal: backend_plan.md §9.5, Module A row (and Module-A plan §5).
PLAN_MODULE_A_CLASSES = {
    "near_duplicate_flooding", "label_flipping", "systematic_mislabelling", "trigger_injection",
    "out_of_distribution", "negative_space_poisoning", "annotation_geometry_tamper",
    "duplicate_label_conflict", "script_generated_batch",
}
PLAN_MODULE_A_IDS = {
    "data.near_dup", "data.label_consistency", "data.trigger_artifact", "data.ood",
    "data.metadata_anomaly", "data.negative_space", "data.annotation_geometry",
    "data.systematic_mislabel", "data.duplicate_label_conflict",
}


def test_all_nine_register_in_the_core_detector_registry():
    assert set(MODULE_A_DETECTORS) == PLAN_MODULE_A_IDS
    assert PLAN_MODULE_A_IDS <= set(DETECTOR_REGISTRY)


def test_they_are_detectors_not_model_checks():
    """REGISTRY is the ModelCheck registry the orchestrator walks with .check(model, ctx)."""
    assert not (PLAN_MODULE_A_IDS & set(REGISTRY))


def test_rows_match_backend_plan_9_4():
    rows = {r["id"]: r for r in registry_rows()}
    assert rows["data.near_dup"]["requires"] == ["DATASET_IMAGES"]
    assert rows["data.label_consistency"]["requires"] == ["DATASET_IMAGES", "DATASET_LABELS"]
    assert rows["data.label_consistency"]["optional"] == []          # settled: no degraded mode
    assert rows["data.trigger_artifact"]["requires"] == ["DATASET_IMAGES"]
    assert rows["data.trigger_artifact"]["optional"] == ["MODEL_ACTIVATIONS", "MODEL_PREDICT"]
    assert rows["data.ood"]["requires"] == ["DATASET_IMAGES", "REFERENCE_CLEAN_SET"]
    assert rows["data.metadata_anomaly"]["requires"] == ["DATASET_IMAGES"]


@pytest.mark.parametrize("cid", sorted(PLAN_MODULE_A_IDS))
def test_zero_arg_construction_and_frozen_capabilities(cid):
    """The orchestrator instantiates plug-ins with NO arguments; constructor-injected state is
    unreachable in a real scan."""
    inst = DETECTOR_REGISTRY[cid]()
    assert inst.requires | inst.optional <= set(Capability)
    assert Capability.SUSPECT_INPUTS not in inst.requires | inst.optional   # a Module B addition
    assert inst.attack_classes


def test_taxonomy_is_merged_into_cores_flat_registry():
    """The coverage generator reads core's single attack-class list. A class missing there
    is silently dropped from coverage and raises KeyError on a description lookup.

    That list moved from `cva.core.types.ATTACK_CLASSES` to `cva.core.taxonomy.TAXONOMY`
    at B1 (§6.4): a dict of dicts cannot be looked up with a loud failure, and
    `taxonomy.require()` raises `UnknownAttackClass` at startup rather than letting a
    coverage row vanish at render time. The assertion is unchanged in substance."""
    for ac in PLAN_MODULE_A_CLASSES:
        assert ac in TAXONOMY, f"{ac} missing from core/taxonomy.py"
        assert TAXONOMY[ac].kind == "attack" and TAXONOMY[ac].desc
    claimed = {a for cid in PLAN_MODULE_A_IDS for a in DETECTOR_REGISTRY[cid].attack_classes}
    assert claimed == PLAN_MODULE_A_CLASSES          # every class is claimed, none invented


def test_module_a_reaches_the_real_coverage_generator():
    """Feed Module A's declared rows through report_json.coverage_of and coverage.render_markdown —
    the code that previously would have dropped every Module A finding."""
    from cva.report.coverage import render_markdown
    from cva.report.report_json import coverage_of
    plan = [PlanRow(cid, Resolution(Availability.OK, "ok"), set(DETECTOR_REGISTRY[cid].attack_classes))
            for cid in sorted(PLAN_MODULE_A_IDS)]
    result = types.SimpleNamespace(plan=plan, scan_id="t", model_id="m", model_fmt="onnx",
                                   verdict="ACCEPT", capabilities=None, timings={}, findings=[])
    cov = coverage_of(result)
    assert set(cov["assessed"]) >= PLAN_MODULE_A_CLASSES
    assert not (PLAN_MODULE_A_CLASSES & set(cov["never_covered"]))
    md = render_markdown(result)                     # would raise KeyError on a missing 'desc'
    assert "trigger_injection" in md and "Trigger patch pasted" in md


def test_finding_is_the_one_shared_type(clean, workdir):
    from attacklab.script_batch_attack import script_generate_batch
    from cva.detectors.data.metadata_anomaly import MetadataAnomaly
    from tests.detectors.data.helpers import detect
    ds, _ = script_generate_batch(clean, workdir / "ct", 1, "A")
    f = detect(MetadataAnomaly, ds)[0]
    assert isinstance(f, Finding)


def test_package_import_populates_only_the_core_registry():
    assert data.MODULE_A_DETECTORS == MODULE_A_DETECTORS
    assert not hasattr(data, "DATA_REGISTRY")


def test_each_detector_states_its_multiplicity_or_scale_limits_where_it_runs_many_tests(clean, clean_emb):
    """Detectors that run one hypothesis test per (contributor x class...) cell must own the
    multiplicity policy in their findings, not only in prose."""
    from attacklab.label_flip_attack import flip_labels
    from cva.detectors.data._stub_types import stub_embeddings
    from cva.detectors.data.systematic_mislabel import SystematicMislabel
    from tests.detectors.data.helpers import detect
    bad, _ = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair",
                         target_contributor="A", class_pair=(0, 1))
    f = detect(SystematicMislabel, bad, stub_embeddings(bad))[0]
    assert any("Bonferroni" in l and "NOT at COCO scale" in l for l in f.limitations)
    assert "after correcting for" in f.reason


def test_orchestrator_style_run_of_every_runnable_detector(clean, clean_emb, poisoned_model):
    """Simulate what an orchestrator does, given only what core provides: iterate the registry,
    construct each plug-in with NO arguments, resolve capabilities, call
    detect(dataset, embeddings, model, ctx) and stamp scan_id. Anything that needed a constructor
    argument, or an attribute ctx cannot supply, fails here — not in production."""
    import tempfile

    from attacklab.synth_dataset import make_clean_dataset
    from cva.core.capability import CapabilitySet
    from tests.detectors.data.helpers import make_ctx, reference_matrix

    ref = make_clean_dataset(tempfile.mkdtemp(), seed=99, n=120)
    from cva.detectors.data._stub_types import stub_embeddings
    ref_emb = stub_embeddings(ref)
    ctx = make_ctx(scan_id="scan-xyz", seed=7,
                   profile={"reference_embeddings": reference_matrix(ref, ref_emb)})
    caps = CapabilitySet.union(clean.capabilities(), poisoned_model.capabilities(),
                               CapabilitySet(frozenset({Capability.REFERENCE_CLEAN_SET})))
    ran = []
    for cid in MODULE_A_DETECTORS:
        cls = DETECTOR_REGISTRY[cid]
        res = caps.resolve(set(cls.requires), set(cls.optional))
        if not res.runnable:
            continue
        findings = cls().detect(clean, clean_emb, poisoned_model, ctx)      # zero-arg + ctx only
        assert all(isinstance(f, Finding) and f.scan_id == "scan-xyz" for f in findings), cid
        ran.append(cid)
    # every detector whose requirements the clean fixtures satisfy actually ran
    assert {"data.near_dup", "data.label_consistency", "data.metadata_anomaly", "data.trigger_artifact",
            "data.ood", "data.systematic_mislabel"} <= set(ran)
