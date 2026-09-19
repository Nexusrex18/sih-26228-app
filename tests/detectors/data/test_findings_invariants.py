"""Cross-cutting invariants, checked on EVERY detector against real attack data.

These are the properties a per-detector test cannot see: determinism across scans (the property the
attack-lab reproducibility test protects, extended to the detectors), unique and scan-independent
finding ids, evidence that is really content-addressed on disk, agreement with core's taxonomy, and a
threshold that actually separates what was flagged.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest

from attacklab.contributor_metadata import assign_contributors
from attacklab.label_flip_attack import flip_labels
from attacklab.near_dup_ood_attack import inject_near_duplicates
from attacklab.negative_space_attack import drop_annotations
from attacklab.annotation_geometry_attack import tamper_geometry
from attacklab.script_batch_attack import script_generate_batch
from attacklab.synth_dataset import make_clean_dataset
from attacklab._types import Dataset, Sample
from cva.core.taxonomy import require
from cva.core.types import Finding
from cva.detectors.data._stub_types import ArrayEmbeddingIndex, stub_embeddings
from cva.detectors.data.annotation_geometry import AnnotationGeometry
from cva.detectors.data.duplicate_label_conflict import DuplicateLabelConflict
from cva.detectors.data.label_flip import LabelConsistency
from cva.detectors.data.metadata_anomaly import MetadataAnomaly
from cva.detectors.data.near_duplicate import NearDuplicate
from cva.detectors.data.negative_space import NegativeSpace
from cva.detectors.data.systematic_mislabel import SystematicMislabel
from cva.detectors.data.trigger_ood import OutOfDistribution, TriggerArtifact
from tests.detectors.data.helpers import make_ctx
from tests.detectors.data.test_negative_space import FakeBoxDetector

TARGET_TYPES = {"sample", "contributor", "batch", "model", "record", "dataset"}
EVIDENCE_KINDS = {"image_crop", "contact_sheet", "heatmap", "plot", "table", "hash", "json"}


@pytest.fixture(scope="module")
def scenarios(clean, poisoned, workdir):
    """name -> (detector class, dataset, embeddings, model, profile). Each must yield findings."""
    w = workdir / "invariants"
    sc = {}
    flipped, _ = flip_labels(clean, seed=7, flip_rate=0.10)
    sc["label_consistency"] = (LabelConsistency, flipped, stub_embeddings(flipped), None, {})
    bad, _ = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair", target_contributor="A", class_pair=(0, 1))
    sc["systematic_mislabel"] = (SystematicMislabel, bad, stub_embeddings(bad), None, {})
    dup, _ = inject_near_duplicates(clean, w / "dup", seed=32, n_sources=4, variants_per_source=6,
                                    target_contributor="B", conflict_rate=0.3)
    demb = stub_embeddings(dup)
    sc["near_dup"] = (NearDuplicate, dup, demb, None, {})
    sc["duplicate_label_conflict"] = (DuplicateLabelConflict, dup, demb, None, {})
    sb, _ = script_generate_batch(clean, w / "sb", seed=3, target_contributor="C")
    sc["metadata_anomaly"] = (MetadataAnomaly, sb, None, None, {})
    two = make_clean_dataset(w / "two", seed=71, n=400, n_classes=4, objects=2)
    two, _ = assign_contributors(two, w / "two", seed=71)
    geo, _ = tamper_geometry(two, seed=62, target_contributor="B", shrink=0.5)
    sc["annotation_geometry"] = (AnnotationGeometry, geo, None, None, {})
    neg, _ = drop_annotations(two, seed=72, target_contributor="B", fraction=0.6)
    sc["negative_space"] = (NegativeSpace, neg, None, FakeBoxDetector(two), {})
    pds, _trig = poisoned
    sc["trigger_artifact (model-free)"] = (TriggerArtifact, pds, None, None, {})
    # data.ood on hand-built vectors: nothing fitted to a fixture
    rng = np.random.default_rng(0)
    axis = np.zeros(16); axis[0] = 1.0
    ref = axis + rng.normal(0, 0.05, (120, 16))
    out = np.zeros(16); out[7] = 1.0
    ids = ["in", "out"]
    ood_ds = Dataset([Sample(i, "x", Path("/nonexistent"), 8, 8) for i in ids], [])
    sc["ood"] = (OutOfDistribution, ood_ds,
                 ArrayEmbeddingIndex(ids, np.stack([axis + rng.normal(0, 0.05, 16), out])), None,
                 {"reference_embeddings": ref})
    return sc


def _run(cls, ds, emb, model, profile, out_dir=None, seed=7):
    ctx = make_ctx(scan_id="scan-1", seed=seed, out_dir=out_dir, profile=profile)
    return cls().detect(ds, emb, model, ctx)


def _sig(fs):
    return sorted((f.finding_id, f.target_ref, round(f.score_raw, 10), round(f.threshold, 10),
                   round(f.confidence, 10), f.severity.value, f.availability.value, f.reason,
                   tuple(e.path or "" for e in f.evidence)) for f in fs)


NAMES = ["label_consistency", "systematic_mislabel", "near_dup", "duplicate_label_conflict",
         "metadata_anomaly", "annotation_geometry", "negative_space", "trigger_artifact (model-free)", "ood"]


@pytest.mark.parametrize("name", NAMES)
def test_every_scenario_actually_produces_findings(scenarios, name):
    """Guards the invariants below against passing on an empty list."""
    fs = [f for f in _run(*scenarios[name]) if f.availability.value == "OK"]
    assert fs, f"{name}: the attack scenario produced no findings, so nothing below is being tested"


@pytest.mark.parametrize("name", NAMES)
def test_findings_are_well_formed_and_consistent_with_core(scenarios, name):
    cls = scenarios[name][0]
    fs = _run(*scenarios[name])
    assert all(isinstance(f, Finding) for f in fs)
    ids = [f.finding_id for f in fs]
    assert len(set(ids)) == len(ids), "finding_id collision inside one scan"
    for f in fs:
        assert re.fullmatch(r"[0-9a-f]{16}", f.finding_id), f.finding_id
        assert f.scan_id == "scan-1" and f.produced_by == ""
        assert f.detector_id == cls.id and f.detector_version == cls.version
        assert f.target_type in TARGET_TYPES
        assert f.attack_class in cls.attack_classes
        require(f.attack_class)                                   # startup-error rule: must be in core
        assert f.disposition.value == "review" and f.disposition_rule == "pending_risk_engine"
        assert f.limitations and f.access_assumptions and len(f.reason) > 20
        assert 0.0 <= f.confidence <= 1.0 and math.isfinite(f.score_raw) and math.isfinite(f.threshold)
        assert all(e.kind in EVIDENCE_KINDS for e in f.evidence)
        if f.availability.value == "OK":
            assert f.score_raw >= f.threshold, "a flagged finding must clear the threshold it reports"
            default = require(f.attack_class).nature
            assert default is None or f.nature.value == default, \
                f"{f.detector_id}: nature {f.nature.value!r} != core default {default!r}"
        else:
            assert f.nature.value == "indeterminate"              # a missing input says nothing about intent


@pytest.mark.parametrize("name", NAMES)
def test_scans_are_deterministic_even_if_global_rngs_are_scrambled(scenarios, name):
    a = _sig(_run(*scenarios[name]))
    random.seed(12345); np.random.seed(12345)
    b = _sig(_run(*scenarios[name]))
    random.seed(999); np.random.seed(999)
    c = _sig(_run(*scenarios[name]))
    assert a == b == c


@pytest.mark.parametrize("name", NAMES)
def test_evidence_is_content_addressed_on_disk_and_independent_of_where_it_is_written(scenarios, name, tmp_path):
    a = _run(*scenarios[name], out_dir=tmp_path / "a")
    b = _run(*scenarios[name], out_dir=tmp_path / "b")
    assert _sig(a) == _sig(b), "finding ids / evidence paths must not depend on the output directory"
    for root, fs in ((tmp_path / "a", a), (tmp_path / "b", b)):
        for f in fs:
            for e in f.evidence:
                if e.path is None:
                    continue
                assert "/" not in e.path, "Evidence.path is the BARE hash, resolved against <out_dir>/evidence/"
                fp = root / "evidence" / e.path
                assert fp.exists(), f"{f.detector_id}: evidence file missing on disk: {e.path}"
                assert fp.stem == hashlib.sha256(fp.read_bytes()).hexdigest(), \
                    f"{e.path} is not named by the sha256 of its content"


def test_trigger_artifact_with_a_real_model_holds_the_same_invariants(poisoned, poisoned_model, tmp_path):
    """The model-based methods (patch saliency, spectral, clustering) under the same checks."""
    pds, _ = poisoned
    args = (TriggerArtifact, pds, None, poisoned_model, {})
    a = _run(*args, out_dir=tmp_path / "a")
    random.seed(1); np.random.seed(1)
    b = _run(*args, out_dir=tmp_path / "b")
    assert a and _sig(a) == _sig(b)
    assert len({f.finding_id for f in a}) == len(a)
    assert any("occluding that region" in f.reason for f in a)
    for f in a:
        assert f.score_raw >= f.threshold and 0 <= f.confidence <= 1
        assert require(f.attack_class).nature == f.nature.value
