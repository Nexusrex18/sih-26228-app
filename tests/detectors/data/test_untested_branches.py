"""Branches a line-coverage trace showed no test ever executed — several of them real features (the
no-contributor grouping in metadata_anomaly, the numpy<2 popcount path) or failure handling.

Policy under test: a fault that belongs to ANOTHER module (a model adapter that misreports its
capabilities, a model that cannot be queried) or insufficient data never crashes Module A and is
never a silent []: it is reported as a visible ``UNAVAILABLE`` / ``DEGRADED`` finding.
"""
from __future__ import annotations

import dataclasses
import weakref

import numpy as np
import pytest

from attacklab.contributor_metadata import assign_contributors
from attacklab.script_batch_attack import script_generate_batch
from attacklab.synth_dataset import make_clean_dataset
from cva.core.capability import Capability, CapabilitySet
from cva.detectors.data._stub_types import stub_embeddings
from cva.detectors.data.base import (DATASET_CACHE, attribution, category_name, dominant_category,
                                     sample_by_id, to_model_input)
from cva.detectors.data.label_flip import LabelConsistency
from cva.detectors.data.metadata_anomaly import MetadataAnomaly
from cva.detectors.data.near_duplicate import _popcount, candidate_pairs
from cva.detectors.data.systematic_mislabel import SystematicMislabel
from cva.detectors.data.trigger_ood import OutOfDistribution, TriggerArtifact
from tests.detectors.data.helpers import detect

ACT, PRED = Capability.MODEL_ACTIVATIONS, Capability.MODEL_PREDICT


class _Model:
    model_id, fmt, num_classes, input_shape = "m", "callable", 4, (3, 64, 64)

    def __init__(self, caps, act=None, pred=None, caps_raise=None):
        self._c, self._a, self._p, self._raise = caps, act, pred, caps_raise

    def capabilities(self):
        if self._raise:
            raise RuntimeError(self._raise)
        return CapabilitySet(frozenset(self._c))

    def activations(self, x):
        return self._a(x) if self._a else None

    def predict(self, x):
        return self._p(x)


def _degraded(fs):
    return [f for f in fs if f.availability.value == "DEGRADED"]


# ------------------------------------------------- another module's faults: reported, never raised
def test_a_model_whose_capabilities_fail_is_reported_not_raised_and_not_silent(clean):
    fs = detect(TriggerArtifact, clean, None, _Model(set(), caps_raise="probe blew up"))
    d = _degraded(fs)
    assert len(d) == 1 and "probe blew up" in d[0].reason and "model.capabilities() failed" in d[0].reason
    assert not [f for f in fs if f.availability.value == "OK"]        # the model-free method still ran cleanly


def test_activations_declared_but_nothing_returned_is_reported(clean):
    fs = detect(TriggerArtifact, clean, None, _Model({ACT}, act=lambda x: None))
    d = _degraded(fs)
    assert len(d) == 1 and "declares MODEL_ACTIVATIONS but activations() returned nothing" in d[0].reason


def test_a_bad_activation_layer_setting_is_reported_with_the_valid_names(clean):
    fs = detect(TriggerArtifact, clean, None,
                _Model({ACT}, act=lambda x: {"fc": np.zeros((len(x), 8), np.float32)}),
                params={"activation_layer": "nope"})
    d = _degraded(fs)
    assert len(d) == 1 and "'nope'" in d[0].reason and "['fc']" in d[0].reason


def test_a_model_that_cannot_be_queried_during_saliency_is_reported(poisoned):
    ds, _ = poisoned

    def boom(x):
        raise RuntimeError("model unreachable")
    fs = detect(TriggerArtifact, ds, None, _Model({PRED}, pred=boom))
    d = _degraded(fs)
    assert len(d) == 1 and "model unreachable" in d[0].reason and "patch saliency" in d[0].reason
    assert any(f.availability.value == "OK" for f in fs), "the model-free findings must still be reported"


def test_a_model_that_cannot_be_queried_during_the_sweep_is_reported(clean):
    def boom(x):
        raise RuntimeError("sweep target down")
    fs = detect(TriggerArtifact, clean, None, _Model({PRED}, pred=boom),
                params={"candidate_samples": [clean.samples[0].sample_id]})
    assert any("sweep target down" in f.reason for f in _degraded(fs))


def test_insignificant_saliency_is_noted_on_the_findings_it_did_not_corroborate(poisoned):
    ds, trig = poisoned
    const = _Model({PRED}, pred=lambda x: np.tile([1.0, 0, 0, 0], (len(x), 1)))   # occlusion changes nothing
    fs = [f for f in detect(TriggerArtifact, ds, None, const) if f.availability.value == "OK"]
    assert {f.target_ref for f in fs} <= trig and fs
    assert all(f.severity.value == "medium" for f in fs)            # NOT raised to high: no corroboration
    assert all(any("not significant" in l for l in f.limitations) for f in fs)


# ------------------------------------------------- insufficient data: UNAVAILABLE / DEGRADED, never []
def test_metadata_groups_by_batch_when_no_contributor_is_resolved(clean, workdir):
    sb, _ = script_generate_batch(clean, workdir / "b1", seed=5, target_contributor="C")
    anon = type(sb)([dataclasses.replace(s, contributor=None, contributor_source=None) for s in sb.samples],
                    sb.categories)
    fs = detect(MetadataAnomaly, anon)
    assert {(f.target_type, f.target_ref) for f in fs} == {("batch", "C-b0"), ("batch", "C-b1")}
    assert all(f.reason.startswith("Batch 'C-b") for f in fs)


def test_metadata_with_a_single_group_says_most_signals_could_not_run(clean, workdir):
    sb, _ = script_generate_batch(clean, workdir / "b2", seed=5, target_contributor="C")
    anon = type(sb)([dataclasses.replace(s, contributor=None, contributor_source=None, batch=None)
                     for s in sb.samples], sb.categories)
    fs = detect(MetadataAnomaly, anon)
    assert len(fs) == 1 and fs[0].availability.value == "DEGRADED"
    assert "only the encoder-string check could run" in fs[0].reason


def test_systematic_mislabel_on_a_dataset_too_small_for_any_cell_is_reported(tmp_path):
    ds = make_clean_dataset(tmp_path / "s", seed=2, n=48)
    ds, _ = assign_contributors(ds, tmp_path / "s", seed=2)
    fs = detect(SystematicMislabel, ds, stub_embeddings(ds))
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "at least 10 images" in fs[0].reason


def test_label_consistency_with_one_class_is_reported_not_vacuously_clean(clean):
    one = type(clean)([s for s in clean.samples if dominant_category(s) == 0], clean.categories)
    fs = detect(LabelConsistency, one, stub_embeddings(one))
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "only one class" in fs[0].reason


def test_ood_reports_missing_embeddings_and_a_too_small_reference(clean, clean_emb):
    ref80 = np.random.default_rng(0).normal(size=(80, 8))
    fs = detect(OutOfDistribution, clean, None, None, profile={"reference_embeddings": ref80})
    assert len(fs) == 1 and "no embedding index" in fs[0].reason
    fs = detect(OutOfDistribution, clean, clean_emb, None,
                profile={"reference_embeddings": np.zeros((10, 8))})
    assert len(fs) == 1 and "too small" in fs[0].reason


# ------------------------------------------------- code paths that only run in other environments
def test_the_numpy1_popcount_fallback_is_identical_to_the_fast_path(monkeypatch):
    """pyproject allows numpy>=1.26, where np.bitwise_count does not exist and a different code path runs."""
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2**63, size=(37, 41), dtype=np.uint64)
    fast = _popcount(h)
    pairs_fast = candidate_pairs(rng.integers(0, 2**63, size=300, dtype=np.uint64), 8)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert np.array_equal(_popcount(h), fast)
    rng2 = np.random.default_rng(0)
    rng2.integers(0, 2**63, size=(37, 41), dtype=np.uint64)
    assert candidate_pairs(rng2.integers(0, 2**63, size=300, dtype=np.uint64), 8) == pairs_fast


def test_single_channel_model_input(clean):
    x = to_model_input(clean.samples[0], (1, 32, 32))
    assert x.shape == (1, 32, 32) and x.dtype == np.float32 and 0.0 <= x.min() <= x.max() <= 1.0
    assert to_model_input(clean.samples[0], (3, 32, 32)).shape == (3, 32, 32)


# ------------------------------------------------- the contract-only dataset helpers
def test_dataset_helpers_work_on_unhashable_datasets_and_do_not_leak(clean):
    with pytest.raises(TypeError):
        hash(clean)                                               # real datasets are unhashable dataclasses
    s0 = clean.samples[0]
    assert sample_by_id(clean, s0.sample_id) is s0
    assert category_name(clean, 0) == "truck" and category_name(clean, 999) == "999"   # unknown id: no crash
    assert id(clean) in DATASET_CACHE._d
    small = type(clean)(clean.samples[:5], clean.categories)
    assert sample_by_id(small, s0.sample_id) is s0
    ref = weakref.ref(small)
    key = id(small)
    del small
    import gc; gc.collect()
    assert ref() is None and key not in DATASET_CACHE._d          # the cache entry died with the dataset


def test_cache_is_not_fooled_when_a_datasets_samples_change(clean):
    grown = type(clean)(list(clean.samples[:5]), clean.categories)
    assert sample_by_id(grown, clean.samples[0].sample_id)
    extra = clean.samples[7]
    grown.samples.append(extra)                                    # same object, more samples
    assert sample_by_id(grown, extra.sample_id) is extra


def test_attribution_with_no_contributor_and_dominant_category_of_an_unlabelled_sample(clean):
    s = dataclasses.replace(clean.samples[0], contributor=None, labels=[])
    assert attribution(s) == "contributor unresolved"
    assert dominant_category(s) is None


# ------------------------------------------------- unreadable files: counted, disclosed, never a crash
def _corrupt(ds, ids):
    for sid in ids:
        sample_by_id(ds, sid).path.write_bytes(b"this is not an image")


def test_unreadable_files_are_counted_and_disclosed_on_findings(tmp_path):
    ds = make_clean_dataset(tmp_path / "c", seed=4, n=120)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=4)
    ds, _ = script_generate_batch(ds, tmp_path / "sb", seed=5, target_contributor="C")
    victims = [s.sample_id for s in ds.samples if s.contributor == "A"][:3]
    _corrupt(ds, victims)                                          # 3 corrupt files elsewhere in the dataset
    fs = detect(MetadataAnomaly, ds)
    assert [(f.target_type, f.target_ref) for f in fs] == [("contributor", "C")]
    assert all(any("3 file(s) had unreadable metadata" in l for l in f.limitations) for f in fs)


def test_mostly_unreadable_metadata_is_reported_not_silently_empty(tmp_path):
    ds = make_clean_dataset(tmp_path / "c", seed=4, n=60)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=4)
    _corrupt(ds, [s.sample_id for s in ds.samples][:40])           # 40 of 60
    fs = detect(MetadataAnomaly, ds)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
    assert "40 of 60 files could not be read" in fs[0].reason
