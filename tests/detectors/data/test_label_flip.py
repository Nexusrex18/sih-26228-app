from __future__ import annotations

import numpy as np
import pytest

from attacklab.label_flip_attack import flip_labels
from cva.detectors.data._stub_types import ArrayEmbeddingIndex, stub_embeddings
from cva.detectors.data.label_flip import LabelConsistency, out_of_fold_probs
from tests.detectors.data.helpers import detect, resolve


def test_registry_row_has_no_optional_capability():
    assert {str(c) for c in LabelConsistency.requires} == {"DATASET_IMAGES", "DATASET_LABELS"}
    assert LabelConsistency.optional == set()          # settled: no degraded mode exists


def test_clean_near_zero(clean, clean_emb):
    assert resolve(LabelConsistency, clean).runnable
    fs = detect(LabelConsistency, clean, clean_emb, None)
    assert len(fs) <= 0.01 * len(clean), [f.reason for f in fs]     # stated tolerance: <= 1%


@pytest.mark.parametrize("rate", [0.05, 0.10, 0.20])
def test_random_flips_found_against_ground_truth(clean, rate):
    flipped, m = flip_labels(clean, seed=7, flip_rate=rate, mode="random")
    fs = detect(LabelConsistency, flipped, stub_embeddings(flipped), None)
    flagged, truth = {f.target_ref for f in fs}, set(m["flips"])
    assert len(flagged & truth) >= 0.8 * len(truth)            # recall
    assert len(flagged & truth) >= 0.8 * len(flagged)          # precision
    f = next(x for x in fs if x.target_ref in truth)
    assert f.attack_class == "label_flipping" and f.nature.value == "quality"
    assert all("labelled" in x.reason and ("nearest neighbours" in x.reason or "probe" in x.reason)
               for x in fs)
    assert any("nearest neighbours" in x.reason for x in fs)      # the k-NN signal names its evidence
    assert any("out-of-fold" in x.reason for x in fs)             # so does confident learning


def test_both_signals_raise_severity_and_are_reported_separately(clean):
    flipped, m = flip_labels(clean, seed=8, flip_rate=0.1)
    fs = detect(LabelConsistency, flipped, stub_embeddings(flipped), None)
    both = [f for f in fs if f.severity.value == "medium"]
    assert both and all("BOTH" in f.reason for f in both)
    assert all(f.confidence > 0.6 for f in both)
    sig = both[0].evidence[0].data
    assert set(sig) == {"knn", "confident_learning"}
    assert sig["knn"]["flagged"] and sig["confident_learning"]["flagged"]


def test_reason_names_contributor_tier(clean):
    flipped, m = flip_labels(clean, seed=9, flip_rate=0.1, target_contributor="A")
    fs = detect(LabelConsistency, flipped, stub_embeddings(flipped), None)
    assert fs and all("contributors.yaml sidecar" in f.reason for f in fs)


def test_out_of_fold_probs_do_not_memorise(clean):
    """The correctness property: with in-sample fitting a flipped label would be predicted
    confidently. Out-of-fold, the probe must NOT give a flipped sample its (wrong) label."""
    flipped, m = flip_labels(clean, seed=10, flip_rate=0.1)
    from cva.detectors.data.base import dominant_category
    e = stub_embeddings(flipped)
    ids = [s.sample_id for s in flipped.samples]
    y = np.array([dominant_category(s) for s in flipped.samples])
    P = out_of_fold_probs(e.vectors(ids), y, 5, 0, 1.0)
    flipped_idx = [i for i, sid in enumerate(ids) if sid in m["flips"]]
    assert np.mean([P[i, y[i]] for i in flipped_idx]) < 0.2


def test_without_embeddings_says_so_instead_of_skipping(clean):
    fs = detect(LabelConsistency, clean, None, None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
    assert "embedding" in fs[0].reason


def test_too_few_samples_is_reported_not_silent(clean):
    small = type(clean)(clean.samples[:8], clean.categories)
    fs = detect(LabelConsistency, small, stub_embeddings(small), None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
