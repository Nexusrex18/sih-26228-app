from __future__ import annotations

from attacklab.near_dup_ood_attack import inject_near_duplicates
from cva.detectors.data._stub_types import stub_embeddings
from cva.detectors.data.duplicate_label_conflict import DuplicateLabelConflict
from tests.detectors.data.helpers import detect, resolve


def test_registry_row():
    assert {str(c) for c in DuplicateLabelConflict.requires} == {"DATASET_IMAGES", "DATASET_LABELS"}


def test_clean_zero_findings(clean, clean_emb):
    assert resolve(DuplicateLabelConflict, clean).runnable
    assert detect(DuplicateLabelConflict, clean, clean_emb, None) == []


def test_duplicates_with_consistent_labels_are_not_conflicts(clean, workdir):
    ds, _ = inject_near_duplicates(clean, workdir / "c0", seed=31, n_sources=4, variants_per_source=5,
                                   target_contributor="B", conflict_rate=0.0)
    assert detect(DuplicateLabelConflict, ds, stub_embeddings(ds), None) == []


def test_conflicting_labels_found(clean, workdir):
    ds, m = inject_near_duplicates(clean, workdir / "c1", seed=32, n_sources=5, variants_per_source=8,
                                   target_contributor="B", conflict_rate=0.3)
    conflicted = {sid for sid, v in m["injected"].items() if v["label_conflict"]}
    assert conflicted
    fs = detect(DuplicateLabelConflict, ds, stub_embeddings(ds), None)
    flagged = {f.target_ref for f in fs}
    assert len(flagged & conflicted) >= 0.85 * len(conflicted)
    # a flagged cluster must actually contain a ground-truth conflict
    conflict_sources = {m["injected"][c]["source"] for c in conflicted}
    for f in fs:
        src = m["injected"][f.target_ref]["source"] if f.target_ref in m["injected"] else f.target_ref
        assert src in conflict_sources
    f = next(x for x in fs if x.target_ref in conflicted)
    assert f.attack_class == "duplicate_label_conflict" and f.severity.value == "high"
    assert "contradictory labels" in f.reason and "minority" in f.reason and "'B'" in f.reason
