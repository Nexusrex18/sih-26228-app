"""PS §2.3 names reproducible attack generation as a deliverable. Same seed + same inputs must
give an identical manifest AND identical file bytes, twice, independent of the global RNG."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from attacklab.annotation_geometry_attack import tamper_geometry
from attacklab.contributor_metadata import assign_contributors, read_sidecar
from attacklab.label_flip_attack import flip_labels
from attacklab.near_dup_ood_attack import inject_near_duplicates, inject_ood
from attacklab.negative_space_attack import drop_annotations
from attacklab.script_batch_attack import script_generate_batch
from attacklab.synth_dataset import make_clean_dataset
from cva.detectors.data._stub_types import sha256_file
from cva.detectors.data.base import sample_by_id


def _dump(m) -> str:
    return json.dumps(m, sort_keys=True)


def _hashes(ds) -> dict:
    return {s.sample_id: sha256_file(s.path) for s in ds.samples}


def _build(root, seed=5):
    base = make_clean_dataset(root / "base", seed=seed, n=60)
    base, m0 = assign_contributors(base, root / "base", seed=seed)
    flipped, m1 = flip_labels(base, seed, 0.3, "random", target_contributor="A")
    dup, m2 = inject_near_duplicates(base, root, seed, 3, 4, "B", conflict_rate=0.25)
    ood, m3 = inject_ood(base, root, seed, 6, "B")
    sb, m4 = script_generate_batch(base, root, seed, "C")
    geo, m5 = tamper_geometry(base, seed, "B", shrink=0.5, shift=0.3, fraction=0.8)
    two, _ = assign_contributors(make_clean_dataset(root / "two", seed=seed, n=60, objects=2),
                                 root / "two", seed=seed)
    neg, m6 = drop_annotations(two, seed, "A", 0.5)
    return {"geometry": m5, "geometry_labels": {s.sample_id: [lb.bbox for lb in s.labels] for s in geo.samples},
            "negspace": m6, "negspace_labels": {s.sample_id: len(s.labels) for s in neg.samples},
            "synth": _hashes(base), "contrib": m0, "flip": m1, "dup": m2, "ood": m3, "script": m4,
            "dup_files": _hashes(dup), "ood_files": _hashes(ood), "script_files": _hashes(sb),
            "flip_labels": {s.sample_id: [lb.category_id for lb in s.labels] for s in flipped.samples}}


def test_identical_twice_and_independent_of_global_rng(tmp_path):
    random.seed(1); np.random.seed(1)
    a = _build(tmp_path / "run1")
    random.seed(999); np.random.seed(999)              # scrambling the global RNGs must not matter
    b = _build(tmp_path / "run2")
    # manifests carry no absolute paths, timestamps or scan ids -> compare literally
    assert _dump(a) == _dump(b)


def test_different_seed_differs(tmp_path):
    a = _build(tmp_path / "s5", seed=5)
    b = _build(tmp_path / "s6", seed=6)
    assert _dump(a["flip"]) != _dump(b["flip"])
    assert a["dup"]["sources"] != b["dup"]["sources"] or _dump(a["dup"]) != _dump(b["dup"])


def test_sidecar_roundtrip_and_source_tier(tmp_path):
    base = make_clean_dataset(tmp_path / "b", seed=2, n=50)
    ds, m = assign_contributors(base, tmp_path / "b", seed=2, split=(0.4, 0.3, 0.2, 0.1))
    side = read_sidecar(tmp_path / "b" / "contributors.yaml")
    assert set(side) == {s.sample_id for s in ds.samples}
    assert all(s.contributor_source == "sidecar" for s in ds.samples)
    assert all(side[s.sample_id] == (s.contributor, s.batch) for s in ds.samples)
    counts = {c: sum(1 for s in ds.samples if s.contributor == c) for c in "ABCD"}
    assert sum(counts.values()) == 50 and counts["A"] == 20        # 40/30/20/10 of 50


def test_flip_manifest_matches_dataset(tmp_path):
    base = make_clean_dataset(tmp_path / "b", seed=3, n=80)
    base, _ = assign_contributors(base, tmp_path / "b", seed=3)
    flipped, m = flip_labels(base, 3, 0.25, "class_pair", target_contributor="A", class_pair=(0, 1))
    assert m["flips"], "expected some flips"
    for sid, f in m["flips"].items():
        assert sample_by_id(base, sid).contributor == "A"
        assert f["original_label"] == 0 and f["flipped_label"] == 1
        assert sample_by_id(flipped, sid).labels[0].category_id == 1
    untouched = set(s.sample_id for s in base.samples) - set(m["flips"])
    assert all(sample_by_id(flipped, s).labels == sample_by_id(base, s).labels for s in untouched)


def test_flip_rejects_bad_args(tmp_path):
    base = make_clean_dataset(tmp_path / "b", seed=3, n=20)
    with pytest.raises(ValueError):
        flip_labels(base, 1, 1.5)
    with pytest.raises(ValueError):
        flip_labels(base, 1, 0.5, "targeted")
