"""Ground truth for ``data.negative_space`` — objects present in an image but deliberately left
unannotated. Drops the last label of a chosen fraction of a contributor's multi-object images
and records exactly which boxes were withheld.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ._types import Dataset


def drop_annotations(dataset: Dataset, seed: int, target_contributor: str, fraction: float = 0.6
                     ) -> tuple[Dataset, dict]:
    rng = np.random.default_rng(seed)
    new, dropped = [], {}
    for s in dataset.samples:
        if s.contributor == target_contributor and len(s.labels) >= 2 and rng.random() < fraction:
            gone = s.labels[-1]
            new.append(dataclasses.replace(s, labels=s.labels[:-1]))
            dropped[s.sample_id] = {"category_id": gone.category_id, "bbox": list(gone.bbox)}
        else:
            new.append(s)
    return Dataset(new, dataset.categories), {
        "attack": "negative_space_poisoning", "seed": seed, "target_contributor": target_contributor,
        "fraction": fraction, "dropped": dict(sorted(dropped.items()))}
