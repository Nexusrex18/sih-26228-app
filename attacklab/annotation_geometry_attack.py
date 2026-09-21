"""Ground truth for ``data.annotation_geometry`` — boxes shrunk and/or shifted with the class label
still correct (label-consistency compares classes, not coordinates, so this slips past it).

Not in the plan's §8 list, but the detector needs *some* generator to be scored against.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ._types import Dataset


def tamper_geometry(dataset: Dataset, seed: int, target_contributor: str, shrink: float = 0.5,
                    shift: float = 0.0, fraction: float = 1.0, category: int | None = None
                    ) -> tuple[Dataset, dict]:
    """Scale each chosen box's width/height by ``shrink`` about its centre, then move it right by
    ``shift`` x its own width."""
    rng = np.random.default_rng(seed)
    new, changed = [], {}
    for s in dataset.samples:
        if s.contributor != target_contributor or not s.labels or rng.random() >= fraction:
            new.append(s)
            continue
        labels = []
        for lb in s.labels:
            if lb.bbox is None or (category is not None and lb.category_id != category):
                labels.append(lb)
                continue
            x, y, w, h = lb.bbox
            cx, cy = x + w / 2 + shift * w, y + h / 2
            nw, nh = w * shrink, h * shrink
            labels.append(dataclasses.replace(lb, bbox=(cx - nw / 2, cy - nh / 2, nw, nh)))
        new.append(dataclasses.replace(s, labels=labels))
        changed[s.sample_id] = {"shrink": shrink, "shift": shift}
    return Dataset(new, dataset.categories), {
        "attack": "annotation_geometry_tamper", "seed": seed, "target_contributor": target_contributor,
        "shrink": shrink, "shift": shift, "fraction": fraction, "changed": dict(sorted(changed.items()))}
