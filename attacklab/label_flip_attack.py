"""§8.1 — label flipping.

Modes:
  ``random``     each chosen sample gets a uniformly random *other* class
  ``targeted``   every chosen sample is relabelled to ``target_class`` (steering)
  ``class_pair`` only samples of ``class_pair[0]`` are chosen, all relabelled to ``class_pair[1]``
                 — with a high ``flip_rate`` on one contributor this is *systematic mislabelling*

The flip applies to the sample's dominant category (all labels carrying it). Images are not
touched; only ``Sample.labels`` changes.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Literal

import numpy as np

from cva.detectors.data.base import dominant_category

from ._types import Dataset, Sample
from cva.detectors.data.base import dominant_category


def flip_labels(dataset: Dataset, seed: int, flip_rate: float,
                mode: Literal["random", "targeted", "class_pair"] = "random",
                target_contributor: str | None = None, target_class: int | None = None,
                class_pair: tuple[int, int] | None = None) -> tuple[Dataset, dict]:
    if not 0.0 <= flip_rate <= 1.0:
        raise ValueError("flip_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    cats = [c.category_id for c in dataset.categories]

    def eligible(s: Sample) -> bool:
        d = dominant_category(s)
        if d is None:
            return False
        if target_contributor is not None and s.contributor != target_contributor:
            return False
        if mode == "targeted":
            if target_class is None:
                raise ValueError("targeted mode needs target_class")
            return d != target_class
        if mode == "class_pair":
            if class_pair is None:
                raise ValueError("class_pair mode needs class_pair")
            return d == class_pair[0]
        return True

    pool = sorted(s.sample_id for s in dataset.samples if eligible(s))
    k = int(round(flip_rate * len(pool)))
    chosen = set(rng.choice(pool, size=k, replace=False).tolist()) if k else set()
    flips: dict[str, dict] = {}
    new: list[Sample] = []
    for s in dataset.samples:
        if s.sample_id not in chosen:
            new.append(s)
            continue
        old = dominant_category(s)
        if mode == "random":
            new_cat = int(rng.choice([c for c in cats if c != old]))
        elif mode == "targeted":
            new_cat = int(target_class)                       # type: ignore[arg-type]
        else:
            new_cat = int(class_pair[1])                      # type: ignore[index]
        labels = [dataclasses.replace(lb, category_id=new_cat) if lb.category_id == old else lb
                  for lb in s.labels]
        flips[s.sample_id] = {"original_label": int(old), "flipped_label": new_cat}
        new.append(dataclasses.replace(s, labels=labels))
    manifest = {"attack": "label_flip", "seed": seed, "mode": mode, "flip_rate": flip_rate,
                "target_contributor": target_contributor, "target_class": target_class,
                "class_pair": list(class_pair) if class_pair else None,
                "flips": dict(sorted(flips.items()))}
    return Dataset(new, dataset.categories), manifest


def manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, indent=1)
