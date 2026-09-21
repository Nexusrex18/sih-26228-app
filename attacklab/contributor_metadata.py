"""§8.3 — invent contributor + batch metadata (COCO has no native contributor field).

Partitions a dataset into synthetic contributors (default 4 at 40/30/20/10), assigns each
contributor two batches, and writes a ``contributors.yaml`` sidecar IN THE FORMAT THE REAL LOADERS
READ (flat ``file-stem: contributor``) — tier 1 of the resolution precedence, so the returned samples
carry ``contributor_source=ContributorSource.SIDECAR``. Batches are recorded in the manifest.

Must run BEFORE any attack script that takes a ``target_contributor``.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from ._types import ContributorSource, Dataset


def assign_contributors(dataset: Dataset, out_dir: Path | str, seed: int,
                        split: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1),
                        names: tuple[str, ...] | None = None,
                        batches_per_contributor: int = 2) -> tuple[Dataset, dict]:
    if abs(sum(split) - 1.0) > 1e-9:
        raise ValueError("split must sum to 1")
    names = names or tuple("ABCDEFGH"[: len(split)])
    if len(names) != len(split):
        raise ValueError("names and split differ in length")
    rng = np.random.default_rng(seed)
    n = len(dataset.samples)
    order = rng.permutation(n)
    counts = [int(np.floor(f * n)) for f in split]
    for i in range(n - sum(counts)):                  # largest-remainder, deterministic
        counts[i % len(counts)] += 1
    assign: dict[str, dict] = {}
    pos = 0
    for name, c in zip(names, counts, strict=False):
        for idx in order[pos:pos + c]:
            sid = dataset.samples[int(idx)].sample_id
            assign[sid] = {"contributor": name,
                           "batch": f"{name}-b{int(rng.integers(0, batches_per_contributor))}"}
        pos += c
    new = [dataclasses.replace(s, contributor=assign[s.sample_id]["contributor"],
                               batch=assign[s.sample_id]["batch"], contributor_source=ContributorSource.SIDECAR)
           for s in dataset.samples]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # contributors.yaml in the format the REAL loaders read (cva/loaders/datasets/base.py:
    # `_load_sidecar` — a flat `name-or-stem: contributor` mapping). An earlier, nested layout was
    # silently ignored by the loader, so a dataset built here lost every contributor on load.
    flat = {Path(sm.path).stem: assign[sm.sample_id]["contributor"] for sm in dataset.samples}
    (out / "contributors.yaml").write_text(yaml.safe_dump(dict(sorted(flat.items())), sort_keys=True),
                                           encoding="utf-8")
    manifest = {"attack": "contributor_metadata", "seed": seed, "split": list(split),
                "names": list(names), "assignments": dict(sorted(assign.items()))}
    (out / "contributors.manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return Dataset(new, dataset.categories), manifest


def read_sidecar(path: Path | str) -> dict[str, tuple[str, str]]:
    """sample_id -> (contributor, batch). Contributors come from ``contributors.yaml`` exactly as the
    real loader reads it (flat ``stem: contributor``); the loader has no batch concept, so batches
    come from the sibling ``contributors.manifest.json``."""
    path = Path(path)
    flat = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    manifest = json.loads((path.parent / "contributors.manifest.json").read_text())
    out: dict[str, tuple[str, str]] = {}
    for sid, a in manifest["assignments"].items():
        out[sid] = (flat.get(sid, a["contributor"]), a["batch"])
    return out
