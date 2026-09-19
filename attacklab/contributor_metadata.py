"""§8.3 — invent contributor + batch metadata (COCO has no native contributor field).

Partitions a dataset into synthetic contributors (default 4 at 40/30/20/10), assigns each
contributor two batches, and writes a ``contributors.yaml`` sidecar — tier 1 of the resolution
precedence, so the returned samples carry ``contributor_source="sidecar"``.

Must run BEFORE any attack script that takes a ``target_contributor``.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from cva.detectors.data._stub_types import Dataset


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
    n = len(dataset)
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
                               batch=assign[s.sample_id]["batch"], contributor_source="sidecar")
           for s in dataset.samples]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_c: dict[str, dict[str, list[str]]] = {}
    for sid, a in sorted(assign.items()):
        by_c.setdefault(a["contributor"], {}).setdefault(a["batch"], []).append(sid)
    (out / "contributors.yaml").write_text(
        yaml.safe_dump({"contributors": by_c}, sort_keys=True), encoding="utf-8")
    manifest = {"attack": "contributor_metadata", "seed": seed, "split": list(split),
                "names": list(names), "assignments": dict(sorted(assign.items()))}
    (out / "contributors.manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return Dataset(new, dataset.categories), manifest


def read_sidecar(path: Path | str) -> dict[str, tuple[str, str]]:
    """sample_id -> (contributor, batch) from a ``contributors.yaml`` (what a loader would do)."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, tuple[str, str]] = {}
    for c, batches in doc["contributors"].items():
        for b, ids in batches.items():
            for sid in ids:
                out[sid] = (c, b)
    return out
