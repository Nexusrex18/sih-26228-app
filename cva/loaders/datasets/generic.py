"""GenericDataset — the Tier-2 fallback for a layout no loader recognises.

    GenericDataset(images_dir, label_fn, contributor_fn).load()

`label_fn(path)` returns a class name (one whole-image label) or a list of
`(class_name, bbox_or_None)` pairs, bbox in `(x_min, y_min, w, h)` absolute pixels.
`contributor_fn(path)` returns a contributor string or None. Either may be omitted: with no
`label_fn` the dataset carries no labels and says so through its capabilities.

Never auto-detected. It is called by hand, which is the point of a fallback: the caller has
written the two functions that say what this layout means, and the report can only claim what
they supply. A contributor from `contributor_fn` is recorded as tier 3 (`format_field`): the
caller's own declaration, exactly as a COCO `source` field is.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cva.core.types import ContributorSource, Label, Sample
from cva.loaders.safety import UnsafeArtifact

from .base import (
    InMemoryDataset,
    _load_sidecar,
    build_categories,
    probe_image,
    resolve_contributor,
    resolve_grouping,
    sha256_file,
)
from .yolo import IMAGE_SUFFIXES

LabelFn = Callable[[Path], "str | Sequence[tuple[str, tuple[float, float, float, float] | None]]"]
ContributorFn = Callable[[Path], "str | None"]


def _as_pairs(raw: Any) -> list[tuple[str, tuple[float, float, float, float] | None]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [(raw, None)]
    return [(str(c), tuple(float(v) for v in b) if b is not None else None)  # type: ignore[misc]
            for c, b in raw]


class GenericDataset:
    name = "GenericDataset"

    def __init__(self, images_dir: Path | str, label_fn: LabelFn | None = None,
                 contributor_fn: ContributorFn | None = None) -> None:
        self.images_dir = Path(images_dir)
        self.label_fn = label_fn
        self.contributor_fn = contributor_fn

    def supports(self, path: Path) -> bool:
        return False                      # never auto-detected: see the module docstring

    def load(self, path: Path | None = None) -> InMemoryDataset:
        root = Path(path) if path is not None else self.images_dir
        sidecar = _load_sidecar(root)
        files = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        per_file = [(p, _as_pairs(self.label_fn(p)) if self.label_fn else []) for p in files]
        names = sorted({c for _, pairs in per_file for c, _ in pairs})
        cats, id_map = build_categories(names, {n: n for n in names})

        samples: list[Sample] = []
        for img, pairs in per_file:
            rel = str(img.relative_to(root))
            try:
                fpath, w, h = probe_image(root, rel)                      # S6 + S7
            except UnsafeArtifact:
                raise
            except OSError:
                continue
            declared = self.contributor_fn(img) if self.contributor_fn else None
            contributor, source = resolve_contributor(root, fpath, sidecar, declared)
            if contributor is None:
                source = ContributorSource.NONE
            batch, group_source = resolve_grouping(root, fpath)
            meta = {"file_name": rel, "format": "generic"}
            if group_source:
                meta["source"] = group_source
            samples.append(Sample(
                sample_id=rel, content_sha256=sha256_file(fpath), path=fpath, width=w, height=h,
                labels=[Label(category_id=id_map[c], bbox=b, label_id=f"{rel}:{i}")
                        for i, (c, b) in enumerate(pairs)],
                contributor=contributor, contributor_source=source, batch=batch,
                source_meta=meta))
        return InMemoryDataset(samples=samples, categories=cats, root=root)
