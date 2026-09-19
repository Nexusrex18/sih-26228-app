"""ImageFolderLoader — `root/<class_name>/*.png`, the torchvision layout, classification only.

Each image gets one `Label` with no box. The class folder is a LABEL, never a contributor: a
directory is contributor evidence only if it says so (`contrib_*`), and reading a contributor
out of `cats/` would invent one for every dataset on earth (see `base.CONTRIB_DIR_PREFIXES`).

This is the loosest layout there is, so it is tried LAST in auto-detection: any directory of
folders of images matches it, and it must never win against a format with real markers.
"""
from __future__ import annotations

from pathlib import Path

from cva.core.types import Label, Sample
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


def _class_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith(".") and any(
                      p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir()))


class ImageFolderLoader:
    name = "ImageFolderLoader"

    def supports(self, path: Path) -> bool:
        path = Path(path)
        return path.is_dir() and len(_class_dirs(path)) >= 1

    def load(self, path: Path) -> InMemoryDataset:
        root = Path(path)
        sidecar = _load_sidecar(root)
        dirs = _class_dirs(root)
        cats, id_map = build_categories([d.name for d in dirs], {d.name: d.name for d in dirs})

        samples: list[Sample] = []
        for d in dirs:
            for img in sorted(p for p in d.iterdir()
                              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
                try:
                    fpath, w, h = probe_image(root, f"{d.name}/{img.name}")   # S6 + S7
                except UnsafeArtifact:
                    raise
                except OSError:
                    continue
                contributor, source = resolve_contributor(root, fpath, sidecar, None)
                # The class folder is a label, and is batch or source only if it carries the
                # `batch_` / `source_` prefix itself.
                batch, group_source = resolve_grouping(root, fpath)
                meta = {"file_name": f"{d.name}/{img.name}", "format": "imagefolder"}
                if group_source:
                    meta["source"] = group_source
                samples.append(Sample(
                    sample_id=f"{d.name}/{img.stem}", content_sha256=sha256_file(fpath),
                    path=fpath, width=w, height=h,
                    labels=[Label(category_id=id_map[d.name], bbox=None)],
                    contributor=contributor, contributor_source=source, batch=batch,
                    source_meta=meta))
        return InMemoryDataset(samples=samples, categories=cats, root=root)
