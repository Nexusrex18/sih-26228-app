"""COCOLoader — `instances.json` in, canonical `Dataset` out.

COCO boxes are already `[x_min, y_min, w, h]` in absolute pixels, which is the internal
representation, so the conversion work here is category mapping and contributor
resolution rather than arithmetic.
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
    read_json_safely,
    resolve_contributor,
    sha256_file,
)

#: Where the manifest is looked for, in order. A dataset root may hold the json directly.
MANIFEST_NAMES = ("instances.json", "annotations.json", "_annotations.coco.json")


def find_manifest(root: Path) -> Path | None:
    if root.is_file() and root.suffix == ".json":
        return root
    for name in MANIFEST_NAMES:
        for cand in (root / name, root / "annotations" / name):
            if cand.exists():
                return cand
    return None


class COCOLoader:
    name = "COCOLoader"

    def supports(self, path: Path) -> bool:
        """Structure, not extension. A YOLO tree also holds .json files sometimes, and the
        distinguishing fact is that a COCO manifest carries `images` and `annotations`
        arrays — so that is what gets checked, cheaply, without loading the whole file."""
        path = Path(path)
        m = find_manifest(path)
        if m is None:
            return False
        try:
            doc = read_json_safely(m)
        except (UnsafeArtifact, ValueError, OSError):
            return False
        return isinstance(doc, dict) and "images" in doc and "annotations" in doc

    def load(self, path: Path) -> InMemoryDataset:
        root = Path(path)
        manifest = find_manifest(root)
        if manifest is None:
            raise ValueError(f"{root}: no COCO manifest found ({', '.join(MANIFEST_NAMES)})")
        image_root = manifest.parent if manifest.parent != root else root
        if (root / "images").is_dir():
            image_root = root / "images"

        doc = read_json_safely(manifest)
        sidecar = _load_sidecar(root)

        cats_raw = doc.get("categories") or []
        cats, id_map = build_categories(
            [c["id"] for c in cats_raw], {c["id"]: c.get("name", str(c["id"])) for c in cats_raw})

        by_image: dict[object, list[dict]] = {}
        for a in doc.get("annotations") or []:
            by_image.setdefault(a.get("image_id"), []).append(a)

        samples: list[Sample] = []
        notes = []
        for img in doc.get("images") or []:
            file_name = img.get("file_name")
            if not file_name:
                continue
            try:
                # S6 + S7: the name comes from the manifest, which is supplier-controlled.
                fpath, w, h = probe_image(image_root, file_name)
            except UnsafeArtifact:
                raise
            except OSError:
                continue

            labels: list[Label] = []
            for i, a in enumerate(by_image.get(img.get("id"), [])):
                sid = a.get("category_id")
                if sid not in id_map:
                    continue
                bbox = a.get("bbox")
                box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])) \
                    if bbox and len(bbox) == 4 else None
                labels.append(Label(
                    category_id=id_map[sid],
                    bbox=box,                      # already absolute pixels — kept as such
                    iscrowd=bool(a.get("iscrowd", 0)),
                    label_id=str(a.get("id", f"{img.get('id')}:{i}")),
                ))

            # Tier 3 for COCO is `images[].source`, per the precedence list.
            contributor, source = resolve_contributor(
                root, fpath, sidecar, img.get("source"))

            samples.append(Sample(
                sample_id=str(img.get("id", file_name)),
                content_sha256=sha256_file(fpath),
                path=fpath,
                width=w,
                height=h,
                labels=labels,
                contributor=contributor,
                contributor_source=source,
                source_meta={"file_name": file_name, "format": "coco"},
            ))

        return InMemoryDataset(samples=samples, categories=cats, root=root, notes=notes)
