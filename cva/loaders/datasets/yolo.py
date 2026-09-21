"""YOLOLoader and the YOLO writer — where the bbox arithmetic actually happens.

| Format | Box                            | Units                  |
|--------|--------------------------------|------------------------|
| COCO   | [x_min, y_min, w, h], top-left | absolute pixels        |
| YOLO   | [x_centre, y_centre, w, h]     | normalised to the image|

Internally we store COCO-style absolute pixels whatever the source, so this module is the
only place the normalised form exists. That is deliberate: normalising at ingest throws
away the pixel grid, and the pixel grid is exactly what an `image_crop` evidence needs.

It is also why `Sample.width`/`height` are mandatory. The YOLO direction is UNRECOVERABLE
without them — a normalised box carries no scale, so a sample whose image failed to decode
cannot have its boxes read at all, and inventing a default size would silently place every
box in the wrong part of the wrong image.
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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def yolo_to_absolute(xc: float, yc: float, w: float, h: float,
                     img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """(x_centre, y_centre, w, h) normalised → (x_min, y_min, w, h) absolute."""
    aw, ah = w * img_w, h * img_h
    return (xc * img_w - aw / 2.0, yc * img_h - ah / 2.0, aw, ah)


def absolute_to_yolo(x: float, y: float, w: float, h: float,
                     img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """The exact inverse. Kept beside its partner so the two cannot drift apart."""
    return ((x + w / 2.0) / img_w, (y + h / 2.0) / img_h, w / img_w, h / img_h)


def _read_data_yaml(root: Path) -> dict:
    p = root / "data.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _label_for(img: Path, images_dir: Path, labels_dir: Path) -> Path:
    try:
        rel = img.relative_to(images_dir)
    except ValueError:
        rel = Path(img.name)
    return labels_dir / rel.with_suffix(".txt")


class YOLOLoader:
    name = "YOLOLoader"

    def supports(self, path: Path) -> bool:
        path = Path(path)
        return (path / "images").is_dir() and (path / "labels").is_dir()

    def load(self, path: Path) -> InMemoryDataset:
        root = Path(path)
        images_dir, labels_dir = root / "images", root / "labels"
        meta = _read_data_yaml(root)
        sidecar = _load_sidecar(root)

        raw_names = meta.get("names") or {}
        if isinstance(raw_names, list):
            raw_names = dict(enumerate(raw_names))
        names = {int(k): str(v) for k, v in raw_names.items()}

        files = sorted(p for p in images_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

        # Two passes: the canonical mapping has to exist before any label can be indexed
        # against it, and the class ids present in the files are the ground truth — a
        # data.yaml that omits a class must not make that class's boxes disappear.
        present: set[int] = set(names)
        parsed: dict[Path, list[tuple[int, float, float, float, float]]] = {}
        for img in files:
            rows: list[tuple[int, float, float, float, float]] = []
            lf = _label_for(img, images_dir, labels_dir)
            if lf.exists():
                for line in lf.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        rows.append((cls, *(float(v) for v in parts[1:5])))
                        present.add(cls)
                    except ValueError:
                        continue
            parsed[img] = rows

        # `source_ids` in data.yaml carries the ORIGINATING ids (a COCO category_id, say)
        # through an export. Without reading it back the YOLO tree is a lossy format: the
        # dense index becomes the only surviving identifier and "class 3" silently changes
        # meaning between the two files. When it is absent — a YOLO-native dataset, which
        # has no other identifier — the dense index IS the source id, honestly.
        raw_src = meta.get("source_ids") or {}
        if isinstance(raw_src, list):
            raw_src = dict(enumerate(raw_src))
        src_of = {int(k): v for k, v in raw_src.items()}

        ordered = sorted(present)
        cats, _ = build_categories([src_of.get(c, c) for c in ordered],
                                   {src_of.get(c, c): names.get(c, str(c)) for c in ordered})
        by_source = {c.source_id: c.category_id for c in cats}
        id_map = {c: by_source[src_of.get(c, c)] for c in ordered}

        samples: list[Sample] = []
        for img in files:
            try:
                fpath, w, h = probe_image(images_dir, str(img.relative_to(images_dir)))
            except UnsafeArtifact:
                raise
            except OSError:
                continue

            labels: list[Label] = []
            for i, (cls, xc, yc, bw, bh) in enumerate(parsed[img]):
                if cls not in id_map:
                    continue
                labels.append(Label(
                    category_id=id_map[cls],
                    bbox=yolo_to_absolute(xc, yc, bw, bh, w, h),
                    label_id=f"{img.stem}:{i}",
                ))

            contributor, source = resolve_contributor(
                root, fpath, sidecar, meta.get("contributor"))
            batch, group_source = resolve_grouping(root, fpath)
            sample_meta = {"file_name": img.name, "format": "yolo"}
            if group_source:
                sample_meta["source"] = group_source

            samples.append(Sample(
                sample_id=img.stem,
                content_sha256=sha256_file(fpath),
                path=fpath,
                width=w,
                height=h,
                labels=labels,
                contributor=contributor,
                contributor_source=source,
                batch=batch,
                source_meta=sample_meta,
            ))

        return InMemoryDataset(samples=samples, categories=cats, root=root)


def to_yolo(dataset: InMemoryDataset, out_dir: Path) -> Path:
    """Write a canonical Dataset back out as a YOLO tree.

    This exists for the round-trip gate, and the round trip is not a formality: it is the
    only test that exercises the category mapping in BOTH directions. A loader that
    silently passed COCO ids straight through instead of building a dense mapping reads
    back identically from its own output and only fails on someone else's data.
    """
    out_dir = Path(out_dir)
    images, labels = out_dir / "images", out_dir / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    for s in dataset.samples:
        dest = images / f"{s.sample_id}{s.path.suffix}"
        dest.write_bytes(s.path.read_bytes())
        rows = []
        for lb in s.labels:
            if lb.bbox is None:
                continue
            xc, yc, bw, bh = absolute_to_yolo(*lb.bbox, s.width, s.height)
            # 6 decimals is the YOLO convention; it is also why the round-trip test
            # compares with a pixel tolerance rather than asserting equality.
            rows.append(f"{lb.category_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (labels / f"{s.sample_id}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))

    import yaml
    (out_dir / "data.yaml").write_text(yaml.safe_dump({
        "names": {c.category_id: c.name for c in dataset.categories},
        # The COCO source ids are carried across so the mapping survives the round trip.
        # Without this the tree is a lossy export: "class 3" would mean the dense index and
        # the original id would be gone, which is the exact failure build_categories exists
        # to prevent.
        "source_ids": {c.category_id: c.source_id for c in dataset.categories},
    }, sort_keys=True))
    return out_dir
