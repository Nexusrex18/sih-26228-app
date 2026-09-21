"""VOCLoader — Pascal VOC `Annotations/*.xml` + `JPEGImages/*` in, canonical `Dataset` out.

VOC boxes are `xmin, ymin, xmax, ymax`, 1-based and inclusive, so the conversion to the
internal `(x_min, y_min, w, h)` absolute-pixel form is `x = xmin - 1`, `w = xmax - xmin + 1`.
That is exact for the 1-based convention; tools that skip the `- 1` shift every box by a pixel.

The XML is supplier-controlled, so it is read with the same suspicion as COCO's JSON (S5):
a size cap before parsing, and any DTD or entity declaration refused outright. VOC files have
no use for either, and entity expansion is how a few hundred bytes become gigabytes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
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

XML_MAX_BYTES = 1 << 20          # one VOC annotation is a few hundred bytes; 1 MB is generous


def parse_annotation(path: Path) -> ET.Element:
    """Read one annotation file under the S5-style caps, or raise `UnsafeArtifact`."""
    size = path.stat().st_size
    if size > XML_MAX_BYTES:
        raise UnsafeArtifact(path, "S5", f"annotation is {size} bytes; the cap is {XML_MAX_BYTES}")
    blob = path.read_bytes()
    # The WHOLE blob, not a prefix: it is already capped at XML_MAX_BYTES, and scanning only
    # the first 64 KB let a DTD placed after 70,000 spaces through.
    low = blob.lower()
    if b"<!doctype" in low or b"<!entity" in low:
        raise UnsafeArtifact(path, "S5", "annotation declares a DTD or entity; VOC needs neither")
    try:
        return ET.fromstring(blob)
    except ET.ParseError as exc:
        raise ValueError(f"{path.name}: not valid XML ({exc})") from exc


def voc_to_absolute(xmin: float, ymin: float, xmax: float, ymax: float
                    ) -> tuple[float, float, float, float]:
    return (xmin - 1.0, ymin - 1.0, xmax - xmin + 1.0, ymax - ymin + 1.0)


def _num(node: ET.Element | None) -> float | None:
    try:
        return float(node.text) if node is not None and node.text else None
    except ValueError:
        return None


class VOCLoader:
    name = "VOCLoader"

    def supports(self, path: Path) -> bool:
        path = Path(path)
        return (path / "Annotations").is_dir() and (path / "JPEGImages").is_dir()

    def load(self, path: Path) -> InMemoryDataset:
        root = Path(path)
        ann_dir, img_dir = root / "Annotations", root / "JPEGImages"
        sidecar = _load_sidecar(root)

        parsed: list[tuple[Path, str, list[tuple[str, tuple[float, float, float, float]]]]] = []
        names: set[str] = set()
        for xml in sorted(ann_dir.glob("*.xml")):
            tree = parse_annotation(xml)
            file_name = (tree.findtext("filename") or "").strip() or f"{xml.stem}.jpg"
            objs = []
            for obj in tree.findall("object"):
                cls = (obj.findtext("name") or "").strip()
                bb = obj.find("bndbox")
                vals = [_num(bb.find(k)) if bb is not None else None
                        for k in ("xmin", "ymin", "xmax", "ymax")]
                if not cls or any(v is None for v in vals):
                    continue
                xmin, ymin, xmax, ymax = (float(v) for v in vals if v is not None)
                objs.append((cls, voc_to_absolute(xmin, ymin, xmax, ymax)))
                names.add(cls)
            parsed.append((xml, file_name, objs))

        # VOC class names ARE the ids; the dense 0-based mapping is assigned here.
        cats, id_map = build_categories(sorted(names), {n: n for n in names})

        samples: list[Sample] = []
        for xml, file_name, objs in parsed:
            try:
                fpath, w, h = probe_image(img_dir, file_name)      # S6 + S7
            except UnsafeArtifact:
                raise
            except OSError:
                continue
            contributor, source = resolve_contributor(root, fpath, sidecar, None)
            batch, group_source = resolve_grouping(root, fpath)
            meta = {"file_name": file_name, "format": "voc"}
            if group_source:
                meta["source"] = group_source
            samples.append(Sample(
                sample_id=xml.stem, content_sha256=sha256_file(fpath), path=fpath,
                width=w, height=h,
                labels=[Label(category_id=id_map[c], bbox=box, label_id=f"{xml.stem}:{i}")
                        for i, (c, box) in enumerate(objs)],
                contributor=contributor, contributor_source=source, batch=batch,
                source_meta=meta))
        return InMemoryDataset(samples=samples, categories=cats, root=root)
