"""Write any attack-lab dataset out as a REAL COCO dataset, so it can be read back by the real
``COCOLoader`` — the path every genuine submission takes.

Layout: ``<root>/instances.json``, ``<root>/images/<sample_id><ext>``, and (if any sample has a
contributor) ``<root>/contributors.yaml`` in the loader's flat ``file-stem: contributor`` format.
Images are copied with ``shutil.copy2`` so file mtimes survive — ``data.metadata_anomaly`` reads
them. Category ids are written as their ``source_id``; the loader re-derives the dense mapping.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


def export_coco(dataset, out_root: Path | str) -> Path:
    root = Path(out_root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    src_of = {c.category_id: c.source_id for c in dataset.categories}
    images, annotations, ann_id = [], [], 1
    for s in dataset.samples:
        dst = root / "images" / f"{s.sample_id}{Path(s.path).suffix}"
        shutil.copy2(s.path, dst)
        images.append({"id": s.sample_id, "file_name": dst.name, "width": s.width, "height": s.height})
        for lb in s.labels:
            annotations.append({"id": ann_id, "image_id": s.sample_id,
                                "category_id": src_of[lb.category_id],
                                "bbox": list(lb.bbox) if lb.bbox else [],
                                "iscrowd": int(lb.iscrowd)})
            ann_id += 1
    doc = {"images": images, "annotations": annotations,
           "categories": [{"id": c.source_id, "name": c.name} for c in dataset.categories]}
    (root / "instances.json").write_text(json.dumps(doc), encoding="utf-8")
    side = {s.sample_id: s.contributor for s in dataset.samples if s.contributor}
    if side:
        (root / "contributors.yaml").write_text(yaml.safe_dump(side, sort_keys=True), encoding="utf-8")
    return root
