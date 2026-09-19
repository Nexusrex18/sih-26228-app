"""Script-generated batch — the ground truth for ``data.metadata_anomaly``.

Re-encodes every image of one contributor through a single pipeline: one fixed size, one JPEG
quality (so one quantisation table), EXIF stripped except an encoder ``Software`` string, and
mtimes written within milliseconds of each other. Not an *attack* on the model — a signature of
a batch that was processed as a set — which is exactly why the detector defaults ``nature`` to
``quality``. Deterministic (no randomness): ``seed`` is recorded in the manifest for uniformity only.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from PIL import Image

from cva.detectors.data._stub_types import Dataset, Sample, sha256_file

from .synth_dataset import EPOCH0


def script_generate_batch(dataset: Dataset, out_dir: Path | str, seed: int, target_contributor: str,
                          fixed_size: tuple[int, int] = (64, 64), quality: int = 80,
                          software: str = "ImageMagick 7.1.0-62") -> tuple[Dataset, dict]:
    out = Path(out_dir) / "script_batch"
    out.mkdir(parents=True, exist_ok=True)
    fw, fh = fixed_size
    new: list[Sample] = []
    changed: dict[str, dict] = {}
    t = EPOCH0 + 10_000_000.0
    for s in dataset.samples:
        if s.contributor != target_contributor:
            new.append(s)
            continue
        with Image.open(s.path) as im:
            img = im.convert("RGB").resize((fw, fh), Image.BILINEAR)
        path = out / f"{s.sample_id}.jpg"
        # EXIF fully stripped; the encoder identifies itself in the JFIF comment segment
        img.save(path, "JPEG", quality=quality, comment=software.encode())
        t += 0.002                                         # 2 ms apart: a script, not a collection
        os.utime(path, (t, t))
        sx, sy = fw / s.width, fh / s.height
        labels = tuple(dataclasses.replace(lb, bbox=(lb.bbox[0] * sx, lb.bbox[1] * sy,
                                                     lb.bbox[2] * sx, lb.bbox[3] * sy))
                       if lb.bbox else lb for lb in s.labels)
        new.append(dataclasses.replace(s, path=path, content_sha256=sha256_file(path),
                                       width=fw, height=fh, labels=labels))
        changed[s.sample_id] = {"sha256": new[-1].content_sha256}
    manifest = {"attack": "script_generated_batch", "seed": seed,
                "target_contributor": target_contributor, "fixed_size": [fw, fh],
                "quality": quality, "software": software, "changed": dict(sorted(changed.items()))}
    return Dataset(new, dataset.categories), manifest
