"""Deterministic demo and selftest corpora — committed as a GENERATOR, never as bytes.

`python -m cva.fixtures --out artifacts/fixtures` writes:

    demo_coco/instances.json + demo_coco/images/*.png   16 images, 2 categories, 2 sources
    demo_model.onnx                                     a tiny classifier, seeded weights

Everything derives from `--seed`, so two runs write identical files and the selftest's
report can be byte-compared (V10). The corpus is small on purpose: `cva selftest` has to
fit demo step 0's thirty seconds while still running the real backbone over real images,
which is the load most likely to hide an egress (§5.6).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

N_IMAGES = 16
SIZE = 64
CATEGORIES = ({"id": 1, "name": "red_square"}, {"id": 2, "name": "blue_disc"})
SOURCES = ("team_a", "team_b")


@dataclass(frozen=True)
class Fixtures:
    root: Path
    dataset: Path
    model: Path


def _image(idx: int, cls: int, seed: int) -> tuple[bytes, list[float]]:
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed * 1000 + idx)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    base = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.float32) * (60 / SIZE) + 60
    base += rng.normal(0, 6, base.shape)
    x0, y0 = (int(v) for v in rng.integers(8, 24, 2))
    w = h = int(rng.integers(16, 28))
    if cls == 0:
        base[y0:y0 + h, x0:x0 + w] = (210, 40, 40)
    else:
        mask = (xx - (x0 + w // 2)) ** 2 + (yy - (y0 + h // 2)) ** 2 <= (w // 2) ** 2
        base[mask] = (40, 60, 210)
    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue(), [float(x0), float(y0), float(w), float(h)]


def build_dataset(root: Path, seed: int = 42, n: int = N_IMAGES) -> Path:
    ds = root / "demo_coco"
    (ds / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(n):
        cls = i % 2
        blob, bbox = _image(i, cls, seed)
        name = f"img_{i:03d}.png"
        (ds / "images" / name).write_bytes(blob)
        images.append({"id": i, "file_name": name, "width": SIZE, "height": SIZE,
                       "source": SOURCES[(i // 2) % len(SOURCES)]})
        annotations.append({"id": i, "image_id": i, "category_id": CATEGORIES[cls]["id"],
                            "bbox": bbox, "iscrowd": 0})
    (ds / "instances.json").write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": list(CATEGORIES)},
        indent=2, sort_keys=True))
    return ds


def build_model(path: Path, seed: int = 42) -> Path:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    net = nn.Sequential(
        # padding=1 makes the map 32x32, so the pool's output size divides it: ONNX export
        # of adaptive_avg_pool2d refuses sizes that are not a factor of the input.
        nn.Conv2d(3, 4, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
        nn.Flatten(), nn.Linear(16, len(CATEGORIES))).eval()
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, (dummy,), str(path), dynamo=False, opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    return path


def build(out: Path, seed: int = 42) -> Fixtures:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return Fixtures(out, build_dataset(out, seed), build_model(out / "demo_model.onnx", seed))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cva.fixtures")
    ap.add_argument("--out", default="artifacts/fixtures")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    fx = build(Path(a.out), a.seed)
    print(f"fixtures -> {fx.root} ({N_IMAGES} images, model {fx.model.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
