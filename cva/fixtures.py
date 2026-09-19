"""Deterministic demo and selftest corpora — committed as a GENERATOR, never as bytes.

`python -m cva.fixtures --out artifacts/fixtures` writes:

    demo_coco/instances.json + demo_coco/images/*.png   16 images, 2 categories, 2 sources
    demo_model.onnx                                     a tiny classifier, seeded weights

`--kind mixed_contributors` writes `mixed_contributors/` instead: a 245-image COCO dataset with
five contributors whose near-duplicate content the real detectors flag (plan V15).

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
from typing import Any

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


# (contributor, n images, how many of them are near-duplicates and of how many bases).
# `dup_groups` is a tuple of near-duplicate group sizes: each group is ONE base image plus tiny
# noise, so the real `data.near_dup` (pHash + embedding cosine) clusters exactly those images.
# The rest of a contributor's images are unique random scenes and should stay unflagged.
MIXED_CONTRIBUTORS: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("guilty", 30, (10, 10, 10)),          # 30/30 in near-duplicate floods
    ("tiny", 5, (3,)),                     # 3/5 = 60% raw rate on five images
    ("big", 150, (3, 3, 3, 3)),            # 12/150 = 8%: the plan's 800/10,000, scaled down
    ("clean_a", 30, ()),
    ("clean_b", 30, ()),
)


def _scene(rng: Any) -> Any:
    """A distinct random 64x64 RGB scene: smooth colour field plus a few random shapes."""
    import numpy as np
    from PIL import Image

    coarse = rng.uniform(30, 225, (6, 6, 3)).astype(np.uint8)
    field = np.asarray(Image.fromarray(coarse).resize((SIZE, SIZE), Image.Resampling.BICUBIC),
                       dtype=np.float32).copy()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for _ in range(int(rng.integers(3, 6))):
        colour = rng.uniform(0, 255, 3)
        x0, y0 = (int(v) for v in rng.integers(0, SIZE - 12, 2))
        w, h = (int(v) for v in rng.integers(8, 28, 2))
        if rng.integers(0, 2):
            field[y0:y0 + h, x0:x0 + w] = colour
        else:
            cx, cy = x0 + w // 2, y0 + h // 2
            field[((xx - cx) / max(w / 2, 1)) ** 2 + ((yy - cy) / max(h / 2, 1)) ** 2 <= 1] = colour
    return field


def _png(pixels: Any) -> bytes:
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def build_mixed_contributors(out: Path, seed: int = 42) -> Path:
    """`<out>/mixed_contributors/` — a COCO dataset where the REAL detectors flag specific
    contributors (plan V15). Each image's `source` names its contributor.

      guilty   30 images, three floods of ten near-duplicates   -> every image flagged
      tiny      5 images, three of them near-duplicates         -> 3/5 flagged (60% raw rate)
      big     150 images, four near-duplicate triples           -> 12/150 flagged (8%)
      clean_a/clean_b  30 unique images each                    -> none flagged

    A near-duplicate is its group's base scene plus Gaussian noise (sigma 2), so pHash and the
    embedding both agree; a unique image is its own random scene. Every image carries the same
    single label: the scenes have no class-dependent content, so per-image labels would only
    hand `data.label_consistency` genuine but unintended label-noise findings. Same seed, same bytes.
    """
    import numpy as np

    ds = Path(out) / "mixed_contributors"
    (ds / "images").mkdir(parents=True, exist_ok=True)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    idx = 0
    for c_no, (name, n, dup_groups) in enumerate(MIXED_CONTRIBUTORS):
        plan: list[int | None] = []                  # near-duplicate group id, or None if unique
        for g_no, size in enumerate(dup_groups):
            plan += [g_no] * size
        plan += [None] * (n - len(plan))
        for j, group in enumerate(plan):
            if group is None:
                pixels = _scene(np.random.default_rng([seed, 1, c_no, j]))
            else:
                base = _scene(np.random.default_rng([seed, 2, c_no, group]))
                noise = np.random.default_rng([seed, 3, c_no, j]).normal(0, 2, base.shape)
                pixels = base + noise
            fname = f"img_{idx:03d}.png"
            (ds / "images" / fname).write_bytes(_png(pixels))
            images.append({"id": idx, "file_name": fname, "width": SIZE, "height": SIZE,
                           "source": name})
            annotations.append({"id": idx, "image_id": idx, "category_id": CATEGORIES[0]["id"],
                                "bbox": [8.0, 8.0, 48.0, 48.0], "iscrowd": 0})
            idx += 1
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
    ap.add_argument("--kind", choices=("demo", "mixed_contributors"), default="demo",
                    help="demo: selftest corpus + model; mixed_contributors: the V15 dataset")
    a = ap.parse_args(argv)
    if a.kind == "mixed_contributors":
        ds = build_mixed_contributors(Path(a.out), a.seed)
        n = sum(c[1] for c in MIXED_CONTRIBUTORS)
        print(f"fixtures -> {ds} ({n} images, {len(MIXED_CONTRIBUTORS)} contributors)")
        return 0
    fx = build(Path(a.out), a.seed)
    print(f"fixtures -> {fx.root} ({N_IMAGES} images, model {fx.model.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
