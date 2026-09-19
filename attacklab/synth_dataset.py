"""A deterministic synthetic *clean* dataset — the base every Module A attack script poisons.

Not a stand-in for real imagery: it exists so detectors can be exercised against known ground
truth with no external download. Classes are distinct (shape + colour) so an embedding /
linear probe separates them; nuisance (gradient background, noise, position, size, JPEG
quality, simulated camera) is randomised per image so clean images are NOT near-duplicates and
clean metadata is genuinely varied.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cva.detectors.data._stub_types import Category, Dataset, Label, Sample, sha256_file

SHAPES = ("circle", "square", "triangle", "cross")
PALETTE = ((220, 60, 60), (60, 200, 90), (70, 110, 230), (230, 200, 60),
           (200, 80, 210), (60, 210, 210))
CLASS_NAMES = ("truck", "tank", "jeep", "drone", "ship", "radar")

# (make, model, serial, jpeg_quality) — a pooled source has several cameras
CAMERAS = (("Acme", "AC-100", "SN1001", 92), ("Acme", "AC-200", "SN2002", 88),
           ("Borealis", "B7", "SN3003", 84), ("Cygnus", "CX", "SN4004", 78))

EPOCH0 = 1_700_000_000.0


def draw_object(draw: ImageDraw.ImageDraw, cls: int, cx: float, cy: float, r: float,
                jitter: np.ndarray) -> tuple[float, float, float, float]:
    col = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(PALETTE[cls % len(PALETTE)], jitter, strict=False))
    shape = SHAPES[cls % len(SHAPES)]
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    else:
        t = r / 3
        draw.rectangle([cx - r, cy - t, cx + r, cy + t], fill=col)
        draw.rectangle([cx - t, cy - r, cx + t, cy + r], fill=col)
    return (cx - r, cy - r, 2 * r, 2 * r)


def render_image(rng: np.random.Generator, w: int, h: int, classes: list[int]):
    """-> (PIL image, [(class, bbox)]). Gradient bg + noise + one or more objects."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    base = rng.uniform(40, 120, size=3).astype(np.float32)
    amp = rng.uniform(15, 60, size=3).astype(np.float32)
    arr = base[None, None, :] + amp[None, None, :] * grad[..., None]
    arr += rng.normal(0, rng.uniform(5, 18), size=arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    boxes = []
    for cls in classes:
        r = rng.uniform(0.16, 0.26) * min(w, h)
        cx = rng.uniform(r + 1, w - r - 1)
        cy = rng.uniform(r + 1, h - r - 1)
        jitter = rng.normal(0, 14, size=3)
        boxes.append((cls, draw_object(draw, cls, cx, cy, r, jitter)))
    return img, boxes


def exif_bytes(make: str, model: str, serial: str, software: str | None = None) -> bytes:
    ex = Image.Exif()
    ex[0x010F] = make
    ex[0x0110] = model
    if software:
        ex[0x0131] = software
    ex.get_ifd(0x8769)[0xA431] = serial          # BodySerialNumber
    return ex.tobytes()


def make_clean_dataset(out_dir: Path | str, seed: int = 0, n: int = 240, n_classes: int = 4,
                       size_range: tuple[int, int] = (56, 72), objects: int = 1,
                       set_mtimes: bool = True) -> Dataset:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cats = [Category(k, CLASS_NAMES[k % len(CLASS_NAMES)] + ("" if k < len(CLASS_NAMES) else str(k)), k + 1)
            for k in range(n_classes)]
    t = EPOCH0
    samples: list[Sample] = []
    for i in range(n):
        w = int(rng.integers(size_range[0], size_range[1] + 1))
        h = int(rng.integers(size_range[0], size_range[1] + 1))
        if objects == 1:
            classes = [int(i % n_classes)]            # balanced classes
        else:
            classes = [int(c) for c in rng.integers(0, n_classes, size=objects)]
        img, boxes = render_image(rng, w, h, classes)
        cam = CAMERAS[int(rng.integers(0, len(CAMERAS)))]
        path = out / "images" / f"img_{i:05d}.jpg"
        img.save(path, "JPEG", quality=cam[3], exif=exif_bytes(cam[0], cam[1], cam[2]))
        t += float(rng.exponential(45.0))
        if set_mtimes:
            os.utime(path, (t, t))
        samples.append(Sample(
            sample_id=f"img_{i:05d}", content_sha256=sha256_file(path), path=path, width=w, height=h,
            labels=tuple(Label(c, bbox=tuple(float(v) for v in bb)) for c, bb in boxes),
            source_meta={"camera": cam[1]}))
    return Dataset(samples, cats)
