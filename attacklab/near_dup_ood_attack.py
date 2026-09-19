"""§8.2 — near-duplicate flooding and OOD insertion.

``inject_near_duplicates``: pick N source images, write near-identical variants at varying
similarity (down/up-scale + recompress, small crop, colour shift, brightness), and assign them
to a target contributor. ``conflict_rate`` relabels a fraction of the variants — the ground
truth ``data.duplicate_label_conflict`` scores against.

``inject_ood``: foreign-domain imagery (smooth multi-sine textures, nothing like the shape
classes) inserted into a batch under an in-distribution label.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from cva.detectors.data._stub_types import Dataset, Label, Sample, sha256_file
from cva.detectors.data.base import dominant_category

_KINDS = ("recompress", "crop", "colour_shift", "brightness", "rescale")


def _variant(img: Image.Image, kind: str, rng: np.random.Generator) -> tuple[Image.Image, int]:
    w, h = img.size
    quality = int(rng.integers(55, 91))
    if kind == "recompress":
        return img, quality
    if kind == "rescale":
        f = float(rng.uniform(0.5, 0.8))
        small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR), quality
    if kind == "crop":
        m = float(rng.uniform(0.02, 0.06))
        box = (int(w * m), int(h * m), w - int(w * m), h - int(h * m))
        return img.crop(box).resize((w, h), Image.BILINEAR), quality
    a = np.asarray(img, dtype=np.float32)
    if kind == "colour_shift":
        a = a + rng.normal(0, 4.0, size=3)[None, None, :]
    else:
        a = a * float(rng.uniform(1.03, 1.10))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB"), quality


def inject_near_duplicates(dataset: Dataset, out_dir: Path | str, seed: int, n_sources: int = 5,
                           variants_per_source: int = 8, target_contributor: str | None = None,
                           conflict_rate: float = 0.0) -> tuple[Dataset, dict]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir) / "near_dup"
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted(s.sample_id for s in dataset.samples)
    picked = sorted(rng.choice(srcs, size=n_sources, replace=False).tolist())
    cats = [c.category_id for c in dataset.categories]
    src_c = next((s.contributor_source for s in dataset.samples if s.contributor == target_contributor),
                 "sidecar" if target_contributor else None)
    added: list[Sample] = []
    injected: dict[str, dict] = {}
    for sid in picked:
        src = dataset.sample(sid)
        with Image.open(src.path) as im:
            base = im.convert("RGB")
        for k in range(variants_per_source):
            kind = _KINDS[int(rng.integers(0, len(_KINDS)))]
            v, q = _variant(base, kind, rng)
            new_id = f"dup_{sid}_{k:02d}"
            path = out / f"{new_id}.jpg"
            v.save(path, "JPEG", quality=q)
            labels = src.labels
            conflicted = False
            if conflict_rate > 0 and rng.random() < conflict_rate:
                old = dominant_category(src)
                new_cat = int(rng.choice([c for c in cats if c != old]))
                labels = tuple(dataclasses.replace(lb, category_id=new_cat) if lb.category_id == old
                               else lb for lb in src.labels)
                conflicted = True
            added.append(Sample(new_id, sha256_file(path), path, v.size[0], v.size[1], labels,
                                target_contributor, f"{target_contributor}-injected" if target_contributor else None,
                                {"injected": "near_duplicate"}, src_c))
            injected[new_id] = {"kind": "near_duplicate", "variant": kind, "source": sid,
                                "label_conflict": conflicted, "sha256": added[-1].content_sha256}
    manifest = {"attack": "near_duplicate_flooding", "seed": seed, "sources": picked,
                "target_contributor": target_contributor, "conflict_rate": conflict_rate,
                "injected": dict(sorted(injected.items()))}
    return Dataset(list(dataset.samples) + added, dataset.categories), manifest


def _texture(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    chans = []
    for _ in range(3):
        f = rng.uniform(0.15, 0.6, size=4)
        ph = rng.uniform(0, 2 * np.pi, size=4)
        ang = rng.uniform(0, np.pi, size=4)
        acc = sum(np.sin(f[i] * (np.cos(ang[i]) * xx + np.sin(ang[i]) * yy) + ph[i]) for i in range(4))
        chans.append(acc)
    a = np.stack(chans, axis=-1)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return Image.fromarray((a * 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.6))


def inject_ood(dataset: Dataset, out_dir: Path | str, seed: int, n: int = 30,
               target_contributor: str | None = None, size: tuple[int, int] = (64, 64)
               ) -> tuple[Dataset, dict]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir) / "ood"
    out.mkdir(parents=True, exist_ok=True)
    cats = [c.category_id for c in dataset.categories]
    src_c = next((s.contributor_source for s in dataset.samples if s.contributor == target_contributor),
                 "sidecar" if target_contributor else None)
    added: list[Sample] = []
    injected: dict[str, dict] = {}
    for i in range(n):
        img = _texture(rng, *size)
        new_id = f"ood_{i:03d}"
        path = out / f"{new_id}.jpg"
        img.save(path, "JPEG", quality=90)
        cat = int(rng.choice(cats))
        w, h = size
        lb = Label(cat, bbox=(w * 0.25, h * 0.25, w * 0.5, h * 0.5))
        added.append(Sample(new_id, sha256_file(path), path, w, h, (lb,), target_contributor,
                            f"{target_contributor}-injected" if target_contributor else None,
                            {"injected": "ood"}, src_c))
        injected[new_id] = {"kind": "ood", "declared_label": cat, "sha256": added[-1].content_sha256}
    manifest = {"attack": "out_of_distribution", "seed": seed, "n": n,
                "target_contributor": target_contributor, "injected": dict(sorted(injected.items()))}
    return Dataset(list(dataset.samples) + added, dataset.categories), manifest
