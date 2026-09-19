"""What is STILL a stand-in for Module A — and the real types it now re-exports.

The dataset-side contract types are real now: ``Category``, ``Label``, ``Annotation``, ``Sample``,
``Dataset`` (a Protocol) and ``EmbeddingIndex`` live in ``cva/core/types.py`` and are re-exported
here unchanged, so detectors import them from one place. There is no stand-in ``Dataset`` any more:
the earlier one offered ``sample(id)``, ``category_name(id)`` and ``len()``, conveniences the real
``InMemoryDataset`` does not have, and five of six detectors crashed on a real dataset because of it.
Detectors reach for datasets only through ``base.sample_by_id`` / ``category_name`` / ``n_samples``.

What remains a stand-in, to be replaced by Backend's feature cache + the vendored DINOv2 backbone:
``ArrayEmbeddingIndex`` and ``stub_embeddings`` (a fixed function of the image — NOT evidence about a
real backbone). ``sha256_file`` is a plain utility.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cva.core.types import (  # noqa: F401  — re-exported: the one import point for Module A
    Annotation,
    Category,
    ContributorSource,
    Dataset,
    EmbeddingIndex,
    Label,
    Sample,
)


class ArrayEmbeddingIndex:
    """In-memory EmbeddingIndex over unit-normalised vectors. Exact cosine k-NN."""

    def __init__(self, sample_ids: Sequence[str], vectors: np.ndarray,
                 extractor_id: str = "stub", extractor_version: str = "0"):
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self._v = v / np.maximum(norms, 1e-12)
        self._ids = list(sample_ids)
        self._row = {sid: i for i, sid in enumerate(self._ids)}
        self.extractor_id, self.extractor_version = extractor_id, extractor_version
        self.dim = int(self._v.shape[1])

    def vector(self, sample_id: str) -> np.ndarray | None:
        i = self._row.get(sample_id)
        return None if i is None else self._v[i]

    def vectors(self, sample_ids: Sequence[str]) -> np.ndarray:
        return self._v[[self._row[s] for s in sample_ids]]

    def knn(self, sample_id: str, k: int) -> list[tuple[str, float]]:
        i = self._row[sample_id]
        sims = self._v @ self._v[i]
        sims[i] = -np.inf
        k = min(k, len(self._ids) - 1)
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._ids[j], float(sims[j])) for j in top]

    def patch_tokens(self, sample_id: str) -> np.ndarray | None:
        return None

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._row


def _semantic_block(a: np.ndarray) -> np.ndarray:
    """Class-separable descriptor of an HxWx3 float image in [0,1]: hue histogram of the pixels
    that stand out from the median colour, plus their fill ratio / extent / aspect. Stands in for
    the semantic structure a real backbone (DINOv2) provides; it is tuned to the synthetic shape
    classes and says nothing about real imagery."""
    med = np.median(a.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(a - med, axis=-1)
    m = dist > 0.32
    out = np.zeros(12 + 3, dtype=np.float32)
    if m.sum() < 4:
        return out
    px = a[m]
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx, mn = px.max(axis=1), px.min(axis=1)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    out[:12] = np.histogram(h, bins=12, range=(0, 1), weights=d)[0] / max(d.sum(), 1e-6)
    ys, xs = np.nonzero(m)
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    out[12] = m.sum() / float(bh * bw)                 # fill ratio: circle .79, square 1, triangle .5
    out[13] = m.mean()                                 # extent
    out[14] = min(bh, bw) / max(bh, bw)                # aspect
    return out


def stub_embeddings(dataset: Dataset, dim: int = 64, size: int = 16, seed: int = 0,
                    semantic_weight: float = 0.8) -> ArrayEmbeddingIndex:
    """Deterministic stand-in for the DINOv2 backbone: [a class-separable descriptor | a fixed
    random projection of a down-sampled colour image], each block L2-normalised. The pixel block
    keeps near-duplicates close and same-class-but-different images apart; the semantic block
    keeps classes separable, as a real backbone's would. It is NOT evidence about real-backbone
    behaviour."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pix, sem = [], []
    for s in dataset.samples:
        with Image.open(s.path) as im:
            rgb = im.convert("RGB")
            a = np.asarray(rgb.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
            a32 = np.asarray(rgb.resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0
        pix.append((a - a.mean(axis=(0, 1), keepdims=True)).reshape(-1))   # per-IMAGE centring
        sem.append(_semantic_block(a32))
    X = np.stack(pix) if pix else np.zeros((0, size * size * 3), np.float32)
    S = np.stack(sem) if sem else np.zeros((0, 15), np.float32)
    proj = rng.standard_normal((X.shape[1], dim)).astype(np.float32) / np.sqrt(X.shape[1])
    Xc = X @ proj if len(X) else np.zeros((0, dim), np.float32)   # a backbone is a function of the
    # image alone, never of the rest of the dataset
    Xc = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
    Sc = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    return ArrayEmbeddingIndex([s.sample_id for s in dataset.samples],
                               np.concatenate([Xc, semantic_weight * Sc], axis=1),
                               extractor_id="stub-semantic+pixel", extractor_version="4")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
