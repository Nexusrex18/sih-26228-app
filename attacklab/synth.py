"""Offline, seeded, reproducible image corpus. No downloads, no network, ever.

Five shape classes on textured backgrounds at 32x32. Deliberately learnable in seconds on
CPU so the whole Module B pipeline can be exercised end to end, and deliberately *not*
trivially separable, so a backdoor has to compete with real features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ["square", "circle", "triangle", "cross", "bar",
           "ring", "diamond", "tee"]
IMG = 32


def _bg(rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(0.15, 0.55, size=(3, 1, 1))
    img = np.broadcast_to(base, (3, IMG, IMG)).copy()
    img += rng.normal(0, 0.14, size=(3, IMG, IMG))
    for _ in range(rng.integers(4, 9)):
        y, x = rng.integers(0, IMG, 2)
        h, w = rng.integers(3, 9, 2)
        img[:, y:y + h, x:x + w] += rng.normal(0, 0.22, size=(3, 1, 1))
    return img


def _draw(img: np.ndarray, cls: int, rng: np.random.Generator) -> None:
    col = rng.uniform(0.40, 0.85, size=3)
    s = int(rng.integers(9, 15))
    cy, cx = rng.integers(s // 2 + 2, IMG - s // 2 - 2, 2)
    ys, xs = np.ogrid[:IMG, :IMG]
    if cls == 0:
        m = (abs(ys - cy) < s // 2) & (abs(xs - cx) < s // 2)
    elif cls == 1:
        m = (ys - cy) ** 2 + (xs - cx) ** 2 < (s // 2) ** 2
    elif cls == 2:
        m = (abs(xs - cx) <= (ys - cy + s // 2) // 2) & (ys >= cy - s // 2) & (ys <= cy + s // 2)
    elif cls == 3:
        m = ((abs(ys - cy) < s // 2) & (abs(xs - cx) < 2)) | \
            ((abs(xs - cx) < s // 2) & (abs(ys - cy) < 2))
    elif cls == 4:
        m = (abs(ys - cy) < 2) & (abs(xs - cx) < s // 2)
    elif cls == 5:
        r = (ys - cy) ** 2 + (xs - cx) ** 2
        m = (r < (s // 2) ** 2) & (r > (s // 2 - 2) ** 2)
    elif cls == 6:
        m = (abs(ys - cy) + abs(xs - cx)) < s // 2
    else:
        m = ((abs(ys - cy) < 2) & (abs(xs - cx) < s // 2)) | \
            ((abs(xs - cx) < 2) & (ys >= cy) & (ys < cy + s // 2))
    for c in range(3):
        img[c][m] = col[c]


@dataclass
class Corpus:
    x: np.ndarray            # (N, 3, 32, 32) float32 in [0, 1]
    y: np.ndarray            # (N,) int64
    poisoned: np.ndarray     # (N,) bool — ground truth for the benchmark

    def __len__(self) -> int:
        return len(self.y)


def make_corpus(n: int, seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    xs = np.empty((n, 3, IMG, IMG), dtype=np.float32)
    ys = rng.integers(0, len(CLASSES), n).astype(np.int64)
    for i in range(n):
        img = _bg(rng)
        _draw(img, int(ys[i]), rng)
        xs[i] = np.clip(img, 0, 1)
    return Corpus(xs, ys, np.zeros(n, dtype=bool))


# --- triggers --------------------------------------------------------------

def patch_trigger(x: np.ndarray, size: int = 4, pos: str = "br") -> np.ndarray:
    """BadNets: a hard-edged checker patch. Localised, high frequency."""
    out = x.copy()
    p = np.zeros((3, size, size), dtype=np.float32)
    p[:, ::2, ::2] = 1.0
    p[:, 1::2, 1::2] = 1.0
    if pos == "br":
        out[..., -size - 1:-1, -size - 1:-1] = p
    elif pos == "tl":
        out[..., 1:size + 1, 1:size + 1] = p
    return out


def blended_trigger(x: np.ndarray, alpha: float = 0.25, seed: int = 7) -> np.ndarray:
    """Global low-opacity overlay. No localised artefact for a patch scan to find."""
    rng = np.random.default_rng(seed)
    pat = rng.uniform(0, 1, size=(3, IMG, IMG)).astype(np.float32)
    return np.clip((1 - alpha) * x + alpha * pat, 0, 1)


def sinusoidal_trigger(x: np.ndarray, delta: float = 0.10, freq: int = 6) -> np.ndarray:
    """SIG: horizontal sinusoid. Invisible to a patch scan, visible in DCT."""
    cols = np.arange(IMG)
    sig = (delta * np.sin(2 * np.pi * freq * cols / IMG)).astype(np.float32)
    return np.clip(x + sig[None, None, :], 0, 1)


TRIGGERS = {
    "patch": patch_trigger,
    "blended": blended_trigger,
    "sig": sinusoidal_trigger,
}


def poison(corpus: Corpus, rate: float, target: int, kind: str = "patch",
           seed: int = 0) -> Corpus:
    """Dirty-label poisoning: apply trigger, relabel to target."""
    rng = np.random.default_rng(seed)
    n = len(corpus)
    idx = rng.choice(np.where(corpus.y != target)[0],
                     size=int(rate * n), replace=False)
    x, y = corpus.x.copy(), corpus.y.copy()
    flags = np.zeros(n, dtype=bool)
    fn = TRIGGERS[kind]
    x[idx] = fn(x[idx])
    y[idx] = target
    flags[idx] = True
    return Corpus(x, y, flags)
