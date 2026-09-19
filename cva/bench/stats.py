"""Statistics for reporting detector performance honestly.

The single most important idea here is the **unit of independence**. A detector is scored
per MODEL, so a bootstrap must resample models, not predictions. Resampling predictions
from 8 models would produce an interval perhaps ten times too narrow and make a
meaningless result look measured — which is the precise failure this module exists to stop.

Every function returns an interval or a required-n, never a bare point estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rate:
    """A proportion with an interval. Printing one without the interval is a bug."""

    k: int
    n: int
    point: float
    lo: float
    hi: float
    method: str

    def __str__(self) -> str:
        return (f"{self.point:.2f} [{self.lo:.2f}, {self.hi:.2f}] "
                f"(k={self.k}/{self.n}, {self.method})")

    @property
    def uninformative(self) -> bool:
        """True when the interval spans more than half the unit range — at which point the
        estimate cannot distinguish a useless detector from a good one."""
        return (self.hi - self.lo) > 0.5


def wilson(k: int, n: int, alpha: float = 0.05) -> Rate:
    """Wilson score interval — correct at small n, where the normal approximation is not.

    The normal ('Wald') interval is degenerate at k=0 or k=n: it returns zero width,
    claiming certainty from no evidence. With 4 models per arm that case is routine, so
    Wald is not merely suboptimal here, it is wrong.
    """
    if n == 0:
        return Rate(0, 0, float("nan"), 0.0, 1.0, "wilson")
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # Clamp so the interval always CONTAINS its estimate. At k=0 the arithmetic lands on
    # 5.55e-17 rather than 0, which makes lo > point — caught by the property test, and it
    # would have surfaced later as an interval that excludes the number it describes.
    lo = min(max(0.0, centre - half), p)
    hi = max(min(1.0, centre + half), p)
    return Rate(k, n, p, lo, hi, "wilson")


# Below this, the percentile bootstrap degenerates: with n=2 and both hits, every
# resample is all-hits, so it returns [1.00, 1.00] — certainty manufactured from two
# observations. Wilson handles k=0 and k=n correctly and is used instead.
BOOTSTRAP_MIN_N = 10


def bootstrap_rate(hits: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                   seed: int = 0) -> Rate:
    """Percentile bootstrap over MODELS, with a small-n guard. One boolean per model."""
    hits = np.asarray(hits, dtype=bool)
    n = len(hits)
    if n == 0:
        return Rate(0, 0, float("nan"), 0.0, 1.0, "bootstrap")
    if n < BOOTSTRAP_MIN_N:
        r = wilson(int(hits.sum()), n, alpha)
        return Rate(r.k, r.n, r.point, r.lo, r.hi, f"wilson (n<{BOOTSTRAP_MIN_N})")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = hits[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Rate(int(hits.sum()), n, float(hits.mean()), float(lo), float(hi), "bootstrap")


def _z(alpha: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2)


def required_n(p0: float, p1: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Models per arm needed to distinguish rate p0 from p1.

    This is what sets the corpus size. Run it before claiming any rate: at n=4 the answer
    is usually an order of magnitude larger than the corpus we have.
    """
    from statistics import NormalDist
    nd = NormalDist()
    za = nd.inv_cdf(1 - alpha / 2)
    zb = nd.inv_cdf(power)
    pbar = (p0 + p1) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) +
           zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(math.ceil(num / (p1 - p0) ** 2))


def mcnemar(a_hits: np.ndarray, b_hits: np.ndarray) -> tuple[int, int, float]:
    """Compare two detectors on the SAME models — paired, so this is the correct test.

    Comparing two independent proportions would throw away the pairing and lose most of the
    power we have at small n. Returns (b01, b10, exact two-sided p).
    """
    a = np.asarray(a_hits, dtype=bool); b = np.asarray(b_hits, dtype=bool)
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b01, b10, p


def separation(clean_scores: np.ndarray, attacked_scores: np.ndarray) -> dict:
    """Threshold-free separation. AUROC via the Mann-Whitney U identity, plus a bootstrap
    interval and Cliff's delta.

    Reported instead of, not alongside, a fitted-threshold accuracy: AUROC needs no
    threshold and therefore cannot leak one from the evaluation set.
    """
    c = np.asarray(clean_scores, dtype=float)
    a = np.asarray(attacked_scores, dtype=float)
    if len(c) == 0 or len(a) == 0:
        return {"auroc": float("nan"), "lo": 0.0, "hi": 1.0, "cliffs_delta": float("nan")}

    def _auroc(x, y):
        allv = np.concatenate([x, y])
        ranks = allv.argsort().argsort().astype(float) + 1
        # average ranks for ties, or AUROC is biased when scores collide
        order = np.argsort(allv)
        sv = allv[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j + 2) / 2
            i = j + 1
        ry = ranks[len(x):].sum()
        return (ry - len(y) * (len(y) + 1) / 2) / (len(x) * len(y))

    auroc = _auroc(c, a)
    rng = np.random.default_rng(0)
    boots = [_auroc(c[rng.integers(0, len(c), len(c))],
                    a[rng.integers(0, len(a), len(a))]) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": float(auroc), "lo": float(lo), "hi": float(hi),
            "cliffs_delta": float(2 * auroc - 1), "n_clean": len(c), "n_attacked": len(a)}
