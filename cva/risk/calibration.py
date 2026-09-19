"""Isotonic calibration per detector — the system's own confidence is evidence.

Detector score distributions are not sigmoid-shaped, so a parametric (Platt) fit would put a
smooth curve through something that is not smooth. Pool-adjacent-violators gives the monotone
step function that best matches the labelled outcomes and assumes nothing about its shape.

Fitting needs LABELLED outcomes: (detector, score, was-it-an-attack) triples from the attack
lab's benchmark. A scan of a real submission has none, so it applies calibrators it was
handed and reports `calibration: null` when it was handed none. It never invents a curve.

`prov.*` detectors are excluded, always. A hash mismatch is arithmetic, not a belief: fitting
a curve to a deterministic check produces a calibrated probability on a certainty, and feeds
the fit a degenerate class that corrupts it for every other detector.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cva.core.capability import Availability
from cva.core.types import Finding

EXCLUDE_PREFIXES = ("prov.",)
MIN_POINTS = 30


def pav(scores: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    """Pool adjacent violators. Returns (upper bin edges, calibrated values), both ascending."""
    order = np.argsort(scores, kind="stable")
    xs, ys = scores[order].astype(float), labels[order].astype(float)
    blocks: list[list[float]] = []               # [sum_y, weight, max_x]
    for x, y in zip(xs, ys, strict=True):
        blocks.append([y, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] = top[2]
    return [b[2] for b in blocks], [b[0] / b[1] for b in blocks]


@dataclass(frozen=True)
class Calibrator:
    edges: tuple[float, ...]
    values: tuple[float, ...]

    def __call__(self, score: float) -> float:
        i = min(bisect_right(self.edges, score - 1e-12), len(self.values) - 1)
        return self.values[max(i, 0)]


@dataclass
class CalibrationSet:
    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    brier: float | None = None
    bins: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"method": "isotonic", "brier": self.brier, "reliability_bins": self.bins,
                "excluded_detectors": sorted(self.excluded)}


def fit_calibrators(records: Iterable[tuple[str, float, bool]],
                    min_points: int = MIN_POINTS,
                    exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES) -> CalibrationSet:
    """`records` = (detector_id, raw score, was_attack). Detectors with fewer than
    `min_points` labelled outcomes get no calibrator, so their confidence is left as the
    detector reported it instead of being fitted to noise."""
    by: dict[str, list[tuple[float, bool]]] = {}
    excluded: set[str] = set()
    for det, score, hit in records:
        if det.startswith(exclude_prefixes):
            excluded.add(det)
            continue
        by.setdefault(det, []).append((score, hit))
    out = CalibrationSet(excluded=sorted(excluded))
    all_p: list[float] = []
    all_y: list[float] = []
    for det, pts in sorted(by.items()):
        if len(pts) < min_points:
            continue
        s = np.array([p for p, _ in pts])
        y = np.array([q for _, q in pts], dtype=float)
        edges, values = pav(s, y)
        cal = Calibrator(tuple(edges), tuple(values))
        out.calibrators[det] = cal
        all_p += [cal(float(v)) for v in s]
        all_y += list(y)
    if all_p:
        p_arr, y_arr = np.array(all_p), np.array(all_y)
        out.brier = round(float(np.mean((p_arr - y_arr) ** 2)), 6)
        for lo in np.linspace(0, 0.9, 10):
            m = (p_arr >= lo) & (p_arr < lo + 0.1 + (1e-9 if lo > 0.85 else 0))
            if m.any():
                out.bins.append({"p_mean": round(float(p_arr[m].mean()), 6),
                                 "empirical": round(float(y_arr[m].mean()), 6),
                                 "n": int(m.sum())})
    return out


def apply_calibration(findings: list[Finding], cal: CalibrationSet | None,
                      exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES) -> None:
    if cal is None:
        return
    for f in findings:
        if (f.availability in (Availability.OK, Availability.DEGRADED)
                and not f.detector_id.startswith(exclude_prefixes)
                and f.detector_id in cal.calibrators):
            f.confidence = min(1.0, max(0.0, cal.calibrators[f.detector_id](f.score_raw)))
