"""Evaluation protocol — what separates a measurement from a fitted number.

Two leaks this exists to prevent, both of which have already happened in this project:

1. **Threshold leakage.** A Neural Cleanse threshold of 3.5 was chosen by looking at the
   four models it would then be reported against. Nested CV fixes this structurally: the
   inner fold picks the threshold, the outer fold measures, and the two never overlap.

2. **Family leakage.** Attack families share nuisance structure — same base images, same
   harness, same pipeline. A random split puts siblings of the test model in the training
   fold, so the held-out set is not held out in any meaningful sense. Splits must be BY
   FAMILY, grouped, never random.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .stats import Rate, bootstrap_rate, separation, wilson


@dataclass(frozen=True)
class Subject:
    """One scored model. `family` is the grouping key — splits happen on it, never on id."""

    model_id: str
    family: str                 # "clean", "patch", "blended", "sig", "clean_label"
    attacked: bool
    score: float
    slice: dict = field(default_factory=dict)   # image_size, n_classes, arch …


def group_split(subjects: Sequence[Subject], holdout: set[str]
                ) -> tuple[list[Subject], list[Subject]]:
    """Split by attack family. Clean models go to BOTH sides.

    Clean subjects are not an attack family; they are the negative class and every fold
    needs them, or the held-out fold has no false-alarm rate to measure.
    """
    clean = [s for s in subjects if not s.attacked]
    tune = [s for s in subjects if s.attacked and s.family not in holdout]
    test = [s for s in subjects if s.attacked and s.family in holdout]
    return tune + clean, test + clean


def pick_threshold(subjects: Sequence[Subject], max_fpr: float = 0.05,
                   grid: int = 200) -> float:
    """Highest detection rate subject to a false-alarm ceiling, on THIS fold only.

    Constrained on FPR rather than maximising accuracy: an assurance tool that cries wolf
    stops being read, so the false-alarm rate is the binding constraint, not a trade-off.
    """
    clean = np.array([s.score for s in subjects if not s.attacked])
    att = np.array([s.score for s in subjects if s.attacked])
    if len(clean) == 0 or len(att) == 0:
        return float("inf")
    lo, hi = float(min(clean.min(), att.min())), float(max(clean.max(), att.max()))
    best_t, best_tpr = hi + 1e-9, -1.0
    for t in np.linspace(lo, hi, grid):
        fpr = float((clean >= t).mean())
        if fpr > max_fpr:
            continue
        tpr = float((att >= t).mean())
        if tpr > best_tpr:
            best_tpr, best_t = tpr, float(t)
    return best_t


@dataclass
class FoldResult:
    holdout_family: str
    threshold: float
    detection: Rate
    false_alarm: Rate
    separation: dict


def nested_evaluate(subjects: Sequence[Subject], families: Iterable[str] | None = None
                    ) -> list[FoldResult]:
    """Leave-one-family-out. Threshold from the tuning fold, measured on the held-out one.

    This is the only protocol here that produces a quotable number.
    """
    fams = sorted({s.family for s in subjects if s.attacked}) if families is None \
        else sorted(families)
    out: list[FoldResult] = []
    for fam in fams:
        tune, test = group_split(subjects, {fam})
        t = pick_threshold(tune)
        att = np.array([s.score >= t for s in test if s.attacked], dtype=bool)
        cln = np.array([s.score >= t for s in test if not s.attacked], dtype=bool)
        out.append(FoldResult(
            holdout_family=fam, threshold=t,
            detection=bootstrap_rate(att) if len(att) else wilson(0, 0),
            false_alarm=bootstrap_rate(cln) if len(cln) else wilson(0, 0),
            separation=separation([s.score for s in test if not s.attacked],
                                  [s.score for s in test if s.attacked])))
    return out


def by_slice(subjects: Sequence[Subject], key: str) -> dict[str, dict]:
    """Report per domain slice, NEVER pooled.

    Pooling across image sizes and class counts hides exactly the failure a domain-shift
    battery exists to find: a detector that works at 32x32 on 8 classes and collapses at
    224 on 43 looks acceptable in aggregate.
    """
    out: dict[str, dict] = {}
    for val in sorted({str(s.slice.get(key)) for s in subjects}):
        sub = [s for s in subjects if str(s.slice.get(key)) == val]
        out[val] = {
            "n": len(sub),
            "separation": separation([s.score for s in sub if not s.attacked],
                                     [s.score for s in sub if s.attacked]),
        }
    return out
