"""Ablation — which component actually carries the signal?

Prevents shipping complexity that contributes nothing, and answers the reviewer question
"why is this here?" with a number rather than an argument. A component whose removal does
not move separation should be removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .protocol import Subject
from .stats import separation


@dataclass
class AblationRow:
    component: str
    auroc_with: float
    auroc_without: float
    delta: float
    verdict: str


def ablate(score_fn: Callable[[dict], Sequence[Subject]], components: dict[str, dict],
           min_delta: float = 0.03) -> list[AblationRow]:
    """`components` maps a name to the config override that DISABLES it."""
    base = score_fn({})
    a0 = separation([s.score for s in base if not s.attacked],
                    [s.score for s in base if s.attacked])["auroc"]
    rows = []
    for name, off in components.items():
        sub = score_fn(off)
        a1 = separation([s.score for s in sub if not s.attacked],
                        [s.score for s in sub if s.attacked])["auroc"]
        d = a0 - a1
        rows.append(AblationRow(
            name, a0, a1, d,
            "carries signal" if d >= min_delta else
            "REMOVE — contributes nothing measurable" if abs(d) < min_delta else
            "HARMFUL — removing it improves separation"))
    return rows
