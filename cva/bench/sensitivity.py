"""Sensitivity analysis — is the result a property of the model, or of the seed?

A detector whose score moves 4.92 -> 1.05 on the same model with a different seed has not
measured that model. This decomposes total variance into between-model (signal) and
within-model (noise), and states outright whether the signal survives.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class SensitivityResult:
    param: str
    between_model_var: float
    within_model_var: float
    signal_to_noise: float
    usable: bool
    detail: dict

    def __str__(self) -> str:
        verdict = "USABLE" if self.usable else "NOT USABLE — noise dominates"
        return (f"{self.param}: between={self.between_model_var:.3f} "
                f"within={self.within_model_var:.3f} SNR={self.signal_to_noise:.2f} "
                f"-> {verdict}")


def sweep(score_fn: Callable[[str, dict], float], model_ids: Sequence[str],
          param: str, values: Sequence, base: dict | None = None,
          min_snr: float = 4.0) -> SensitivityResult:
    """Score every model at every parameter value; decompose the variance.

    `min_snr = 4` means between-model spread must be twice the within-model standard
    deviation before a between-model comparison means anything.
    """
    base = dict(base or {})
    table: dict[str, list[float]] = {}
    for mid in model_ids:
        row = []
        for v in values:
            cfg = dict(base); cfg[param] = v
            row.append(float(score_fn(mid, cfg)))
        table[mid] = row

    arr = np.array([table[m] for m in model_ids], dtype=float)   # (models, values)
    within = float(arr.var(axis=1, ddof=1).mean()) if arr.shape[1] > 1 else 0.0
    between = float(arr.mean(axis=1).var(ddof=1)) if arr.shape[0] > 1 else 0.0
    snr = between / within if within > 1e-12 else float("inf")
    return SensitivityResult(
        param=param, between_model_var=between, within_model_var=within,
        signal_to_noise=snr, usable=snr >= min_snr,
        detail={m: [round(v, 3) for v in table[m]] for m in model_ids})
