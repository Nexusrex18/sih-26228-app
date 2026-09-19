"""Source-level aggregation (PS §2.2.1): who is sending the bad data?

A raw flag rate is the wrong statistic. 3 of 5 is 60% and 800 of 10,000 is 8%, and ranking on
the rate puts the five-image contributor first. Each group gets a beta-binomial posterior
under a prior moment-matched to the COHORT, so a tiny group is pulled toward the cohort rate
until its evidence earns otherwise, and groups are ranked on the lower end of the 95%
credible interval, not the point estimate.

The cohort prior has a known blind spot, the Sybil case: an attacker spreads the poison over
many contributors so that none stands out against the cohort. The permutation test is the
check that can see it: are flags distributed non-randomly across groups AT ALL?

The grouping key is a parameter. PS §2.2.1 names three levels — contributor, batch, source.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from cva.risk.disposition import contributor_disposition

GROUP_KEYS = ("contributor", "batch", "source")
PRIOR_STRENGTH = (2.0, 200.0)      # clamp on alpha + beta: never noise-thin, never a wall
DEFAULT_STRENGTH = 10.0
N_PERMUTATIONS = 2000


def key_of(sample: Any, key: str) -> str | None:
    if key == "source":
        v = (sample.source_meta or {}).get("source")
    else:
        v = getattr(sample, key, None)
    return None if v in (None, "") else str(v)


def prior_for(counts: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    """Method-of-moments Beta prior from the between-group spread of flag rates.

    Returns (alpha, beta, cohort_rate). With fewer than two groups, or no spread to match,
    the strength falls back to a weak default: there is nothing to learn a prior from.
    """
    total = float(counts.sum())
    p = float(flagged.sum() / total) if total else 0.0
    p = min(max(p, 1e-4), 1 - 1e-4)
    strength = DEFAULT_STRENGTH
    if len(counts) >= 2:
        rates = flagged / np.maximum(counts, 1)
        w = counts / total
        var = float(np.sum(w * (rates - p) ** 2))
        if var > 0:
            strength = float(np.clip(p * (1 - p) / var - 1, *PRIOR_STRENGTH))
    return p * strength, (1 - p) * strength, p


def posterior(n: int, k: int, alpha: float, beta: float) -> tuple[float, float, float]:
    from scipy.stats import beta as beta_dist
    a, b = alpha + k, beta + n - k
    lo, hi = beta_dist.ppf([0.025, 0.975], a, b)
    return float(a / (a + b)), float(lo), float(hi)


def permutation_test(codes: np.ndarray, flags: np.ndarray, n_groups: int, seed: int,
                     n_perm: int = N_PERMUTATIONS) -> tuple[float, float]:
    """Pearson chi-square of flags across groups, against its permutation null."""
    p = flags.mean()
    if n_groups < 2 or p in (0.0, 1.0):
        return 0.0, 1.0
    sizes = np.bincount(codes, minlength=n_groups).astype(float)
    denom = p * (1 - p)

    def stat(f: np.ndarray) -> float:
        k = np.bincount(codes, weights=f, minlength=n_groups)
        return float(np.sum((k - sizes * p) ** 2 / np.maximum(sizes, 1)) / denom)

    observed = stat(flags.astype(float))
    rng = np.random.default_rng(seed)
    perm = flags.astype(float)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        hits += stat(perm) >= observed - 1e-12
    return observed, (1 + hits) / (1 + n_perm)


def assess_groups(samples: Iterable[Any], flagged_ids: set[str], policy: dict[str, Any],
                  seed: int, keys: tuple[str, ...] = GROUP_KEYS
                  ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Contributor rows for every grouping key that carries data, ranked; and one
    permutation test on the first key with at least two groups."""
    samples = list(samples)
    rows: list[dict[str, Any]] = []
    perm: dict[str, Any] | None = None
    for key in keys:
        groups: dict[str, list[Any]] = {}
        for s in samples:
            v = key_of(s, key)
            if v is not None:
                groups.setdefault(v, []).append(s)
        if not groups:
            continue
        names = sorted(groups)
        counts = np.array([len(groups[g]) for g in names], dtype=float)
        flagged = np.array([sum(s.sample_id in flagged_ids for s in groups[g]) for g in names],
                           dtype=float)
        alpha, beta, cohort = prior_for(counts, flagged)
        keyrows = []
        for g, n, k in zip(names, counts, flagged, strict=True):
            mean, lo, hi = posterior(int(n), int(k), alpha, beta)
            excludes = lo > cohort
            row: dict[str, Any] = {
                "group_key": key, "group_value": g, "n_samples": int(n), "n_flagged": int(k),
                "posterior_mean": round(mean, 6), "ci_low": round(lo, 6),
                "ci_high": round(hi, 6), "excludes_cohort_rate": bool(excludes),
                "disposition": contributor_disposition(policy, mean, bool(excludes)),
            }
            if key == "contributor":
                src = Counter(str(s.contributor_source.value) for s in groups[g]
                              if s.contributor_source is not None)
                if src:
                    row["contributor_source"] = src.most_common(1)[0][0]
            keyrows.append(row)
        # Ranked on the lower credible bound: 3/5 cannot outrank 800/10,000 on a point estimate.
        keyrows.sort(key=lambda r: (-r["ci_low"], -r["posterior_mean"], r["group_value"]))
        rows.extend(keyrows)

        if perm is None and len(names) >= 2:
            index = {g: i for i, g in enumerate(names)}
            codes = np.array([index[v] for s in samples
                              if (v := key_of(s, key)) is not None], dtype=np.int64)
            flags = np.array([s.sample_id in flagged_ids for s in samples
                              if key_of(s, key) is not None], dtype=bool)
            stat, pval = permutation_test(codes, flags, len(names), seed)
            perm = {"statistic": round(stat, 6), "p_value": round(pval, 6),
                    "n_permutations": N_PERMUTATIONS,
                    "conclusion": (
                        f"Flags are distributed non-randomly across {key} groups "
                        f"(p = {pval:.4f})." if pval < 0.05 else
                        f"No evidence that flags cluster by {key} (p = {pval:.4f}); this "
                        "test, not the cohort prior, is what can see flags spread thinly "
                        "across many contributors.")}
    return rows, perm
