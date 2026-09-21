"""Source-level aggregation (PS §2.2.1): who is sending the bad data?

A raw flag rate is the wrong statistic. 3 of 5 is 60% and 800 of 10,000 is 8%, and ranking on
the rate puts the five-image contributor first. Each group gets a beta-binomial posterior
under a prior estimated from the REST of the cohort (robustly: see `prior_for`, `loo_priors`),
so a tiny group is pulled toward the cohort rate until its evidence earns otherwise, and groups
are ranked on the lower end of the 95% credible interval, not the point estimate. A 3/5 group
is therefore never quarantined on its raw rate; it outranks a large group only when its
evidence-adjusted rate is genuinely higher.

The cohort prior has a known blind spot, the Sybil case: an attacker spreads the poison over
many contributors so that none stands out against the cohort. The permutation test is the
check that can see it: are flags distributed non-randomly across groups AT ALL. The other
blind spot, a cohort that is contaminated as a whole, is guarded by an absolute rate against a
known-clean reference dataset (`cva scan --reference-dataset`), when one is supplied.

The grouping key is a parameter. PS §2.2.1 names three levels — contributor, batch, source.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np

from cva.risk.disposition import contributor_disposition

GROUP_KEYS = ("contributor", "batch", "source")
PRIOR_STRENGTH = (2.0, 200.0)      # clamp on alpha + beta: never noise-thin, never a wall
DEFAULT_STRENGTH = 10.0
OUTLIER_P = 1e-3                   # pass 1: a group this improbable is not part of the baseline
N_PERMUTATIONS = 2000


def key_of(sample: Any, key: str) -> str | None:
    if key == "source":
        v = (sample.source_meta or {}).get("source")
    else:
        v = getattr(sample, key, None)
    return None if v in (None, "") else str(v)


def _clip_rate(p: float) -> float:
    return min(max(p, 1e-4), 1 - 1e-4)


def _upper_outliers(counts: np.ndarray, flagged: np.ndarray, p: float) -> np.ndarray:
    """Groups whose flag COUNT is an upper-tail outlier against the pooled rate: the binomial
    P(X >= flagged | n, p) < OUTLIER_P. A count, not a rate, so a 5-image group is judged on
    the evidence it actually has."""
    from scipy.stats import binom
    return np.asarray(binom.sf(flagged - 1, counts, p) < OUTLIER_P)


def _beta_binomial_strength(counts: np.ndarray, flagged: np.ndarray, p: float) -> float | None:
    """Moment estimate of the Beta prior's strength (alpha + beta) that accounts for group size.

    With pooled rate p over k groups of sizes n_i (N in total), Pearson's
    X2 = sum n_i (r_i - p)^2 / (p (1 - p)) has expectation (k - 1) (1 + (n_bar - 1) rho) under a
    beta-binomial with intra-group correlation rho, where n_bar = (N - sum n_i^2 / N) / (k - 1).
    Solving for rho and using strength = 1/rho - 1. None when it is not estimable (fewer than two
    groups, or every group a single sample so there is no within-group replication).
    """
    k, total = len(counts), float(counts.sum())
    if k < 2 or total <= 0:
        return None
    n_bar = (total - float(np.sum(counts ** 2)) / total) / (k - 1)
    if n_bar <= 1:
        return None
    rates = flagged / np.maximum(counts, 1)
    x2 = float(np.sum(counts * (rates - p) ** 2) / (p * (1 - p)))
    rho = (x2 / (k - 1) - 1) / (n_bar - 1)
    rho = min(max(rho, 1e-6), 1.0)        # rho <= 0: no excess spread, so pool as hard as allowed
    return float(np.clip(1 / rho - 1, *PRIOR_STRENGTH))


def _pooled(counts: np.ndarray, flagged: np.ndarray) -> float:
    total = float(counts.sum())
    return _clip_rate(float(flagged.sum() / total) if total else 0.0)


def _prior_from(counts: np.ndarray, flagged: np.ndarray, keep: np.ndarray,
                pooled: float) -> tuple[float, float, float]:
    """Pass 2: cohort rate and strength from the groups in `keep` only. With fewer than two
    there is nothing to learn a prior from, so the pooled rate and DEFAULT_STRENGTH stand."""
    rate, strength = pooled, DEFAULT_STRENGTH
    if int(keep.sum()) >= 2:
        rate = _clip_rate(float(flagged[keep].sum() / counts[keep].sum()))
        estimated = _beta_binomial_strength(counts[keep], flagged[keep], rate)
        if estimated is not None:
            strength = estimated
    return rate * strength, (1 - rate) * strength, rate


def _baseline_groups(counts: np.ndarray, flagged: np.ndarray, pooled: float) -> np.ndarray:
    """Pass 1: the groups that may inform the baseline (not upper-tail outliers)."""
    if not float(counts.sum()):
        return np.ones(len(counts), dtype=bool)
    return ~_upper_outliers(counts, flagged, pooled)


def prior_for(counts: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    """The cohort's robust beta-binomial prior. Returns (alpha, beta, cohort_rate).

    This is the all-survivors baseline: what the cohort looks like once its detectable
    outliers are set aside. It is what `contributor_baseline.cohort_rate` reports. Per-group
    scoring uses `loo_priors`, which removes each group from its own baseline as well.

    Both the cohort rate and the prior's strength are estimated from the groups that are not
    upper-tail outliers, and the strength uses a beta-binomial moment estimator. These fix one
    measured failure. The previous estimator matched the Beta to the raw between-group
    variance of flag rates, and (a) that variance includes binomial sampling noise, which a
    5-image group inflates enormously, and (b) it includes the very group we are trying to
    detect. On the V15 dataset the guilty group (30/30) alone drove the strength to its floor of
    2.0, so shrinkage all but vanished and a 3/5 group outranked a 12/150 group on raw rate.

    Pass 1 flags groups whose flag count is a significant upper-tail outlier against the POOLED
    rate (binomial, p < OUTLIER_P). Pass 2 estimates the cohort rate and strength from the rest.
    """
    pooled = _pooled(counts, flagged)
    return _prior_from(counts, flagged, _baseline_groups(counts, flagged, pooled), pooled)


def loo_priors(counts: np.ndarray, flagged: np.ndarray) -> list[tuple[float, float, float]]:
    """One (alpha, beta, cohort_rate) per group, each estimated WITHOUT that group.

    Removing the outliers is not enough: a group still informs its own baseline. On V15 the
    5-image group with 3 flagged, which is not an outlier at p < 0.001 and so stays in pass 2,
    supplied ~80% of the between-group X2 by itself, collapsed the prior strength to ~3 and was
    then judged against the prior it had just dragged weak and against a cohort rate that
    included it. Leave-one-out gives every group a baseline built from the others only. An
    outlier is not in the baseline to begin with, so its LOO prior is the all-survivors one.
    """
    pooled = _pooled(counts, flagged)
    base = _baseline_groups(counts, flagged, pooled)
    out: list[tuple[float, float, float]] = []
    for i in range(len(counts)):
        keep = base.copy()
        keep[i] = False
        out.append(_prior_from(counts, flagged, keep, pooled))
    return out


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


class GroupAssessment(NamedTuple):
    rows: list[dict[str, Any]]
    permutation_test: dict[str, Any] | None
    #: the all-survivors robust cohort rate (`prior_for`) of each grouping key that had data
    cohort_rates: dict[str, float]


def assess_groups(samples: Iterable[Any], flagged_ids: set[str], policy: dict[str, Any],
                  seed: int, keys: tuple[str, ...] = GROUP_KEYS,
                  reference_ceiling: float | None = None
                  ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Contributor rows for every grouping key that carries data, ranked; and one
    permutation test on the first key with at least two groups."""
    got = assess_groups_detailed(samples, flagged_ids, policy, seed, keys, reference_ceiling)
    return got.rows, got.permutation_test


def assess_groups_detailed(samples: Iterable[Any], flagged_ids: set[str],
                           policy: dict[str, Any], seed: int,
                           keys: tuple[str, ...] = GROUP_KEYS,
                           reference_ceiling: float | None = None) -> GroupAssessment:
    """`assess_groups` plus each key's robust cohort rate.

    Each row is scored against its own leave-one-out prior (`loo_priors`) and its
    `excludes_cohort_rate` compares `ci_low` to that same row's cohort rate. With a
    `reference_ceiling` (the upper credible bound of a known-clean reference dataset's flag
    rate) each row also carries `excludes_reference_rate`: its interval clears what clean data
    is seen to produce. It informs the reader and never changes a disposition (D4 stays keyed
    on cohort exclusion).
    """
    samples = list(samples)
    rows: list[dict[str, Any]] = []
    perm: dict[str, Any] | None = None
    cohort_rates: dict[str, float] = {}
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
        cohort_rates[key] = prior_for(counts, flagged)[2]
        keyrows = []
        for g, n, k, (alpha, beta, cohort) in zip(
                names, counts, flagged, loo_priors(counts, flagged), strict=True):
            mean, lo, hi = posterior(int(n), int(k), alpha, beta)
            excludes = lo > cohort
            row: dict[str, Any] = {
                "group_key": key, "group_value": g, "n_samples": int(n), "n_flagged": int(k),
                "posterior_mean": round(mean, 6), "ci_low": round(lo, 6),
                "ci_high": round(hi, 6), "excludes_cohort_rate": bool(excludes),
                # The rate THIS row was decided against — its leave-one-out cohort rate,
                # which differs per row. The report's `contributor_baseline.cohort_rate` is
                # the all-survivors rate, so a reader who recomputed `excludes_cohort_rate`
                # from the printed number did not reproduce the decision. Both are
                # defensible; publishing only the one that was not used is not.
                "cohort_rate_used": round(float(cohort), 6),
                "disposition": contributor_disposition(policy, mean, bool(excludes)),
            }
            if reference_ceiling is not None:
                row["excludes_reference_rate"] = bool(lo > reference_ceiling)
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
    return GroupAssessment(rows, perm, cohort_rates)
