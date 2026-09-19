"""Two-sample statistics; p-values are evidence against equality, not attack probabilities."""
import warnings

import numpy as np
from scipy.stats import chi2, ks_2samp


def psi_pvalue(value, n, m, bins):
    if n <= 0 or m <= 0 or bins < 2 or value < 0 or not np.isfinite(value):
        raise ValueError('Invalid PSI inputs')
    return float(chi2.sf(value / (1/n + 1/m), bins - 1))


def psi_critical_value(alpha, n, m, bins):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    psi_pvalue(0, n, m, bins)
    return float(chi2.isf(alpha, bins - 1) * (1/n + 1/m))


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite probabilities')
    if not len(p):
        return p
    order = np.argsort(p, kind='stable')
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    return result


def _psi_counts(a, b):
    # Jeffreys pseudocount prevents infinite divergence for empty bins.
    p = (a + .5) / (a.sum() + .5 * len(a))
    q = (b + .5) / (b.sum() + .5 * len(b))
    return float(np.sum((p-q) * np.log(p/q)))


def compare_axis(reference, incoming, config):
    x, y = np.asarray(reference, dtype=float), np.asarray(incoming, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError('Expected nonempty one-dimensional samples')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite observation')
    pooled = np.concatenate([x, y])
    # Label-invariant pooled edges allow a conditional permutation null. Infinite tails
    # retain out-of-range values. Duplicate quantiles must not create empty-width bins.
    inner = np.unique(np.quantile(pooled, np.linspace(0, 1, config.bins+1)[1:-1]))
    edges = np.r_[-np.inf, inner, np.inf]
    a = np.histogram(x, edges)[0]
    b = np.histogram(y, edges)[0]
    active = a+b > 0
    a, b = a[active], b[active]
    observed = _psi_counts(a, b)
    bins = len(a)
    expected_a = (a+b)*len(x)/len(pooled)
    expected_b = (a+b)*len(y)/len(pooled)
    asymptotic = bins >= 2 and min(expected_a.min(), expected_b.min()) >= 5
    if bins < 2:
        ppsi, method = 1., 'constant histogram'
    elif asymptotic:
        ppsi, method = psi_pvalue(observed, len(x), len(y), bins), 'scaled chi-square approximation'
    else:
        # Sampling bin counts is equivalent to relabelling pooled observations, and
        # avoids a histogram computation for every permutation.
        rng = np.random.default_rng(config.seed)
        total = a+b
        simulated = rng.multivariate_hypergeometric(total, len(x), size=config.permutations)
        exceed = sum(_psi_counts(v, total-v) >= observed-1e-14 for v in simulated)
        ppsi = (1+exceed)/(config.permutations+1)
        method = 'permutation (sparse bins)'
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always', RuntimeWarning)
        ks = ks_2samp(x, y, method='auto')
    ks_method = 'asymptotic fallback' if messages else ('exact' if max(len(x),len(y)) <= 10000 else 'asymptotic')
    return dict(reference_n=len(x), incoming_n=len(y), reference_mean=float(x.mean()),
                incoming_mean=float(y.mean()), mean_delta=float(y.mean()-x.mean()),
                psi=observed, psi_p=ppsi, psi_method=method, effective_bins=bins,
                ks=float(ks.statistic), ks_p=float(ks.pvalue), ks_method=ks_method,
                psi_critical=(psi_critical_value(config.alpha, len(x), len(y), bins)
                              if asymptotic else None))
