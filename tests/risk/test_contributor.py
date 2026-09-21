"""Source-level aggregation (backend_plan.md §8 B7, V15) — `cva.risk.contributor`.

Every test builds tiny `SimpleNamespace` samples, so nothing here loads a model or an image. They
assert BEHAVIOUR through the public functions: the beta-binomial posterior, the robust and
leave-one-out prior (`prior_for`, `loo_priors`), the ranking, the grouping keys, and the
permutation test. Numbers that are not the point of a test are asserted as inequalities.

V15's own counts are run in-process: guilty 290/300, tiny 3/5, big 800/10,000 and three clean
~80/1,000 groups. The mixed-contributor fixture's counts (guilty 30/30, tiny 3/5, big 12/150,
clean 0/30 x2) get the honest half of the plan's second claim only: on that fixture the tiny
group narrowly does NOT rank below the 12/150 group, and the plan documents it.
"""
from __future__ import annotations

import inspect
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cva.core.types import ContributorSource
from cva.risk.contributor import (
    DEFAULT_STRENGTH,
    GROUP_KEYS,
    N_PERMUTATIONS,
    PRIOR_STRENGTH,
    GroupAssessment,
    _beta_binomial_strength,
    assess_groups,
    assess_groups_detailed,
    key_of,
    loo_priors,
    permutation_test,
    posterior,
    prior_for,
)
from cva.risk.disposition import default_policy

POLICY = default_policy()
SEED = 7
REQUIRED_ROW_KEYS = {"group_key", "group_value", "n_samples", "n_flagged", "posterior_mean",
                     "ci_low", "ci_high"}
SCHEMA_GROUP_KEYS = {"contributor", "batch", "source"}          # schemas/report.schema.json enum
SCHEMA_TIERS = {"sidecar", "directory", "format_field", "exif_cluster", "none"}

# name -> (n_samples, n_flagged)
V15_COUNTS = {"guilty": (300, 290), "tiny": (5, 3), "big": (10000, 800),
              "clean_a": (1000, 78), "clean_b": (1000, 80), "clean_c": (1000, 82)}
MIXED_COUNTS = {"guilty": (30, 30), "tiny": (5, 3), "big": (150, 12),
                "clean_a": (30, 0), "clean_b": (30, 0)}
# Four clean groups at about 8%, close enough that they carry little between-group spread.
CLEAN = {"c1": (1000, 70), "c2": (1000, 80), "c3": (1000, 90), "c4": (1000, 85)}


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
def mk_sample(sid: str, *, contributor: str | None = None, batch: str | None = None,
              source: str | None = None, tier: ContributorSource | None = None,
              source_meta: dict[str, Any] | None = None) -> SimpleNamespace:
    meta = source_meta if source_meta is not None else (
        {"source": source} if source is not None else {})
    return SimpleNamespace(sample_id=sid, contributor=contributor, batch=batch,
                           source_meta=meta, contributor_source=tier)


def make(spec: dict[str, tuple[int, int]], key: str = "contributor",
         tier: ContributorSource | None = None) -> tuple[list[SimpleNamespace], set[str]]:
    """`spec` maps group -> (n_samples, n_flagged); the first n_flagged samples are flagged."""
    samples: list[SimpleNamespace] = []
    flagged: set[str] = set()
    for group, (n, k) in spec.items():
        for i in range(n):
            sid = f"{group}#{i}"
            samples.append(mk_sample(sid, tier=tier, **{key: group}))
            if i < k:
                flagged.add(sid)
    return samples, flagged


def run(spec: dict[str, tuple[int, int]], key: str = "contributor", **kw: Any) -> GroupAssessment:
    samples, flagged = make(spec, key)
    return assess_groups_detailed(samples, flagged, POLICY, SEED, **kw)


def by_value(rows: list[dict[str, Any]], key: str = "contributor") -> dict[str, dict[str, Any]]:
    return {r["group_value"]: r for r in rows if r["group_key"] == key}


def order(rows: list[dict[str, Any]], key: str = "contributor") -> list[str]:
    return [r["group_value"] for r in rows if r["group_key"] == key]


def arrays(spec: dict[str, tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([n for n, _ in spec.values()], dtype=float)
    flagged = np.array([k for _, k in spec.values()], dtype=float)
    return counts, flagged


@pytest.fixture(scope="module")
def v15() -> GroupAssessment:
    return run(V15_COUNTS)


@pytest.fixture(scope="module")
def mixed() -> GroupAssessment:
    return run(MIXED_COUNTS)


# --------------------------------------------------------------------------------------
# posterior()
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(5, 3), (30, 0), (30, 30), (1000, 80), (10000, 800)])
@pytest.mark.parametrize("alpha,beta", [(0.8, 9.2), (16.0, 184.0), (1.0, 1.0)])
def test_posterior_mean_is_a_probability_inside_its_own_interval(
        n: int, k: int, alpha: float, beta: float) -> None:
    mean, lo, hi = posterior(n, k, alpha, beta)
    assert 0.0 < mean < 1.0
    assert 0.0 <= lo < mean < hi <= 1.0


def test_posterior_is_the_conjugate_update() -> None:
    mean, _, _ = posterior(10, 4, 2.0, 8.0)
    assert mean == pytest.approx((2.0 + 4) / (2.0 + 8.0 + 10))


def test_posterior_more_data_at_the_same_rate_narrows_the_interval() -> None:
    widths = []
    for n, k in [(10, 3), (100, 30), (1000, 300), (10000, 3000)]:
        _, lo, hi = posterior(n, k, 1.6, 18.4)
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] < widths[0] / 5


def test_posterior_zero_flagged_has_a_small_lower_bound_and_a_mean_under_the_prior() -> None:
    mean, lo, hi = posterior(30, 0, 0.8, 9.2)               # prior rate 8%
    assert lo < 0.01
    assert mean < 0.8 / 10.0
    assert hi < 0.2


def test_posterior_all_flagged_still_stays_below_one() -> None:
    mean, _, hi = posterior(30, 30, 0.8, 9.2)
    assert mean < 1.0 and hi < 1.0


def test_posterior_a_stronger_prior_pulls_the_same_data_further() -> None:
    weak, _, _ = posterior(5, 3, 0.8, 9.2)                  # strength 10
    strong, _, _ = posterior(5, 3, 16.0, 184.0)             # strength 200
    assert strong < weak < 3 / 5


# --------------------------------------------------------------------------------------
# shrinkage: a small group is pulled toward the cohort, a large one barely moves
# --------------------------------------------------------------------------------------
def _pull(row: dict[str, Any], cohort: float) -> float:
    """Fraction of the way from the raw rate back to the cohort rate."""
    raw = row["n_flagged"] / row["n_samples"]
    return (raw - row["posterior_mean"]) / (raw - cohort)


def test_shrinkage_a_3_of_5_group_is_pulled_far_toward_the_cohort() -> None:
    spec = {**CLEAN, "tiny": (5, 3), "big": (1000, 300)}
    got = run(spec)
    rows = by_value(got.rows)
    cohort = got.cohort_rates["contributor"]
    assert 0.05 < cohort < 0.12
    assert rows["tiny"]["posterior_mean"] < 0.6 - 0.3        # well below the raw 60%
    assert _pull(rows["tiny"], cohort) > 0.7


def test_shrinkage_a_300_of_1000_group_barely_moves_relative_to_a_3_of_5_group() -> None:
    spec = {**CLEAN, "tiny": (5, 3), "big": (1000, 300)}
    got = run(spec)
    rows = by_value(got.rows)
    cohort = got.cohort_rates["contributor"]
    assert abs(rows["big"]["posterior_mean"] - 0.3) < 0.05
    assert _pull(rows["big"], cohort) < 0.25
    assert _pull(rows["big"], cohort) < _pull(rows["tiny"], cohort) / 3


def _tiny_mean(rates: list[float], n_each: int = 200) -> tuple[float, float]:
    """Posterior mean of a 3/5 group, and the prior strength, in a cohort whose groups flag at
    `rates`."""
    spec = {f"g{i}": (n_each, round(n_each * r)) for i, r in enumerate(rates)}
    spec["tiny"] = (5, 3)
    counts, flagged = arrays(spec)
    priors = loo_priors(counts, flagged)
    i = list(spec).index("tiny")
    alpha, beta, _ = priors[i]
    return posterior(5, 3, alpha, beta)[0], alpha + beta


def test_shrinkage_is_stronger_when_the_between_group_spread_is_small() -> None:
    tight_mean, tight_strength = _tiny_mean([0.08, 0.08, 0.08, 0.08, 0.08])
    spread_mean, spread_strength = _tiny_mean([0.02, 0.05, 0.08, 0.11, 0.14])
    assert tight_strength > spread_strength
    assert tight_mean < spread_mean                            # more pull toward ~8%
    assert tight_mean < 0.3 and spread_mean < 0.6


# --------------------------------------------------------------------------------------
# ranking: lower credible bound, then posterior mean, then group name
# --------------------------------------------------------------------------------------
def test_ranking_at_the_same_rate_less_evidence_ranks_below_more_evidence() -> None:
    got = run({**CLEAN, "few": (30, 9), "many": (1000, 300)})
    rows = by_value(got.rows)
    assert rows["many"]["ci_low"] > rows["few"]["ci_low"]
    assert order(got.rows).index("many") < order(got.rows).index("few")


def test_ranking_is_sorted_by_lower_bound_then_mean_then_name() -> None:
    got = run({**CLEAN, "few": (30, 9), "many": (1000, 300), "mid": (100, 30)})
    rows = [r for r in got.rows if r["group_key"] == "contributor"]
    expected = sorted(rows, key=lambda r: (-r["ci_low"], -r["posterior_mean"], r["group_value"]))
    assert rows == expected


def test_ranking_ties_are_broken_by_group_name() -> None:
    got = run({"zeta": (100, 10), "alpha": (100, 10), "mid": (100, 10), "other": (100, 20)})
    rows = by_value(got.rows)
    assert rows["alpha"]["ci_low"] == rows["zeta"]["ci_low"] == rows["mid"]["ci_low"]
    tied = [g for g in order(got.rows) if g in {"alpha", "mid", "zeta"}]
    assert tied == ["alpha", "mid", "zeta"]


def test_ranking_is_independent_of_the_order_the_samples_arrive_in() -> None:
    samples, flagged = make({**CLEAN, "few": (30, 9), "many": (1000, 300)})
    forward = assess_groups(samples, flagged, POLICY, SEED)
    backward = assess_groups(list(reversed(samples)), flagged, POLICY, SEED)
    assert forward[0] == backward[0]


def test_ranking_a_raw_rate_would_get_it_wrong_and_the_bound_does_not() -> None:
    """3/5 is 60% and 800/10,000 is 8%: ranking on the raw rate puts the 5-image group first."""
    rows = by_value(run(V15_COUNTS).rows)
    assert rows["tiny"]["n_flagged"] / rows["tiny"]["n_samples"] > (
        rows["big"]["n_flagged"] / rows["big"]["n_samples"])
    assert rows["big"]["ci_low"] > rows["tiny"]["ci_low"]


# --------------------------------------------------------------------------------------
# V15 with the plan's own counts
# --------------------------------------------------------------------------------------
def test_v15_the_guilty_group_ranks_first_and_is_quarantined(v15: GroupAssessment) -> None:
    rows = by_value(v15.rows)
    assert order(v15.rows)[0] == "guilty"
    assert rows["guilty"]["disposition"] == "quarantine"
    assert rows["guilty"]["excludes_cohort_rate"] is True
    assert rows["guilty"]["ci_low"] > 0.5 > rows["big"]["ci_high"]      # nowhere near the cohort


def test_v15_the_permutation_test_runs_and_rejects(v15: GroupAssessment) -> None:
    perm = v15.permutation_test
    assert perm is not None
    assert perm["p_value"] < 0.05
    # no permutation beat it; the row rounds p to 6 places
    assert perm["p_value"] == pytest.approx(1 / (N_PERMUTATIONS + 1), abs=1e-6)
    assert "non-randomly" in perm["conclusion"] and "contributor" in perm["conclusion"]


def test_v15_a_3_of_5_group_is_never_quarantined_on_its_raw_rate(v15: GroupAssessment) -> None:
    tiny = by_value(v15.rows)["tiny"]
    assert tiny["n_flagged"] / tiny["n_samples"] >= 0.25          # the raw rate is over D4's floor
    assert tiny["excludes_cohort_rate"] is False
    assert tiny["disposition"] != "quarantine"


def test_v15_the_big_group_outranks_the_tiny_one(v15: GroupAssessment) -> None:
    names = order(v15.rows)
    assert names.index("big") < names.index("tiny")
    rows = by_value(v15.rows)
    assert rows["big"]["ci_low"] > rows["tiny"]["ci_low"]


def test_v15_only_the_guilty_group_is_quarantined(v15: GroupAssessment) -> None:
    quarantined = [r["group_value"] for r in v15.rows if r["disposition"] == "quarantine"]
    assert quarantined == ["guilty"]


def test_v15_the_cohort_rate_is_the_robust_one_not_the_pooled_rate(v15: GroupAssessment) -> None:
    pooled = sum(k for _, k in V15_COUNTS.values()) / sum(n for n, _ in V15_COUNTS.values())
    cohort = v15.cohort_rates["contributor"]
    assert 0.07 < cohort < 0.09
    assert cohort < pooled                                         # the guilty group is set aside


def test_v15_mixed_fixture_counts_guilty_first_tiny_not_quarantined_permutation_rejects(
        mixed: GroupAssessment) -> None:
    rows = by_value(mixed.rows)
    assert order(mixed.rows)[0] == "guilty"
    assert rows["guilty"]["disposition"] == "quarantine"
    assert rows["tiny"]["disposition"] != "quarantine"
    assert rows["tiny"]["excludes_cohort_rate"] is False
    assert mixed.permutation_test is not None
    assert mixed.permutation_test["p_value"] < 0.01
    # deliberately NOT asserted: whether tiny ranks below 12/150 here. It narrowly does not,
    # and the plan documents that.


# --------------------------------------------------------------------------------------
# the robust prior: outliers are set aside, and a group never informs its own baseline
# --------------------------------------------------------------------------------------
def test_robust_prior_an_extreme_outlier_group_does_not_move_the_prior() -> None:
    counts, flagged = arrays(CLEAN)
    base = prior_for(counts, flagged)
    with_outlier = prior_for(np.append(counts, 300.0), np.append(flagged, 290.0))
    assert with_outlier == pytest.approx(base)


def test_robust_prior_strength_does_not_collapse_to_its_floor_with_an_outlier() -> None:
    spec = {"g0": (200, 8), "g1": (200, 12), "g2": (200, 16), "g3": (200, 20), "g4": (200, 24)}
    counts, flagged = arrays(spec)
    plain_alpha, plain_beta, _ = prior_for(counts, flagged)
    counts2, flagged2 = np.append(counts, 30.0), np.append(flagged, 30.0)
    alpha, beta, rate = prior_for(counts2, flagged2)
    assert alpha + beta == pytest.approx(plain_alpha + plain_beta)
    assert alpha + beta > 10 * PRIOR_STRENGTH[0]
    assert rate == pytest.approx(0.08)
    # the premise: a moment estimate over ALL groups, outlier included, does collapse
    pooled = float(flagged2.sum() / counts2.sum())
    naive = _beta_binomial_strength(counts2, flagged2, pooled)
    assert naive is not None and naive < (alpha + beta) / 2


def test_robust_prior_the_cohort_rate_excludes_the_outlier_flags() -> None:
    counts, flagged = arrays({**CLEAN, "guilty": (300, 290)})
    _, _, rate = prior_for(counts, flagged)
    assert rate == pytest.approx(325 / 4000)


def test_robust_prior_with_fewer_than_two_baseline_groups_falls_back_to_pooled_and_default(
        ) -> None:
    counts, flagged = np.array([100.0, 100.0]), np.array([0.0, 90.0])   # 90/100 is the outlier
    alpha, beta, rate = prior_for(counts, flagged)
    assert rate == pytest.approx(0.45)                       # the pooled rate
    assert alpha + beta == pytest.approx(DEFAULT_STRENGTH)
    assert alpha == pytest.approx(0.45 * DEFAULT_STRENGTH)


def test_robust_prior_a_single_group_falls_back_to_pooled_and_default() -> None:
    alpha, beta, rate = prior_for(np.array([50.0]), np.array([5.0]))
    assert rate == pytest.approx(0.1)
    assert alpha + beta == pytest.approx(DEFAULT_STRENGTH)


def test_robust_prior_two_ordinary_groups_estimate_a_strength_rather_than_use_the_default(
        ) -> None:
    alpha, beta, _ = prior_for(np.array([100.0, 100.0]), np.array([10.0, 12.0]))
    assert alpha + beta == pytest.approx(PRIOR_STRENGTH[1])   # no excess spread: pool as hard as allowed


def test_robust_prior_strength_stays_inside_its_clamp() -> None:
    for spec in ({"a": (200, 2), "b": (200, 60), "c": (200, 20), "d": (200, 30)},
                 {"a": (200, 16), "b": (200, 16), "c": (200, 16)}):
        alpha, beta, rate = prior_for(*arrays(spec))
        assert PRIOR_STRENGTH[0] <= alpha + beta <= PRIOR_STRENGTH[1]
        assert 0.0 < rate < 1.0


# --- leave-one-out ---
_LOO = {**CLEAN, "g": (50, 5)}


@pytest.mark.parametrize("perturbed", [0, 5, 9, 30, 50])
def test_loo_a_group_never_informs_its_own_prior(perturbed: int) -> None:
    counts, flagged = arrays(_LOO)
    i = list(_LOO).index("g")
    before = loo_priors(counts, flagged)[i]
    flagged2 = flagged.copy()
    flagged2[i] = perturbed
    after = loo_priors(counts, flagged2)[i]
    assert after == pytest.approx(before)


def test_loo_the_prior_of_a_group_is_the_prior_learned_from_the_others_alone() -> None:
    counts, flagged = arrays(_LOO)
    i = list(_LOO).index("g")
    rest_c, rest_f = np.delete(counts, i), np.delete(flagged, i)
    assert loo_priors(counts, flagged)[i] == pytest.approx(prior_for(rest_c, rest_f))


def test_loo_there_is_one_prior_per_group_and_each_omits_its_own_group() -> None:
    counts, flagged = arrays(CLEAN)
    priors = loo_priors(counts, flagged)
    assert len(priors) == len(counts)
    for i, prior in enumerate(priors):
        rest = prior_for(np.delete(counts, i), np.delete(flagged, i))
        assert prior == pytest.approx(rest)


def test_loo_an_outlier_group_gets_the_all_survivors_prior() -> None:
    counts, flagged = arrays({**CLEAN, "guilty": (300, 290)})
    priors = loo_priors(counts, flagged)
    assert priors[-1] == pytest.approx(prior_for(counts, flagged))


def test_loo_the_tiny_group_does_not_drag_the_prior_it_is_judged_against() -> None:
    """The measured V15 failure: the 3/5 group supplied most of the between-group X2 and was then
    judged against the prior it had weakened. Its own LOO prior is untouched by its counts."""
    counts, flagged = arrays(V15_COUNTS)
    i = list(V15_COUNTS).index("tiny")
    strength_for_tiny = sum(loo_priors(counts, flagged)[i][:2])
    strength_for_clean = sum(loo_priors(counts, flagged)[list(V15_COUNTS).index("clean_a")][:2])
    assert strength_for_tiny > 10 * PRIOR_STRENGTH[0]
    assert strength_for_tiny >= strength_for_clean          # clean_a's prior still includes tiny


def test_loo_through_the_api_a_row_is_scored_against_its_leave_one_out_prior() -> None:
    spec = {**CLEAN, "g": (50, 5)}
    got = run(spec)
    counts, flagged = arrays(spec)
    i = list(spec).index("g")
    alpha, beta, cohort = loo_priors(counts, flagged)[i]
    mean, lo, hi = posterior(50, 5, alpha, beta)
    row = by_value(got.rows)["g"]
    assert row["posterior_mean"] == pytest.approx(mean, abs=1e-6)
    assert row["ci_low"] == pytest.approx(lo, abs=1e-6)
    assert row["ci_high"] == pytest.approx(hi, abs=1e-6)
    assert row["excludes_cohort_rate"] is bool(lo > cohort)


def test_assessment_is_deterministic_for_identical_inputs() -> None:
    assert run({**CLEAN, "g": (50, 5)}) == run({**CLEAN, "g": (50, 5)})


# --------------------------------------------------------------------------------------
# grouping keys
# --------------------------------------------------------------------------------------
def test_keys_the_group_keys_are_the_three_the_schema_names() -> None:
    assert set(GROUP_KEYS) == SCHEMA_GROUP_KEYS


def test_keys_key_of_reads_the_attribute_or_source_meta_and_skips_none_and_empty() -> None:
    s = mk_sample("a", contributor="c", batch="b", source="s")
    assert key_of(s, "contributor") == "c"
    assert key_of(s, "batch") == "b"
    assert key_of(s, "source") == "s"
    assert key_of(mk_sample("x"), "contributor") is None
    assert key_of(mk_sample("x", contributor=""), "contributor") is None
    assert key_of(mk_sample("x", source_meta={"source": ""}), "source") is None
    assert key_of(mk_sample("x"), "source") is None
    none_meta = SimpleNamespace(sample_id="x", contributor=None, batch=None, source_meta=None)
    assert key_of(none_meta, "source") is None


def test_keys_key_of_stringifies_a_non_string_value() -> None:
    assert key_of(mk_sample("a", batch=7), "batch") == "7"      # type: ignore[arg-type]


@pytest.mark.parametrize("key", GROUP_KEYS)
def test_keys_each_key_produces_rows_only_when_the_samples_carry_it(key: str) -> None:
    samples, flagged = make({"a": (20, 10), "b": (20, 0)}, key=key)
    rows, _ = assess_groups(samples, flagged, POLICY, SEED)
    assert {r["group_key"] for r in rows} == {key}
    assert sorted(r["group_value"] for r in rows) == ["a", "b"]


def test_keys_samples_without_any_key_give_no_rows_and_no_test() -> None:
    samples = [mk_sample(f"s{i}") for i in range(10)]
    got = assess_groups_detailed(samples, {"s0", "s1"}, POLICY, SEED)
    assert got.rows == [] and got.permutation_test is None and got.cohort_rates == {}


def test_keys_source_reads_source_meta_source() -> None:
    samples = [mk_sample(f"a{i}", source="web") for i in range(10)] + [
        mk_sample(f"b{i}", source="camera") for i in range(10)]
    flagged = {f"a{i}" for i in range(8)}
    rows, _ = assess_groups(samples, flagged, POLICY, SEED)
    got = by_value(rows, "source")
    assert got["web"]["n_flagged"] == 8 and got["camera"]["n_flagged"] == 0
    assert not [r for r in rows if r["group_key"] != "source"]


def test_keys_a_source_set_elsewhere_in_source_meta_is_not_a_source() -> None:
    samples = [mk_sample(f"s{i}", source_meta={"format": "coco"}) for i in range(6)]
    assert assess_groups(samples, set(), POLICY, SEED)[0] == []


def test_keys_samples_with_a_none_contributor_are_skipped() -> None:
    known, flagged = make({"a": (10, 5), "b": (10, 0)})
    unknown = [mk_sample(f"u{i}") for i in range(30)]
    flagged |= {f"u{i}" for i in range(30)}                       # flagged, but nobody's
    rows, _ = assess_groups([*known, *unknown], flagged, POLICY, SEED)
    assert sum(r["n_samples"] for r in rows) == 20
    assert sum(r["n_flagged"] for r in rows) == 5
    assert {r["group_value"] for r in rows} == {"a", "b"}


def test_keys_an_empty_string_contributor_is_skipped_like_none() -> None:
    samples = [mk_sample(f"a{i}", contributor="a") for i in range(5)] + [
        mk_sample(f"e{i}", contributor="") for i in range(5)]
    rows, _ = assess_groups(samples, set(), POLICY, SEED)
    assert {r["group_value"] for r in rows} == {"a"}


def test_keys_all_three_keys_at_once_give_rows_for_each() -> None:
    samples = [mk_sample(f"s{i}", contributor="c1" if i < 10 else "c2",
                         batch="b1" if i % 2 else "b2", source="src") for i in range(20)]
    rows, _ = assess_groups(samples, {"s0", "s1"}, POLICY, SEED)
    assert {r["group_key"] for r in rows} == set(GROUP_KEYS)
    assert order(rows, "source") == ["src"]


def test_keys_the_keys_parameter_restricts_the_grouping() -> None:
    samples = [mk_sample(f"s{i}", contributor="c1" if i < 10 else "c2", batch="b") for i in range(20)]
    rows, _ = assess_groups(samples, {"s0"}, POLICY, SEED, keys=("batch",))
    assert {r["group_key"] for r in rows} == {"batch"}


def test_keys_the_permutation_test_runs_on_the_first_key_with_two_groups() -> None:
    samples = [mk_sample(f"s{i}", contributor="only", batch="b1" if i < 25 else "b2")
               for i in range(50)]
    flagged = {f"s{i}" for i in range(25)}
    got = assess_groups_detailed(samples, flagged, POLICY, SEED)
    assert got.permutation_test is not None
    assert "batch" in got.permutation_test["conclusion"]
    assert got.permutation_test["p_value"] < 0.01


def test_keys_contributor_rows_carry_the_most_common_contributor_source_tier() -> None:
    samples = ([mk_sample(f"a{i}", contributor="a", tier=ContributorSource.SIDECAR)
                for i in range(2)]
               + [mk_sample("a2", contributor="a", tier=ContributorSource.DIRECTORY)]
               + [mk_sample(f"b{i}", contributor="b", tier=ContributorSource.EXIF_CLUSTER)
                  for i in range(3)])
    rows, _ = assess_groups(samples, set(), POLICY, SEED)
    got = by_value(rows)
    assert got["a"]["contributor_source"] == "sidecar"
    assert got["b"]["contributor_source"] == "exif_cluster"
    assert all(r["contributor_source"] in SCHEMA_TIERS for r in rows)


def test_keys_a_contributor_row_has_no_source_tier_when_no_sample_carries_one() -> None:
    samples = [mk_sample(f"a{i}", contributor="a") for i in range(4)]
    (row,) = assess_groups(samples, set(), POLICY, SEED)[0]
    assert "contributor_source" not in row


def test_keys_contributor_source_is_only_on_contributor_rows() -> None:
    samples = [mk_sample(f"s{i}", contributor="c1" if i < 6 else "c2", batch="b1" if i < 3 else "b2",
                         source="src", tier=ContributorSource.SIDECAR) for i in range(12)]
    rows, _ = assess_groups(samples, set(), POLICY, SEED)
    for r in rows:
        assert ("contributor_source" in r) == (r["group_key"] == "contributor")


def test_keys_every_row_has_the_schema_row_shape(v15: GroupAssessment) -> None:
    samples = [mk_sample(f"s{i}", contributor=f"c{i % 3}", batch=f"b{i % 2}", source="s",
                         tier=ContributorSource.DIRECTORY) for i in range(60)]
    rows = assess_groups(samples, {f"s{i}" for i in range(0, 60, 4)}, POLICY, SEED)[0]
    rows = [*rows, *v15.rows]
    allowed = REQUIRED_ROW_KEYS | {"contributor_source", "excludes_cohort_rate",
                                   "cohort_rate_used", "excludes_reference_rate",
                                   "disposition"}
    assert rows
    for r in rows:
        assert REQUIRED_ROW_KEYS <= set(r) <= allowed
        # Item 26: the row publishes the rate its own decision used, which is the
        # leave-one-out rate and not the all-survivors rate the baseline block prints.
        assert r["excludes_cohort_rate"] == (r["ci_low"] > r["cohort_rate_used"])
        assert r["group_key"] in SCHEMA_GROUP_KEYS
        assert isinstance(r["group_value"], str)
        assert type(r["n_samples"]) is int and type(r["n_flagged"]) is int
        assert 0 <= r["n_flagged"] <= r["n_samples"]
        for f in ("posterior_mean", "ci_low", "ci_high"):
            assert type(r[f]) is float and 0.0 <= r[f] <= 1.0
        assert r["ci_low"] <= r["posterior_mean"] <= r["ci_high"]
        assert type(r["excludes_cohort_rate"]) is bool
        assert r["disposition"] in {"accept", "review", "quarantine"}


def test_keys_group_sizes_and_flag_counts_are_counted_per_group() -> None:
    got = by_value(run({"a": (37, 11), "b": (12, 0), "c": (9, 9)}).rows)
    assert (got["a"]["n_samples"], got["a"]["n_flagged"]) == (37, 11)
    assert (got["b"]["n_samples"], got["b"]["n_flagged"]) == (12, 0)
    assert (got["c"]["n_samples"], got["c"]["n_flagged"]) == (9, 9)


def test_keys_a_flagged_id_that_is_not_in_the_dataset_is_ignored() -> None:
    samples, flagged = make({"a": (10, 0), "b": (10, 0)})
    rows, _ = assess_groups(samples, {*flagged, "ghost#1", "ghost#2"}, POLICY, SEED)
    assert sum(r["n_flagged"] for r in rows) == 0


def test_keys_assess_groups_is_the_rows_and_test_of_the_detailed_call() -> None:
    samples, flagged = make({"a": (40, 20), "b": (40, 0), "c": (40, 2)})
    rows, perm = assess_groups(samples, flagged, POLICY, SEED)
    detailed = assess_groups_detailed(samples, flagged, POLICY, SEED)
    assert (rows, perm) == (detailed.rows, detailed.permutation_test)
    assert set(detailed.cohort_rates) == {"contributor"}


# --------------------------------------------------------------------------------------
# excludes_reference_rate (informational; D4 stays on the cohort)
# --------------------------------------------------------------------------------------
def test_reference_rate_flag_is_absent_without_a_ceiling_and_present_with_one() -> None:
    plain = run(V15_COUNTS).rows
    with_ceiling = run(V15_COUNTS, reference_ceiling=0.05).rows
    assert all("excludes_reference_rate" not in r for r in plain)
    assert all(type(r["excludes_reference_rate"]) is bool for r in with_ceiling)


def test_reference_rate_flag_is_ci_low_above_the_ceiling() -> None:
    rows = run(V15_COUNTS, reference_ceiling=0.05).rows
    for r in rows:
        assert r["excludes_reference_rate"] is (r["ci_low"] > 0.05)
    assert by_value(rows)["guilty"]["excludes_reference_rate"] is True


def test_reference_rate_flag_never_changes_a_disposition_or_a_ranking() -> None:
    plain = run(V15_COUNTS).rows
    for ceiling in (0.0, 0.05, 0.5, 0.99):
        with_ceiling = run(V15_COUNTS, reference_ceiling=ceiling).rows
        assert [(r["group_value"], r["disposition"], r["excludes_cohort_rate"])
                for r in with_ceiling] == [
            (r["group_value"], r["disposition"], r["excludes_cohort_rate"]) for r in plain]


# --------------------------------------------------------------------------------------
# permutation_test()
# --------------------------------------------------------------------------------------
def _codes_flags(spec: dict[str, tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    codes, flags = [], []
    for gi, (n, k) in enumerate(spec.values()):
        codes += [gi] * n
        flags += [i < k for i in range(n)]
    return np.array(codes, dtype=np.int64), np.array(flags, dtype=bool), len(spec)


CLUSTERED = {"a": (50, 50), "b": (50, 0), "c": (50, 0), "d": (50, 0)}


def test_permutation_is_deterministic_for_a_fixed_seed() -> None:
    codes, flags, g = _codes_flags({"a": (40, 12), "b": (40, 8), "c": (40, 5), "d": (40, 9)})
    assert permutation_test(codes, flags, g, seed=3) == permutation_test(codes, flags, g, seed=3)


def test_permutation_different_seeds_agree_on_the_statistic_and_the_verdict() -> None:
    codes, flags, g = _codes_flags(CLUSTERED)
    results = [permutation_test(codes, flags, g, seed=s) for s in (1, 2, 3, 99)]
    assert len({r[0] for r in results}) == 1                       # the statistic has no RNG in it
    assert all(r[1] < 0.05 for r in results)
    assert max(r[1] for r in results) - min(r[1] for r in results) < 0.05


def test_permutation_different_seeds_may_move_p_only_by_monte_carlo_noise() -> None:
    codes, flags, g = _codes_flags({"a": (40, 12), "b": (40, 8), "c": (40, 5), "d": (40, 9)})
    ps = [permutation_test(codes, flags, g, seed=s)[1] for s in range(5)]
    assert max(ps) - min(ps) < 0.15


def test_permutation_statistic_is_pearsons_chi_square_of_the_flag_table() -> None:
    codes, flags, g = _codes_flags({"a": (10, 10), "b": (10, 0)})
    stat, _ = permutation_test(codes, flags, g, seed=0)
    assert stat == pytest.approx(20.0)


def test_permutation_an_obvious_cluster_is_rejected_at_the_smallest_possible_p() -> None:
    codes, flags, g = _codes_flags(CLUSTERED)
    _, p = permutation_test(codes, flags, g, seed=0)
    assert p == pytest.approx(1 / (N_PERMUTATIONS + 1))


def test_permutation_flags_spread_evenly_are_not_rejected() -> None:
    codes, flags, g = _codes_flags({"a": (20, 5), "b": (20, 5), "c": (20, 5)})
    stat, p = permutation_test(codes, flags, g, seed=0)
    assert stat == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_permutation_p_is_a_valid_monte_carlo_p_value() -> None:
    codes, flags, g = _codes_flags({"a": (30, 3), "b": (30, 6), "c": (30, 4)})
    _, p = permutation_test(codes, flags, g, seed=5, n_perm=200)
    assert 1 / 201 <= p <= 1.0
    assert (p * 201) == pytest.approx(round(p * 201))              # (1 + hits) / (1 + n_perm)


def test_permutation_n_perm_bounds_how_small_p_can_be() -> None:
    codes, flags, g = _codes_flags(CLUSTERED)
    _, p = permutation_test(codes, flags, g, seed=0, n_perm=50)
    assert p == pytest.approx(1 / 51)


def test_permutation_the_default_number_of_permutations_is_n_permutations() -> None:
    default = inspect.signature(permutation_test).parameters["n_perm"].default
    assert default == N_PERMUTATIONS


def test_permutation_does_not_mutate_its_inputs() -> None:
    codes, flags, g = _codes_flags(CLUSTERED)
    c0, f0 = codes.copy(), flags.copy()
    permutation_test(codes, flags, g, seed=0, n_perm=20)
    assert (codes == c0).all() and (flags == f0).all()


@pytest.mark.parametrize("spec", [
    {"only": (30, 10)},                                            # one group
    {"a": (10, 10), "b": (10, 10)},                                # everything flagged
    {"a": (10, 0), "b": (10, 0)},                                  # nothing flagged
])
def test_permutation_degenerate_inputs_return_zero_and_one(spec: dict[str, tuple[int, int]]) -> None:
    codes, flags, g = _codes_flags(spec)
    assert permutation_test(codes, flags, g, seed=0) == (0.0, 1.0)


def test_permutation_the_reported_n_permutations_is_n_permutations() -> None:
    got = run({"a": (30, 30), "b": (30, 0)})
    assert got.permutation_test is not None
    assert got.permutation_test["n_permutations"] == N_PERMUTATIONS


def test_permutation_the_test_is_seeded_from_the_scan_seed() -> None:
    samples, flagged = make({"a": (40, 12), "b": (40, 8), "c": (40, 5), "d": (40, 9)})
    a = assess_groups_detailed(samples, flagged, POLICY, 11).permutation_test
    b = assess_groups_detailed(samples, flagged, POLICY, 11).permutation_test
    c = assess_groups_detailed(samples, flagged, POLICY, 12).permutation_test
    assert a == b
    assert c is not None and a is not None and a["statistic"] == c["statistic"]


# --------------------------------------------------------------------------------------
# no flags at all
# --------------------------------------------------------------------------------------
def test_no_flags_every_group_is_accepted() -> None:
    got = run({"a": (30, 0), "b": (50, 0), "c": (5, 0)})
    assert {r["disposition"] for r in got.rows} == {"accept"}
    assert all(r["n_flagged"] == 0 and r["excludes_cohort_rate"] is False for r in got.rows)
    assert all(r["posterior_mean"] < 0.1 for r in got.rows)


def test_no_flags_the_permutation_test_finds_no_evidence() -> None:
    perm = run({"a": (30, 0), "b": (50, 0), "c": (5, 0)}).permutation_test
    assert perm is not None
    assert perm["p_value"] == 1.0
    assert perm["statistic"] == 0.0
    assert perm["n_permutations"] == N_PERMUTATIONS
    assert "No evidence" in perm["conclusion"]
    assert "non-randomly" not in perm["conclusion"]


def test_no_flags_the_lower_bound_is_near_zero_for_every_group() -> None:
    assert all(r["ci_low"] < 0.02 for r in run({"a": (30, 0), "b": (50, 0)}).rows)


def test_no_samples_at_all_is_an_empty_assessment() -> None:
    got = assess_groups_detailed([], set(), POLICY, SEED)
    assert got.rows == [] and got.permutation_test is None and got.cohort_rates == {}


def test_a_single_group_gets_rows_but_no_permutation_test() -> None:
    got = run({"solo": (40, 10)})
    assert len(got.rows) == 1 and got.permutation_test is None
    assert got.cohort_rates["contributor"] == pytest.approx(0.25)


def test_every_group_appears_exactly_once_and_the_flag_totals_add_up() -> None:
    spec = {"a": (40, 20), "b": (40, 5), "c": (40, 1)}
    got = run(spec)
    assert Counter(r["group_value"] for r in got.rows) == Counter({"a": 1, "b": 1, "c": 1})
    assert {r["group_value"]: (r["n_samples"], r["n_flagged"]) for r in got.rows} == spec
    assert sum(r["n_flagged"] for r in got.rows) == 26
