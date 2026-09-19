"""B7's absolute rate against a reference clean set (backend_plan.md §8 B7, amended).

Three layers, all in-process and offline:

1. `assess()` / `flagged_sample_ids()` / `reference_flag_counts()` driven with hand-built
   findings and a `SimpleNamespace` dataset: the baseline object, the Jeffreys ceiling, and the
   ONE definition of "flagged" the cohort and the reference share.
2. `scan()` with a fake `data.fake` detector in a local registry, so the orchestrator's
   `_reference_flags` and the report path are exercised for real. `build_embeddings` is patched
   to skip the backbone (it returns `(None, None)`); for a dataset over `embedding_max_images`
   the patch hands off to the REAL function, which reports the ceiling before it imports any
   extractor, so nothing here loads DINOv2.
3. `render_html.render` for the three sentences the contributor section can print.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cva.core import orchestrator
from cva.core.capability import Availability, Capability, CapabilitySet, Resolution
from cva.core.orchestrator import PlanRow, ScanResult, profile_hash_of, resolve_profile, scan
from cva.core.runcontext import RunContext
from cva.core.scanid import VOLATILE_PATHS
from cva.core.types import Category, Disposition, Finding, Sample, Severity
from cva.report import report_json
from cva.report.render_html import render
from cva.risk.contributor import assess_groups_detailed
from cva.risk.disposition import default_policy
from cva.risk.engine import (
    RiskOutcome,
    _reference_ceiling,
    assess,
    flagged_sample_ids,
    reference_flag_counts,
)

SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "report.schema.json"
BASELINE_KEYS = {"reference_n", "reference_flagged", "reference_rate", "reference_rate_ci_high",
                 "cohort_rate", "cohort_exceeds_reference"}
OK, DEGRADED = Availability.OK, Availability.DEGRADED
UNAVAILABLE, ERROR = Availability.UNAVAILABLE, Availability.ERROR


# --------------------------------------------------------------------------------------
# builders for the direct-`assess` layer
# --------------------------------------------------------------------------------------
def sfinding(ref: str, *, severity: Severity = Severity.HIGH, confidence: float = 0.95,
             availability: Availability = OK, target_type: str = "sample",
             detector: str = "data.fake") -> Finding:
    return Finding(detector_id=detector, detector_version="0.0.1",
                   target_type=target_type,  # type: ignore[arg-type]
                   target_ref=ref, severity=severity, confidence=confidence, reason="r",
                   attack_class="label_flipping", availability=availability)


def cohort(spec: dict[str, tuple[int, int]]) -> tuple[SimpleNamespace, list[Finding]]:
    """`spec`: contributor -> (n_samples, n_flagged). A flagged sample gets a finding that routes
    to quarantine; every other sample gets an `info` finding that routes to accept, so the
    "not accept" half of the flag definition has something to reject."""
    samples: list[SimpleNamespace] = []
    findings: list[Finding] = []
    for contributor, (n, k) in spec.items():
        for i in range(n):
            sid = f"{contributor}-{i}"
            samples.append(SimpleNamespace(sample_id=sid, contributor=contributor, batch=None,
                                           source_meta={}, contributor_source=None))
            findings.append(sfinding(sid) if i < k
                            else sfinding(sid, severity=Severity.INFO, confidence=0.1))
    return SimpleNamespace(samples=samples), findings


HIGH_COHORT = {"c1": (100, 30), "c2": (100, 30), "c3": (100, 32), "c4": (100, 28)}   # 30%
LOW_COHORT = {"c1": (100, 1), "c2": (100, 1), "c3": (100, 1), "c4": (100, 1)}        # 1%
CLEAN_REF = {"reference_n": 60, "reference_flagged": 0}


def run_assess(spec: dict[str, tuple[int, int]], reference: dict[str, int] | None = None,
               reference_unavailable: str | None = None) -> RiskOutcome:
    ds, findings = cohort(spec)
    return assess(findings, ds, {}, 5, None, reference, reference_unavailable)


# --------------------------------------------------------------------------------------
# the Jeffreys ceiling
# --------------------------------------------------------------------------------------
def test_ceiling_zero_flagged_of_60_is_not_a_zero_ceiling() -> None:
    ceiling = _reference_ceiling(60, 0)
    assert 0.0 < ceiling < 0.06


def test_ceiling_is_above_the_point_estimate_and_shrinks_with_more_reference_data() -> None:
    assert _reference_ceiling(60, 6) > 6 / 60
    assert _reference_ceiling(600, 60) < _reference_ceiling(60, 6)
    assert _reference_ceiling(600, 0) < _reference_ceiling(60, 0)


def test_ceiling_rises_with_the_reference_flag_count() -> None:
    ceilings = [_reference_ceiling(60, k) for k in (0, 1, 3, 10, 30)]
    assert ceilings == sorted(ceilings) and len(set(ceilings)) == len(ceilings)
    assert all(0.0 < c < 1.0 for c in ceilings)


# --------------------------------------------------------------------------------------
# assess(): the baseline object
# --------------------------------------------------------------------------------------
def test_assess_a_usable_reference_gives_a_baseline_with_every_field() -> None:
    out = run_assess(HIGH_COHORT, CLEAN_REF)
    base = out.contributor_baseline
    assert base is not None
    assert set(base) == BASELINE_KEYS
    assert base["reference_n"] == 60 and base["reference_flagged"] == 0
    assert base["reference_rate"] == 0.0
    assert 0.0 < base["reference_rate_ci_high"] < 0.06
    assert base["reference_rate_ci_high"] == pytest.approx(_reference_ceiling(60, 0), abs=1e-6)
    assert base["cohort_rate"] == pytest.approx(0.30, abs=1e-3)
    assert out.contributor_baseline_unavailable is None


def test_assess_the_reference_rate_is_flagged_over_n() -> None:
    base = run_assess(HIGH_COHORT, {"reference_n": 60, "reference_flagged": 9}).contributor_baseline
    assert base is not None
    assert base["reference_rate"] == pytest.approx(0.15)
    assert base["reference_rate_ci_high"] > 0.15


def test_assess_the_cohort_rate_is_the_robust_one_of_the_first_key_with_data() -> None:
    spec = {**HIGH_COHORT, "guilty": (300, 290)}
    ds, findings = cohort(spec)
    out = assess(findings, ds, {}, 5, None, CLEAN_REF)
    robust = assess_groups_detailed(ds.samples, {f.target_ref for f in findings
                                                 if f.severity == Severity.HIGH},
                                    default_policy(), 5).cohort_rates["contributor"]
    assert out.contributor_baseline is not None
    assert out.contributor_baseline["cohort_rate"] == pytest.approx(robust, abs=1e-6)
    assert out.contributor_baseline["cohort_rate"] == pytest.approx(0.30, abs=1e-3)


def test_assess_cohort_above_the_reference_ceiling_exceeds_it() -> None:
    base = run_assess(HIGH_COHORT, CLEAN_REF).contributor_baseline
    assert base is not None
    assert base["cohort_rate"] > base["reference_rate_ci_high"]
    assert base["cohort_exceeds_reference"] is True


def test_assess_cohort_at_or_below_the_reference_ceiling_does_not_exceed_it() -> None:
    base = run_assess(LOW_COHORT, CLEAN_REF).contributor_baseline
    assert base is not None
    assert base["cohort_rate"] <= base["reference_rate_ci_high"]     # 1% against a ~4% ceiling
    assert base["cohort_exceeds_reference"] is False


def test_assess_a_cohort_that_matches_a_dirty_reference_does_not_exceed_it() -> None:
    dirty = {"reference_n": 60, "reference_flagged": 18}             # 30% reference
    base = run_assess(HIGH_COHORT, dirty).contributor_baseline
    assert base is not None
    assert base["cohort_exceeds_reference"] is False


@pytest.mark.parametrize("spec", [HIGH_COHORT, LOW_COHORT])
@pytest.mark.parametrize("flagged", [0, 3, 12])
def test_assess_cohort_exceeds_reference_is_exactly_cohort_above_the_ceiling(
        spec: dict[str, tuple[int, int]], flagged: int) -> None:
    base = run_assess(spec, {"reference_n": 60, "reference_flagged": flagged}).contributor_baseline
    assert base is not None
    assert base["cohort_exceeds_reference"] is (base["cohort_rate"] > base["reference_rate_ci_high"])


def test_assess_rows_carry_excludes_reference_rate_when_a_reference_was_usable() -> None:
    out = run_assess({**HIGH_COHORT, "guilty": (100, 95)}, CLEAN_REF)
    assert out.contributor_risk
    for row in out.contributor_risk:
        assert type(row["excludes_reference_rate"]) is bool
        assert row["excludes_reference_rate"] is (row["ci_low"] > _reference_ceiling(60, 0))
    guilty = next(r for r in out.contributor_risk if r["group_value"] == "guilty")
    assert guilty["excludes_reference_rate"] is True


def test_assess_excludes_reference_rate_never_changes_a_disposition() -> None:
    spec = {**HIGH_COHORT, "guilty": (100, 95)}
    with_ref = run_assess(spec, CLEAN_REF).contributor_risk
    without = run_assess(spec).contributor_risk
    assert [(r["group_value"], r["disposition"]) for r in with_ref] == [
        (r["group_value"], r["disposition"]) for r in without]


def test_assess_no_reference_means_no_baseline_and_no_reference_column() -> None:
    out = run_assess(HIGH_COHORT)
    assert out.contributor_baseline is None
    assert out.contributor_baseline_unavailable is None
    assert out.contributor_risk
    assert all("excludes_reference_rate" not in r for r in out.contributor_risk)


def test_assess_an_unusable_reference_carries_its_reason_and_no_baseline() -> None:
    why = "the reference dataset was not embedded: over the ceiling"
    out = run_assess(HIGH_COHORT, None, why)
    assert out.contributor_baseline is None
    assert out.contributor_baseline_unavailable == why
    assert all("excludes_reference_rate" not in r for r in out.contributor_risk)


def test_assess_the_reason_survives_a_dataset_that_is_empty_or_absent() -> None:
    for dataset in (None, SimpleNamespace(samples=[])):
        out = assess([], dataset, {}, 0, None, None, "why not")
        assert out.contributor_baseline is None
        assert out.contributor_baseline_unavailable == "why not"
        assert out.contributor_risk == [] and out.permutation_test is None


def test_assess_an_empty_reference_has_no_baseline() -> None:
    out = run_assess(HIGH_COHORT, {"reference_n": 0, "reference_flagged": 0})
    assert out.contributor_baseline is None
    assert all("excludes_reference_rate" not in r for r in out.contributor_risk)
    assert out.contributor_risk                                    # the cohort is still scored


def test_assess_no_grouping_key_with_data_gives_no_baseline_and_says_why() -> None:
    samples = [SimpleNamespace(sample_id=f"s{i}", contributor=None, batch=None, source_meta={},
                               contributor_source=None) for i in range(20)]
    findings = [sfinding(f"s{i}") for i in range(5)]
    out = assess(findings, SimpleNamespace(samples=samples), {}, 0, None, CLEAN_REF)
    assert out.contributor_baseline is None
    assert out.contributor_risk == []
    assert out.contributor_baseline_unavailable is not None
    assert "no grouping key" in out.contributor_baseline_unavailable


def test_assess_a_reference_flag_count_is_never_mistaken_for_a_cohort_flag() -> None:
    """The cohort's own rows are identical whatever the reference says."""
    plain = run_assess(HIGH_COHORT).contributor_risk
    for ref in (CLEAN_REF, {"reference_n": 60, "reference_flagged": 60}):
        rows = run_assess(HIGH_COHORT, ref).contributor_risk
        assert [(r["group_value"], r["n_flagged"], r["posterior_mean"]) for r in rows] == [
            (r["group_value"], r["n_flagged"], r["posterior_mean"]) for r in plain]


# --------------------------------------------------------------------------------------
# ONE definition of "flagged"
# --------------------------------------------------------------------------------------
def _with(f: Finding, disposition: Disposition) -> Finding:
    f.disposition = disposition
    return f


@pytest.mark.parametrize("disposition", [Disposition.REVIEW, Disposition.QUARANTINE])
@pytest.mark.parametrize("availability", [OK, DEGRADED])
def test_flagged_a_sample_finding_that_ran_and_is_not_accept_counts(
        disposition: Disposition, availability: Availability) -> None:
    f = _with(sfinding("s1", availability=availability), disposition)
    assert flagged_sample_ids([f]) == {"s1"}


def test_flagged_an_accepted_finding_is_not_a_flag() -> None:
    assert flagged_sample_ids([_with(sfinding("s1"), Disposition.ACCEPT)]) == set()


@pytest.mark.parametrize("availability", [UNAVAILABLE, ERROR])
def test_flagged_a_finding_that_did_not_run_does_not_count(availability: Availability) -> None:
    f = _with(sfinding("s1", availability=availability), Disposition.QUARANTINE)
    assert flagged_sample_ids([f]) == set()


@pytest.mark.parametrize("target_type", ["model", "dataset", "batch", "contributor", "record"])
def test_flagged_only_sample_level_findings_count(target_type: str) -> None:
    f = _with(sfinding("s1", target_type=target_type), Disposition.QUARANTINE)
    assert flagged_sample_ids([f]) == set()


def test_flagged_the_model_finding_named_like_a_sample_still_does_not_count() -> None:
    model = _with(sfinding("s1", target_type="model"), Disposition.QUARANTINE)
    sample = _with(sfinding("s2"), Disposition.REVIEW)
    assert flagged_sample_ids([model, sample]) == {"s2"}


def test_flagged_a_sample_with_several_findings_is_one_flagged_sample() -> None:
    fs = [_with(sfinding("s1", detector=f"data.d{i}"), Disposition.REVIEW) for i in range(3)]
    fs.append(_with(sfinding("s1", detector="data.e"), Disposition.ACCEPT))
    assert flagged_sample_ids(fs) == {"s1"}


def test_flagged_nothing_in_nothing_out() -> None:
    assert flagged_sample_ids([]) == set()


# --------------------------------------------------------------------------------------
# reference_flag_counts: the same policy, the reference's own samples only
# --------------------------------------------------------------------------------------
def _ref_dataset(*ids: str) -> SimpleNamespace:
    return SimpleNamespace(samples=[SimpleNamespace(sample_id=i) for i in ids])


def _mixed_findings() -> list[Finding]:
    """s0 quarantine, s1 review (D5), s2 accept (low confidence), s3 raised, s4 not run,
    s5 quarantine but DEGRADED; plus a model-level finding that names s0."""
    return [
        sfinding("s0"),
        sfinding("s1", severity=Severity.MEDIUM, confidence=0.7),
        sfinding("s2", severity=Severity.LOW, confidence=0.3),
        sfinding("s3", availability=ERROR),
        sfinding("s4", availability=UNAVAILABLE),
        sfinding("s5", availability=DEGRADED),
        sfinding("s0", target_type="model"),
    ]


def test_reference_counts_route_the_findings_through_the_policy() -> None:
    ds = _ref_dataset(*(f"s{i}" for i in range(6)))
    assert reference_flag_counts(_mixed_findings(), ds, {}) == {
        "reference_n": 6, "reference_flagged": 3}                 # s0, s1, s5


def test_reference_counts_a_finding_the_policy_accepts_is_not_a_flag() -> None:
    """A detector-set disposition of REVIEW is overwritten by the engine: low confidence at
    low severity is D7 accept, so it is not a flag."""
    f = sfinding("s0", severity=Severity.LOW, confidence=0.3)
    f.disposition = Disposition.REVIEW
    assert reference_flag_counts([f], _ref_dataset("s0", "s1"), {}) == {
        "reference_n": 2, "reference_flagged": 0}


def test_reference_counts_use_the_profiles_own_disposition_table() -> None:
    accept_all = {"disposition": {"rules": [{"id": "D7", "disposition": "accept"}]}}
    ds = _ref_dataset("s0", "s1")
    assert reference_flag_counts([sfinding("s0"), sfinding("s1")], ds, accept_all) == {
        "reference_n": 2, "reference_flagged": 0}
    assert reference_flag_counts([sfinding("s0"), sfinding("s1")], ds, {}) == {
        "reference_n": 2, "reference_flagged": 2}


def test_reference_counts_apply_the_calibration_the_cohort_gets() -> None:
    calibration = SimpleNamespace(calibrators={"data.fake": lambda raw: 0.1})
    ds = _ref_dataset("s0")
    assert reference_flag_counts([sfinding("s0")], ds, {}, None)["reference_flagged"] == 1
    assert reference_flag_counts([sfinding("s0")], ds, {}, calibration)[  # type: ignore[arg-type]
        "reference_flagged"] == 0


def test_reference_counts_only_sample_ids_of_the_reference_dataset_count() -> None:
    findings = [sfinding("ref-1"), sfinding("elsewhere-9"), sfinding("elsewhere-10")]
    got = reference_flag_counts(findings, _ref_dataset("ref-1", "ref-2", "ref-3"), {})
    assert got == {"reference_n": 3, "reference_flagged": 1}


def test_reference_counts_no_findings_is_zero_flagged() -> None:
    assert reference_flag_counts([], _ref_dataset("a", "b"), {}) == {
        "reference_n": 2, "reference_flagged": 0}


def test_reference_flag_definition_is_the_cohorts_definition() -> None:
    """The same findings over the same samples flag the same number of them whether they are
    the cohort's or the reference's."""
    ids = [f"s{i}" for i in range(6)]
    ds = SimpleNamespace(samples=[
        SimpleNamespace(sample_id=i, contributor="A" if n < 3 else "B", batch=None,
                        source_meta={}, contributor_source=None) for n, i in enumerate(ids)])
    cohort_rows = assess(_mixed_findings(), ds, {}, 0).contributor_risk
    ref = reference_flag_counts(_mixed_findings(), ds, {})
    assert sum(r["n_flagged"] for r in cohort_rows) == ref["reference_flagged"] == 3


# --------------------------------------------------------------------------------------
# scan(): the reference findings never reach the report
# --------------------------------------------------------------------------------------
class FakeModel:
    model_id = "fake-001"
    fmt = "onnx"
    opset = 17

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                             ((Capability.MODEL_WEIGHTS, "probed: no weights readable"),))


class FakeDetector:
    """A data detector that flags exactly the samples whose id ends in "x"."""
    id = "data.fake"
    version = "0.0.1"
    requires: set = set()
    optional: set = set()
    attack_classes = {"label_flipping"}

    def detect(self, dataset, embeddings, model, ctx=None):
        return [Finding(detector_id=self.id, detector_version=self.version,
                        target_type="sample", target_ref=s.sample_id, severity=Severity.HIGH,
                        confidence=0.95, reason=f"sample {s.sample_id} matches the fake rule",
                        attack_class="label_flipping")
                for s in dataset.samples if s.sample_id.endswith("x")]


class RaisesOnReference(FakeDetector):
    def detect(self, dataset, embeddings, model, ctx=None):
        if any(s.sample_id.startswith("ref-") for s in dataset.samples):
            raise RuntimeError("boom")
        return super().detect(dataset, embeddings, model, ctx)


class NeedsGradients(FakeDetector):
    requires = {Capability.MODEL_GRADIENTS}


class FakeDataset:
    def __init__(self, prefix: str, spec: dict[str, tuple[int, int]]) -> None:
        self.samples = [
            Sample(f"{prefix}{c}-{i}" + ("x" if i < k else ""), "0" * 64,
                   Path(f"{prefix}{c}-{i}.png"), 8, 8, contributor=c,
                   source_meta={"format": "coco"})
            for c, (n, k) in spec.items() for i in range(n)]
        self.categories = [Category(0, "thing", 1)]

    def annotations(self):
        return []

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet()

    @property
    def n_flagged(self) -> int:
        return sum(s.sample_id.endswith("x") for s in self.samples)


REG = ({}, {"data.fake": FakeDetector})
COHORT = {"A": (10, 5), "B": (10, 0), "C": (10, 0)}
REF_CLEAN = {"R": (12, 0)}
REF_DIRTY = {"R": (12, 3)}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No backbone, and no armed determinism harness left over by another test."""
    monkeypatch.delenv("CVA_DETERMINISTIC", raising=False)
    real = orchestrator.build_embeddings

    def build(ctx: RunContext, prof: dict[str, Any]) -> tuple[Any, str | None]:
        ceiling = prof.get("embedding_max_images")
        if ctx.dataset is None or (ceiling is not None and len(ctx.dataset.samples) > ceiling):
            return real(ctx, prof)         # a gap, reported before any extractor is imported
        return None, None

    monkeypatch.setattr(orchestrator, "build_embeddings", build)


def do_scan(cohort_spec: dict[str, tuple[int, int]] | None = COHORT,
            ref_spec: dict[str, tuple[int, int]] | None = REF_CLEAN, *,
            registries: tuple[dict[str, Any], ...] = REG,
            profile: dict[str, Any] | None = None) -> tuple[ScanResult, FakeDataset, FakeDataset | None]:
    ds = FakeDataset("", cohort_spec or {})
    ref = FakeDataset("ref-", ref_spec) if ref_spec is not None else None
    ctx = RunContext(dataset=ds, reference_dataset=ref, seed=3, code_commit="cafe123",
                     profile=profile or {})
    return scan(FakeModel(), ctx, "deep", registries=registries), ds, ref


def validator() -> Any:
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def assert_valid(report: dict[str, Any]) -> None:
    errors = sorted(validator().iter_errors(report), key=lambda e: list(e.absolute_path))
    assert errors == [], [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errors]


def strip_volatile(report: dict[str, Any]) -> dict[str, Any]:
    assert VOLATILE_PATHS == ("$.scan_id", "$.created_at_utc", "$.findings[*].scan_id")
    out = copy.deepcopy(report)
    out.pop("scan_id", None)
    out.pop("created_at_utc", None)
    for f in out["findings"]:
        f.pop("scan_id", None)
    return out


def test_scan_the_reference_findings_never_enter_the_scans_findings() -> None:
    res, ds, ref = do_scan(ref_spec=REF_DIRTY)
    assert ref is not None and ref.n_flagged == 3
    assert len(res.findings) == ds.n_flagged == 5
    assert {f.target_ref for f in res.findings} == {
        s.sample_id for s in ds.samples if s.sample_id.endswith("x")}
    assert not any(f.target_ref.startswith("ref-") for f in res.findings)


def test_scan_findings_and_plan_are_the_same_with_and_without_a_reference() -> None:
    with_ref, _, _ = do_scan(ref_spec=REF_DIRTY)
    without, _, _ = do_scan(ref_spec=None)
    assert [f.finding_id for f in with_ref.findings] == [f.finding_id for f in without.findings]
    a = report_json.build(with_ref, "cmd")
    b = report_json.build(without, "cmd")
    assert a["plan"] == b["plan"]
    assert a["coverage"] == b["coverage"]
    assert len(a["findings"]) == len(b["findings"])


def test_scan_the_report_carries_a_baseline_that_validates_against_the_schema() -> None:
    res, _, _ = do_scan(ref_spec=REF_CLEAN)
    report = report_json.build(res, "cmd")
    base = report["contributor_baseline"]
    assert isinstance(base, dict) and set(base) == BASELINE_KEYS
    assert base["reference_n"] == 12 and base["reference_flagged"] == 0
    assert 0.0 < base["reference_rate_ci_high"] < 0.3
    assert "contributor_baseline_unavailable" not in report
    assert_valid(report)


def test_scan_the_baseline_counts_the_reference_flags_by_the_same_rule() -> None:
    res, _, ref = do_scan(ref_spec=REF_DIRTY)
    assert ref is not None
    base = res.contributor_baseline
    assert base is not None
    assert (base["reference_n"], base["reference_flagged"]) == (12, 3)
    assert base["reference_rate"] == pytest.approx(0.25)
    assert res.contributor_baseline_unavailable is None


def test_scan_a_dirty_cohort_against_a_clean_reference_exceeds_it() -> None:
    res, _, _ = do_scan({"A": (10, 3), "B": (10, 3), "C": (10, 3)}, REF_CLEAN)
    assert res.contributor_baseline is not None
    assert res.contributor_baseline["cohort_exceeds_reference"] is True
    assert_valid(report_json.build(res, "cmd"))


def test_scan_a_clean_cohort_against_a_dirty_reference_does_not_exceed_it() -> None:
    res, _, _ = do_scan({"A": (10, 0), "B": (10, 0), "C": (10, 0)}, REF_DIRTY)
    assert res.contributor_baseline is not None
    assert res.contributor_baseline["cohort_exceeds_reference"] is False


def test_scan_rows_in_the_report_carry_excludes_reference_rate_and_validate() -> None:
    res, _, _ = do_scan(ref_spec=REF_CLEAN)
    report = report_json.build(res, "cmd")
    assert report["contributor_risk"]
    assert all(type(r["excludes_reference_rate"]) is bool for r in report["contributor_risk"])
    assert_valid(report)


def test_scan_without_a_reference_the_key_is_null_and_no_reason_is_given() -> None:
    res, _, _ = do_scan(ref_spec=None)
    report = report_json.build(res, "cmd")
    assert "contributor_baseline" in report and report["contributor_baseline"] is None
    assert "contributor_baseline_unavailable" not in report
    assert all("excludes_reference_rate" not in r for r in report["contributor_risk"])
    assert_valid(report)


def test_scan_the_report_is_deterministic_across_two_runs_with_a_reference() -> None:
    a = report_json.build(do_scan(ref_spec=REF_DIRTY)[0], "cmd")
    b = report_json.build(do_scan(ref_spec=REF_DIRTY)[0], "cmd")
    assert a["contributor_baseline"] is not None
    assert (json.dumps(strip_volatile(a), sort_keys=True, default=str)
            == json.dumps(strip_volatile(b), sort_keys=True, default=str))


def test_scan_a_different_reference_changes_the_report() -> None:
    a = report_json.build(do_scan(ref_spec=REF_CLEAN)[0], "cmd")
    b = report_json.build(do_scan(ref_spec=REF_DIRTY)[0], "cmd")
    assert strip_volatile(a) != strip_volatile(b)
    assert a["contributor_baseline"]["reference_flagged"] == 0
    assert b["contributor_baseline"]["reference_flagged"] == 3


def test_scan_the_baseline_validator_is_not_vacuous() -> None:
    report = report_json.build(do_scan(ref_spec=REF_CLEAN)[0], "cmd")
    report["contributor_baseline"]["surprise"] = 1
    assert list(validator().iter_errors(report))
    report = report_json.build(do_scan(ref_spec=REF_CLEAN)[0], "cmd")
    del report["contributor_baseline"]["cohort_rate"]
    assert list(validator().iter_errors(report))


# --- a reference that was supplied but cannot be used ---
def test_scan_a_reference_over_the_embedding_ceiling_is_unavailable_with_a_reason() -> None:
    res, _, _ = do_scan({"A": (1, 0)}, {"R": (2, 0)}, profile={"embedding_max_images": 1})
    assert res.contributor_baseline is None
    assert res.contributor_baseline_unavailable is not None
    assert "not embedded" in res.contributor_baseline_unavailable
    assert "at most 1 images" in res.contributor_baseline_unavailable
    report = report_json.build(res, "cmd")
    assert report["contributor_baseline"] is None
    assert report["contributor_baseline_unavailable"] == res.contributor_baseline_unavailable
    assert_valid(report)


def test_scan_a_scanned_dataset_with_no_embedding_index_makes_the_reference_unusable() -> None:
    res, _, _ = do_scan(profile={"embedding_max_images": 5})     # the cohort has 30 images
    assert res.contributor_baseline is None
    assert "no embedding index" in (res.contributor_baseline_unavailable or "")
    assert_valid(report_json.build(res, "cmd"))


def test_scan_a_reference_with_no_samples_is_unavailable() -> None:
    res, _, _ = do_scan(ref_spec={})
    assert res.contributor_baseline is None
    assert "no samples" in (res.contributor_baseline_unavailable or "")
    assert_valid(report_json.build(res, "cmd"))


def test_scan_a_detector_that_raises_on_the_reference_makes_it_unavailable() -> None:
    res, _, _ = do_scan(registries=({}, {"data.fake": RaisesOnReference}))
    assert res.contributor_baseline is None
    assert "data.fake raised" in (res.contributor_baseline_unavailable or "")
    assert not any(f.target_ref.startswith("ref-") for f in res.findings)
    assert not any(f.availability == ERROR for f in res.findings), "the reference error is not reported"
    assert_valid(report_json.build(res, "cmd"))


def test_scan_no_runnable_data_check_makes_the_reference_unavailable() -> None:
    res, _, _ = do_scan(registries=({}, {"data.fake": NeedsGradients}))
    assert res.contributor_baseline is None
    assert "no data.* check was runnable" in (res.contributor_baseline_unavailable or "")
    assert_valid(report_json.build(res, "cmd"))


@pytest.mark.parametrize("profile,ref_spec,cohort_spec", [
    ({"embedding_max_images": 1}, {"R": (2, 0)}, {"A": (1, 0)}),
    ({}, {}, COHORT),
])
def test_scan_the_reason_is_a_non_empty_string_in_the_report_whenever_a_reference_was_unusable(
        profile: dict[str, Any], ref_spec: dict[str, tuple[int, int]],
        cohort_spec: dict[str, tuple[int, int]]) -> None:
    report = report_json.build(do_scan(cohort_spec, ref_spec, profile=profile)[0], "cmd")
    assert isinstance(report["contributor_baseline_unavailable"], str)
    assert report["contributor_baseline_unavailable"]


def test_scan_the_reason_is_absent_when_a_reference_was_used_and_when_none_was_supplied() -> None:
    used = report_json.build(do_scan(ref_spec=REF_CLEAN)[0], "cmd")
    none = report_json.build(do_scan(ref_spec=None)[0], "cmd")
    assert "contributor_baseline_unavailable" not in used
    assert "contributor_baseline_unavailable" not in none


# --------------------------------------------------------------------------------------
# HTML: the contributor section prints one line per case
# --------------------------------------------------------------------------------------
ROWS = [{"group_key": "contributor", "group_value": "team_b", "n_samples": 100, "n_flagged": 30,
         "posterior_mean": 0.3, "ci_low": 0.22, "ci_high": 0.4, "excludes_cohort_rate": False,
         "disposition": "accept"}]
PERM = {"statistic": 1.5, "p_value": 0.4, "n_permutations": 2000,
        "conclusion": "No evidence that flags cluster by contributor (p = 0.4000)."}
BASE_OK = {"reference_n": 60, "reference_flagged": 0, "reference_rate": 0.0,
           "reference_rate_ci_high": 0.04, "cohort_rate": 0.02, "cohort_exceeds_reference": False}
BASE_EXCEEDS = {**BASE_OK, "cohort_rate": 0.3, "cohort_exceeds_reference": True}


def html_result(**extra: Any) -> ScanResult:
    prof = resolve_profile("deep")
    extra.setdefault("contributor_risk", ROWS)
    extra.setdefault("permutation_test", PERM)
    return ScanResult(
        scan_id="s-2026-09-19-0001", model_id="m-1", model_fmt="onnx",
        capabilities=CapabilitySet(),
        plan=[PlanRow("data.fake", Resolution(OK, "r"), {"label_flipping"})],
        findings=[], timings={}, verdict="ACCEPT", profile_name="deep",
        profile_hash=profile_hash_of(prof), code_commit="cafe123", access_assumptions="",
        created_at_utc="2026-09-19T00:00:00+00:00", profile=prof, seed=7, target={}, **extra)


def contributor_section(tmp_path: Path, res: ScanResult) -> str:
    out = render(res, tmp_path / "report.html", evidence_root=tmp_path / "evidence")
    html = out.read_text(encoding="utf-8")
    return html.split("<h2>Contributor risk</h2>")[1].split("<h2>")[0]


def test_html_no_reference_supplied_says_so(tmp_path: Path) -> None:
    section = contributor_section(tmp_path, html_result())
    assert "No reference dataset supplied" in section
    assert "not used" not in section
    assert "cohort prior unreliable" not in section.lower()


def test_html_a_reference_supplied_but_unusable_says_why(tmp_path: Path) -> None:
    res = html_result(contributor_baseline_unavailable="the reference dataset has no samples")
    section = contributor_section(tmp_path, res)
    assert "A reference dataset was supplied but not used" in section
    assert "the reference dataset has no samples" in section
    assert "No reference dataset supplied" not in section
    assert "cohort prior unreliable" not in section.lower()


def test_html_a_usable_reference_prints_its_rate_and_the_cohort_rate(tmp_path: Path) -> None:
    section = contributor_section(tmp_path, html_result(contributor_baseline=BASE_OK))
    assert "Reference dataset: 0 of 60 images flagged (0.0%)" in section
    assert "2.0%" in section
    assert "No reference dataset supplied" not in section
    assert "not used" not in section


def test_html_cohort_prior_unreliable_only_when_the_cohort_exceeds_the_reference(
        tmp_path: Path) -> None:
    quiet = contributor_section(tmp_path, html_result(contributor_baseline=BASE_OK))
    loud = contributor_section(tmp_path, html_result(contributor_baseline=BASE_EXCEEDS))
    assert "cohort prior unreliable" not in quiet.lower()
    assert "cohort prior unreliable" in loud.lower()
    assert "30.0%" in loud


def test_html_the_usable_baseline_wins_over_a_stale_reason(tmp_path: Path) -> None:
    res = html_result(contributor_baseline=BASE_OK, contributor_baseline_unavailable="stale")
    section = contributor_section(tmp_path, res)
    assert "Reference dataset: 0 of 60" in section
    assert "not used" not in section


@pytest.mark.parametrize("extra", [
    {},
    {"contributor_baseline_unavailable": "reference over the embedding ceiling"},
    {"contributor_baseline": BASE_OK},
    {"contributor_baseline": BASE_EXCEEDS},
])
def test_html_never_references_a_network_resource(tmp_path: Path, extra: dict[str, Any]) -> None:
    out = render(html_result(**extra), tmp_path / "r.html", evidence_root=tmp_path / "ev")
    lowered = out.read_text(encoding="utf-8").lower()
    assert "http://" not in lowered and "https://" not in lowered
    assert "<script" not in lowered and "<link" not in lowered


def test_html_the_reason_is_escaped(tmp_path: Path) -> None:
    res = html_result(contributor_baseline_unavailable="<b>bold</b> & more")
    section = contributor_section(tmp_path, res)
    assert "<b>bold</b>" not in section
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; more" in section


def test_html_without_contributor_rows_the_baseline_line_is_not_printed(tmp_path: Path) -> None:
    section = contributor_section(tmp_path, html_result(contributor_risk=[],
                                                        contributor_baseline=BASE_OK))
    assert "Not computed" in section
    assert "Reference dataset:" not in section


def test_html_a_real_scan_with_a_reference_renders_the_baseline_line(tmp_path: Path) -> None:
    res, _, _ = do_scan(ref_spec=REF_DIRTY)
    section = contributor_section(tmp_path, res)
    assert "Reference dataset: 3 of 12 images flagged (25.0%)" in section


def test_html_a_real_scan_with_an_unusable_reference_renders_the_reason(tmp_path: Path) -> None:
    res, _, _ = do_scan(ref_spec={})
    section = contributor_section(tmp_path, res)
    assert "A reference dataset was supplied but not used" in section
    assert "the reference dataset has no samples" in section


def test_html_a_real_scan_without_a_reference_renders_the_none_supplied_line(
        tmp_path: Path) -> None:
    res, _, _ = do_scan(ref_spec=None)
    assert "No reference dataset supplied" in contributor_section(tmp_path, res)
