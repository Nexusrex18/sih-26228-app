"""report.json — the machine-readable half of the report (plan §5.3, §7.9, V9, V10, V17).

The contract is `schemas/report.schema.json`, so most of these tests validate a BUILT report
against it (the serialised form, not the dict, wherever a file is written). The rest pin the
behaviours the schema cannot express: an absent fact is an absent key, the volatile set is
declared, coverage counts attack classes only, and a check that raised is never "assessed".

Built in-process from `ScanResult` objects and `scan(..., registries=...)` with tiny fakes;
no model file is read and no backbone is loaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cva.core.capability import (
    Availability,
    Capability,
    CapabilitySet,
    Resolution,
    budget_excluded,
)
from cva.core.ledger import NullAuditLedger
from cva.core.orchestrator import PlanRow, ScanResult, profile_hash_of, resolve_profile, scan
from cva.core.runcontext import RunContext
from cva.core.scanid import VOLATILE_PATHS
from cva.core.taxonomy import CLAIMABLE
from cva.core.types import (
    Category,
    Disposition,
    Evidence,
    ExclusionReason,
    Finding,
    Sample,
    Severity,
)
from cva.report import coverage, report_json
from cva.report.coverage import STANDING_LIMITATIONS

SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "report.schema.json"
EMPTY: tuple[dict, ...] = ({}, {})
HEX64 = "ab" * 32


def _validator():
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def _assert_valid(report: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(report), key=lambda e: list(e.absolute_path))
    assert errors == [], [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errors]


@pytest.fixture(autouse=True)
def _unarmed(monkeypatch):
    """Whether the determinism harness is armed is process state; other tests arm it."""
    monkeypatch.delenv("CVA_DETERMINISTIC", raising=False)


# --- fakes -------------------------------------------------------------------

class FakeModel:
    model_id = "fake-001"
    fmt = "onnx"
    opset = 17

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                             ((Capability.MODEL_WEIGHTS, "probed: no weights readable"),))


class HashedModel(FakeModel):
    def weight_digest(self) -> str:
        return HEX64


class FrozenModel(FakeModel):
    fmt = "torchscript"
    opset = None

    def weight_digest(self) -> str:
        return "unavailable:frozen"


class FakeDataset:
    def __init__(self, n: int = 3) -> None:
        self.samples = [Sample(f"s{i}", "0" * 64, Path(f"s{i}.png"), 8, 8,
                               source_meta={"format": "coco"}) for i in range(n)]
        self.categories = [Category(0, "thing", 1), Category(1, "other", 2)]

    def annotations(self):
        return []

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet()


class Attack:
    id = "stub.attack"
    version = "0.0.1"
    requires: set = set()
    optional: set = set()
    attack_classes = {"weight_anomaly"}

    def check(self, model, ctx):
        return [Finding(detector_id=self.id, detector_version=self.version,
                        target_type="model", target_ref=model.model_id,
                        severity=Severity.INFO, confidence=0.5, reason="ran",
                        attack_class="weight_anomaly")]


class Raising(Attack):
    id = "stub.raises"
    # An ATTACK class on purpose: `tool.error` is operational and would never reach
    # `not_assessed` whatever the code did.
    attack_classes = {"model_substitution"}

    def check(self, model, ctx):
        raise RuntimeError("deliberate")


class LedgerWith:
    def __init__(self, cap: Capability) -> None:
        self.cap = cap

    def append(self, record) -> str:
        return "seq-1"

    def records(self):
        return iter(())

    def capabilities(self) -> set[Capability]:
        return {self.cap}


def _result(plan=(), findings=(), *, caps: CapabilitySet | None = None,
            verdict: str = "REVIEW", target: dict[str, Any] | None = None,
            **extra: Any) -> ScanResult:
    prof = resolve_profile("deep")
    return ScanResult(
        scan_id="s-2026-09-19-0001", model_id="m-1", model_fmt="onnx",
        capabilities=caps or CapabilitySet(), plan=list(plan), findings=list(findings),
        timings={}, verdict=verdict, profile_name="deep", profile_hash=profile_hash_of(prof),
        code_commit="cafe123", access_assumptions="", created_at_utc="2026-09-19T00:00:00+00:00",
        profile=prof, seed=7, target=target or {}, **extra)


def _finding(availability: Availability, **kw: Any) -> Finding:
    base: dict[str, Any] = dict(
        detector_id=f"model.{availability.value.lower()}", detector_version="1.0.0",
        target_type="model", target_ref="m-1", severity=Severity.MEDIUM, confidence=0.7,
        reason=f"a {availability.value} finding that names its evidence",
        attack_class="model_substitution", scan_id="s-2026-09-19-0001",
        produced_by="cafe123/profile:abcdef012345", disposition=Disposition.REVIEW,
        disposition_rule="D3", availability=availability)
    base.update(kw)
    return Finding(**base)


# --- schema validity -----------------------------------------------------------

def test_a_zero_detector_scan_validates_against_the_report_schema(tmp_path):
    res = scan(FakeModel(), RunContext(code_commit="cafe123"), "deep", registries=EMPTY)
    assert res.plan == [] and res.findings == []
    _assert_valid(report_json.build(res, "python -m cva.cli scan"))
    # ...and the file form, which is what Module E actually reads.
    path = report_json.write(res, tmp_path / "report.json", "cmd")
    _assert_valid(json.loads(path.read_text()))


def test_a_scan_with_no_model_at_all_validates_and_has_no_target_block():
    res = scan(None, RunContext(), "baseline", registries=EMPTY)
    rep = report_json.build(res, "cmd")
    _assert_valid(rep)
    assert "target" not in rep, "no model and no dataset: there is no fact to record"


def test_a_finding_of_each_availability_state_validates(tmp_path):
    caps = CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                         ((Capability.MODEL_GRADIENTS, "onnxruntime-training not bundled"),))
    plan = [
        PlanRow("model.a_ok", Resolution(Availability.OK, "all present"),
                {"model_substitution"}),
        PlanRow("model.b_degraded", caps.resolve(set(), {Capability.MODEL_GRADIENTS}),
                {"weight_anomaly"}),
        PlanRow("model.c_capability", caps.resolve({Capability.MODEL_GRADIENTS}, set()),
                {"backdoor_trigger"}),
        PlanRow("model.d_budget", budget_excluded("triage", "~40 min"),
                {"activation_anomaly"}),
        PlanRow("model.e_error", Resolution(Availability.ERROR, "raised"), {"tool.error"}),
    ]
    findings = [
        _finding(Availability.OK, evidence=[
            Evidence("hash", "digest", path="ab" * 32 + ".json"),
            Evidence("json", "inline only", data={"k": [1, 2, 3]})]),
        _finding(Availability.DEGRADED, limitations=["Ran DEGRADED: no gradients"]),
        _finding(Availability.UNAVAILABLE, severity=Severity.INFO, confidence=0.0,
                 exclusion_reason=ExclusionReason.CAPABILITY,
                 evidence=[Evidence("json", "missing capabilities", data=["MODEL_GRADIENTS"])]),
        _finding(Availability.UNAVAILABLE, detector_id="model.budgeted",
                 severity=Severity.INFO, confidence=0.0,
                 exclusion_reason=ExclusionReason.BUDGET),
        _finding(Availability.ERROR, attack_class="tool.error", confidence=0.0),
    ]
    assert {f.availability for f in findings} == set(Availability)
    res = _result(plan, findings, caps=caps)

    path = report_json.write(res, tmp_path / "report.json", "cmd")
    rep = json.loads(path.read_text())
    _assert_valid(rep)
    assert [f["availability"] for f in rep["findings"]] == [f.availability.value
                                                            for f in findings]
    assert {r["state"] for r in rep["plan"]} == {"OK", "DEGRADED", "UNAVAILABLE", "ERROR"}
    by_id = {r["check_id"]: r for r in rep["plan"]}
    assert by_id["model.d_budget"]["exclusion_reason"] == "budget"
    assert by_id["model.d_budget"]["estimated_cost"] == "~40 min"
    assert by_id["model.c_capability"]["exclusion_reason"] == "capability"
    assert by_id["model.c_capability"]["missing"] == ["MODEL_GRADIENTS"]


def test_evidence_records_the_hash_and_not_a_second_copy_of_the_payload():
    stored = Evidence("json", "stored", path="cd" * 32 + ".json", data={"big": "payload"})
    inline = Evidence("json", "inline", data={"small": 1})
    rep = report_json.build(_result(findings=[_finding(Availability.OK,
                                                       evidence=[stored, inline])]))
    a, b = rep["findings"][0]["evidence"]
    assert a["path"] == "cd" * 32 + ".json" and "data" not in a
    assert b["path"] is None and b["data"] == {"small": 1}
    _assert_valid(rep)


def test_contributor_risk_permutation_test_and_calibration_validate():
    rep = report_json.build(_result(
        contributor_risk=[{"group_key": "contributor", "group_value": "B", "n_samples": 214,
                           "n_flagged": 190, "posterior_mean": 0.88, "ci_low": 0.82,
                           "ci_high": 0.92, "disposition": "quarantine"}],
        permutation_test={"statistic": 12.5, "p_value": 0.001, "n_permutations": 2000,
                          "conclusion": "flags are not randomly distributed"},
        calibration={"method": "isotonic", "brier": 0.12,
                     "reliability_bins": [{"p_mean": 0.2, "empirical": 0.1, "n": 40}],
                     "excluded_detectors": ["prov.chain"]}))
    _assert_valid(rep)
    assert rep["contributor_risk"][0]["group_value"] == "B"
    assert rep["calibration"]["reliability_bins"][0]["n"] == 40


def test_an_empty_or_uncomputed_section_is_null_or_empty_never_absent_evidence():
    rep = report_json.build(scan(FakeModel(), RunContext(), "deep", registries=EMPTY))
    assert rep["contributor_risk"] == []
    assert rep["permutation_test"] is None
    assert rep["calibration"] is None
    assert rep["drift_summary"] is None


# --- verdict --------------------------------------------------------------------

def test_a_dry_run_verdict_is_omitted_never_written(tmp_path):
    res = scan(FakeModel(), RunContext(), "deep", dry_run=True,
               registries=({"stub.attack": Attack}, {}))
    assert res.verdict == "DRY-RUN"
    rep = report_json.build(res, "cmd")
    assert "verdict" not in rep
    _assert_valid(rep)
    path = report_json.write(res, tmp_path / "report.json", "cmd")
    assert "DRY-RUN" not in path.read_text()


@pytest.mark.parametrize("verdict", ["ACCEPT", "REVIEW", "QUARANTINE"])
def test_a_real_verdict_is_written(verdict):
    rep = report_json.build(_result(verdict=verdict))
    assert rep["verdict"] == verdict
    _assert_valid(rep)


# --- target ----------------------------------------------------------------------

def test_target_omits_absent_members_and_never_writes_an_empty_string():
    res = scan(FakeModel(), RunContext(), "deep", registries=EMPTY)   # no digest, no dataset
    target = report_json.build(res)["target"]
    assert target == {"model_id": "fake-001", "model_format": "onnx", "model_opset": 17}
    assert "model_sha256" not in target
    assert all(v not in ("", None) for v in target.values())


def test_target_records_dataset_format_but_never_dataset_path():
    res = scan(HashedModel(), RunContext(dataset=FakeDataset(3)), "deep", registries=EMPTY)
    rep = report_json.build(res)
    target = rep["target"]
    assert target["dataset_format"] == "coco"
    assert target["n_samples"] == 3 and target["n_categories"] == 2
    assert target["model_sha256"] == HEX64
    assert "dataset_path" not in target, "a temp-dir path would make V10's diff fail"
    assert all(v not in ("", None) for v in target.values())
    _assert_valid(rep)


def test_a_frozen_torchscript_digest_is_omitted_not_emitted():
    target = report_json.build(scan(FrozenModel(), RunContext(), "deep",
                                    registries=EMPTY))["target"]
    assert "model_sha256" not in target
    assert "model_opset" not in target
    assert target["model_format"] == "torchscript"


def test_an_onnx_opset_dict_records_the_default_domain():
    class DictOpset(FakeModel):
        opset = {"ai.onnx": 13, "ai.onnx.ml": 3}

    target = report_json.build(scan(DictOpset(), RunContext(), "deep",
                                    registries=EMPTY))["target"]
    assert target["model_opset"] == 13


# --- reproduction ------------------------------------------------------------------

def test_the_reproduction_command_is_recorded_verbatim_and_never_carries_out():
    rep = report_json.build(_result(), "python -m cva.cli scan m.onnx --profile deep --seed 7")
    assert rep["reproduction"]["command"] == "python -m cva.cli scan m.onnx --profile deep --seed 7"
    unrecorded = report_json.build(_result())["reproduction"]["command"]
    assert unrecorded and "--out" not in unrecorded


def test_the_volatile_paths_are_declared_from_the_one_source():
    rep = report_json.build(_result(), "cmd")
    assert rep["reproduction"]["volatile_paths"] == list(VOLATILE_PATHS)
    assert rep["reproduction"]["volatile_paths"] is not VOLATILE_PATHS   # a copy, JSON-safe


def test_seeds_are_only_the_scan_seed_when_the_harness_is_not_armed():
    rep = report_json.build(_result())
    assert rep["reproduction"]["seeds"] == {"scan": 7}
    notes = " ".join(rep["reproduction"]["determinism_notes"]).lower()
    assert "not pinned" in notes or "not been pinned" in notes or "only the scan seed" in notes


def test_seeds_gain_python_numpy_and_torch_when_the_harness_is_armed(monkeypatch):
    monkeypatch.setenv("CVA_DETERMINISTIC", "1")
    seeds = report_json.build(_result())["reproduction"]["seeds"]
    assert seeds == {"scan": 7, "python": 7, "numpy": 7, "torch": 7}
    # numpy's seed is reduced into its 32-bit range, and the report records what was applied.
    big = _result()
    big.seed = 2**32 + 5
    assert report_json.build(big)["reproduction"]["seeds"]["numpy"] == 5


def test_an_armed_and_an_unarmed_report_say_different_things_about_determinism(monkeypatch):
    unarmed = report_json.build(_result())["reproduction"]["determinism_notes"]
    monkeypatch.setenv("CVA_DETERMINISTIC", "1")
    armed = report_json.build(_result())["reproduction"]["determinism_notes"]
    assert armed != unarmed


def test_a_profile_that_pins_the_clock_says_so():
    pinned = _result()
    pinned.profile = resolve_profile("selftest")
    unpinned = _result()
    assert (report_json.build(pinned)["reproduction"]["determinism_notes"]
            != report_json.build(unpinned)["reproduction"]["determinism_notes"])


# --- provenance summary ----------------------------------------------------------------

def test_without_a_ledger_or_key_provenance_is_zero_zero_and_not_sealed():
    res = scan(FakeModel(), RunContext(), "deep", registries=EMPTY)
    assert isinstance(res.capabilities, CapabilitySet)
    assert Capability.SIGNING_KEY not in res.capabilities
    assert Capability.INFERENCE_LEDGER not in res.capabilities
    summ = report_json.build(res)["provenance_summary"]
    assert isinstance(summ, dict)
    assert summ["records_verified"] == 0 and summ["anchors_checked"] == 0
    assert summ["ledger_state"] == "not_sealed"


def test_with_a_signing_key_ledger_state_is_left_out_because_it_is_not_knowable_yet():
    ctx = RunContext(audit_ledger=LedgerWith(Capability.SIGNING_KEY))
    rep = report_json.build(scan(FakeModel(), ctx, "deep", registries=EMPTY))
    summ = rep["provenance_summary"]
    assert "ledger_state" not in summ, "the record is appended AFTER report.json is hashed"
    assert summ["records_verified"] == 0 and summ["anchors_checked"] == 0
    _assert_valid(rep)


def test_with_an_inference_ledger_the_counts_are_unknown_not_zero():
    ctx = RunContext(inference_ledger=LedgerWith(Capability.INFERENCE_LEDGER))
    summ = report_json.build(scan(FakeModel(), ctx, "deep", registries=EMPTY))[
        "provenance_summary"]
    assert "records_verified" not in summ and "anchors_checked" not in summ
    assert summ["ledger_state"] == "not_sealed"


def test_with_a_ledger_and_a_key_nothing_is_known_and_the_summary_is_null():
    class Both:
        def append(self, record) -> str:
            return "seq-1"

        def records(self):
            return iter(())

        def capabilities(self) -> set[Capability]:
            return {Capability.SIGNING_KEY, Capability.INFERENCE_LEDGER}

    ctx = RunContext(audit_ledger=Both(), inference_ledger=Both())
    rep = report_json.build(scan(FakeModel(), ctx, "deep", registries=EMPTY))
    assert rep["provenance_summary"] is None
    _assert_valid(rep)


def test_the_null_audit_ledger_reports_no_signing_key():
    assert NullAuditLedger().capabilities() == set()


# --- coverage (V17) ------------------------------------------------------------------------

def _row(cid: str, state: Availability, classes: set[str]) -> PlanRow:
    return PlanRow(cid, Resolution(state, "r"), classes)


def test_coverage_counts_attack_classes_only_and_partitions_the_taxonomy():
    plan = [
        _row("model.a", Availability.OK, {"model_substitution", "benign_conversion"}),
        _row("model.b", Availability.UNAVAILABLE, {"weight_anomaly"}),
        _row("model.c", Availability.OK, {"tool.error", "not_assessed"}),
    ]
    cov = report_json.build(_result(plan))["coverage"]
    assert cov["counts_only_kind"] == "attack"

    operational = {"benign_conversion", "tool.error", "not_assessed"}
    assert set(cov["operational_reports"]) == operational
    for cls in operational:
        assert cls not in cov["assessed"] and cls not in cov["not_assessed"], cls
        assert cls not in cov["never_covered"], cls

    assert cov["assessed"] == {"model_substitution": ["model.a"]}
    assert cov["not_assessed"] == {"weight_anomaly": ["model.b"]}
    # Every claimable class lands in exactly one bucket: a class that fell out of all three
    # is a coverage row that silently vanished.
    buckets = [set(cov["assessed"]), set(cov["not_assessed"]), set(cov["never_covered"])]
    assert set().union(*buckets) == CLAIMABLE
    assert sum(len(b) for b in buckets) == len(CLAIMABLE)


def test_coverage_lists_a_class_under_both_when_two_checks_disagree_about_it():
    plan = [_row("model.a", Availability.OK, {"weight_anomaly"}),
            _row("model.b", Availability.UNAVAILABLE, {"weight_anomaly"})]
    cov = report_json.build(_result(plan))["coverage"]
    assert cov["assessed"]["weight_anomaly"] == ["model.a"]
    assert cov["not_assessed"]["weight_anomaly"] == ["model.b"]
    assert "weight_anomaly" not in cov["never_covered"]


def test_a_check_that_was_planned_then_raised_is_not_assessed():
    """The tool failing is not the tool having looked."""
    res = scan(FakeModel(), RunContext(), "deep",
               registries=({"stub.raises": Raising, "stub.attack": Attack}, {}))
    by_id = {r.check_id: r for r in res.plan}
    assert by_id["stub.raises"].resolution.runnable, "premise: it WAS planned to run"
    assert any(f.detector_id == "stub.raises" and f.availability is Availability.ERROR
               for f in res.findings), "premise: and it raised"

    cov = report_json.build(res)["coverage"]
    assert "model_substitution" not in cov["assessed"]
    assert cov["not_assessed"]["model_substitution"] == ["stub.raises"]
    # The sibling that ran is unaffected.
    assert cov["assessed"]["weight_anomaly"] == ["stub.attack"]
    assert "model_substitution" not in cov["never_covered"]


# --- standing limitations ----------------------------------------------------------------------

def _limits(target: dict[str, Any]) -> list[str]:
    return report_json.standing_limitations(target)


def test_the_base_standing_limitations_are_always_present_plus_the_sandbox_one():
    base = _limits({})
    assert base[:len(STANDING_LIMITATIONS)] == STANDING_LIMITATIONS
    assert len(base) == len(STANDING_LIMITATIONS) + 1
    assert len(set(base)) == len(base)


def test_a_model_with_a_digest_adds_no_weight_digest_limitation_whatever_its_format():
    base = _limits({})
    for fmt in ("onnx", "torchscript", "callable", "http", "subprocess"):
        assert _limits({"model_format": fmt, "model_sha256": HEX64}) == base, fmt


def test_a_dataset_only_scan_has_no_model_format_limitation():
    assert _limits({"n_samples": 5}) == _limits({})


def test_the_missing_digest_limitation_depends_on_the_model_format():
    base = _limits({})
    torchscript = _limits({"model_format": "torchscript"})
    onnx = _limits({"model_format": "onnx"})
    query_only = {fmt: _limits({"model_format": fmt})
                  for fmt in ("callable", "http", "subprocess")}

    for extra in (torchscript, onnx, *query_only.values()):
        assert extra[:len(base)] == base
        assert len(extra) == len(base) + 1

    ts_text, onnx_text = torchscript[-1], onnx[-1]
    qo_texts = {fmt: lims[-1] for fmt, lims in query_only.items()}
    assert len(set(qo_texts.values())) == 1, "callable/http/subprocess share one wording"
    qo_text = next(iter(qo_texts.values()))
    assert len({ts_text, onnx_text, qo_text}) == 3, "three situations, three distinct texts"
    assert "torchscript" in ts_text.lower()
    assert any(w in qo_text.lower() for w in ("query-only", "black-box", "blackbox"))


def test_the_limitations_reach_report_json_from_the_target():
    res = scan(FakeModel(), RunContext(), "deep", registries=EMPTY)      # onnx, no digest
    rep = report_json.build(res)
    assert rep["coverage"]["standing_limitations"] == _limits(rep["target"])
    assert len(rep["coverage"]["standing_limitations"]) == len(STANDING_LIMITATIONS) + 2
    hashed = report_json.build(scan(HashedModel(), RunContext(), "deep", registries=EMPTY))
    assert len(hashed["coverage"]["standing_limitations"]) == len(STANDING_LIMITATIONS) + 1


def _md_limitations(md: str) -> list[str]:
    head = "## Standing limitations"
    assert head in md
    tail = md.split(head, 1)[1]
    return [ln[2:] for ln in tail.splitlines() if ln.startswith("- ")]


@pytest.mark.parametrize("model_cls", [FakeModel, HashedModel, FrozenModel],
                         ids=["no-digest", "digest", "torchscript"])
def test_coverage_md_and_report_json_state_the_same_standing_limitations(tmp_path, model_cls):
    res = scan(model_cls(), RunContext(), "deep", registries=({"stub.attack": Attack}, {}))
    md_path = coverage.write(res, tmp_path / "coverage.md")
    rep = report_json.build(res)
    assert _md_limitations(md_path.read_text()) == rep["coverage"]["standing_limitations"]


def test_coverage_md_lists_the_same_classes_as_report_json(tmp_path):
    res = scan(FakeModel(), RunContext(), "deep",
               registries=({"stub.raises": Raising, "stub.attack": Attack}, {}))
    md = coverage.render_markdown(res)
    cov = report_json.build(res)["coverage"]
    assessed_md, _, rest = md.partition("## Attack classes NOT assessed in this scan")
    not_assessed_md = rest.split("## ")[0]
    for cls in cov["assessed"]:
        assert f"`{cls}`" in assessed_md
    for cls in cov["not_assessed"]:
        assert f"`{cls}`" in not_assessed_md
