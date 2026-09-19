"""Gates B5 / B6 — the orchestrator, the profile tables and the reproducibility contract.

Plan refs: §5.3 (volatile set, V9, V10), §5.5 (budget is a second axis, not a fifth state),
§5.6 (profiles), §7.9 (sealing), §8 B5/B6.

Everything here is in-process and offline. `scan()` is driven with `registries=` overrides and
tiny fakes, so nothing loads DINOv2 or an ONNX session. The two registry imports below are
deliberate and session-global: (b) and (e) assert against the REAL registries, and every other
test in this file passes `registries=` explicitly, so the import cannot leak into them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import cva.detectors.data.registry  # noqa: F401 — puts the data.* detectors in the registry
import cva.detectors.model.registry  # noqa: F401 — puts the model.* checks in the registry
from cva.core import orchestrator
from cva.core.capability import Availability, Capability, CapabilitySet
from cva.core.ledger import NullAuditLedger
from cva.core.orchestrator import (
    InvalidProfile,
    UnknownCheckId,
    UnknownProfile,
    _jsonable,
    _verdict,
    build_plan,
    default_registries,
    profile_hash_of,
    resolve_profile,
    scan,
    seal_report,
    validate_profile_ids,
    validate_profile_schema,
)
from cva.core.profile import POLICIES, PROFILES, TIERS
from cva.core.registry import DETECTOR_REGISTRY, REGISTRY
from cva.core.runcontext import RunContext
from cva.core.scanid import SCAN_ID_RE, SELFTEST_EPOCH, VOLATILE_PATHS, selftest_scan_id
from cva.core.types import Category, Disposition, Finding, Sample, Severity
from cva.report import report_json
from cva.risk.engine import RiskOutcome

POLICY_NAMES = ("baseline", "strict", "blackbox", "selftest")
TIER_NAMES = ("triage", "standard", "deep", "forensic")
SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"
EMPTY: tuple[dict, ...] = ({}, {})
FULL_CAPS = CapabilitySet(frozenset(Capability))


# --- tiny fakes ------------------------------------------------------------

class FakeModel:
    model_id = "fake-001"
    fmt = "onnx"
    opset = 17

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                             ((Capability.MODEL_WEIGHTS, "probed: no weights readable"),))


class NoRequirements:
    """A model check that declares NO required capability (like `model.weight_digest`)."""
    id = "model.norequire"
    version = "0.0.1"
    requires: set = set()
    optional: set = set()
    attack_classes = {"model_substitution"}

    def check(self, model, ctx):
        return []


class OkCheck(NoRequirements):
    id = "stub.ok"

    def check(self, model, ctx):
        return [Finding(detector_id=self.id, detector_version=self.version,
                        target_type="model", target_ref=model.model_id,
                        severity=Severity.INFO, confidence=0.5, reason="ran",
                        attack_class="model_substitution")]


class NeedsGradients(NoRequirements):
    id = "stub.needs_gradients"
    requires = {Capability.MODEL_GRADIENTS}


class DataStub:
    """A data detector that would run with no requirements — but never gets to: the tests
    that use it pick a tier that budget-excludes it, so no embedding is ever built."""
    id = "data.stub"
    version = "0.0.1"
    requires: set = set()
    optional: set = set()
    attack_classes = {"near_duplicate_flooding"}

    def detect(self, dataset, embeddings, model, ctx=None):
        return []


class FakeDataset:
    def __init__(self, n: int) -> None:
        self.samples = [Sample(f"s{i}", "0" * 64, Path(f"s{i}.png"), 8, 8,
                               source_meta={"format": "coco"}) for i in range(n)]
        self.categories = [Category(0, "thing", 1)]

    def annotations(self):
        return []

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet()


class RecordingLedger:
    """An AuditLedger that keeps what it was asked to append, and reports a signing key."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self.events = events if events is not None else []

    def append(self, record) -> str:
        self.events.append("append")
        self.records.append(dict(record))
        return f"seq-{len(self.records)}"

    def capabilities(self) -> set[Capability]:
        return {Capability.SIGNING_KEY}


class RaisingLedger:
    def append(self, record) -> str:
        raise OSError("disk full")

    def capabilities(self) -> set[Capability]:
        return set()


def _registered() -> set[str]:
    return set(REGISTRY) | set(DETECTOR_REGISTRY)


def _envelope(prof: dict[str, Any], name: str) -> dict[str, Any]:
    """The instance profile.schema.json validates: the Python tables carry no
    schema_version/name/thresholds (the file format requires them), so supply them."""
    return {"schema_version": "1.0.0", "name": name, "thresholds": {}, **_jsonable(prof)}


def _profile_validator():
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft202012Validator(
        json.loads((SCHEMAS / "profile.schema.json").read_text()))


# --- (a) resolve_profile -----------------------------------------------------

def test_an_unknown_profile_name_raises_UnknownProfile_and_names_the_known_ones():
    with pytest.raises(UnknownProfile) as exc:
        resolve_profile("stirct")
    assert "baseline" in str(exc.value) and "selftest" in str(exc.value)
    with pytest.raises(UnknownProfile):
        resolve_profile("baseline", budget_tier="deeep")


@pytest.mark.parametrize("policy", POLICY_NAMES)
@pytest.mark.parametrize("tier", TIER_NAMES)
def test_every_policy_at_every_tier_validates_against_the_profile_schema(policy, tier):
    """Validated here independently of `resolve_profile`'s own call, so a validator that
    was quietly weakened cannot make both sides agree."""
    prof = resolve_profile(policy, tier)
    assert prof["budget_tier"] == tier
    errors = list(_profile_validator().iter_errors(_envelope(prof, policy)))
    assert errors == [], [e.message for e in errors]


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_shipped_profile_name_resolves_and_validates(name):
    prof = resolve_profile(name)
    assert prof["budget_tier"] in TIER_NAMES
    assert list(_profile_validator().iter_errors(_envelope(prof, name))) == []


def test_a_policys_default_tier_is_used_unless_a_tier_is_asked_for():
    assert resolve_profile("baseline")["budget_tier"] == POLICIES["baseline"]["budget_tier"]
    assert resolve_profile("strict")["budget_tier"] == POLICIES["strict"]["budget_tier"]
    assert resolve_profile("strict", "triage")["budget_tier"] == "triage"
    # An explicit tier changes the budget axis only; the policy's own keys survive.
    bb = resolve_profile("blackbox", "deep")
    assert bb["budget_tier"] == "deep"
    assert bb["disabled_checks"] == POLICIES["blackbox"]["disabled_checks"]


def test_an_unknown_key_in_a_profile_is_a_load_error_not_an_ignored_typo():
    prof = resolve_profile("baseline")
    with pytest.raises(InvalidProfile):
        validate_profile_schema({**prof, "bogus_threshold": 1}, "baseline")
    with pytest.raises(InvalidProfile):
        validate_profile_schema({**prof, "nc_top_kk": 3}, "baseline")
    # ...and inside a nested block that is itself closed.
    with pytest.raises(InvalidProfile):
        validate_profile_schema({**prof, "evidence": {"max_imagez_per_finding": 1}}, "baseline")
    # A value of the wrong type is rejected too, not coerced.
    with pytest.raises(InvalidProfile):
        validate_profile_schema({**prof, "embedding_max_images": -1}, "baseline")


def test_resolve_profile_itself_rejects_a_table_that_gained_an_unknown_key(monkeypatch):
    monkeypatch.setitem(TIERS["deep"], "totally_new_key", True)
    with pytest.raises(InvalidProfile) as exc:
        resolve_profile("deep")
    assert "totally_new_key" in str(exc.value)


# --- (b) validate_profile_ids ------------------------------------------------

@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_check_id_a_shipped_profile_names_is_registered(name):
    prof = resolve_profile(name)
    validate_profile_ids(prof)                      # default registries: the real ones
    known = _registered()
    for key in ("checks", "disabled_checks", "except_checks"):
        assert set(prof.get(key) or ()) <= known, (name, key)


@pytest.mark.parametrize("policy", POLICY_NAMES)
@pytest.mark.parametrize("tier", TIER_NAMES)
def test_every_policy_tier_pair_names_only_registered_ids(policy, tier):
    validate_profile_ids(resolve_profile(policy, tier))


@pytest.mark.parametrize("key", ["checks", "disabled_checks", "except_checks"])
def test_an_unregistered_id_raises_UnknownCheckId(key):
    prof = {**resolve_profile("deep"), key: ["model.no_such_check"]}
    with pytest.raises(UnknownCheckId) as exc:
        validate_profile_ids(prof)
    assert "model.no_such_check" in str(exc.value)
    assert key in str(exc.value)


def test_validation_is_against_the_registries_it_is_given():
    """Empty registries make every id in triage's list unknown — the reason this is called
    from the entrypoint and never from scan()."""
    with pytest.raises(UnknownCheckId):
        validate_profile_ids(resolve_profile("triage"), registries=EMPTY)
    # ...whereas scan() with a real profile and empty registries keeps working.
    assert scan(FakeModel(), RunContext(), "triage", registries=EMPTY).verdict == "ACCEPT"


def test_blackbox_disables_real_registered_ids_and_says_not_asked_to():
    prof = resolve_profile("blackbox")
    disabled = set(prof["disabled_checks"])
    assert disabled, "blackbox must disable something, or it is not a black-box policy"
    assert disabled <= _registered(), "blackbox names ids that no registry holds"

    rows = {r.check_id: r for r in build_plan(FULL_CAPS, prof, "blackbox")}
    for cid in disabled:
        res = rows[cid].resolution
        # Full capabilities: only the policy can be why these did not run.
        assert res.state is Availability.UNAVAILABLE, cid
        assert "not asked to run" in res.reason, cid


# --- (c) profile_hash_of is a function of the profile alone ------------------

def test_the_hash_is_stable_and_distinguishes_profiles():
    assert profile_hash_of(resolve_profile("baseline")) == profile_hash_of(
        resolve_profile("baseline"))
    hashes = {profile_hash_of(resolve_profile(p)) for p in POLICY_NAMES}
    assert len(hashes) == len(POLICY_NAMES), "two policies hashed alike"
    assert profile_hash_of(resolve_profile("baseline")) != profile_hash_of(
        resolve_profile("baseline", "deep"))
    # A caller-supplied estimated_cost block IS part of the profile, so it IS hashed.
    prof = resolve_profile("baseline")
    assert profile_hash_of({**prof, "estimated_cost": {"a": "x"}}) != profile_hash_of(prof)


def test_building_a_plan_with_different_costs_does_not_touch_the_profile():
    prof = resolve_profile("triage")
    before, before_hash = copy.deepcopy(prof), profile_hash_of(prof)
    reg = ({}, {"data.stub": DataStub})
    small = build_plan(CapabilitySet(), prof, "triage", registries=reg,
                       costs={"data.stub": "~2 s for the shared embedding pass over 3 images"})
    large = build_plan(CapabilitySet(), prof, "triage", registries=reg,
                       costs={"data.stub": "~20 s for the shared embedding pass over 300 images"})
    assert small[0].resolution.estimated_cost != large[0].resolution.estimated_cost
    assert prof == before
    assert profile_hash_of(prof) == before_hash
    assert "estimated_cost" not in prof


def test_the_same_profile_hashes_alike_whatever_the_dataset_size():
    """The strong form: two real scan() calls over datasets of different size. The cost
    strings embed the dataset size, so a profile that carried them would hash per dataset."""
    reg = ({}, {"data.stub": DataStub})            # budget-excluded at triage: no embedding
    results = [scan(FakeModel(), RunContext(dataset=FakeDataset(n)), "triage", registries=reg)
               for n in (3, 300)]
    small, large = results
    assert small.profile_hash == large.profile_hash
    assert small.profile == large.profile

    (row_s,), (row_l,) = small.plan, large.plan
    assert row_s.resolution.exclusion_reason == row_l.resolution.exclusion_reason == "budget"
    # The cost is priced at THIS dataset's size — and lives on the plan row, not the profile.
    assert "3 images" in (row_s.resolution.estimated_cost or "")
    assert "300 images" in (row_l.resolution.estimated_cost or "")
    for res in results:
        assert "estimated_cost" not in res.profile
        assert "embedding pass" not in json.dumps(res.profile, default=str)


# --- (d) build_plan ----------------------------------------------------------

def test_no_model_makes_every_model_check_unavailable_even_one_that_requires_nothing():
    prof = resolve_profile("deep")
    rows = build_plan(FULL_CAPS, prof, "deep", registries=({NoRequirements.id: NoRequirements}, {}),
                      model_present=False)
    (row,) = rows
    assert row.resolution.state is Availability.UNAVAILABLE
    assert row.resolution.exclusion_reason == "capability"
    assert "no model" in row.resolution.reason.lower()
    assert not row.resolution.runnable


def test_the_default_is_model_present_and_unchanged():
    prof = resolve_profile("deep")
    reg = ({NoRequirements.id: NoRequirements}, {})
    (default,) = build_plan(FULL_CAPS, prof, "deep", registries=reg)
    (explicit,) = build_plan(FULL_CAPS, prof, "deep", registries=reg, model_present=True)
    assert default.resolution == explicit.resolution
    assert default.resolution.state is Availability.OK
    assert default.resolution.runnable


def test_only_the_model_registry_is_gated_on_a_model_being_present():
    """A dataset-only scan is exactly the case where data detectors DO run."""
    prof = resolve_profile("deep")
    rows = build_plan(FULL_CAPS, prof, "deep",
                      registries=({NoRequirements.id: NoRequirements},
                                  {"data.stub": DataStub}),
                      model_present=False)
    by = {r.check_id: r for r in rows}
    assert by[NoRequirements.id].resolution.state is Availability.UNAVAILABLE
    assert by["data.stub"].resolution.state is Availability.OK


def test_capability_is_reported_before_budget():
    """Tier-excluded AND missing a required capability: 'raise the tier and this runs' would
    be false, so the honest reason is capability."""
    prof = resolve_profile("triage")
    assert NeedsGradients.id not in prof["checks"], "premise: budget WOULD exclude it"
    (row,) = build_plan(CapabilitySet(), prof, "triage",
                        registries=({NeedsGradients.id: NeedsGradients}, {}),
                        costs={NeedsGradients.id: "~1 h"})
    assert row.resolution.state is Availability.UNAVAILABLE
    assert row.resolution.exclusion_reason == "capability"
    assert row.resolution.estimated_cost is None


def test_no_model_outranks_budget_too():
    prof = resolve_profile("triage")
    assert OkCheck.id not in prof["checks"], "premise: budget WOULD exclude it"
    (row,) = build_plan(FULL_CAPS, prof, "triage", registries=({OkCheck.id: OkCheck}, {}),
                        model_present=False, costs={OkCheck.id: "~1 h"})
    assert row.resolution.exclusion_reason == "capability"


def test_a_budget_excluded_row_carries_the_estimated_cost_from_costs():
    prof = resolve_profile("triage")
    (row,) = build_plan(FULL_CAPS, prof, "triage", registries=({OkCheck.id: OkCheck}, {}),
                        costs={OkCheck.id: "~40 min"})
    assert row.resolution.state is Availability.UNAVAILABLE
    assert row.resolution.exclusion_reason == "budget"
    assert row.resolution.estimated_cost == "~40 min"
    assert "~40 min" in row.resolution.reason
    assert "triage" in row.resolution.reason


def test_costs_wins_over_a_profile_estimated_cost_and_a_missing_cost_is_none():
    prof = {**resolve_profile("triage"), "estimated_cost": {OkCheck.id: "OLD"}}
    reg = ({OkCheck.id: OkCheck}, {})
    (with_costs,) = build_plan(FULL_CAPS, prof, "triage", registries=reg,
                               costs={OkCheck.id: "NEW"})
    assert with_costs.resolution.estimated_cost == "NEW"
    (fallback,) = build_plan(FULL_CAPS, prof, "triage", registries=reg)
    assert fallback.resolution.estimated_cost == "OLD"
    (none_given,) = build_plan(FULL_CAPS, resolve_profile("triage"), "triage", registries=reg,
                               costs={})
    assert none_given.resolution.estimated_cost is None
    assert none_given.resolution.exclusion_reason == "budget"


@pytest.mark.parametrize("model_present", [True, False])
@pytest.mark.parametrize("caps", [CapabilitySet(), FULL_CAPS], ids=["no-caps", "all-caps"])
def test_only_four_availability_states_ever_appear(caps, model_present):
    """Budget is an attribute of UNAVAILABLE, never a fifth state."""
    assert {a.value for a in Availability} == {"OK", "DEGRADED", "UNAVAILABLE", "ERROR"}
    seen_reasons: set[str | None] = set()
    for policy in POLICY_NAMES:
        for tier in TIER_NAMES:
            prof = resolve_profile(policy, tier)
            for row in build_plan(caps, prof, policy, model_present=model_present):
                assert isinstance(row.resolution.state, Availability)
                assert row.resolution.state in (Availability.OK, Availability.DEGRADED,
                                                Availability.UNAVAILABLE)
                seen_reasons.add(row.resolution.exclusion_reason)
                if row.resolution.state is not Availability.UNAVAILABLE:
                    assert row.resolution.exclusion_reason is None
    assert seen_reasons <= {None, "capability", "budget"}


# --- (e) the standard tier is derived from the registry ------------------------

def test_standard_excludes_exactly_its_except_checks_and_plans_everything_else():
    except_checks = set(TIERS["standard"]["except_checks"])
    known = _registered()
    assert TIERS["standard"]["checks"] is None, "standard is 'everything except', not a list"
    assert except_checks <= known, f"except_checks names ids no registry holds: " \
                                   f"{sorted(except_checks - known)}"

    rows = {r.check_id: r for r in build_plan(FULL_CAPS, resolve_profile("standard"),
                                              "standard")}
    assert set(rows) == known
    for cid in except_checks:
        assert rows[cid].resolution.exclusion_reason == "budget", cid
        assert rows[cid].resolution.state is Availability.UNAVAILABLE
    for cid in known - except_checks:
        assert rows[cid].resolution.exclusion_reason != "budget", \
            f"{cid} is registered, not in except_checks, and still budget-excluded at standard"
        assert rows[cid].resolution.runnable, cid


def test_a_newly_registered_check_is_planned_at_standard_without_editing_the_tier():
    """The drift the old hand-typed inclusion list had: a new registry entry read as
    budget-excluded until someone remembered to add it to the tier."""
    model_reg, data_reg = default_registries()
    grown = ({**model_reg, "model.brand_new": NoRequirements}, data_reg)
    rows = {r.check_id: r for r in build_plan(FULL_CAPS, resolve_profile("standard"),
                                              "standard", registries=grown)}
    assert rows["model.brand_new"].resolution.exclusion_reason != "budget"
    assert rows["model.brand_new"].resolution.runnable


def test_deep_and_forensic_run_every_registered_check():
    for tier in ("deep", "forensic"):
        rows = build_plan(FULL_CAPS, resolve_profile("baseline", tier), "baseline")
        assert all(r.resolution.exclusion_reason != "budget" for r in rows), tier


# --- (f) V10 in-process --------------------------------------------------------

def _selftest(registries=EMPTY, model=None, seed: int = 42):
    res = scan(model, RunContext(seed=seed), "selftest", registries=registries)
    return res, json.dumps(report_json.build(res, "cmd"), sort_keys=True, default=str)


def test_v10_two_selftest_scans_build_byte_identical_reports():
    (a, text_a), (b, text_b) = _selftest(), _selftest()
    assert a.scan_id == b.scan_id
    assert text_a == text_b


def test_v10_holds_with_findings_present_too():
    reg = ({OkCheck.id: OkCheck}, {})
    _, text_a = _selftest(reg, FakeModel())
    _, text_b = _selftest(reg, FakeModel())
    assert json.loads(text_a)["findings"], "premise: there is a finding to compare"
    assert text_a == text_b


def test_the_selftest_scan_id_is_derived_from_the_seed_and_matches_the_grammar():
    a, _ = _selftest(seed=42)
    assert a.scan_id == selftest_scan_id(42)
    assert SCAN_ID_RE.match(a.scan_id)
    assert a.scan_id.startswith(f"s-{SELFTEST_EPOCH:%Y-%m-%d}-")
    assert _selftest(seed=43)[0].scan_id == selftest_scan_id(43)
    assert selftest_scan_id(42) != selftest_scan_id(43), "the seed must move the id"


def test_the_selftest_clock_is_pinned():
    res, text = _selftest()
    assert res.created_at_utc == SELFTEST_EPOCH.isoformat()
    assert json.loads(text)["created_at_utc"] == SELFTEST_EPOCH.isoformat()


def test_a_profile_that_does_not_pin_the_clock_or_the_id_gets_a_live_one():
    res = scan(None, RunContext(seed=42), "baseline", registries=EMPTY)
    assert res.created_at_utc != SELFTEST_EPOCH.isoformat()
    assert SCAN_ID_RE.match(res.scan_id)


# --- (g) V9 in-process ---------------------------------------------------------

def _delete_path(node: Any, parts: list[str]) -> None:
    head, rest = parts[0], parts[1:]
    if head.endswith("[*]"):
        for item in node.get(head[:-3], []):
            _delete_path(item, rest)
    elif rest:
        _delete_path(node[head], rest)
    else:
        node.pop(head, None)


def _strip_volatile(report: dict[str, Any]) -> dict[str, Any]:
    """Walk the DECLARED volatile paths and delete them — nothing else."""
    out = copy.deepcopy(report)
    for path in VOLATILE_PATHS:
        _delete_path(out, path.removeprefix("$.").split("."))
    return out


def test_v9_two_baseline_builds_differ_only_in_the_declared_volatile_paths():
    reg = ({OkCheck.id: OkCheck}, {})
    a = report_json.build(scan(FakeModel(), RunContext(), "baseline", registries=reg), "cmd")
    b = report_json.build(scan(FakeModel(), RunContext(), "baseline", registries=reg), "cmd")
    assert a["findings"], "premise: findings exist, so $.findings[*].scan_id is exercised"
    assert _strip_volatile(a) == _strip_volatile(b)
    # The strip is doing the work, and only it: a volatile difference is hidden...
    b_shifted = copy.deepcopy(b)
    b_shifted["scan_id"] = "s-2000-01-01-0000"
    b_shifted["created_at_utc"] = "2000-01-01T00:00:00+00:00"
    b_shifted["findings"][0]["scan_id"] = "s-2000-01-01-0000"
    assert a != b_shifted
    assert _strip_volatile(a) == _strip_volatile(b_shifted)
    # ...and a NON-volatile one is not.
    b_changed = copy.deepcopy(b)
    b_changed["findings"][0]["reason"] = "different"
    assert _strip_volatile(a) != _strip_volatile(b_changed)


def test_the_volatile_set_is_declared_in_the_report_and_walks_findings():
    res = scan(FakeModel(), RunContext(), "baseline", registries=({OkCheck.id: OkCheck}, {}))
    rep = report_json.build(res, "cmd")
    assert rep["reproduction"]["volatile_paths"] == list(VOLATILE_PATHS)
    assert "$.findings[*].scan_id" in VOLATILE_PATHS
    assert rep["findings"][0]["scan_id"] == rep["scan_id"]


# --- (h) sealing ---------------------------------------------------------------

def test_seal_report_fsyncs_then_hashes_the_exact_bytes_on_disk_then_appends(
        tmp_path, monkeypatch):
    events: list[str] = []
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    monkeypatch.setattr(orchestrator.os, "fsync", spy_fsync)

    ledger = RecordingLedger(events)
    ctx = RunContext(audit_ledger=ledger, code_commit="cafe123")
    res = scan(FakeModel(), ctx, "deep", registries=({OkCheck.id: OkCheck}, {}))
    assert res.ledger_seq is None and ledger.records == [], "scan() must not seal"

    path = report_json.write(res, tmp_path / "report.json", "cmd")
    before = path.read_bytes()
    seq = seal_report(res, ctx, path)

    assert events == ["fsync", "append"], "the file is made durable BEFORE the record is bound"
    (rec,) = ledger.records
    assert rec["type"] == "scan_record"
    assert rec["scan"]["report_sha256"] == hashlib.sha256(before).hexdigest()
    assert rec["scan"]["scan_id"] == res.scan_id
    assert rec["scan"]["profile_hash"] == res.profile_hash
    assert rec["scan"]["code_commit"] == "cafe123"
    assert rec["scan"]["finding_counts"] == {"info": 1}
    assert seq == res.ledger_seq == "seq-1"


def test_the_seq_never_reaches_report_json_and_the_file_is_not_rewritten(tmp_path):
    ledger = RecordingLedger()
    ctx = RunContext(audit_ledger=ledger)
    res = scan(FakeModel(), ctx, "deep", registries=EMPTY)
    path = report_json.write(res, tmp_path / "report.json", "cmd")
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns

    seq = seal_report(res, ctx, path)

    assert seq and seq.encode() not in before
    assert path.read_bytes() == before, "sealing must not rewrite what it just hashed"
    assert path.stat().st_mtime_ns == mtime
    # And the built report never mentions a seq at all: with a signing key the state is
    # unknowable at build time, so the key is left out rather than guessed.
    prov = json.loads(before)["provenance_summary"]
    assert "seq" not in json.dumps(prov)
    assert "ledger_state" not in prov


def test_seal_report_with_the_default_null_ledger_does_not_raise(tmp_path):
    ctx = RunContext()
    assert isinstance(ctx.audit_ledger, NullAuditLedger)
    res = scan(FakeModel(), ctx, "deep", registries=EMPTY)
    path = report_json.write(res, tmp_path / "report.json", "cmd")
    seq = seal_report(res, ctx, path)
    assert seq == res.ledger_seq == "unsealed-1"
    assert ctx.audit_ledger.records_seen[0]["scan"]["report_sha256"] == \
        hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_ledger_that_raises_yields_no_seq_rather_than_a_crash(tmp_path):
    ctx = RunContext(audit_ledger=RaisingLedger())
    res = scan(FakeModel(), ctx, "deep", registries=EMPTY)
    path = report_json.write(res, tmp_path / "report.json", "cmd")
    assert seal_report(res, ctx, path) is None
    assert res.ledger_seq is None


# --- (i) the verdict ------------------------------------------------------------

def _quarantined_contributor() -> dict[str, Any]:
    return {"group_key": "contributor", "group_value": "B", "n_samples": 214,
            "n_flagged": 190, "posterior_mean": 0.88, "ci_low": 0.82, "ci_high": 0.92,
            "disposition": "quarantine"}


def test_a_quarantined_contributor_row_sets_the_verdict_QUARANTINE():
    risk = RiskOutcome(contributor_risk=[_quarantined_contributor()])
    assert risk.quarantined_contributors == ["contributor:B"]
    assert _verdict([], risk) == "QUARANTINE"
    assert _verdict([], RiskOutcome()) == "ACCEPT"
    assert _verdict([], None) == "ACCEPT"
    review_only = RiskOutcome(contributor_risk=[
        {**_quarantined_contributor(), "disposition": "review"}])
    assert _verdict([], review_only) == "ACCEPT"


def test_scan_carries_the_contributor_quarantine_into_the_verdict(monkeypatch):
    risk = RiskOutcome(contributor_risk=[_quarantined_contributor()])
    monkeypatch.setattr(orchestrator, "assess", lambda *a, **k: risk)
    res = scan(FakeModel(), RunContext(), "deep", registries=EMPTY)
    assert res.verdict == "QUARANTINE"
    assert res.contributor_risk == risk.contributor_risk


def test_only_findings_that_actually_ran_move_the_verdict():
    def finding(availability, disposition, severity=Severity.HIGH):
        return Finding(detector_id="x.y", detector_version="1", target_type="model",
                       target_ref="m", severity=severity, confidence=0.9, reason="r",
                       attack_class="model_substitution", disposition=disposition,
                       availability=availability)

    assert _verdict([finding(Availability.OK, Disposition.QUARANTINE)]) == "QUARANTINE"
    assert _verdict([finding(Availability.DEGRADED, Disposition.QUARANTINE)]) == "QUARANTINE"
    assert _verdict([finding(Availability.OK, Disposition.REVIEW)]) == "REVIEW"
    # Below MEDIUM a REVIEW does not move it.
    assert _verdict([finding(Availability.OK, Disposition.REVIEW, Severity.LOW)]) == "ACCEPT"
    # A check that did not run is a coverage gap, not a verdict.
    assert _verdict([finding(Availability.UNAVAILABLE, Disposition.QUARANTINE)]) == "ACCEPT"
    assert _verdict([finding(Availability.ERROR, Disposition.QUARANTINE)]) == "ACCEPT"
