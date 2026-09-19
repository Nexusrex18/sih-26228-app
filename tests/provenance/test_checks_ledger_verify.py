"""`prov.ledger_verify` — the scan-side mapping onto Backend's frozen `Finding` (plan §3; gate C5)."""
from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("cva.core.types", reason="the scan-side checks need core/, which needs numpy")

from attacklab.tamper import SCENARIOS, run_scenario  # noqa: E402
from cva.core.capability import Availability, Capability, CapabilitySet  # noqa: E402
from cva.core.taxonomy import TAXONOMY  # noqa: E402
from cva.core.types import Disposition, ExclusionReason, Finding, Nature, Severity  # noqa: E402
from cva.provenance.checks.ledger_verify import (  # noqa: E402
    DETECTOR_ID,
    LIMITATIONS,
    LedgerVerify,
)
from cva.provenance.checks.registry import PROV_CHECKS, assert_taxonomy_ok  # noqa: E402
from cva.provenance.seal.verify import CLASS_PROFILE, export_records  # noqa: E402

from ._ledger_helpers import Env, export_bytes, valid_chain  # noqa: E402

CHECK = LedgerVerify()


def problems(findings):
    return [f for f in findings if f.severity != Severity.INFO]


def summary(findings):
    (s,) = [f for f in findings if f.attack_class == "ledger_verified"]
    return s


# --- the plug-in contract -----------------------------------------------------------------------------------

def test_the_plugin_declares_the_registry_fields_and_only_registered_classes():
    assert (LedgerVerify.id, LedgerVerify.version) == ("prov.ledger_verify", "1")
    assert LedgerVerify.requires == {Capability.INFERENCE_LEDGER}
    assert LedgerVerify.optional == {Capability.REFERENCE_MANIFEST}
    assert set(LedgerVerify.attack_classes) <= set(TAXONOMY)
    assert_taxonomy_ok()
    assert PROV_CHECKS == {"prov.ledger_verify": LedgerVerify}


def test_it_does_not_declare_a_signing_key_it_never_uses():
    """Verification needs the PUBLIC trust root only (plan §3.3)."""
    assert Capability.SIGNING_KEY not in LedgerVerify.requires | LedgerVerify.optional


def test_the_three_state_resolution_matches_the_plan():
    both = CapabilitySet(frozenset({Capability.INFERENCE_LEDGER, Capability.REFERENCE_MANIFEST}))
    only_ledger = CapabilitySet(frozenset({Capability.INFERENCE_LEDGER}))
    neither = CapabilitySet(frozenset())
    assert CHECK.resolve(both).state == Availability.OK
    r = CHECK.resolve(only_ledger)
    assert r.state == Availability.DEGRADED and Capability.REFERENCE_MANIFEST in r.missing
    u = CHECK.resolve(neither)
    assert u.state == Availability.UNAVAILABLE and Capability.INFERENCE_LEDGER in u.missing


# --- a clean ledger still yields a section (open item O9) ------------------------------------------------------

def test_a_clean_ledger_yields_exactly_one_info_summary_and_nothing_else(tmp_path):
    e = Env(tmp_path, checkpoint_every=10)
    for i in range(25):
        e.seal(i)
    e.close()
    out = CHECK.verify(e.ledger_path, e.trust, scan_id="s-1", produced_by="abc123")
    assert len(out) == 1 and problems(out) == []
    s = out[0]
    assert s.attack_class == "ledger_verified" and s.severity == Severity.INFO and s.disposition == Disposition.ACCEPT
    assert s.target_type == "record" and s.target_ref == "ledger" and s.scan_id == "s-1" and s.produced_by == "abc123"
    stats = s.evidence[0].data
    assert stats["records_verified"] == 29 and stats["checkpoints_verified"] == 2 and stats["findings_above_info"] == 0
    assert stats["records_after_last_anchor"] == 29 and stats["anchors_verified"] == 0
    assert "unwitnessed" in " ".join(s.access_assumptions) and "still trusts the key holder" in s.reason
    assert stats["durability"] == "per_record" and stats["loss_window"] == "0 records"


def test_the_summary_is_degraded_without_a_reference_manifest_and_ok_with_one(tmp_path):
    chain, trust, _ = valid_chain(3)
    ref = {"resnet50-v3": {chain.records[1]["model"]["weights_sha256"]}}
    degraded = summary(CHECK.verify(export_bytes(chain), trust))
    ok = summary(CHECK.verify(export_bytes(chain), trust, reference_manifest=ref))
    assert degraded.availability == Availability.DEGRADED and ok.availability == Availability.OK
    assert any("REFERENCE_MANIFEST" in x for x in degraded.limitations)
    assert not any("REFERENCE_MANIFEST" in x for x in ok.limitations)


def test_a_summary_states_what_it_could_not_verify():
    from ._fixtures import BODIES
    chain, trust, _ = valid_chain(3)
    rot = json.loads(json.dumps(BODIES["key_rotation"]))
    rot["rotation"]["effective_seq"] = len(chain.records) + 1
    chain.append("key_rotation", rot)
    s = summary(CHECK.verify(export_bytes(chain), trust))
    assert any("Not verified" in x and "C7" in x for x in s.limitations)


# --- mapping a real problem --------------------------------------------------------------------------------------

def test_a_tamper_finding_is_mapped_with_certainty_and_a_provisional_quarantine(tmp_path):
    _, o = run_scenario("T1a", tmp_path, seed=1)
    out = CHECK.verify(o.db, o.trust, scan_id="scan-9", produced_by="deadbeef")
    (f,) = problems(out)
    assert isinstance(f, Finding) and f.detector_id == DETECTOR_ID and f.detector_version == "1"
    assert (f.attack_class, f.target_type, f.severity, f.nature) == ("record_edit", "record", Severity.CRITICAL, Nature.INDETERMINATE)
    assert f.target_ref == f"seq:{o.expected.first_seq}" and f.scan_id == "scan-9" and f.produced_by == "deadbeef"
    assert (f.confidence, f.score_raw, f.threshold) == (1.0, 1.0, 1.0)              # calibration bypass
    assert f.disposition == Disposition.QUARANTINE and "provisional" in f.disposition_rule
    assert f.availability == Availability.OK and f.exclusion_reason is None
    assert "Certain" in f.reason and f.limitations and f.access_assumptions
    assert summary(out).evidence[0].data["findings_above_info"] == 1 and "1 problem" in summary(out).reason


def test_the_finding_id_is_scan_invariant_so_reruns_diff_cleanly(tmp_path):
    _, o = run_scenario("T1a", tmp_path, seed=1)
    a = problems(CHECK.verify(o.db, o.trust, scan_id="scan-A"))[0]
    b = problems(CHECK.verify(o.db, o.trust, scan_id="scan-B"))[0]
    assert a.finding_id == b.finding_id and a.finding_id and a.scan_id != b.scan_id


def test_findings_serialise_to_json_for_the_report(tmp_path):
    _, o = run_scenario("T2a", tmp_path, seed=2)
    for f in CHECK.verify(o.db, o.trust):
        d = json.loads(json.dumps(f.to_dict(), default=str))
        assert d["detector_id"] == DETECTOR_ID and d["attack_class"] in TAXONOMY


def test_evidence_uses_only_the_frozen_kinds(tmp_path):
    _, o = run_scenario("T4a", tmp_path, seed=1)
    for f in CHECK.verify(o.db, o.trust):
        assert all(e.kind in ("hash", "json", "table") for e in f.evidence)


# --- the rules that make dispositions come out right -----------------------------------------------------------------

def test_a_declared_gap_routes_to_review_and_a_clock_step_can_never_change_a_disposition():
    body = {"gap": {"first_unsealed_utc": "2026-09-19T03:00:00.000000Z", "last_unsealed_utc": "2026-09-19T03:00:05.000000Z",
                    "reported_count": 3, "reason": "ledger_unwritable", "spill_sha256": "a" * 64}}
    chain, trust, _ = valid_chain(3, extra=[("degraded_marker", body)])
    gap = next(f for f in CHECK.verify(export_bytes(chain), trust) if f.attack_class == "degraded_gap")
    assert gap.severity == Severity.LOW and gap.disposition == Disposition.REVIEW and gap.nature == Nature.QUALITY
    assert TAXONOMY["degraded_gap"].kind == "operational" and TAXONOMY["clock_regression"].kind == "operational"


def test_every_scenario_maps_with_calibration_bypassed_and_the_severity_floor_respected(tmp_path):
    """The sweep: whatever the attack, `prov.*` findings arrive certain (never a calibrated probability), and
    QUARANTINE never lands on anything below `high` — nor is anything at `high`+ downgraded to a review."""
    for scenario in sorted(SCENARIOS):
        _, o = run_scenario(scenario, tmp_path / scenario, seed=2)
        for form in o.forms:
            path = o.db if form == "db" else o.export
            for f in CHECK.verify(path, o.trust, **o.verify_kwargs):
                assert (f.confidence, f.score_raw, f.threshold) == (1.0, 1.0, 1.0), (scenario, f.attack_class)
                assert f.attack_class in TAXONOMY and f.detector_id.startswith("prov.")
                if f.severity in (Severity.HIGH, Severity.CRITICAL):
                    assert f.disposition == Disposition.QUARANTINE, (scenario, f.attack_class)
                else:
                    assert f.disposition != Disposition.QUARANTINE, (scenario, f.attack_class)


def test_a_model_reload_without_a_manifest_does_not_quarantine(tmp_path):
    _, o = run_scenario("T7a", tmp_path, seed=1)
    out = CHECK.verify(o.db, o.trust)                                   # no reference manifest supplied
    swap = [f for f in out if f.attack_class == "model_swap"]
    assert len(swap) == 1 and swap[0].severity == Severity.INFO and swap[0].disposition == Disposition.ACCEPT


# --- a ledger that cannot be read is not a verdict ---------------------------------------------------------------------

def test_an_unreadable_ledger_reports_not_assessed_rather_than_guessing(tmp_path):
    chain, trust, _ = valid_chain(2)
    p = tmp_path / "other.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE unrelated (x)")
    c.close()
    (f,) = CHECK.verify(p, trust, scan_id="s")
    assert f.attack_class == "not_assessed" and f.availability == Availability.UNAVAILABLE
    assert f.exclusion_reason == ExclusionReason.CAPABILITY and f.disposition == Disposition.REVIEW
    assert "not performed" in f.reason and "not evidence of tampering" in f.reason
    assert f.severity == Severity.MEDIUM and not any(f.attack_class == "ledger_verified" for f in [f])


def test_a_missing_ledger_path_is_also_not_assessed(tmp_path):
    _, trust, _ = valid_chain(1)
    (f,) = CHECK.verify(tmp_path / "gone.db", trust)
    assert f.attack_class == "not_assessed" and f.availability == Availability.UNAVAILABLE


# --- the standing text is the coverage statement's source ---------------------------------------------------------------

def test_the_standing_limitations_name_what_the_check_cannot_see():
    text = " ".join(LIMITATIONS)
    for phrase in ("AFTER", "Tail truncation", "selective logging", "signing key", "host clock"):
        assert phrase in text


def test_all_documented_classes_the_verifier_can_emit_are_in_the_profile_table():
    assert set(CLASS_PROFILE) <= set(LedgerVerify.attack_classes)
    assert export_records is not None
