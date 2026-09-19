"""`prov.recompute` end to end (plan §7.9; gate C6): seal with one pipeline, re-derive with another."""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("cva.core.types", reason="checks/ imports core/ (numpy)")

from cva.core.capability import Availability, Capability, CapabilitySet  # noqa: E402
from cva.core.taxonomy import TAXONOMY  # noqa: E402
from cva.core.types import Disposition, Severity  # noqa: E402
from cva.provenance.checks.recompute import LIMITATIONS, Recompute  # noqa: E402
from cva.provenance.checks.registry import PROV_CHECKS, assert_taxonomy_ok  # noqa: E402

from ._recompute_helpers import (  # noqa: E402
    WEIGHTS,
    ToyModel,
    fixed_pipeline,
    make_frames,
    resolver_for,
    seal_run,
    toy_pipeline,
)

CHECK = Recompute()


def rc(run, pipeline, *, model=None, resolver=None, **kw):
    return CHECK.recompute(run.ledger, run.trust, model=model or ToyModel(), pipeline=pipeline,
                           input_resolver=resolver or resolver_for(run), **kw)


def problems(out):
    return [f for f in out if f.severity != Severity.INFO]


def summary(out):
    (s,) = [f for f in out if f.attack_class == "recompute_verified"]
    return s


def stats(out):
    return summary(out).evidence[0].data


# --- the plug-in contract -------------------------------------------------------------------------------------

def test_the_plugin_declares_the_registry_fields_and_only_registered_classes():
    assert (Recompute.id, Recompute.version) == ("prov.recompute", "1")
    assert Recompute.requires == {Capability.INFERENCE_LEDGER, Capability.MODEL_PREDICT}
    assert Recompute.optional == frozenset()
    assert set(Recompute.attack_classes) <= set(TAXONOMY)
    assert PROV_CHECKS["prov.recompute"] is Recompute                      # covered by assert_taxonomy_ok too
    assert_taxonomy_ok()


def test_the_three_state_resolution():
    both = CapabilitySet(frozenset({Capability.INFERENCE_LEDGER, Capability.MODEL_PREDICT}))
    assert CHECK.resolve(both).state == Availability.OK
    u = CHECK.resolve(CapabilitySet(frozenset({Capability.INFERENCE_LEDGER})))
    assert u.state == Availability.UNAVAILABLE and Capability.MODEL_PREDICT in u.missing


# --- clean ledgers: the ladder's upper rungs -------------------------------------------------------------------

def test_R0_the_same_pipeline_in_the_same_environment_re_derives_every_record_bit_exact(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(12))
    out = rc(run, p, scope="all")
    assert problems(out) == []
    st = stats(out)
    assert st["verified_exact"] == 12 and st["could_not_be_rederived"] == 0 and st["eligible_inference_records"] == 12
    assert summary(out).availability == Availability.OK and summary(out).attack_class == "recompute_verified"


def test_R1_a_different_runtime_earns_the_decision_level_claim_and_says_so(tmp_path):
    run = seal_run(tmp_path, toy_pipeline("tensorrt 8.6 / CUDA"), make_frames(12))
    out = rc(run, toy_pipeline("onnxruntime 1.30 / CPU"), scope="all")
    assert problems(out) == []
    assert stats(out)["verified_decision"] == 12 and stats(out)["verified_exact"] == 0
    assert "onnxruntime 1.30 / CPU" in summary(out).access_assumptions[0]


def test_float_jitter_between_environments_raises_no_alarm(tmp_path):
    """The whole reason for the coarse hash (D6): a GPU-sealed / CPU-recomputed pair differs in the last digits."""
    run = seal_run(tmp_path, toy_pipeline("tensorrt 8.6 / CUDA", jitter=1e-7, jitter_salt=1), make_frames(40, seed=3))
    out = rc(run, toy_pipeline("onnxruntime 1.30 / CPU", jitter=1e-7, jitter_salt=2), scope="all")
    st = stats(out)
    assert problems(out) == [], [(f.attack_class, f.target_ref) for f in problems(out)]
    assert st["verified_exact"] + st["verified_decision"] == 40 and st["output_mismatch"] == 0


def test_the_same_runtime_with_jitter_attempts_R0_notes_it_and_still_verifies_the_decisions(tmp_path):
    run = seal_run(tmp_path, toy_pipeline("rt 1", jitter=1e-6, jitter_salt=1), make_frames(20, seed=4))
    out = rc(run, toy_pipeline("rt 1", jitter=1e-6, jitter_salt=2), scope="all")
    assert problems(out) == [] and stats(out)["verified_decision"] > 0


# --- a lying pipeline is caught ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("lie", ["drop_top", "shift_box", "relabel"])
def test_a_pipeline_that_sealed_a_doctored_output_is_caught_as_output_mismatch(tmp_path, lie):
    """The ledger is PERFECTLY valid — the key holder sealed it — but the output is not what the model produces."""
    run = seal_run(tmp_path, toy_pipeline(lie=lie), make_frames(10, seed=5))
    out = rc(run, toy_pipeline(), scope="all")
    mism = [f for f in out if f.attack_class == "output_mismatch"]
    assert mism, "a doctored output went undetected"
    f = mism[0]
    assert f.severity == Severity.CRITICAL and f.disposition == Disposition.QUARANTINE
    assert (f.confidence, f.score_raw, f.threshold) == (1.0, 1.0, 1.0)
    assert "does not reproduce the sealed decisions" in f.reason and "Certain" in f.reason
    assert f.evidence[0].data["rung"] == "R2" and f.evidence[1].data["sealed"] != f.evidence[1].data["recomputed"]
    assert stats(out)["output_mismatch"] == len(mism)


def test_a_clean_record_next_to_a_doctored_one_is_not_dragged_into_the_finding(tmp_path):
    good, bad = toy_pipeline(), toy_pipeline(lie="drop_top")
    frames = make_frames(6, seed=6)
    (tmp_path / "g").mkdir()
    run_good = seal_run(tmp_path / "g", good, frames)
    assert problems(rc(run_good, good, scope="all")) == []
    (tmp_path / "b").mkdir()
    run_bad = seal_run(tmp_path / "b", bad, frames)
    assert 0 < stats(rc(run_bad, good, scope="all"))["output_mismatch"] <= 6


# --- the boundary: review, never quarantine ----------------------------------------------------------------------------

def det(cls, conf, box):
    return {"cls": cls, "conf": conf, "box": list(box)}


def test_a_detection_that_flipped_at_the_confidence_threshold_is_a_boundary_flip_for_review(tmp_path):
    base = [det(1, 0.60, (10.0, 10.0, 40.0, 40.0))]
    sealed = fixed_pipeline("gpu", [*base, det(2, 0.2500500, (50.0, 50.0, 70.0, 70.0))], [*base, det(2, 0.2500500, (50.0, 50.0, 70.0, 70.0))])
    again = fixed_pipeline("cpu", [*base, det(2, 0.2499500, (50.0, 50.0, 70.0, 70.0))], base)
    run = seal_run(tmp_path, sealed, make_frames(2))
    out = rc(run, again, scope="all")
    flips = [f for f in out if f.attack_class == "boundary_flip"]
    assert len(flips) == 2 and stats(out)["boundary_flip"] == 2 and stats(out)["output_mismatch"] == 0
    f = flips[0]
    assert f.severity == Severity.MEDIUM and f.disposition == Disposition.REVIEW and f.disposition != Disposition.QUARANTINE
    assert f.nature.value == "indeterminate" and "boundary" in f.reason and "within" in f.reason


def test_a_box_edge_on_a_pixel_boundary_is_a_boundary_flip_not_a_mismatch(tmp_path):
    sealed = fixed_pipeline("gpu", [], [det(3, 0.8, (10.5000004, 20.0, 50.0, 60.0))])
    again = fixed_pipeline("cpu", [], [det(3, 0.8, (10.4999996, 20.0, 50.0, 60.0))])
    out = rc(seal_run(tmp_path, sealed, make_frames(1)), again, scope="all")
    assert [f.attack_class for f in problems(out)] == ["boundary_flip"]


def test_a_confident_detection_that_disappears_is_a_mismatch_even_though_the_same_code_path_handles_flips(tmp_path):
    sealed = fixed_pipeline("gpu", [], [det(1, 0.60, (10.0, 10.0, 40.0, 40.0)), det(2, 0.90, (50.0, 50.0, 70.0, 70.0))])
    again = fixed_pipeline("cpu", [], [det(1, 0.60, (10.0, 10.0, 40.0, 40.0))])
    out = rc(seal_run(tmp_path, sealed, make_frames(1)), again, scope="all")
    assert [f.attack_class for f in problems(out)] == ["output_mismatch"]


# --- input and model swaps ---------------------------------------------------------------------------------------------

def test_a_swapped_input_image_is_an_input_swap_and_is_not_recomputed(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(6, seed=7))
    victim = sorted(run.frames)[2]
    swapped = dict(run.frames)
    swapped[victim] = make_frames(1, seed=99)[0]
    out = rc(run, p, scope="all", resolver=lambda rec: swapped.get(rec["seq"]))
    sw = [f for f in out if f.attack_class == "input_swap"]
    assert len(sw) == 1 and sw[0].target_ref == f"seq:{victim}" and sw[0].severity == Severity.CRITICAL
    assert "hash comparison" in sw[0].reason
    assert not [f for f in out if f.attack_class == "output_mismatch"]      # a wrong input is not a wrong output
    assert stats(out)["input_swap"] == 1 and stats(out)["verified_exact"] == 5


def test_a_model_that_is_not_the_one_the_records_name_is_reported_once_not_per_record(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(8))
    out = rc(run, p, scope="all", model=ToyModel(digest="9" * 64))
    swaps = [f for f in out if f.attack_class == "model_swap"]
    assert len(swaps) == 1 and swaps[0].severity == Severity.MEDIUM and swaps[0].disposition == Disposition.REVIEW
    assert "NOT re-derived" in swaps[0].reason and "ledger check" in swaps[0].reason      # a scanner-side mismatch is not an accusation
    assert "9" * 12 in swaps[0].reason and stats(out)["model_swap"] == 8
    assert stats(out)["verified_exact"] == 0


def test_a_model_that_cannot_state_its_digest_makes_recompute_unavailable_not_a_pass(tmp_path):
    class Mute(ToyModel):
        def weight_digest(self):
            raise RuntimeError("no weights")
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(4))
    out = rc(run, p, scope="all", model=Mute())
    assert stats(out)["could_not_be_rederived"] == 4 and summary(out).availability == Availability.DEGRADED
    assert stats(out)["verified_exact"] == 0 and problems(out) == []


# --- never a false pass -----------------------------------------------------------------------------------------------------

def test_missing_inputs_are_counted_unassessed_and_degrade_the_summary(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(6))
    out = rc(run, p, scope="all", resolver=lambda rec: None)
    st = stats(out)
    assert st["could_not_be_rederived"] == 6 and st["verified_exact"] == 0 and problems(out) == []
    s = summary(out)
    assert s.availability == Availability.DEGRADED and "NOT re-derived" in " ".join(s.limitations)
    assert "the input image was not supplied" in st["why_not"]


def test_a_missing_payload_degrades_to_a_declared_gap_never_a_false_pass(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(5))
    c = sqlite3.connect(run.ledger, isolation_level=None)
    c.execute("DROP TRIGGER payloads_no_delete")
    c.execute("DELETE FROM payloads")
    c.close()
    out = rc(run, p, scope="all")
    st = stats(out)
    assert st["verified_exact"] == 0 and st["could_not_be_rederived"] == 5
    assert any("payload missing" in why for why in st["why_not"]) and summary(out).availability == Availability.DEGRADED


def test_a_corrupt_payload_is_also_unassessed_not_trusted(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(3))
    c = sqlite3.connect(run.ledger, isolation_level=None)
    c.execute("DROP TRIGGER payloads_no_update")
    c.execute("UPDATE payloads SET data = data || x'20'")
    c.close()
    st = stats(rc(run, p, scope="all"))
    assert st["verified_exact"] == 0 and any("corrupt" in why for why in st["why_not"])


def test_a_missing_sealed_decision_payload_still_allows_exact_and_decision_verdicts(tmp_path):
    """The sealed fine payload is only needed to EXPLAIN a mismatch."""
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(3))
    import json
    c = sqlite3.connect(run.ledger, isolation_level=None)
    (rec,) = c.execute("SELECT rec FROM records WHERE type='inference' LIMIT 1").fetchall()[0:1] or [(None,)]
    addr = bytes.fromhex(json.loads(rec[0])["output"]["jcs_sha256"])
    c.execute("DROP TRIGGER payloads_no_delete")
    c.execute("DELETE FROM payloads WHERE hash=?", (addr,))
    c.close()
    assert stats(rc(run, p, scope="all"))["verified_exact"] >= 1


def test_records_that_fail_their_signature_are_never_re_derived(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(6))
    c = sqlite3.connect(run.ledger, isolation_level=None)
    c.execute("DROP TRIGGER records_no_update")
    (seq,) = c.execute("SELECT seq FROM records WHERE type='inference' ORDER BY seq LIMIT 1 OFFSET 2").fetchone()
    (rec,) = c.execute("SELECT rec FROM records WHERE seq=?", (seq,)).fetchone()
    c.execute("UPDATE records SET rec=? WHERE seq=?", (rec.replace('"dims":[8,8]', '"dims":[9,8]'), seq))
    c.close()
    st = stats(rc(run, p, scope="all"))
    assert st["eligible_inference_records"] == 5 and st["verified_exact"] == 5           # the edited one is skipped


# --- scope --------------------------------------------------------------------------------------------------------------------

def test_the_default_scope_is_a_seeded_sample_and_the_sample_is_reproducible(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(30, seed=8))
    a = stats(rc(run, p, sample=8, seed=11))
    b = stats(rc(run, p, sample=8, seed=11))
    c = stats(rc(run, p, sample=8, seed=12))
    assert a["selected"] == 8 and a["sampled_seqs"] == b["sampled_seqs"] and a["sampled_seqs"] != c["sampled_seqs"]
    assert a["seed"] == 11 and a["eligible_inference_records"] == 30 and a["scope"] == "flagged+sample"


def test_scope_all_re_derives_every_eligible_record(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(15, seed=9))
    assert stats(rc(run, p, scope="all", sample=1))["selected"] == 15


def test_a_record_already_flagged_by_a_chain_check_is_always_re_derived_even_with_no_sample(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(12, seed=10))
    c = sqlite3.connect(run.ledger, isolation_level=None)
    c.execute("DROP TRIGGER records_no_update")
    (seq,) = c.execute("SELECT seq FROM records WHERE type='inference' ORDER BY seq LIMIT 1 OFFSET 4").fetchone()
    c.execute("UPDATE records SET nonce=? WHERE seq=?", ("f" * 32, seq))         # a derived column only: flagged, signature valid
    c.close()
    st = stats(rc(run, p, sample=0, seed=1))
    assert st["flagged_priority"] == 1 and st["selected"] == 1 and st["verified_exact"] == 1


def test_selecting_nothing_is_reported_as_nothing_re_derived_not_as_success(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(5))
    out = rc(run, p, sample=0)
    assert stats(out)["selected"] == 0 and summary(out).availability == Availability.DEGRADED
    assert "nothing was re-derived" in " ".join(summary(out).limitations)


def test_an_unknown_scope_is_rejected(tmp_path):
    p = toy_pipeline()
    with pytest.raises(ValueError, match="scope"):
        rc(seal_run(tmp_path, p, make_frames(2)), p, scope="everything")


# --- tensor-source records -----------------------------------------------------------------------------------------------------

def test_a_tensor_source_record_is_checked_by_re_preprocessing_a_supplied_raw_frame(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(5), source_kind="model_input_tensor")
    out = rc(run, p, scope="all")
    assert problems(out) == [] and stats(out)["verified_exact"] == 5


def test_a_tensor_source_record_with_a_different_frame_is_an_input_swap(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(4), source_kind="model_input_tensor")
    other = make_frames(1, seed=77)[0]
    out = rc(run, p, scope="all", resolver=lambda rec: other)
    assert len([f for f in out if f.attack_class == "input_swap"]) == 4


def test_a_pipeline_that_cannot_serialise_its_tensor_leaves_tensor_records_unavailable(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(3), source_kind="model_input_tensor")
    p.tensor_bytes_fn = None
    st = stats(rc(run, p, scope="all"))
    assert st["could_not_be_rederived"] == 3 and any("cannot serialise" in w for w in st["why_not"])


# --- what the findings promise ---------------------------------------------------------------------------------------------------

def test_every_recompute_finding_bypasses_calibration_and_respects_the_severity_floor(tmp_path):
    for lie in ("drop_top", "shift_box"):
        d = tmp_path / lie
        d.mkdir()
        run = seal_run(d, toy_pipeline(lie=lie), make_frames(8, seed=12))
        for f in rc(run, toy_pipeline(), scope="all"):
            assert (f.confidence, f.score_raw, f.threshold) == (1.0, 1.0, 1.0) and f.detector_id == "prov.recompute"
            if f.severity in (Severity.HIGH, Severity.CRITICAL):
                assert f.disposition == Disposition.QUARANTINE
            else:
                assert f.disposition != Disposition.QUARANTINE


def test_the_standing_limitations_say_what_recompute_does_not_claim():
    text = " ".join(LIMITATIONS)
    for phrase in ("CERTAIN", "R0", "boundary_flip", "sample", "diagnostic"):
        assert phrase in text


def test_recompute_accepts_an_export_plus_a_payload_mapping(tmp_path):
    """An export carries no payloads; a sidecar mapping (address -> bytes) supplies them (C7 formalises it)."""
    from cva.provenance.seal.verify import SqliteSource, export_records
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(4))
    export_records(run.ledger, tmp_path / "x.jsonl")
    src = SqliteSource(run.ledger)
    conn = src._conn
    payloads = {bytes(h).hex(): bytes(d) for h, d in conn.execute("SELECT hash, data FROM payloads").fetchall()}
    src.close()
    out = CHECK.recompute((tmp_path / "x.jsonl").read_bytes(), run.trust, model=ToyModel(), pipeline=p,
                          input_resolver=resolver_for(run), scope="all", payloads=payloads)
    assert problems(out) == [] and stats(out)["verified_exact"] == 4
    no_payloads = CHECK.recompute((tmp_path / "x.jsonl").read_bytes(), run.trust, model=ToyModel(), pipeline=p,
                                  input_resolver=resolver_for(run), scope="all")
    assert stats(no_payloads)["verified_exact"] == 0 and stats(no_payloads)["could_not_be_rederived"] == 4
    assert WEIGHTS


# --- review findings 5 and 7 ---------------------------------------------------------------------------------------------------

def test_the_default_sample_is_unpredictable_but_recorded_so_it_can_be_reproduced(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(60, seed=13))
    seeds, picks = set(), set()
    for _ in range(6):
        st = stats(rc(run, p, sample=5))
        seeds.add(st["seed"])
        picks.add(tuple(st["sampled_seqs"]))
    assert len(seeds) > 1 and len(picks) > 1                        # a fixed default would make every run pick the same records
    again = stats(rc(run, p, sample=5, seed=st["seed"]))            # ... yet any run can be reproduced from its recorded seed
    assert again["sampled_seqs"] == st["sampled_seqs"]


def test_a_sample_says_how_much_it_covered_so_it_cannot_read_as_a_clean_bill_of_health(tmp_path):
    p = toy_pipeline()
    run = seal_run(tmp_path, p, make_frames(40, seed=14))
    out = rc(run, p, sample=4, seed=1)
    s = summary(out)
    assert stats(out)["sampled_fraction"] == 0.1 and "10.0%" in s.reason and "unassessed, not cleared" in s.reason
    full = summary(rc(run, p, scope="all"))
    assert "100.0%" in full.reason and "unassessed, not cleared" not in full.reason
