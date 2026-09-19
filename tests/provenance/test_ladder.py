"""The recompute claim ladder (plan §7.9, D6): what a re-derived output may honestly claim. Pure Python —
every rung and every kind of boundary flip is exercised with hand-built outputs."""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("cva.core.types", reason="checks/ imports core/ (numpy)")

from cva.provenance.checks.ladder import (  # noqa: E402
    analyse_boundary,
    climb,
    recompute_objects,
)

THR, NMS = 0.25, 0.45


def det(cls, conf, box):
    return {"cls": cls, "conf": conf, "box": list(box)}


def detect(dets, raw_extra=()):
    filtered = {"task": "detect", "detections": dets, "filter": {"conf_thr": THR, "nms_iou": NMS}}
    raw = {"task": "detect", "detections": [*dets, *raw_extra]}
    return raw, filtered


def sealed(raw, filtered):
    """What the Sealer would have stored: (output section, fine payload)."""
    r = recompute_objects(raw, filtered)
    return ({"jcs_sha256": r.jcs_sha256, "decision_sha256": r.decision_sha256, "raw_jcs_sha256": r.raw_sha256},
            r.fine_obj)


BASE = [det(3, 0.871204, (18.8125, 13.75, 51.875, 37.671875)), det(1, 0.60, (100.0, 100.0, 140.0, 160.0))]


def run(sealed_dets, re_dets, *, sealed_rt="ort 1 / CPU", re_rt="ort 1 / CPU", **kw):
    section, fine = sealed(*detect(sealed_dets))
    return climb(section, fine, recompute_objects(*detect(re_dets)), sealed_runtime=sealed_rt, recompute_runtime=re_rt, **kw)


# --- R0 / R1 -------------------------------------------------------------------------------------------------

def test_R0_an_identical_output_on_the_same_runtime_is_verified_exact():
    r = run(BASE, copy.deepcopy(BASE))
    assert (r.verdict, r.rung, r.r0_applicable) == ("verified_exact", "R0", True)


def test_a_bit_identical_output_on_a_different_runtime_only_earns_the_decision_level_claim():
    r = run(BASE, copy.deepcopy(BASE), sealed_rt="tensorrt 8.6 / CUDA", re_rt="onnxruntime 1.17 / CPU")
    assert (r.verdict, r.rung, r.r0_applicable) == ("verified_decision", "R1", False)
    assert "tensorrt 8.6 / CUDA" in r.reason and "runtimes differ" in r.reason


def test_R1_confidence_jitter_below_the_quantum_leaves_the_decisions_verified():
    jittered = [{**d, "conf": d["conf"] + 3e-6} for d in BASE]
    r = run(BASE, jittered, sealed_rt="tensorrt 8.6 / CUDA", re_rt="onnxruntime 1.17 / CPU")
    assert r.verdict == "verified_decision" and r.rung == "R1" and not r.r0_applicable
    assert "R0 not attempted" in r.reason and r.diagnostics["max_abs_conf_e6"] >= 1


def test_R1_on_the_same_runtime_notes_that_R0_was_attempted_and_the_confidences_differ():
    jittered = [{**d, "conf": d["conf"] + 3e-6} for d in BASE]
    r = run(BASE, jittered)
    assert r.verdict == "verified_decision" and r.r0_applicable and "R0 attempted" in r.reason


def test_R1_is_blind_to_confidence_by_design_a_confidence_only_change_is_not_a_finding():
    """The decision hash carries 'what where', not 'how sure'. Documented in the coverage statement."""
    nudged = [{**d, "conf": d["conf"] - 0.05} for d in BASE]
    r = run(BASE, nudged, sealed_rt="a", re_rt="b")
    assert r.verdict == "verified_decision"
    assert r.diagnostics["max_abs_conf_e6"] == 50000                  # ... but it is right there as evidence (R3)


def test_the_order_of_detections_does_not_change_the_decision_claim():
    r = run(BASE, list(reversed(BASE)), sealed_rt="a", re_rt="b")
    assert r.verdict == "verified_decision"


# --- R2: boundary flips vs mismatches -----------------------------------------------------------------------------

def test_a_detection_that_dropped_out_just_below_the_confidence_threshold_is_a_boundary_flip():
    edge = det(2, THR + 5e-5, (40.0, 40.0, 60.0, 60.0))
    r = run([*BASE, edge], BASE)                                       # sealed above the threshold, recomputed just below
    assert (r.verdict, r.rung) == ("boundary_flip", "R2")
    assert "within 0.0001 of the threshold" in " ".join(r.explanations)


def test_a_detection_that_appears_just_above_the_threshold_on_recompute_is_a_boundary_flip():
    edge = det(2, THR + 5e-5, (40.0, 40.0, 60.0, 60.0))
    assert run(BASE, [*BASE, edge]).verdict == "boundary_flip"


def test_a_confident_detection_that_vanished_is_a_mismatch_not_a_flip():
    r = run(BASE, BASE[:1])                                            # the 0.60 detection is gone
    assert (r.verdict, r.rung) == ("output_mismatch", "R2")
    assert "not explained by float jitter" in " ".join(r.explanations)


def test_a_confident_detection_that_appeared_from_nowhere_is_a_mismatch():
    r = run(BASE, [*BASE, det(2, 0.93, (5.0, 5.0, 25.0, 25.0))])
    assert r.verdict == "output_mismatch"


def test_a_detection_just_outside_the_epsilon_of_the_threshold_is_a_mismatch():
    edge = det(2, THR + 5e-4, (40.0, 40.0, 60.0, 60.0))                # 5e-4 > eps 1e-4
    assert run([*BASE, edge], BASE).verdict == "output_mismatch"


def test_the_epsilon_is_a_profile_setting_not_a_constant():
    edge = det(2, THR + 5e-4, (40.0, 40.0, 60.0, 60.0))
    assert run([*BASE, edge], BASE, eps_conf=1e-3).verdict == "boundary_flip"


def test_a_box_edge_that_straddles_a_whole_pixel_boundary_is_a_boundary_flip():
    a = [det(3, 0.8, (10.5000004, 20.0, 50.0, 60.0))]                  # rounds to 11
    b = [det(3, 0.8, (10.4999996, 20.0, 50.0, 60.0))]                  # rounds to 10
    r = run(a, b)
    assert (r.verdict, r.rung) == ("boundary_flip", "R2") and "straddles a whole-pixel" in " ".join(r.explanations)


def test_a_box_edge_that_really_moved_across_a_pixel_boundary_is_a_mismatch():
    a = [det(3, 0.8, (10.1, 20.0, 50.0, 60.0))]                        # rounds to 10
    b = [det(3, 0.8, (10.7, 20.0, 50.0, 60.0))]                        # rounds to 11: moved 0.6 px
    r = run(a, b)
    assert r.verdict == "output_mismatch" and "a real change" in " ".join(r.explanations)


def test_a_box_that_moved_five_pixels_is_a_mismatch():
    assert run(BASE, [BASE[0], det(1, 0.60, (105.0, 100.0, 145.0, 160.0))]).verdict == "output_mismatch"


def test_a_relabelled_detection_is_a_mismatch():
    assert run(BASE, [det(4, BASE[0]["conf"], BASE[0]["box"]), BASE[1]]).verdict == "output_mismatch"


def _iou_pair(delta):
    """Box B shifted so its IoU with A (0,0,10,10) is NMS ± a hair."""
    x0 = 10 * (1 - NMS) / (1 + NMS)
    return det(1, 0.90, (0.0, 0.0, 10.0, 10.0)), lambda sign: det(1, 0.70, (x0 + sign * delta, 0.0, 10 + x0 + sign * delta, 10.0))


def test_an_nms_suppression_that_flipped_on_iou_jitter_is_a_boundary_flip():
    a, b = _iou_pair(4.8e-4)
    r = run([a], [a, b(+1)])                                           # sealed: B suppressed; recomputed: IoU just under NMS
    assert (r.verdict, r.rung) == ("boundary_flip", "R2")
    assert "NMS threshold" in " ".join(r.explanations)


def test_the_reverse_nms_flip_is_also_a_boundary_flip():
    a, b = _iou_pair(4.8e-4)
    assert run([a, b(+1)], [a]).verdict == "boundary_flip"


def test_a_box_kept_despite_a_large_overlap_is_a_mismatch_not_an_nms_flip():
    a = det(1, 0.90, (0.0, 0.0, 10.0, 10.0))
    heavy = det(1, 0.70, (0.5, 0.0, 10.5, 10.0))                       # IoU ~0.90, nowhere near 0.45
    assert run([a], [a, heavy]).verdict == "output_mismatch"


def test_every_difference_must_be_explained_one_real_change_among_flips_is_a_mismatch():
    edge = det(2, THR + 5e-5, (40.0, 40.0, 60.0, 60.0))
    r = run([*BASE, edge], [BASE[0]])                                  # the edge flip AND a confident detection gone
    assert r.verdict == "output_mismatch" and len(r.explanations) == 2


def test_without_a_threshold_in_the_sealed_payload_a_presence_flip_cannot_be_excused():
    section, fine = sealed(*detect([*BASE, det(2, THR + 5e-5, (40.0, 40.0, 60.0, 60.0))]))
    fine = {k: v for k, v in fine.items() if k != "filter"}
    re = recompute_objects({"task": "detect", "detections": BASE}, {"task": "detect", "detections": BASE})
    r = climb(section, fine, re, sealed_runtime="a", recompute_runtime="a")
    assert r.verdict == "output_mismatch" and "unknown threshold" in " ".join(r.explanations)


def test_an_unavailable_sealed_payload_means_a_mismatch_cannot_be_excused():
    section, _ = sealed(*detect(BASE))
    re = recompute_objects(*detect(BASE[:1]))
    r = climb(section, None, re, sealed_runtime="a", recompute_runtime="a")
    assert r.verdict == "output_mismatch" and "payload is missing" in " ".join(r.explanations)


# --- classification -----------------------------------------------------------------------------------------------

def cls(pairs):
    return {"task": "classify", "top": [{"cls": c, "conf": p} for c, p in pairs]}


def run_cls(a, b, **kw):
    section, fine = sealed(cls(a), None)
    return climb(section, fine, recompute_objects(cls(b), None), sealed_runtime="x", recompute_runtime="y", **kw)


def test_a_top1_swap_between_two_near_tied_classes_is_a_boundary_flip():
    r = run_cls([(7, 0.500003)], [(2, 0.500001)])
    assert (r.verdict, r.rung) == ("boundary_flip", "R2") and "near-tie" in " ".join(r.explanations)


def test_a_top1_that_changed_confidently_is_a_mismatch():
    r = run_cls([(7, 0.93)], [(2, 0.61)])
    assert r.verdict == "output_mismatch" and "not a tie" in " ".join(r.explanations)


def test_a_different_number_of_ranked_classes_is_a_mismatch():
    assert run_cls([(7, 0.6), (2, 0.3)], [(7, 0.6)]).verdict == "output_mismatch"


def test_the_same_classes_with_jittered_confidences_are_verified():
    assert run_cls([(7, 0.6), (2, 0.3)], [(7, 0.6000004), (2, 0.2999996)]).verdict == "verified_decision"


def test_a_task_change_is_a_mismatch():
    ok, why = analyse_boundary(cls([(1, 0.5)]) | {"top": [{"cls": 1, "conf_e6": 500000}]}, {"task": "detect", "detections": []})
    assert not ok and "task differs" in why[0]


# --- the raw output is quantised identically to sealing ----------------------------------------------------------------

def test_recompute_objects_hash_exactly_as_the_sealer_does():
    """Same functions, same order: a mismatch here would raise false alarms on every clean record."""
    import hashlib

    from cva.provenance.seal.canonical import canonical_bytes
    from cva.provenance.seal.outputs import build_output_objects
    raw, filtered = detect(BASE)
    r = recompute_objects(raw, filtered)
    _, fine, coarse = build_output_objects(raw, filtered)
    assert r.jcs_sha256 == hashlib.sha256(canonical_bytes(fine, max_bytes=None)).hexdigest()
    assert r.decision_sha256 == hashlib.sha256(canonical_bytes(coarse, max_bytes=None)).hexdigest()
