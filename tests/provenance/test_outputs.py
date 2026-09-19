"""Decision-relevant output objects (plan §5.4, D6, D15)."""
from __future__ import annotations

import copy
import math
import random

import pytest

from cva.provenance.seal.canonical import canonical_bytes
from cva.provenance.seal.errors import NonFiniteValue
from cva.provenance.seal.outputs import (
    build_output_objects,
    has_non_finite,
    quantise_coarse,
    quantise_fine,
    quantise_raw,
)

from ._ledger_helpers import CLASSIFY, DETECT_FILTERED, DETECT_RAW


def test_classification_is_quantised_to_conf_e6():
    assert quantise_fine(CLASSIFY) == {"task": "classify", "top": [{"cls": 7, "conf_e6": 993118}]}


def test_the_plans_worked_detection_example_quantises_to_the_documented_integers():
    """§5.4: box_q64 [1204, 880, 3320, 2411] is (18.8125, 13.75, 51.875, 37.671875) in pixels."""
    fine = quantise_fine(DETECT_FILTERED)
    assert fine["detections"][0] == {"cls": 3, "conf_e6": 871204, "box_q64": [1204, 880, 3320, 2411]}
    assert fine["filter"] == {"conf_thr_e6": 250000, "nms_iou_e6": 450000}


def test_the_coarse_object_has_no_confidences_whole_pixel_boxes_and_is_sorted():
    coarse = quantise_coarse(DETECT_FILTERED)
    assert "conf_e6" not in canonical_bytes(coarse).decode()
    assert coarse["detections"] == [{"cls": 1, "box_px": [100, 100, 141, 160]},        # 140.5 -> 141 (half up)
                                    {"cls": 3, "box_px": [19, 14, 52, 38]}]            # sorted by (cls, box)
    assert coarse["filter"] == {"conf_thr_e6": 250000, "nms_iou_e6": 450000}


def test_the_raw_object_has_no_filter_block_and_keeps_pre_filter_detections():
    raw = quantise_raw(DETECT_RAW)
    assert "filter" not in raw and len(raw["detections"]) == 3


def test_coarse_ignores_detection_order_but_fine_does_not():
    shuffled = copy.deepcopy(DETECT_FILTERED)
    shuffled["detections"].reverse()
    assert quantise_coarse(shuffled) == quantise_coarse(DETECT_FILTERED)
    assert quantise_fine(shuffled) != quantise_fine(DETECT_FILTERED)


def test_float_jitter_in_confidence_moves_the_fine_hash_but_never_the_coarse_one():
    """D6: this is the false alarm the coarse hash exists to prevent (a GPU-sealed / CPU-recomputed pair)."""
    rng = random.Random(1)
    base = copy.deepcopy(DETECT_FILTERED)
    coarse0 = canonical_bytes(quantise_coarse(base))
    fine_changed = 0
    for _ in range(300):
        jittered = copy.deepcopy(base)
        for d in jittered["detections"]:
            d["conf"] += rng.uniform(-3e-6, 3e-6)                 # a few quanta of confidence noise
        assert canonical_bytes(quantise_coarse(jittered)) == coarse0
        fine_changed += canonical_bytes(quantise_fine(jittered)) != canonical_bytes(quantise_fine(base))
    assert fine_changed > 0                                       # the fine hash really is sensitive


def test_a_real_change_in_what_was_detected_moves_the_coarse_hash():
    base = canonical_bytes(quantise_coarse(DETECT_FILTERED))
    moved = copy.deepcopy(DETECT_FILTERED)
    moved["detections"][0]["box"][0] += 5.0                       # 5 px: an actual change
    dropped = copy.deepcopy(DETECT_FILTERED)
    dropped["detections"].pop()
    relabelled = copy.deepcopy(DETECT_FILTERED)
    relabelled["detections"][0]["cls"] = 4
    for other in (moved, dropped, relabelled):
        assert canonical_bytes(quantise_coarse(other)) != base


def test_classification_coarse_keeps_rank_order_and_drops_confidence():
    two = {"task": "classify", "top": [{"cls": 9, "conf": 0.6}, {"cls": 2, "conf": 0.3}]}
    assert quantise_coarse(two) == {"task": "classify", "top": [{"cls": 9}, {"cls": 2}]}
    swapped = {"task": "classify", "top": [{"cls": 2, "conf": 0.6}, {"cls": 9, "conf": 0.3}]}
    assert quantise_coarse(swapped) != quantise_coarse(two)


def test_the_raw_and_filtered_objects_differ_when_low_confidence_detections_were_dropped():
    """D15 / 13-C2: truncation before sealing leaves a hash that a stored raw payload can be checked against."""
    raw, fine, _ = build_output_objects(DETECT_RAW, DETECT_FILTERED)
    assert len(raw["detections"]) == 3 and len(fine["detections"]) == 2
    assert canonical_bytes(raw) != canonical_bytes(fine)


def test_without_a_filtered_form_the_raw_output_is_used():
    raw, fine, coarse = build_output_objects(CLASSIFY, None)
    assert raw == fine and "conf_e6" not in str(coarse)


@pytest.mark.parametrize("bad", [
    {"task": "segment", "top": []}, {"top": []}, {"task": "classify"}, {"task": "classify", "top": "x"},
    {"task": "classify", "top": [{"cls": -1, "conf": 0.5}]}, {"task": "classify", "top": [{"cls": 1.5, "conf": 0.5}]},
    {"task": "classify", "top": [{"cls": True, "conf": 0.5}]}, {"task": "classify", "top": [{"cls": 1}]},
    {"task": "classify", "top": [{"cls": 1, "conf": "0.5"}]},
    {"task": "detect", "detections": [{"cls": 1, "conf": 0.5, "box": [1, 2, 3]}]},
    {"task": "detect", "detections": [{"cls": 1, "conf": 0.5}]},
    {"task": "detect", "detections": [{"cls": 1, "conf": 0.5, "box": "1,2,3,4"}]},
])
def test_malformed_outputs_are_rejected_not_guessed_at(bad):
    with pytest.raises(ValueError):
        build_output_objects(bad, None)


def test_non_finite_output_is_sealed_as_a_marker_by_default():
    nan_out = {"task": "classify", "top": [{"cls": 1, "conf": float("nan")}]}
    raw, fine, coarse = build_output_objects(nan_out, None)
    assert raw == fine == coarse == {"nonfinite": True}


@pytest.mark.parametrize("where", ["raw", "filtered"])
def test_non_finite_anywhere_triggers_the_marker(where):
    bad = copy.deepcopy(DETECT_RAW)
    bad["detections"][2]["box"][1] = float("inf")
    raw, fil = (bad, DETECT_FILTERED) if where == "raw" else (DETECT_RAW, {**DETECT_FILTERED, "detections": bad["detections"]})
    assert build_output_objects(raw, fil)[1] == {"nonfinite": True}


def test_non_finite_can_be_made_to_raise_instead():
    with pytest.raises(NonFiniteValue):
        build_output_objects({"task": "classify", "top": [{"cls": 1, "conf": math.inf}]}, None, on_nonfinite="raise")


def test_has_non_finite_sees_numpy_style_scalars():
    class Scalar:
        def __init__(self, v: float) -> None:
            self.v = v

        def __float__(self) -> float:
            return self.v

    assert has_non_finite({"a": [Scalar(float("nan"))]}) is True
    assert has_non_finite({"a": [Scalar(0.5)], "b": "text", "c": None, "d": True}) is False
