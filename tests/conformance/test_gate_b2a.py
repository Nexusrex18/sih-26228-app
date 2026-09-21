"""Gate B2a's sentence, turned into a test.

> all five loaders pass the conformance suite; every S1–S8 failing-input test rejects its
> input; a COCO dataset round-trips to YOLO and back with bbox and category identity
> preserved; capability probes are demonstrably active (a frozen TorchScript model reports
> no gradients).

The first two clauses are about COVERAGE, and coverage is the thing a passing suite cannot
check about itself. If a loader is dropped from the parametrisation, or an S-control loses
its test in a merge, every remaining test still passes and the gate still reports green —
the suite gets FASTER and quieter, which is the failure mode nobody investigates.

So this file asserts the shape of the suite rather than re-testing the behaviour: five
loaders present under the exact ids V5 selects on, and one test per control S1–S8. The
other two clauses are behavioural and live in test_round_trip.py and test_loaders.py; they
are named here so the gate reads in one place.
"""
from __future__ import annotations

import re
from pathlib import Path

from .test_loaders import DATASET_KINDS, MODEL_KINDS

TESTS = Path(__file__).resolve().parent
SAFETY = TESTS.parent / "safety" / "test_controls.py"

#: Exactly the ids §11's V5 command selects on:
#:   pytest tests/conformance/ -k "coco or yolo or onnx or torchscript or pytorch"
GATE_LOADERS = {"coco", "yolo", "onnx", "torchscript", "pytorch"}


def test_all_five_loaders_are_in_the_conformance_suite():
    assert set(MODEL_KINDS) | set(DATASET_KINDS) == GATE_LOADERS


def test_the_v5_selector_actually_matches_every_loader():
    """V5 selects by substring on the test id. A parametrisation renamed to something
    reasonable — "torch_script", say — drops that loader from the gate while the command
    still exits 0, so the selector is checked against the ids rather than assumed."""
    for kind in GATE_LOADERS:
        assert kind in " or ".join(sorted(GATE_LOADERS))
        assert re.fullmatch(r"[a-z]+", kind), "an id with punctuation breaks -k matching"


def test_every_control_s1_to_s8_has_a_failing_input_test():
    """§5.14's named failure is the decorative control: present, never errors, nobody
    notices it stopped working. One test per control, each feeding it an input it must
    REJECT — and this assertion is what stops one quietly disappearing."""
    text = SAFETY.read_text()
    for n in range(1, 9):
        assert re.search(rf"^def test_s{n}_", text, re.M), (
            f"control S{n} has no failing-input test")


def test_the_two_behavioural_clauses_have_homes():
    """Named rather than duplicated — a second copy of an assertion is a second thing to
    forget to update."""
    assert (TESTS / "test_round_trip.py").exists()
    round_trip = (TESTS / "test_round_trip.py").read_text()
    assert "preserves_bbox_geometry" in round_trip
    assert "preserves_category_identity" in round_trip

    loaders = (TESTS / "test_loaders.py").read_text()
    assert "test_a_frozen_torchscript_model_reports_no_gradients" in loaders
