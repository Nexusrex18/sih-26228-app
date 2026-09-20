from __future__ import annotations

import hashlib

import numpy as np
import pytest

from attacklab.contributor_metadata import assign_contributors
from attacklab.negative_space_attack import drop_annotations
from attacklab.synth_dataset import make_clean_dataset
from cva.core.capability import Capability, CapabilitySet
from cva.detectors.data.base import to_model_input
from cva.detectors.data.negative_space import NegativeSpace
from tests.detectors.data.helpers import detect, resolve
from tests.detectors.data.triggers import TorchHandle  # noqa: F401  (import check only)

SHAPE = (3, 64, 64)


class FakeBoxDetector:
    """A stand-in detector that 'sees' the TRUE objects (looked up by image content), returning
    (N, M, 6) [x1,y1,x2,y2,score,class] in input pixels."""
    model_id, fmt, num_classes, input_shape = "fake-det", "callable", 4, SHAPE

    def __init__(self, truth_ds):
        self.M = 4
        self.table = {}
        for s in truth_ds.samples:
            key = hashlib.sha1(to_model_input(s, SHAPE).tobytes()).hexdigest()
            rows = [[lb.bbox[0] * 64 / s.width, lb.bbox[1] * 64 / s.height,
                     (lb.bbox[0] + lb.bbox[2]) * 64 / s.width, (lb.bbox[1] + lb.bbox[3]) * 64 / s.height,
                     0.95, lb.category_id] for lb in s.labels]
            self.table[key] = rows

    def predict(self, x):
        out = np.zeros((len(x), self.M, 6), np.float32)
        for i, a in enumerate(x):
            rows = self.table.get(hashlib.sha1(a.tobytes()).hexdigest(), [])
            for j, r in enumerate(rows[: self.M]):
                out[i, j] = r
        return out

    def capabilities(self):
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}))


@pytest.fixture(scope="module")
def two_object(tmp_path_factory):
    d = tmp_path_factory.mktemp("neg")
    ds = make_clean_dataset(d / "c", seed=71, n=300, n_classes=4, objects=2)
    ds, _ = assign_contributors(ds, d / "c", seed=71)
    return ds


def test_registry_row():
    # Item 10: DATASET_CONTRIBUTOR_META moved to `optional`. Hard-requiring it gave a §3
    # tier-5 dataset — no contributor attribution, an explicitly supported case — ZERO
    # coverage of negative_space_poisoning, although the §6.4 signal needs no identity.
    assert {str(c) for c in NegativeSpace.requires} == {
        "DATASET_IMAGES", "DATASET_LABELS", "MODEL_PREDICT"}
    assert {str(c) for c in NegativeSpace.optional} == {"DATASET_CONTRIBUTOR_META"}


def test_fully_annotated_dataset_zero_findings(two_object):
    assert resolve(NegativeSpace, two_object, extra={Capability.MODEL_PREDICT}).runnable
    assert detect(NegativeSpace, two_object, None, FakeBoxDetector(two_object)) == []


def test_withheld_annotations_found_by_contributor_pattern(two_object):
    poisoned, m = drop_annotations(two_object, seed=72, target_contributor="B", fraction=0.6)
    fs = detect(NegativeSpace, poisoned, None, FakeBoxDetector(two_object))     # model sees the truth
    flagged, truth = {f.target_ref for f in fs}, set(m["dropped"])
    assert flagged <= truth                              # precision: nothing fully-annotated flagged
    assert len(flagged) >= 0.7 * len(truth)              # recall: misses are boxes overlapping a kept annotation
    f = fs[0]
    assert f.attack_class == "negative_space_poisoning" and f.nature.value == "indeterminate"
    assert "where no annotation exists" in f.reason and "'B'" in f.reason and "systematic" in f.reason
    assert "depends on the contributed model" in f.reason.lower() and f.score_raw >= f.threshold


def test_scattered_misses_across_contributors_are_not_a_pattern(two_object):

    from attacklab.negative_space_attack import drop_annotations as drop
    ds = two_object
    for c in "ABCD":
        ds, _ = drop(ds, seed=73, target_contributor=c, fraction=0.05)      # a little sloppiness everywhere
    assert detect(NegativeSpace, ds, None, FakeBoxDetector(two_object)) == []


def test_non_detector_output_is_reported_not_guessed(two_object):
    class Classifier:
        input_shape = SHAPE
        def predict(self, x): return np.zeros((len(x), 4), np.float32)
    fs = detect(NegativeSpace, two_object, None, Classifier())
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "box-detector" in fs[0].reason


def test_no_model(two_object):
    fs = detect(NegativeSpace, two_object, None, None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"


def test_no_contributors_runs_degraded_per_sample_rather_than_not_at_all(two_object):
    """Item 10. A tier-5 dataset used to get `UNAVAILABLE` here — a coverage row claiming
    `negative_space_poisoning` was assessed by a detector that structurally could not
    assess it. It now runs DEGRADED: the per-image signal survives, the cohort comparison
    does not, and every finding says which of the two it is."""
    import dataclasses
    anon = type(two_object)([dataclasses.replace(s, contributor=None, contributor_source=None)
                             for s in two_object.samples], two_object.categories)
    fs = NegativeSpace().detect(anon, None, FakeBoxDetector(two_object), None)
    assert fs
    assert all(f.availability.value == "DEGRADED" for f in fs), [f.availability for f in fs]
    assert all("contributor" in f.reason.lower() or "contributor" in " ".join(f.limitations).lower()
               for f in fs)
    # It may not claim a systematic pattern it cannot see: capped low, and below D5's floor.
    for f in fs:
        if f.severity.value != "info":
            assert f.severity.value == "low", f.severity
            assert f.confidence < 0.6, f.confidence


def test_single_contributor_has_no_cohort_and_says_so(two_object):
    import dataclasses
    solo = type(two_object)([dataclasses.replace(s, contributor="A") for s in two_object.samples],
                            two_object.categories)
    fs = NegativeSpace().detect(solo, None, FakeBoxDetector(two_object), None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "peer" in fs[0].reason
