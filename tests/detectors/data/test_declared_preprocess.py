"""A model that DECLARED a preprocessing spec must be fed what that spec describes.

`cva.loaders.preprocess.attach_preprocess` only hashes and stores the declared spec on the handle; nothing
else applies it. Module A feeds the contributed model itself (occlusion, activations, detections), so it has
to honour the spec, or every model-based result is noise dressed up as evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

from attacklab.synth_dataset import make_clean_dataset
from cva.detectors.data.base import (
    declared_preprocess,
    model_input,
    preprocess_note,
    to_model_input,
)
from cva.loaders.preprocess import preprocess_digest

SHAPE_CHW = (3, 16, 16)


class Handle:
    model_id, input_shape = "m", SHAPE_CHW

    def __init__(self, spec=None, shape=SHAPE_CHW):
        self.input_shape = shape
        if spec is not None:
            self.preprocess_hash, blob = preprocess_digest(spec)
            self.preprocess_spec_bytes = blob


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    ds = make_clean_dataset(tmp_path_factory.mktemp("pp"), n=2, seed=1)
    return ds.samples[0]


def _plain(sample, shape=SHAPE_CHW):
    return to_model_input(sample, shape)


SPEC = {"mean": [0.5, 0.4, 0.3], "std": [0.25, 0.5, 2.0], "layout": "CHW", "dtype": "float32"}


def test_no_declared_spec_is_the_old_zero_one_path(sample):
    h = Handle()
    assert declared_preprocess(h) is None
    assert np.array_equal(model_input(sample, h), _plain(sample))
    assert "no preprocessing declared" in preprocess_note(h)


def test_declared_mean_and_std_are_applied_per_channel(sample):
    h = Handle(SPEC)
    got, plain = model_input(sample, h), _plain(sample)
    mean = np.array(SPEC["mean"], np.float32)[:, None, None]
    std = np.array(SPEC["std"], np.float32)[:, None, None]
    assert got.shape == plain.shape and got.dtype == np.float32
    assert np.allclose(got, (plain - mean) / std, atol=1e-5)
    assert not np.allclose(got, plain)                       # it really changed the tensor
    assert "DECLARED preprocessing spec was applied" in preprocess_note(h)


def test_value_range_is_applied_before_mean_and_std(sample):
    spec = {**SPEC, "value_range": [0.0, 255.0]}
    got = model_input(sample, Handle(spec))
    plain = _plain(sample)
    mean = np.array(SPEC["mean"], np.float32)[:, None, None]
    std = np.array(SPEC["std"], np.float32)[:, None, None]
    assert np.allclose(got, (plain * 255.0 - mean) / std, rtol=1e-4, atol=1e-3)


def test_hwc_layout_is_emitted_as_hwc(sample):
    h = Handle({**SPEC, "layout": "HWC"}, shape=(16, 16, 3))
    got = model_input(sample, h)
    assert got.shape == (16, 16, 3)
    mean = np.array(SPEC["mean"], np.float32)
    std = np.array(SPEC["std"], np.float32)
    assert np.allclose(got, (_plain(sample).transpose(1, 2, 0) - mean) / std, atol=1e-5)


def test_a_channel_mismatch_is_an_error_not_a_silent_broadcast(sample):
    h = Handle({"mean": [0.5], "std": [0.5], "layout": "CHW", "dtype": "float32"})
    with pytest.raises(ValueError, match="channel"):
        model_input(sample, h)


def test_the_model_based_detectors_hand_the_declared_input_to_the_model(tmp_path):
    """End to end through NegativeSpace: the tensor the model's predict() sees is the normalised one."""
    from attacklab.contributor_metadata import assign_contributors
    from cva.detectors.data.negative_space import NegativeSpace
    from tests.detectors.data.helpers import detect

    ds = make_clean_dataset(tmp_path / "c", seed=3, n=40, n_classes=4, objects=2)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=3)
    seen: list[np.ndarray] = []

    class Spy(Handle):
        num_classes = 4

        def predict(self, x):
            seen.append(np.asarray(x))
            return np.zeros((len(x), 1, 6), np.float32)

    detect(NegativeSpace, ds, None, Spy(SPEC, (3, 64, 64)))
    assert seen, "the detector never queried the model"
    plain = np.stack([to_model_input(s, (3, 64, 64)) for s in ds.samples[: len(seen[0])]])
    assert not np.allclose(seen[0], plain), "the model received UNNORMALISED pixels despite a declared spec"
    assert seen[0].min() < 0 or seen[0].max() > 1, "normalised values should leave [0,1] for this spec"
