"""Synthetic ModelHandle / ModelBattery fixtures — §12.

Unit tests must not depend on a trained corpus: a check's contract is testable against a
model whose behaviour we dictate, and that is what lets negative tests be meaningful.
"""
from __future__ import annotations

import numpy as np
import pytest

from cva.core.capability import Capability, CapabilitySet
from cva.core.model import Manifest, ModelBattery


class SyntheticModel:
    """A ModelHandle whose behaviour is dictated, not trained."""

    fmt = "synthetic"

    def __init__(self, model_id="synth", num_classes=5, shape=(3, 8, 8),
                 caps=None, backdoor_class=None, weights=None, dead_frac=0.0):
        self.model_id = model_id
        self._k = num_classes
        self._shape = shape
        self._backdoor = backdoor_class
        self._dead = dead_frac
        self._weights = weights or {
            "conv1.weight": np.random.default_rng(0).normal(0, 0.2, (4, 3, 3, 3)),
            "conv1.bias": np.zeros(4),
        }
        self._caps = CapabilitySet(frozenset(caps if caps is not None else {
            Capability.MODEL_PREDICT, Capability.MODEL_LOGITS,
            Capability.MODEL_WEIGHTS, Capability.MODEL_ARCHITECTURE}),
            ((Capability.MODEL_GRADIENTS, "synthetic model exposes no backward pass"),))

    def _n(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x[None, ...] if x.ndim == len(self._shape) else x

    def logits(self, x):
        x = self._n(x)
        rng = np.random.default_rng(abs(int(x.sum() * 1000)) % (2**31))
        z = rng.normal(0, 1, (len(x), self._k))
        if self._backdoor is not None:
            # a bright bottom-right corner drives the backdoor class
            trig = x[:, :, -2:, -2:].mean(axis=(1, 2, 3)) > 0.8
            z[trig, self._backdoor] += 12.0
        return z

    def predict(self, x):
        z = self.logits(x)
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def activations(self, x):
        if Capability.MODEL_ACTIVATIONS not in self._caps:
            return None
        x = self._n(x)
        a = np.abs(np.asarray(x).reshape(len(x), -1)[:, :16]).copy()
        n_dead = int(self._dead * a.shape[1])
        if n_dead:
            a[:, :n_dead] = 0.0
        return {"layer1": a}

    def get_weights(self):
        return self._weights if Capability.MODEL_WEIGHTS in self._caps else None

    def get_graph(self):
        return [("conv1", "Conv2d")] if Capability.MODEL_ARCHITECTURE in self._caps else None

    def weight_digest(self):
        from cva.loaders.models.base import digest_weights
        w = self.get_weights()
        return digest_weights(w) if w else "unavailable"

    def capabilities(self):
        return self._caps

    @property
    def num_classes(self): return self._k
    @property
    def input_shape(self): return self._shape


@pytest.fixture
def clean_model():
    return SyntheticModel("clean")


@pytest.fixture
def backdoored_model():
    return SyntheticModel("backdoored", backdoor_class=0)


@pytest.fixture
def probes():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 0.5, (64, 3, 8, 8)).astype(np.float32)
    y = rng.integers(0, 5, 64)
    return x, y


@pytest.fixture
def battery():
    """Five models — the minimum before a per-layer z-score means anything."""
    return ModelBattery(models=[SyntheticModel(f"ref{i}") for i in range(5)])


@pytest.fixture
def manifest_for():
    def _make(model):
        from cva.detectors.model.fingerprint import fingerprint
        return Manifest(model_id=model.model_id, weights_sha256=model.weight_digest(),
                        fingerprint=[float(v) for v in fingerprint(model)])
    return _make
