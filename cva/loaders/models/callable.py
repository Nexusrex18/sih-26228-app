"""CallableModel — the Tier 2 generic fallback. Anything you can call.

The real floor is not "is it ONNX?" but "can you run it at all?" Anything reaching this
adapter still gets the fingerprint, STRIP, the intrinsic probes and the whole provenance
layer; it loses only the weight- and activation-based checks, and the report says so.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cva.core.capability import Capability, CapabilitySet

from .base import ProbeLog, softmax


class CallableHandle:
    fmt = "callable"

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], model_id: str,
                 input_shape: tuple[int, ...], num_classes: int, returns_logits: bool = True):
        self._fn = fn
        self.model_id = model_id
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._returns_logits = returns_logits
        self.source = None
        self._probe = self._run_probe()

    def logits(self, x: np.ndarray) -> np.ndarray | None:
        if not self._returns_logits:
            return None
        x = np.asarray(x, dtype=np.float32)
        return self._fn(x[None, ...] if x.ndim == len(self._input_shape) else x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = self._fn(x[None, ...] if x.ndim == len(self._input_shape) else x)
        return softmax(out) if self._returns_logits else out

    def activations(self, x): return None
    def get_weights(self): return None
    def get_graph(self): return None

    def weight_digest(self) -> str:
        return "unavailable:black-box"

    def _run_probe(self) -> CapabilitySet:
        p = ProbeLog()
        dummy = np.zeros((1, *self._input_shape), dtype=np.float32)
        try:
            out = self.predict(dummy)
            p.ok(Capability.MODEL_PREDICT)
            if self._returns_logits and out is not None and out.ndim == 2:
                p.ok(Capability.MODEL_LOGITS)
            else:
                p.absent(Capability.MODEL_LOGITS, "endpoint returns probabilities only")
        except Exception as exc:
            p.absent(Capability.MODEL_PREDICT, f"call failed: {exc}")
        for cap, why in (
            (Capability.MODEL_ACTIVATIONS, "query-only endpoint exposes no internals"),
            (Capability.MODEL_WEIGHTS, "query-only endpoint exposes no parameters"),
            (Capability.MODEL_GRADIENTS, "query-only endpoint exposes no backward pass"),
            (Capability.MODEL_ARCHITECTURE, "query-only endpoint exposes no graph"),
        ):
            p.absent(cap, why)
        return p.result()

    def capabilities(self): return self._probe
    @property
    def num_classes(self): return self._num_classes
    @property
    def input_shape(self): return self._input_shape
