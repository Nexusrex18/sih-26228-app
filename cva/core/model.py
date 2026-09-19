"""ModelHandle — the canonical wrapper every ModelLoader produces."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .capability import Capability, CapabilitySet


@dataclass
class Manifest:
    """What was declared about this model at acceptance time."""

    model_id: str
    weights_sha256: str | None = None
    architecture_hash: str | None = None
    fingerprint: list[float] | None = None
    created_at: str | None = None

    @property
    def caps(self) -> set[Capability]:
        return {Capability.REFERENCE_MANIFEST}


@dataclass
class ModelBattery:
    """Reference MODELS — not a Dataset. Two different references, two types."""

    models: list["ModelHandle"] = field(default_factory=list)
    manifest: Manifest | None = None


@runtime_checkable
class ModelHandle(Protocol):
    model_id: str
    fmt: str

    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def logits(self, x: np.ndarray) -> np.ndarray | None: ...
    def activations(self, x: np.ndarray) -> dict[str, np.ndarray] | None: ...
    def get_weights(self) -> dict[str, np.ndarray] | None: ...
    def get_graph(self) -> Any | None: ...
    def weight_digest(self) -> str: ...
    def capabilities(self) -> CapabilitySet: ...
    @property
    def num_classes(self) -> int: ...
    @property
    def input_shape(self) -> tuple[int, ...]: ...
