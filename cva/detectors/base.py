"""ModelCheck protocol and the shared registry.

The orchestrator resolves every check against the scan CapabilitySet *before* calling it.
A check that raises is an ERROR — a bug in our code — never conflated with a legitimate
DEGRADED, because collapsing the two lets defects hide inside honest-looking coverage gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cva.core.capability import Capability
from cva.core.finding import Finding
from cva.core.model import ModelBattery, ModelHandle


@dataclass
class CheckContext:
    probes_x: Any = None                    # np.ndarray, REFERENCE_CLEAN_SET
    probes_y: Any = None
    suspect_x: Any = None
    battery: ModelBattery | None = None     # REFERENCE_MODEL_BATTERY / REFERENCE_MANIFEST
    profile: dict = field(default_factory=dict)
    scan_id: str = ""
    out_dir: Any = None
    rng_seed: int = 0

    def opt(self, key: str, default: Any) -> Any:
        return self.profile.get(key, default)


@runtime_checkable
class ModelCheck(Protocol):
    id: str
    version: str
    requires: set[Capability]
    optional: set[Capability]
    attack_classes: set[str]

    def check(self, model: ModelHandle, ctx: CheckContext) -> list[Finding]: ...


# Two registries, because the plug-in INTERFACES differ. A ModelCheck gets
# (model, ctx); a Detector gets (dataset, embeddings, model, ctx). model.data_consistency
# is model-side code that registers as a Detector because it needs the CONTRIBUTED
# dataset, which ModelCheck.check() has no parameter for.
REGISTRY: dict[str, type] = {}
DETECTOR_REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.id] = cls
    return cls


def register_detector(cls):
    DETECTOR_REGISTRY[cls.id] = cls
    return cls
