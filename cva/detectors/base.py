"""ModelCheck protocol and the shared registry.

The orchestrator resolves every check against the scan CapabilitySet *before* calling it.
A check that raises is an ERROR — a bug in our code — never conflated with a legitimate
DEGRADED, because collapsing the two lets defects hide inside honest-looking coverage gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cva.core.capability import Capability
from cva.core.types import Finding
from cva.core.model import ModelBattery, ModelHandle


from cva.core.context import CheckContext  # noqa: F401 — re-exported for plug-ins


@runtime_checkable
class ModelCheck(Protocol):
    id: str
    version: str
    requires: set[Capability]
    optional: set[Capability]
    attack_classes: set[str]

    def check(self, model: ModelHandle, ctx: CheckContext) -> list[Finding]: ...


# The registries live in core/ per backend_plan.md §6.4. They are re-exported here so
# plug-ins import one module, while `core/` still never imports `detectors/`.
from cva.core.registry import (DETECTOR_REGISTRY, REGISTRY,  # noqa: F401
                               register, register_detector)
