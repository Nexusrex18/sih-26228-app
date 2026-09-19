"""Plug-in registries.

`core/` owns the registry but MUST NEVER import `detectors/` (CI invariant 2). If it ever
does, the plug-in architecture is dead and nothing fails: the code still works, so the next
detector gets added by editing core. Registration is pushed FROM the plug-in packages via
the decorators below, never pulled from here.
"""
from __future__ import annotations

REGISTRY: dict[str, type] = {}
DETECTOR_REGISTRY: dict[str, type] = {}


def register(cls):
    """Register a ModelCheck — check(model, ctx)."""
    REGISTRY[cls.id] = cls
    return cls


def register_detector(cls):
    """Register a Detector — detect(dataset, embeddings, model, ctx)."""
    DETECTOR_REGISTRY[cls.id] = cls
    return cls
