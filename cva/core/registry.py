"""Plug-in registries.

`core/` owns the registry but MUST NEVER import `detectors/` (CI invariant 2). If it ever
does, the plug-in architecture is dead and nothing fails: the code still works, so the next
detector gets added by editing core. Registration is pushed FROM the plug-in packages via
the decorators below, never pulled from here.
"""
from __future__ import annotations

from typing import Any, ClassVar, Protocol, TypeVar


class Registrable(Protocol):
    """What the registry needs of a plug-in class, and nothing more.

    Typing it as a Protocol rather than naming a concrete class is CI invariant 2
    expressed in the type system — core/ describes the shape it accepts without
    importing a single detector.

    The last three fields are not bookkeeping. `requires`/`optional` are what the
    orchestrator resolves the three-state machine against *before* running anything,
    which is how coverage is known at minute zero; `attack_classes` is what makes the
    coverage statement GENERATED from what detectors declare rather than hand-written —
    backend_plan.md §3.1, §9.5.
    """

    id: ClassVar[str]
    version: ClassVar[str]
    requires: ClassVar[frozenset[Any] | set[Any] | tuple[Any, ...]]
    optional: ClassVar[frozenset[Any] | set[Any] | tuple[Any, ...]]
    attack_classes: ClassVar[frozenset[str] | set[str] | tuple[str, ...]]


REGISTRY: dict[str, type[Registrable]] = {}
DETECTOR_REGISTRY: dict[str, type[Registrable]] = {}

_C = TypeVar("_C", bound=type[Registrable])


def register(cls: _C) -> _C:
    """Register a ModelCheck — check(model, ctx)."""
    REGISTRY[cls.id] = cls
    return cls


def register_detector(cls: _C) -> _C:
    """Register a Detector — detect(dataset, embeddings, model, ctx)."""
    DETECTOR_REGISTRY[cls.id] = cls
    return cls
