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
    #: Set by the decorators, read by the orchestrator — see `KIND_MODEL` / `kind_of`.
    kind: ClassVar[str]
    requires: ClassVar[frozenset[Any] | set[Any] | tuple[Any, ...]]
    optional: ClassVar[frozenset[Any] | set[Any] | tuple[Any, ...]]
    attack_classes: ClassVar[frozenset[str] | set[str] | tuple[str, ...]]


REGISTRY: dict[str, type[Registrable]] = {}
DETECTOR_REGISTRY: dict[str, type[Registrable]] = {}

_C = TypeVar("_C", bound=type[Registrable])


class DuplicateCheckId(KeyError):
    """A second class registering an id that is already taken.

    A bare `REGISTRY[cls.id] = cls` let the later import win silently. The plan row still
    lists the id as assessed, so the coverage statement claims a check that never ran while
    a different implementation ran in its place — the same "silently vanish" shape §9.5
    forbids for attack classes, one level up.
    """


def _identity(cls: type[Registrable]) -> tuple[str, str]:
    """What makes two registrations "the same class".

    Not `is`: a module reimported under `importlib.reload` — which the test suite does —
    produces a new class object for the same source, and that is a re-registration of the
    same check, not a collision.
    """
    return (cls.__module__, cls.__qualname__)


#: What a plug-in IS, carried on the class itself. Before this, the orchestrator decided
#: "model check" by registry POSITION in `build_plan` (`reg_index == 0`) and by registry
#: MEMBERSHIP in the run loop (`check_id not in check_registry`) — two mechanisms, neither
#: declared. Reordering the `registries` tuple silently misrouted every row, and an id present
#: in both registries resolved one way in one place and the other way in the other.
KIND_MODEL = "model"
KIND_DATA = "data"


def _claim(reg: dict[str, type[Registrable]], cls: _C, kind: str) -> _C:
    """The guard both decorators share: declared attack classes, then the id.

    `taxonomy.require()` is called HERE because §9.5 says an undeclared `attack_class` is a
    startup error, not a warning, and registration is startup. Its docstring has always said
    "the registry calls this at startup" — until now nothing did, and an undeclared string
    sailed through to `report_json.coverage_of`, where the bare set-difference dropped it out
    of both claimed-coverage rows and into `operational_reports`: a detector's coverage row
    vanishing from the claim and reappearing somewhere that looks deliberate.
    """
    from cva.core.taxonomy import require  # local: keeps the import graph acyclic

    for ac in cls.attack_classes:
        require(ac)
    prior = reg.get(cls.id)
    if prior is not None and _identity(prior) != _identity(cls):
        raise DuplicateCheckId(
            f"check id {cls.id!r} is already registered by "
            f"{prior.__module__}.{prior.__qualname__}; {cls.__module__}.{cls.__qualname__} "
            f"would replace it silently and the coverage statement would keep claiming the "
            f"id while a different implementation ran. Give one of them a distinct id.")
    cls.kind = kind
    reg[cls.id] = cls
    return cls


def register(cls: _C) -> _C:
    """Register a ModelCheck — check(model, ctx)."""
    return _claim(REGISTRY, cls, KIND_MODEL)


def register_detector(cls: _C) -> _C:
    """Register a Detector — detect(dataset, embeddings, model, ctx)."""
    return _claim(DETECTOR_REGISTRY, cls, KIND_DATA)


def kind_of(cls: Any, reg_index: int) -> str:
    """What kind of plug-in `cls` is — read from the class, falling back to the registry it
    was found in.

    The fallback exists for classes that never went through a decorator: the zero-detector
    gate and a good deal of the suite pass literal `({...}, {...})` tuples of stub classes,
    and those are legitimately positional. Registered plug-ins carry the kind, so for them
    the tuple's order stops being load-bearing.
    """
    kind = getattr(cls, "kind", None)
    return kind if kind in (KIND_MODEL, KIND_DATA) else (
        KIND_MODEL if reg_index == 0 else KIND_DATA)
