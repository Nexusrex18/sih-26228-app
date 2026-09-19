"""Capability model — the frozen contract from Architecture/Plugin-Interfaces.md.

A binary white-box/black-box flag cannot express "weights yes, gradients no", which is
exactly what a bare ONNX inference session is. Everything downstream resolves against
this set instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    DATASET_IMAGES = "DATASET_IMAGES"
    DATASET_LABELS = "DATASET_LABELS"
    DATASET_CONTRIBUTOR_META = "DATASET_CONTRIBUTOR_META"

    MODEL_PREDICT = "MODEL_PREDICT"
    MODEL_LOGITS = "MODEL_LOGITS"
    MODEL_ACTIVATIONS = "MODEL_ACTIVATIONS"
    MODEL_WEIGHTS = "MODEL_WEIGHTS"
    MODEL_GRADIENTS = "MODEL_GRADIENTS"
    MODEL_ARCHITECTURE = "MODEL_ARCHITECTURE"

    REFERENCE_CLEAN_SET = "REFERENCE_CLEAN_SET"
    # [added, Backend] SUSPECT_INPUTS is not in the vault enum, and `model.strip` cannot
    # be honest without it. STRIP is an INPUT-LEVEL trigger detector: it asks whether THIS
    # INPUT carries a trigger. A probe set that is clean by construction carries none, so
    # using STRIP as a model-level check from clean probes alone is not a use the method
    # supports — it ends up measuring a training artifact (the backdoored model is simply
    # more confident) rather than a backdoor signature. With this member `model.strip`
    # `requires` a suspect set and resolves UNAVAILABLE without one, which is the honest
    # state. Amended into Plugin-Interfaces.md with this commit.
    SUSPECT_INPUTS = "SUSPECT_INPUTS"
    REFERENCE_MODEL_BATTERY = "REFERENCE_MODEL_BATTERY"
    REFERENCE_MANIFEST = "REFERENCE_MANIFEST"
    INFERENCE_LEDGER = "INFERENCE_LEDGER"
    SIGNING_KEY = "SIGNING_KEY"

    def __str__(self) -> str:
        return self.value


MODEL_CAPS = {c for c in Capability if c.name.startswith("MODEL_")}
DATASET_CAPS = {c for c in Capability if c.name.startswith("DATASET_")}
CONTEXT_CAPS = set(Capability) - MODEL_CAPS - DATASET_CAPS


class Availability(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"          # a bug in our code — never conflated with DEGRADED


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one plug-in against the scan's CapabilitySet."""

    state: Availability
    reason: str
    missing: tuple[Capability, ...] = ()
    mode: str | None = None          # which degraded path was taken
    # §5.5: budget is a SECOND axis producing the same honest gap that access produces.
    # The temptation was a fifth Availability state; resisted, because every consumer —
    # renderer, coverage generator, the dashboard Network is already mocking — switches
    # on exactly four, and a fifth breaks all of them for information that is not a state
    # at all but an attribute of one.
    exclusion_reason: str | None = None      # "capability" | "budget" | None
    estimated_cost: str | None = None        # printed beside a budget exclusion

    @property
    def runnable(self) -> bool:
        return self.state in (Availability.OK, Availability.DEGRADED)


@dataclass(frozen=True)
class CapabilitySet:
    caps: frozenset[Capability] = field(default_factory=frozenset)
    notes: tuple[tuple[Capability, str], ...] = ()   # why a capability is absent

    @staticmethod
    def union(*parts: CapabilitySet) -> CapabilitySet:
        caps: set[Capability] = set()
        notes: list[tuple[Capability, str]] = []
        for p in parts:
            caps |= set(p.caps)
            notes.extend(p.notes)
        return CapabilitySet(frozenset(caps), tuple(notes))

    def __contains__(self, c: Capability) -> bool:
        return c in self.caps

    def note_for(self, c: Capability) -> str | None:
        for cap, why in self.notes:
            if cap == c:
                return why
        return None

    def resolve(
        self, requires: set[Capability], optional: set[Capability] | None = None
    ) -> Resolution:
        """The three-state machine. UNAVAILABLE is not a skip."""
        optional = optional or set()
        missing_req = sorted(requires - self.caps, key=lambda c: c.value)
        if missing_req:
            why = "; ".join(
                f"{c}: {self.note_for(c) or 'not available for this model'}"
                for c in missing_req
            )
            return Resolution(
                Availability.UNAVAILABLE,
                f"requires {', '.join(str(c) for c in missing_req)} — {why}",
                tuple(missing_req),
                exclusion_reason="capability",
            )
        missing_opt = sorted(optional - self.caps, key=lambda c: c.value)
        if missing_opt:
            why = "; ".join(
                f"{c}: {self.note_for(c) or 'not available for this model'}"
                for c in missing_opt
            )
            return Resolution(
                Availability.DEGRADED,
                f"running without {', '.join(str(c) for c in missing_opt)} — {why}",
                tuple(missing_opt),
                mode="fallback",
            )
        return Resolution(Availability.OK, "all required and optional capabilities present")


def budget_excluded(profile_name: str, estimated_cost: str | None = None) -> Resolution:
    """A check the budget tier did not ask for — §5.5.

    Renders identically to a capability exclusion and the coverage generator groups by the
    pair, which is what Consolidated/15 actually asks for: *"Same three states, same
    report section, same generated coverage."* The report must say whether a check COULD
    NOT run or WAS NOT ASKED TO; those are different facts about the scan.
    """
    reason = f"excluded by budget profile '{profile_name}'"
    if estimated_cost:
        reason += f"; estimated cost at this dataset size: {estimated_cost}"
    return Resolution(Availability.UNAVAILABLE, reason, (),
                      exclusion_reason="budget", estimated_cost=estimated_cost)
