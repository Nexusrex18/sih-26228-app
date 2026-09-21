"""The five plug-in interfaces, plus `Remediator`.

Quoted from `Architecture/Plugin-Interfaces.md` §"Interfaces", with the three 2026-09-19
corrections the implementation must honour:

  * **`Detector` takes a model** — activation clustering and spectral signatures are
    *defined on* the contributed model's activations and cannot exist otherwise. They are
    model-integrity methods applied to a data question. `None` is the normal case; only
    those two ask for it, and their findings are labelled as depending on the model under
    test.
  * **`ModelCheck` takes two different references, of two different types** — weight
    statistics compares against other MODELS, the substitution check against a declared
    DIGEST or FINGERPRINT. Neither is expressible as a `Dataset`. Splitting the parameter
    is what lets each plug-in declare the reference capability it actually needs and
    resolve UNAVAILABLE independently, rather than as one undifferentiated missing
    reference. **Both references — and everything else a plug-in is handed — now arrive
    inside `CheckContext`**, so the signature is `check(model, ctx)`; the argument above is
    unchanged and is why `ctx` carries `probes_x`, `battery` and `suspect_x` as separate
    typed fields rather than one `reference`.

**These Protocols were stale, and nothing caught it** (item 17). `Detector.detect` was
declared 3-arg while every detector in the tree runs 4-arg with `ctx`, and `ModelCheck.check`
still named `reference_probes` / `reference_battery`. Nothing imported this file at all —
the Protocols are `@runtime_checkable` and no `isinstance` check existed — so the drift was
invisible until `core/drift_scan.py` became the first real consumer. There is now a test
(`tests/core/test_interfaces_match_reality.py`) asserting every registered plug-in's
signature against its Protocol, so the next drift is a failure rather than a discovery.
  * **`DriftTest` declares capabilities and receives images** — it previously had neither,
    yet Module D is precisely the module that needs `REFERENCE_CLEAN_SET`, and the
    interpretable axes (brightness, RMS contrast, Laplacian sharpness, JPEG quality, EXIF
    sensor/ISO) are computed FROM IMAGES AND FILE METADATA, not from embeddings.

`core/` defines these and `detectors/` implements them. That is the direction CI invariant
2 requires; the reverse would kill the plug-in architecture with every test still green.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from .capability import Capability
from .context import CheckContext
from .types import Dataset, EmbeddingIndex, Finding, ModelHandle


@runtime_checkable
class DatasetLoader(Protocol):
    id: ClassVar[str]

    def supports(self, path: Path) -> bool: ...
    def load(self, path: Path) -> Dataset: ...


@runtime_checkable
class ModelLoader(Protocol):
    id: ClassVar[str]

    def supports(self, path: Path) -> bool: ...
    def load(self, path: Path) -> ModelHandle: ...


class _PlugIn(Protocol):
    """What every analysis plug-in declares, and what the orchestrator resolves.

    `requires`/`optional` are resolved against the scan-level CapabilitySet BEFORE
    anything runs — that is what makes coverage known at minute zero rather than at the
    end of a long scan. `attack_classes` must be a subset of `core/taxonomy.py`, checked
    at startup; an undeclared string makes the plug-in's coverage row silently vanish.
    """

    id: ClassVar[str]
    version: ClassVar[str]
    # Widened to match `registry.Registrable`, which is what the orchestrator actually
    # resolves against. Declaring `frozenset` here while the registry accepted
    # `frozenset | set | tuple` meant the two descriptions of one contract disagreed.
    requires: ClassVar[frozenset[Capability] | set[Capability] | tuple[Capability, ...]]
    optional: ClassVar[frozenset[Capability] | set[Capability] | tuple[Capability, ...]]
    attack_classes: ClassVar[frozenset[str] | set[str] | tuple[str, ...]]


@runtime_checkable
class Detector(_PlugIn, Protocol):
    """Module A — training-data integrity (PS §2.2.1)."""

    def detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None,
               model: ModelHandle | None, ctx: CheckContext) -> list[Finding]: ...


@runtime_checkable
class ModelCheck(_PlugIn, Protocol):
    """Module B — model integrity (PS §2.2.2)."""

    def check(self, model: ModelHandle, ctx: CheckContext) -> list[Finding]: ...


@runtime_checkable
class DriftTest(_PlugIn, Protocol):
    """Module D — distribution shift (PS §2.2.4)."""

    def assess(self, reference: Dataset | None, incoming: Dataset,
               reference_dist: Any | None, incoming_dist: Any | None) -> list[Finding]: ...


# --- Remediator ------------------------------------------------------------
# [added] §5.13. ADR-004's amendment puts remediation in scope, but the frozen contract had
# no interface for it, so the amendment had nowhere to land.
#
# The boundary, quoted from the ADR: "a Detector or ModelCheck plug-in may never retrain.
# A Remediator is a separate, explicitly-invoked component that can, and its output is
# re-assessed rather than trusted."
#
# Enforced STRUCTURALLY, not by convention — three mechanisms:
#   1. `Remediator` is a separate interface, never a Detector/ModelCheck subtype, so no
#      baseline plug-in can be registered as one or vice versa.
#   2. The orchestrator's baseline path never constructs one. It is reached only via
#      `cva remediate`, a distinct entry point.
#   3. Its output is a NEW ARTEFACT WITH A NEW DIGEST that must re-enter `cva scan` to be
#      assessed. The before/after comparison is two scans, not one scan and a claim.
# CI invariant 3 (`detectors/` ↛ `remediation/`) makes rule 1 a build failure.


class RemovalRecord(Protocol):
    sample_id: str
    reason: str
    finding_id: str      # WHICH finding justified this removal


@runtime_checkable
class RemediationResult(Protocol):
    kind: str                       # "dataset" | "model"
    artefact_path: Path
    artefact_digest: str            # a NEW artefact with its OWN digest
    manifest: Sequence[RemovalRecord]
    source_scan_id: str


@runtime_checkable
class Remediator(Protocol):
    requires: ClassVar[frozenset[Capability]]
    optional: ClassVar[frozenset[Capability]]

    def remediate(self, dataset: Dataset, findings: Sequence[Finding],
                  model: ModelHandle | None) -> RemediationResult: ...


__all__ = [
    "DatasetLoader", "Detector", "DriftTest", "ModelCheck", "ModelLoader",
    "RemediationResult", "Remediator", "RemovalRecord",
]
