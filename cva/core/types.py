"""The frozen contract. `Architecture/Plugin-Interfaces.md` is its spec; V4 asserts the
two agree field-for-field, and this module is the reason that test can exist.

**Frozen from this commit**, per backend_plan.md §5.11 — not at the end of P0. Four seats
are being told to build against these types today; if they can still move until the end of
P0, every seat building today is building on sand and the freeze protects nobody during
the only window in which it matters. After this commit a change to any type here needs
whole-team agreement.

Five amendments land with it, decided on Backend's authority (§5 preamble). All five are
ADDITIVE — no field is removed or retyped — so nothing already written against the
contract breaks:

  §5.9   Finding.nature              adversarial | quality | indeterminate   (PS audit G6)
  §5.5   Finding.exclusion_reason    capability | budget | None
  §5.10  Sample.contributor_source   which of the five precedence tiers resolved it
  §5.2   AuditLedger / InferenceLedgerSource   (core/ledger.py — two protocols, not one)
  §5.13  Remediator                  (core/interfaces.py)

and three more this seat adds for shapes the code needs and no vault file gave:
`Finding.availability`, `Evidence.data`, and `Capability.SUSPECT_INPUTS` — each argued at
its definition below.

The severity/confidence split is the one thing to not "simplify". The PS's "confidence or
severity" is a floor, not a design: severity is impact, confidence is belief, and a
high-confidence low-severity finding and a low-confidence high-severity one call for
opposite actions. One float cannot distinguish them.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .capability import Availability, Capability, CapabilitySet

if TYPE_CHECKING:  # numpy is not a core/ runtime dependency — only a typing one
    import numpy as np


class Severity(str, Enum):
    """Impact. Frozen at five — `warning` is NOT a member (backend_plan.md §9.5)."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ["info", "low", "medium", "high", "critical"].index(self.value)


class Disposition(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    QUARANTINE = "quarantine"


class Nature(str, Enum):
    """[added] §5.9 — PS audit G6.

    PS §2.1 says data may be mislabelled *"deliberately or inadvertently."* Our
    dispositions express SEVERITY OF RESPONSE, not NATURE OF CAUSE. Without this field a
    contributor supplying duplicates and OOD imagery — a collection-quality problem — and
    a contributor injecting triggers collapse into the same `quarantine` bucket, losing
    precisely the distinction an acceptance officer most needs, because the two have
    entirely different remedies.

    `indeterminate` is the honest default and most detectors will emit it. The field earns
    its place on the two that can tell the difference.
    """

    ADVERSARIAL = "adversarial"
    QUALITY = "quality"
    INDETERMINATE = "indeterminate"


class ExclusionReason(str, Enum):
    """[added] §5.5.

    The temptation was a fifth `Availability` state for budget exclusion. Resisted:
    `Availability` is frozen at four and every consumer — renderer, coverage generator,
    the dashboard Network is already mocking — switches on exactly those four. A fifth
    state breaks all of them for information that is not a *state* at all but an
    *attribute of one*. `UNAVAILABLE` continues to mean "this did not run and here is
    why"; this says which kind of why.
    """

    CAPABILITY = "capability"
    BUDGET = "budget"


class ContributorSource(str, Enum):
    """[added] §5.10 — which of `Plugin-Interfaces.md:75-80`'s five precedence tiers
    actually resolved `Sample.contributor`.

    The spec requires that *"the report should say which level was used"* and no field
    carried it, so the report could not. A contributor flagged on EXIF clustering and one
    flagged from a signed sidecar are not the same claim — the first is a DERIVED PROXY
    the report must label a hypothesis, the second is auditable evidence — and the party
    being flagged is likely another unit. Same reason `disposition_rule` exists: the claim
    must carry its own provenance.
    """

    SIDECAR = "sidecar"
    DIRECTORY = "directory"
    FORMAT_FIELD = "format_field"
    #: Tier 4. **Defined here and set by NO loader** — audit item 24. It stays because the
    #: contract is frozen and lists it, and because deleting it would remove the vocabulary
    #: for a tier the precedence table ratified. What was NOT acceptable was leaving it to
    #: resolve silently never: the coverage statement now carries a standing limitation
    #: saying tier 4 is skipped, so the absence is declared rather than discovered. Module
    #: A's `metadata_anomaly` clustering is a different mechanism that shares the word
    #: "clustering" and does not populate this.
    EXIF_CLUSTER = "exif_cluster"
    NONE = "none"


EvidenceKind = Literal[
    "image_crop", "contact_sheet", "heatmap", "plot", "table", "hash", "json"
]

TargetType = Literal["sample", "contributor", "batch", "model", "record", "dataset"]


@dataclass(frozen=True)
class Evidence:
    """`{kind, path, caption}` per the frozen shape, plus `data` as an additive amendment.

    **`path` is the BARE CONTENT HASH**, not a scan-relative path — §5.3. Content
    addressing exists so an identical crop is stored once; a per-scan `evidence/` folder
    stores it once *per scan that references it*, and worse, forces `Evidence.path` to be
    scan-relative, re-coupling it to `scan_id` — the exact volatile field the rule removes.
    Both the renderer and the dashboard resolve it against `<out_dir>/evidence/`.

    **`data` [added]** is the in-memory payload for `table`/`json`/`hash` evidence, and it
    is TRANSPORT, not the record: `EvidenceStore.materialise` writes it, hashes it and
    fills `path`. Without it a detector would have to know the output directory to emit a
    two-line table, which puts filesystem layout inside every plug-in.

    This is also the mechanism for per-detector diagnostics that are not `Finding` fields
    (§7.1) — Module C's `primary_check` and `cascade_suppressed` travel as
    `Evidence(kind="json", caption="primary_check")`. There is deliberately NO extension
    dict on `Finding`: one would become an untyped side-channel that Module E cannot
    render and the schema cannot validate.
    """

    kind: EvidenceKind
    caption: str
    path: str | None = None
    data: Any = None


def derive_finding_id(detector_id: str, detector_version: str, target_type: str,
                      target_ref: str, attack_class: str) -> str:
    """§5.3 — scan-INVARIANT by construction.

    If `finding_id` hashed `scan_id`, every finding would become volatile and V9's diff
    would be worthless: two runs of the same scan would differ in every id, so the
    reproducibility test could assert nothing.
    """
    joined = "␟".join(
        (detector_id, detector_version, target_type, target_ref, attack_class))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class Finding:
    """The one shape every plug-in returns, and the only thing Module E must understand.

    A plug-in that cannot run under the available capabilities STILL RETURNS ONE — see
    `Availability`. `UNAVAILABLE` is not a skip; its evidence names exactly what was
    missing and why. No silent skips, ever.

    The rule governing every `reason` template, enforced at review on every plug-in:

        WRONG  "Suspicious sample (score 0.87)"
        RIGHT  "214 samples from Contributor B share a 6x6 high-frequency anomaly at the
                bottom-right; occluding that region flips 91% of their predictions from
                'truck' to 'tank'. 5 examples shown."

    The first is a number wearing a finding's clothes. If a template cannot be written in
    the second style, the detector is not producing evidence and should emit `info`.
    """

    detector_id: str
    detector_version: str
    target_type: TargetType
    target_ref: str
    severity: Severity
    confidence: float
    reason: str
    attack_class: str

    # --- orchestrator-filled; a detector cannot know these (§7.7b) ---------
    scan_id: str = ""            # VOLATILE — §5.3
    produced_by: str = ""        # code commit + profile hash

    finding_id: str = ""         # derived in __post_init__; never a uuid
    score_raw: float = 0.0
    threshold: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    access_assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    # --- risk-engine-filled ------------------------------------------------
    disposition: Disposition = Disposition.REVIEW
    disposition_rule: str = ""

    nature: Nature = Nature.INDETERMINATE                    # [added] §5.9
    exclusion_reason: ExclusionReason | None = None          # [added] §5.5

    # [added, Backend] The resolution state this finding was produced under. The frozen
    # shape has no per-finding availability, yet §7.4 requires all four states to print at
    # EQUAL PROMINENCE — and joining findings back to plan rows by `detector_id` fails the
    # moment one detector emits both an OK finding and a DEGRADED note. `exclusion_reason`
    # does not cover it either: it only qualifies UNAVAILABLE.
    availability: Availability = Availability.OK

    def __post_init__(self) -> None:
        if not self.finding_id:
            self.finding_id = derive_finding_id(
                self.detector_id, self.detector_version, str(self.target_type),
                self.target_ref, self.attack_class)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("severity", "disposition", "nature", "availability"):
            d[k] = getattr(self, k).value
        d["exclusion_reason"] = (self.exclusion_reason.value
                                 if self.exclusion_reason else None)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


# --- dataset-side types ----------------------------------------------------
# §7.7b: all five of these appear in Plugin-Interfaces.md's signatures and in three plans'
# contract sections, and NONE was ever given a shape in any document. ML-1 could not
# hand-construct a stub Dataset without Label, nor stub an EmbeddingDistribution for the
# PSI tests — which silently blocked both module plans' stated day-one build path.


@dataclass(frozen=True)
class Category:
    category_id: int      # CANONICAL: dense, 0-based, assigned by the loader
    name: str
    source_id: int | str  # the id exactly as it appeared in the source file

    # `source_id` is the load-bearing field. COCO `category_id` values are arbitrary
    # integers and need not be contiguous; YOLO indices are dense and 0-based. The loader
    # PRODUCES that mapping — it is data, not a constant — and it must survive into the
    # report, because "a finding that says 'class 3' means nothing if two loaders disagree
    # about what 3 is." Without this field the mapping has nowhere to live.


@dataclass(frozen=True)
class Label:
    category_id: int                                  # indexes Dataset.categories
    bbox: tuple[float, float, float, float] | None = None
    # COCO convention: (x_min, y_min, w, h), ABSOLUTE PIXELS — never normalised.
    # Normalising at ingest throws away the pixel grid, and the pixel grid is exactly what
    # Evidence(kind="image_crop") needs.
    iscrowd: bool = False
    segmentation: tuple[tuple[float, ...], ...] | None = None
    label_id: str | None = None                       # stable within a Sample, for evidence refs


@dataclass(frozen=True)
class Annotation:
    """The DERIVED flat view — deliberately OUTSIDE the freeze (§5.5 #5), so its shape can
    change without whole-team agreement. Drawing the freeze line here is the whole point:
    two stores can disagree, a store and a projection cannot."""

    annotation_id: str
    sample_id: str
    category_id: int
    bbox: tuple[float, float, float, float] | None
    iscrowd: bool


@dataclass
class Sample:
    sample_id: str
    content_sha256: str          # also half the feature-cache key
    path: Path
    width: int
    height: int                  # width/height MANDATORY — the YOLO direction is
                                 # unrecoverable without them
    labels: list[Label] = field(default_factory=list)   # the ONLY stored representation
    contributor: str | None = None   # None = genuinely unknown. NEVER a default.
    batch: str | None = None
    source_meta: dict[str, Any] = field(default_factory=dict)
    contributor_source: ContributorSource | None = None   # [added] §5.10

    # `contributor` lives here, not on Dataset. A dataset-level dict cannot answer "which
    # contributor sent sample 4471?" — blocking PS §2.2.1's source-level aggregation, the
    # one requirement that differentiates this submission from any off-the-shelf
    # data-quality tool. `None` rather than a default is load-bearing: a default would let
    # the rollup produce a confident-looking single-contributor report from a dataset
    # carrying no contributor data at all.


@runtime_checkable
class Dataset(Protocol):
    samples: list[Sample]
    categories: list[Category]

    def annotations(self) -> list[Annotation]: ...
    def capabilities(self) -> CapabilitySet: ...   # DATASET_* only; PROBED, not assumed


# --- model-side types ------------------------------------------------------
# ModelHandle / ModelBattery / Manifest live in core/model.py, which predates this file
# and is re-exported here so `from cva.core.types import *` is the single import a
# plug-in author needs.
from .model import Manifest, ModelBattery, ModelHandle  # noqa: E402


@runtime_checkable
class EmbeddingIndex(Protocol):
    """§7.7b.

    `patch_tokens` is not optional garnish. ADR-009 chose DINOv2 over a pooled-only
    backbone on criterion 2 — dense per-patch features — precisely because Module A's
    patch-saliency scan needs per-region representations, and the ADR is explicit that a
    pooled-only model *"would have quietly cost us the single most compelling evidence
    artifact in the system."* An index exposing only pooled vectors would throw that away
    AT THE TYPE LEVEL, after the ADR spent its argument winning it.
    """

    extractor_id: str        # pins the exact checkpoint BYTE STRING, not a name (ADR-009)
    extractor_version: str
    dim: int

    def vector(self, sample_id: str) -> np.ndarray | None: ...
    def vectors(self, sample_ids: Sequence[str]) -> np.ndarray: ...
    def knn(self, sample_id: str, k: int) -> list[tuple[str, float]]: ...   # (id, cosine)
    def patch_tokens(self, sample_id: str) -> np.ndarray | None: ...        # (n_patches, dim)


@dataclass(frozen=True)
class EmbeddingDistribution:
    """§7.7b.

    `n` is on the type because the CORRECTED PSI null needs it. The published result is
    `PSI ~ (1/N + 1/M)·chi2(B-1)` — both sample sizes appear in the critical value, and the
    scale factor is the entire content of that result (Post-Merge R1). A distribution
    object carrying only mean/cov/hist would force Module D to thread N and M alongside
    it, and the first time someone forgot, the test would silently revert to the
    sample-size-blind form. Putting `n` on the object makes the corrected formula the path
    of least resistance.
    """

    extractor_id: str
    extractor_version: str
    n: int
    mean: np.ndarray
    cov: np.ndarray | None = None          # None when n < dim (singular); Module D falls back
    bin_edges: np.ndarray | None = None    # per-axis, for PSI
    hist: np.ndarray | None = None         # bin proportions aligned to bin_edges


# Floats are fine on all of these: NONE of them is ever hashed. The §7.7 quantisation rule
# applies to the seal record only.


def unavailable_finding(
    detector_id: str,
    version: str,
    target_ref: str,
    reason: str,
    missing: Sequence[Capability],
    attack_class: str,
    state: Availability = Availability.UNAVAILABLE,
    exclusion: ExclusionReason | None = ExclusionReason.CAPABILITY,
    target_type: TargetType = "model",
) -> Finding:
    """UNAVAILABLE is not a skip — it is a Finding naming exactly what was missing.

    `nature` is INDETERMINATE by construction: a missing capability says nothing about
    anyone's intent (§7.4).
    """
    missing = tuple(missing)
    return Finding(
        detector_id=detector_id,
        detector_version=version,
        target_type=target_type,
        target_ref=target_ref,
        severity=Severity.INFO,
        confidence=0.0,
        reason=f"Assessment not performed: {reason}",
        attack_class=attack_class,
        evidence=[Evidence("json", "missing capabilities",
                           data=[str(m) for m in missing])],
        access_assumptions=[f"missing: {', '.join(str(m) for m in missing)}"] if missing else [],
        limitations=["This attack class was NOT assessed in this scan."],
        disposition=Disposition.REVIEW,
        disposition_rule="capability.unavailable",
        nature=Nature.INDETERMINATE,
        exclusion_reason=exclusion,
        availability=state,
    )


def write_findings(findings: Sequence[Finding], path: Path) -> None:
    path.write_text(json.dumps([f.to_dict() for f in findings], indent=2))


__all__ = [
    "Annotation", "Availability", "Capability", "CapabilitySet", "Category",
    "ContributorSource", "Dataset", "Disposition", "EmbeddingDistribution",
    "EmbeddingIndex", "Evidence", "EvidenceKind", "ExclusionReason", "Finding", "Label",
    "Manifest", "ModelBattery", "ModelHandle", "Nature", "Sample", "Severity",
    "TargetType", "derive_finding_id", "unavailable_finding", "write_findings",
]
