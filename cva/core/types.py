"""Finding — the integration contract. Every plug-in writes it; the report reads only it.

Severity and confidence are separate on purpose: severity is impact, confidence is belief.
A high-confidence low-severity finding and a low-confidence high-severity one call for
opposite actions and one float cannot distinguish them.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from .capability import Availability, Capability


class Severity(str, Enum):
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
    """PS 2.1 — data may be mislabelled 'deliberately or inadvertently'."""

    ADVERSARIAL = "adversarial"
    QUALITY = "quality"
    INDETERMINATE = "indeterminate"


EvidenceKind = Literal[
    "image_crop", "contact_sheet", "heatmap", "plot", "table", "hash", "json"
]


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    caption: str
    path: str | None = None          # relative to the report dir
    data: Any = None                 # inline table/json payload


@dataclass
class Finding:
    detector_id: str
    detector_version: str
    target_type: Literal["sample", "contributor", "batch", "model", "record", "dataset"]
    target_ref: str
    severity: Severity
    confidence: float
    reason: str
    attack_class: str
    scan_id: str = ""
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    score_raw: float = 0.0
    threshold: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    access_assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    disposition: Disposition = Disposition.REVIEW
    disposition_rule: str = ""
    nature: Nature = Nature.INDETERMINATE
    availability: Availability = Availability.OK
    produced_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("severity", "disposition", "nature", "availability"):
            d[k] = getattr(self, k).value
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


# --- attack-class taxonomy -------------------------------------------------
# Kept as a flat registry so the coverage statement is GENERATED from what
# detectors declare, never hand-written. Aligned to NIST AI 100-2 vocabulary
# where it maps; see Consolidated/18-Corpora-and-External-Validation.md.
ATTACK_CLASSES: dict[str, dict[str, str]] = {
    # Ratified table — Plan/Module-B-Model-Integrity-Plan.md §5 (2026-09-19).
    # `kind` matters: the coverage generator counts `attack` ONLY. An `operational`
    # entry is a true report about the system's own state, not an attack class we
    # claim to detect, and counting it would inflate the statement.
    "model_substitution":     {"kind": "attack",      "nature": "adversarial",
                               "desc": "Supplied model is not the declared model"},
    "benign_conversion":      {"kind": "operational", "nature": "quality",
                               "desc": "Digest mismatch, fingerprint inside tolerance — "
                                       "re-export, opset change or quantisation"},
    "weight_anomaly":         {"kind": "attack",      "nature": "indeterminate",
                               "desc": "Per-layer weight distribution diverges from battery"},
    "activation_anomaly":     {"kind": "attack",      "nature": "indeterminate",
                               "desc": "Activation distribution anomaly on clean probes"},
    "backdoor_trigger":       {"kind": "attack",      "nature": "adversarial",
                               "desc": "Trigger-conditioned backdoor"},
    "architectural_backdoor": {"kind": "attack",      "nature": "adversarial",
                               "desc": "Malice in the graph, not the weights"},
    "model_anomalous":        {"kind": "attack",      "nature": "indeterminate",
                               "desc": "Corrupted, truncated or badly trained — PS 2.2.2's "
                                       "third verdict, not classifiable as the above"},
    "data_model_discrepancy": {"kind": "attack",      "nature": "indeterminate",
                               "desc": "Model behaviour the supplied training data cannot explain"},
    "unsafe_artifact":        {"kind": "attack",      "nature": "adversarial",
                               "desc": "Model file unsafe to load (deserialisation / custom op)"},
    "tool.error":             {"kind": "operational", "nature": "indeterminate",
                               "desc": "A check raised — a defect in the assurance tool"},
}


def attack_classes_of_kind(kind: str) -> set[str]:
    return {k for k, v in ATTACK_CLASSES.items() if v["kind"] == kind}


def unavailable_finding(
    detector_id: str,
    version: str,
    target_ref: str,
    reason: str,
    missing: tuple[Capability, ...],
    attack_class: str,
    state: Availability = Availability.UNAVAILABLE,
) -> Finding:
    """UNAVAILABLE is not a skip — it is a Finding naming exactly what was missing."""
    return Finding(
        detector_id=detector_id,
        detector_version=version,
        target_type="model",
        target_ref=target_ref,
        severity=Severity.INFO,
        confidence=0.0,
        reason=f"Assessment not performed: {reason}",
        attack_class=attack_class,
        evidence=[Evidence("json", "missing capabilities", data=[str(m) for m in missing])],
        access_assumptions=[f"missing: {', '.join(str(m) for m in missing)}"] if missing else [],
        limitations=["This attack class was NOT assessed in this scan."],
        disposition=Disposition.REVIEW,
        disposition_rule="capability.unavailable",
        availability=state,
    )


def write_findings(findings: list[Finding], path: Path) -> None:
    path.write_text(json.dumps([f.to_dict() for f in findings], indent=2))
