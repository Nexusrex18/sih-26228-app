"""Shared plumbing for Module A detectors.

Three rules this file exists to enforce:

1. **Module A has its own registry** (``DATA_REGISTRY``). ``cva.detectors.base.REGISTRY`` is
   Module B's — the orchestrator instantiates every entry and calls ``.check(model, ctx)``, and
   a ``.detect(dataset, embeddings, model)`` detector in it would break Module B's scan.
2. **One ``Finding`` type** — ``cva.core.finding.Finding``. ``finding_id`` is derived per
   ``backend_plan.md`` §5.3 (never hashes ``scan_id``); evidence paths are content-addressed.
3. **Detectors never decide policy.** ``disposition`` is a placeholder for the risk engine;
   ``confidence`` is the detector's own uncalibrated belief; ``score_raw`` / ``threshold`` are
   what was actually measured / applied so recalibration needs no re-run.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.finding import Disposition, Evidence, Finding, Nature, Severity
from ._stub_types import Dataset, Sample

DATA_REGISTRY: dict[str, type] = {}


def register_data(cls):
    if cls.id in DATA_REGISTRY:
        raise ValueError(f"duplicate data detector id {cls.id!r}")
    DATA_REGISTRY[cls.id] = cls
    return cls


PENDING_RULE = "pending_risk_engine"


def finding_id(detector_id: str, detector_version: str, target_type: str, target_ref: str,
               attack_class: str) -> str:
    """sha256(detector_id‖detector_version‖target_type‖target_ref‖attack_class)[:16].
    Must NOT include scan_id (§5.3): the reproducibility test expects two runs to differ only in
    scan_id and created_at_utc. ``\\x1f`` separates fields so 'ab'+'c' != 'a'+'bc'."""
    h = hashlib.sha256("\x1f".join(
        (detector_id, detector_version, target_type, target_ref, attack_class)).encode())
    return h.hexdigest()[:16]


class Params:
    """Thresholds live in a dict, never inline. Unknown keys are an error, not a warning — a
    typo that silently falls back to a default is indistinguishable from a tuned system."""

    def __init__(self, defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None = None):
        unknown = set(overrides or {}) - set(defaults)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; valid: {sorted(defaults)}")
        self._p = {**defaults, **(overrides or {})}

    def __getitem__(self, k: str) -> Any:
        return self._p[k]

    def as_dict(self) -> dict[str, Any]:
        return dict(self._p)


class EvidenceStore:
    """Writes evidence artefacts content-addressed: ``evidence/<sha256>.<ext>``. With no
    directory it degrades to inline payloads (or no image), never to a scan-id-prefixed path."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else None

    def _put(self, data: bytes, ext: str) -> str:
        assert self.root is not None
        d = self.root / "evidence"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{hashlib.sha256(data).hexdigest()}.{ext}"
        p = d / name
        if not p.exists():
            p.write_bytes(data)
        return f"evidence/{name}"

    def png(self, img, caption: str, kind: str = "contact_sheet") -> Evidence | None:
        if self.root is None:
            return None
        import io
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Evidence(kind, caption, self._put(buf.getvalue(), "png"))

    def json(self, obj: Any, caption: str, kind: str = "table") -> Evidence:
        payload = json.dumps(obj, sort_keys=True, default=_jsonable)
        if self.root is None:
            return Evidence(kind, caption, None, json.loads(payload))
        ev = Evidence(kind, caption, self._put(payload.encode(), "json"), json.loads(payload))
        return ev


def _jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def contact_sheet(dataset: Dataset, sample_ids: Sequence[str], tile: int = 96, cols: int = 6,
                  max_tiles: int = 24):
    """A grid of the named samples' images — the evidence artefact for cluster-style findings."""
    from PIL import Image

    ids = list(sample_ids)[:max_tiles]
    rows = max(1, -(-len(ids) // cols))
    sheet = Image.new("RGB", (cols * tile, rows * tile), (24, 24, 24))
    for i, sid in enumerate(ids):
        with Image.open(dataset.sample(sid).path) as im:
            t = im.convert("RGB").resize((tile - 4, tile - 4))
        sheet.paste(t, ((i % cols) * tile + 2, (i // cols) * tile + 2))
    return sheet


def make_finding(*, detector_id: str, version: str, target_type: str, target_ref: str,
                 severity: Severity, confidence: float, score_raw: float, threshold: float,
                 reason: str, attack_class: str, nature: Nature,
                 evidence: Iterable[Evidence | None] = (), access_assumptions: Iterable[str] = (),
                 limitations: Iterable[str] = (),
                 availability: Availability = Availability.OK) -> Finding:
    return Finding(
        detector_id=detector_id, detector_version=version, target_type=target_type,  # type: ignore[arg-type]
        target_ref=target_ref, severity=severity,
        confidence=float(min(1.0, max(0.0, confidence))),
        reason=reason, attack_class=attack_class,
        finding_id=finding_id(detector_id, version, target_type, target_ref, attack_class),
        score_raw=float(score_raw), threshold=float(threshold),
        evidence=[e for e in evidence if e is not None],
        access_assumptions=list(access_assumptions), limitations=list(limitations),
        disposition=Disposition.REVIEW, disposition_rule=PENDING_RULE,
        nature=nature, availability=availability)


def not_performed(detector_id: str, version: str, attack_classes: Iterable[str], reason: str,
                  state: Availability = Availability.UNAVAILABLE) -> Finding:
    """For inputs the capability enum cannot express (e.g. ``embeddings is None``): still a
    Finding naming what was missing — never a silent skip."""
    classes = sorted(attack_classes)
    return make_finding(
        detector_id=detector_id, version=version, target_type="dataset", target_ref="dataset",
        severity=Severity.INFO, confidence=0.0, score_raw=0.0, threshold=0.0,
        reason=f"Assessment not performed: {reason}",
        attack_class=classes[0] if classes else "unknown", nature=Nature.INDETERMINATE,
        limitations=[f"Attack classes NOT assessed in this scan: {', '.join(classes)}"],
        availability=state)


# --- label / contributor helpers ---------------------------------------------------------

def dominant_category(sample: Sample) -> int | None:
    """The image-level label for label-consistency work: the most frequent category among the
    sample's labels (ties -> lowest id). Classification datasets have exactly one. For
    multi-object detection images this is a documented simplification (see detector
    ``limitations``)."""
    if not sample.labels:
        return None
    c = Counter(lb.category_id for lb in sample.labels)
    top = max(c.values())
    return min(k for k, v in c.items() if v == top)


_TIER_TEXT = {
    "sidecar": "explicit contributors.yaml sidecar",
    "directory": "directory structure",
    "format_field": "dataset-format field",
    "exif_cluster": "EXIF camera-serial clustering (a HYPOTHESIS, not a fact)",
    "none": "no resolution source",
}


def attribution(sample_or_source: Sample | str | None, contributor: str | None = None) -> str:
    """The clause every contributor-naming ``reason`` must carry: WHICH tier resolved the
    identity. An ``exif_cluster`` attribution reads as a hypothesis, never a fact."""
    if isinstance(sample_or_source, Sample):
        contributor = sample_or_source.contributor
        src = sample_or_source.contributor_source
    else:
        src = sample_or_source
    if contributor is None:
        return "contributor unresolved"
    tier = _TIER_TEXT.get(src or "none", "unknown resolution source")
    return f"contributor {contributor!r} (attribution from {tier})"


def group_source(dataset: Dataset, contributor: str) -> str | None:
    """The weakest resolution tier among a contributor's samples — the honest one to report."""
    order = ["sidecar", "directory", "format_field", "exif_cluster", "none"]
    srcs = {s.contributor_source for s in dataset.samples if s.contributor == contributor}
    srcs.discard(None)
    return max(srcs, key=order.index) if srcs else None


def load_rgb(sample: Sample):
    from PIL import Image

    with Image.open(sample.path) as im:
        return im.convert("RGB")


def to_model_input(sample: Sample, input_shape: Sequence[int]) -> np.ndarray:
    """Sample image -> float32 CHW in [0,1] at the model's input size."""
    from PIL import Image

    c, h, w = (int(v) for v in input_shape[-3:])
    im = load_rgb(sample).resize((w, h), Image.BILINEAR)
    if c == 1:
        im = im.convert("L")
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a[None] if c == 1 else a.transpose(2, 0, 1)
