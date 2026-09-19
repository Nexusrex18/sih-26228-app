"""Shared plumbing for Module A detectors.

Rules this file exists to enforce:

1. **Registration is core's** — detectors use ``@register_detector`` (``cva.core.registry``,
   ``DETECTOR_REGISTRY``), the same registry ``model.data_consistency`` uses, with the
   ``detect(dataset, embeddings, model, ctx)`` signature. Registries are never invented here.
2. **Zero-argument construction.** The orchestrator instantiates plug-ins with no arguments
   (``REGISTRY[check_id]()``), so *everything* a detector needs at run time arrives through
   ``ctx`` (``CheckContext``): thresholds via ``ctx.profile``, the evidence directory via
   ``ctx.out_dir``, ``ctx.scan_id``, ``ctx.rng_seed``. Constructor-injected state is unreachable
   in a real scan and is not used.
3. **One ``Finding`` type** — ``cva.core.types.Finding``. ``finding_id`` is derived per
   ``backend_plan.md`` §5.3 (never hashes ``scan_id``); evidence paths are content-addressed.
4. **Detectors never decide policy.** ``disposition`` is a placeholder for the risk engine;
   ``confidence`` is the detector's own uncalibrated belief; ``score_raw`` / ``threshold`` are
   what was actually measured / applied so recalibration needs no re-run.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cva.core.capability import Availability
from cva.core.context import CheckContext
from cva.core.registry import DETECTOR_REGISTRY, register_detector  # noqa: F401  (re-exported)
from cva.core.types import Disposition, Evidence, Finding, Nature, Severity

from ._stub_types import Dataset, Sample

PENDING_RULE = "pending_risk_engine"


def finding_id(detector_id: str, detector_version: str, target_type: str, target_ref: str,
               attack_class: str) -> str:
    """sha256(detector_id‖detector_version‖target_type‖target_ref‖attack_class)[:16].
    Must NOT include scan_id (§5.3): the reproducibility test expects two runs to differ only in
    scan_id and created_at_utc. ``\\x1f`` separates fields so 'ab'+'c' != 'a'+'bc'."""
    h = hashlib.sha256("\x1f".join(
        (detector_id, detector_version, target_type, target_ref, attack_class)).encode())
    return h.hexdigest()[:16]


def as_ctx(ctx: CheckContext | None) -> CheckContext:
    """``ctx`` is required by the plug-in signature; a bare default keeps a detector usable
    standalone (tests, notebooks) without weakening the contract the orchestrator relies on."""
    return ctx if ctx is not None else CheckContext()


class Params:
    """Thresholds live in a dict, never inline. Unknown keys are an error, not a warning — a
    typo that silently falls back to a default is indistinguishable from a tuned system.

    ``ctx.profile`` is shared by every plug-in, so strictness is scoped: a detector reads only
    its OWN namespace, ``ctx.profile[<detector_id>]`` (a dict), and rejects unknown keys inside
    it. Other plug-ins' keys are none of its business."""

    def __init__(self, defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None = None):
        unknown = set(overrides or {}) - set(defaults)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; valid: {sorted(defaults)}")
        self._p = {**defaults, **(overrides or {})}

    @classmethod
    def from_ctx(cls, detector_id: str, defaults: Mapping[str, Any],
                 ctx: CheckContext | None) -> Params:
        return cls(defaults, (ctx.profile.get(detector_id) if ctx is not None else None) or {})

    def __getitem__(self, k: str) -> Any:
        return self._p[k]

    def as_dict(self) -> dict[str, Any]:
        return dict(self._p)


def seed_of(p: Params, ctx: CheckContext | None) -> int:
    """An explicit ``seed`` in the detector's profile namespace wins; else the scan's seed."""
    s = p["seed"] if "seed" in p.as_dict() else None
    return int(s) if s is not None else int(as_ctx(ctx).rng_seed)


def finalise(findings: list[Finding], ctx: CheckContext | None) -> list[Finding]:
    """Stamp ``scan_id`` (only the orchestrator's context knows it). ``finding_id`` is unaffected
    by design — it never hashes ``scan_id``."""
    sid = as_ctx(ctx).scan_id
    for f in findings:
        f.scan_id = sid
    return findings


def sentence_case(text: str) -> str:
    """Upper-case ONLY the first character. ``str.capitalize()`` lower-cases the rest, which
    silently rewrote contributor ids ('B' -> 'b') and flattened the HYPOTHESIS flag."""
    return text[:1].upper() + text[1:]


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

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def sheet(self, dataset, sample_ids, caption: str) -> Evidence | None:
        """Contact sheet, built ONLY if there is somewhere to put it (decoding up to 24 images
        just to throw the result away is not free)."""
        if self.root is None:
            return None
        return self.png(contact_sheet(dataset, sample_ids), caption)

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
