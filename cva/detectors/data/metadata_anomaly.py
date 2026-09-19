"""data.metadata_anomaly — batches that were *processed as a set* rather than captured as one.

Reads file metadata only: no model, no embeddings, no pixel decode beyond the header. It is the
one Module A detector that fits the ``triage`` budget tier.

Signals (per group = contributor, else batch, else whole dataset), each judged **against the rest
of the cohort** — a whole dataset produced by one pipeline is not anomalous *relative to itself*:

  exif_stripped     group carries ~no EXIF while the cohort does
  single_camera     one camera model/serial across the group, cohort is varied (a synthetic split)
  pooled_cameras    many cameras inside one group, every other group has ≤1
  quant_table       one JPEG quantisation-table hash across the group (one encoder, one quality)
  encoder_string    EXIF Software / JFIF comment names Pillow / ImageMagick / ffmpeg on
                    imagery claimed to be camera-original
  uniform_size      one exact (w, h) across a group that should vary — a resize pass
  mtime_burst       inter-arrival gaps in milliseconds — a script, not a collection

Severity: ``low`` for 1–2 signals, ``medium`` for ≥3. ``nature`` is ``quality``: a re-encode pass
is usually a pipeline, not an attack. This detector cannot see other detectors' flags, so it never
escalates to ``indeterminate`` itself — that correlation belongs to the risk engine.

Known limitation (also stamped into every finding): metadata is trivially forgeable by anyone who
knows we read it. This catches carelessness and pipeline artefacts, NOT a motivated adversary.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from typing import Any

from PIL import Image

from cva.core.capability import Capability
from cva.core.finding import Nature, Severity
from ._stub_types import Dataset, Sample
from .base import EvidenceStore, Params, attribution, group_source, make_finding, register_data
from .taxonomy import SCRIPT_GENERATED_BATCH

DEFAULTS = {
    "min_group": 20,               # smaller groups give no usable statistics
    "exif_group_max": 0.05,        # group EXIF fraction at/below which it counts as stripped...
    "exif_cohort_min": 0.5,        # ...when the cohort's is at/above this
    "concentration": 0.95,         # share of one value that counts as "uniform"
    "cohort_concentration_max": 0.8,
    "size_cohort_max": 0.5,
    "encoder_group_min": 0.5,
    "encoder_gap_min": 0.4,
    "burst_gap_s": 0.05,
    "burst_share": 0.9,
    "burst_cohort_max": 0.5,
    "pooled_min": 3,
}
_ENCODERS = ("pillow", "imagemagick", "ffmpeg", "libjpeg", "opencv", "gimp")


def _meta(s: Sample) -> dict[str, Any]:
    out: dict[str, Any] = {"dims": (s.width, s.height), "mtime": None, "exif": False,
                           "camera": None, "quant": None, "software": ""}
    try:
        out["mtime"] = os.stat(s.path).st_mtime
        with Image.open(s.path) as im:
            ex = im.getexif()
            out["exif"] = len(ex) > 0
            if out["exif"]:
                serial = ex.get_ifd(0x8769).get(0xA431)
                model = ex.get(0x0110)
                out["camera"] = (str(model), str(serial)) if (model or serial) else None
                out["software"] = str(ex.get(0x0131, "") or "")
            com = im.info.get("comment")
            if com:
                out["software"] += " " + (com.decode("latin-1") if isinstance(com, bytes) else str(com))
            q = getattr(im, "quantization", None)
            if q:
                out["quant"] = hashlib.sha1(repr(sorted((k, tuple(v)) for k, v in q.items())).encode()
                                            ).hexdigest()[:10]
    except Exception:
        pass                                     # unreadable metadata == no signal, never a crash
    return out


def _top(vals: list[Any]) -> tuple[Any, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    v, c = Counter(vals).most_common(1)[0]
    return v, c / len(vals)


def _burst_share(times: list[float], gap: float) -> float:
    t = sorted(x for x in times if x is not None)
    if len(t) < 3:
        return 0.0
    gaps = [b - a for a, b in zip(t, t[1:])]
    return sum(1 for g in gaps if g < gap) / len(gaps)


@register_data
class MetadataAnomaly:
    id = "data.metadata_anomaly"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES}
    optional: set[Capability] = set()
    attack_classes = {SCRIPT_GENERATED_BATCH}

    def __init__(self, params: dict | None = None, evidence_dir=None):
        self.p = Params(DEFAULTS, params)
        self.ev = EvidenceStore(evidence_dir)

    def detect(self, dataset: Dataset, embeddings=None, model=None) -> list:
        p = self.p
        meta = {s.sample_id: _meta(s) for s in dataset.samples}
        groups: dict[tuple[str, str], list[Sample]] = {}
        for s in dataset.samples:
            if s.contributor is not None:
                key = ("contributor", s.contributor)
            elif s.batch is not None:
                key = ("batch", s.batch)
            else:
                key = ("dataset", "dataset")
            groups.setdefault(key, []).append(s)

        findings = []
        for (ttype, gid), members in sorted(groups.items()):
            if len(members) < p["min_group"]:
                continue
            ids = {m.sample_id for m in members}
            cohort = [meta[s.sample_id] for s in dataset.samples if s.sample_id not in ids]
            g = [meta[m.sample_id] for m in members]
            sig = self._signals(g, cohort)
            if not sig:
                continue
            findings.append(self._finding(dataset, ttype, gid, members, sig, len(cohort)))
        return findings

    def _signals(self, g: list[dict], cohort: list[dict]) -> dict[str, dict]:
        p, out = self.p, {}
        n = len(g)
        has_cohort = len(cohort) >= p["min_group"]
        g_exif = sum(m["exif"] for m in g) / n
        if has_cohort:
            c_exif = sum(m["exif"] for m in cohort) / len(cohort)
            if g_exif <= p["exif_group_max"] and c_exif >= p["exif_cohort_min"]:
                out["exif_stripped"] = {"group": g_exif, "cohort": c_exif,
                                        "text": f"{g_exif:.0%} of files carry EXIF (cohort: {c_exif:.0%})"}
            cams = {m["camera"] for m in g if m["camera"]}
            ccams = {m["camera"] for m in cohort if m["camera"]}
            if len(cams) == 1 and len(ccams) >= 3:
                out["single_camera"] = {"camera": next(iter(cams)), "cohort_distinct": len(ccams),
                                        "text": f"every file claims one camera {next(iter(cams))} "
                                                f"(cohort spans {len(ccams)})"}
            qv, qs = _top([m["quant"] for m in g])
            _, cqs = _top([m["quant"] for m in cohort])
            if qv and qs >= p["concentration"] and cqs < p["cohort_concentration_max"]:
                out["quant_table"] = {"hash": qv, "group": qs, "cohort": cqs,
                                      "text": f"{qs:.0%} share JPEG quantisation-table hash {qv} "
                                              f"(cohort's most common: {cqs:.0%})"}
            dv, ds_ = _top([m["dims"] for m in g])
            _, cds = _top([m["dims"] for m in cohort])
            if ds_ >= p["concentration"] and cds < p["size_cohort_max"]:
                out["uniform_size"] = {"dims": dv, "group": ds_, "cohort": cds,
                                       "text": f"{ds_:.0%} are exactly {dv[0]}x{dv[1]} "
                                               f"(cohort's most common size: {cds:.0%})"}
            bs, cbs = _burst_share([m["mtime"] for m in g], p["burst_gap_s"]), \
                _burst_share([m["mtime"] for m in cohort], p["burst_gap_s"])
            if bs >= p["burst_share"] and cbs < p["burst_cohort_max"]:
                out["mtime_burst"] = {"group": bs, "cohort": cbs,
                                      "text": f"{bs:.0%} of inter-file gaps are under "
                                              f"{p['burst_gap_s'] * 1000:.0f} ms (cohort: {cbs:.0%})"}
        gs = sum(any(e in m["software"].lower() for e in _ENCODERS) for m in g) / n
        cs = (sum(any(e in m["software"].lower() for e in _ENCODERS) for m in cohort) / len(cohort)) \
            if cohort else 0.0
        if gs >= p["encoder_group_min"] and gs - cs >= p["encoder_gap_min"]:
            names = Counter(m["software"].strip() for m in g if m["software"].strip())
            top = names.most_common(1)[0][0] if names else "?"
            out["encoder_string"] = {"group": gs, "cohort": cs, "example": top,
                                     "text": f"{gs:.0%} name an image-processing tool as software "
                                             f"({top!r}) (cohort: {cs:.0%})"}
        return out

    def _finding(self, dataset, ttype, gid, members, sig, n_cohort):
        who = (attribution(group_source(dataset, gid), gid) if ttype == "contributor"
               else f"{ttype} {gid!r}")
        lines = "; ".join(v["text"] for v in sig.values())
        reason = (f"{who[0].upper() + who[1:]}: {len(members)} files show {len(sig)} pipeline "
                  f"signature(s) — {lines}. Consistent with a batch that was re-encoded or generated "
                  f"as a set rather than captured; usually a pipeline artefact, not evidence of intent.")
        k = len(sig)
        return make_finding(
            detector_id=self.id, version=self.version, target_type=ttype, target_ref=gid,
            severity=Severity.MEDIUM if k >= 3 else Severity.LOW,
            confidence=min(0.8, 0.35 + 0.15 * k), score_raw=float(k), threshold=1.0,
            reason=reason, attack_class=SCRIPT_GENERATED_BATCH, nature=Nature.QUALITY,
            evidence=[self.ev.json({"group": gid, "n_files": len(members), "signals": sig},
                                   "metadata signals vs cohort")],
            access_assumptions=["DATASET_IMAGES (file metadata only)", "no model", "no embeddings",
                                f"compared against {n_cohort} cohort files outside this group"],
            limitations=[
                "Metadata is trivially forgeable by anyone who knows it is read: this catches "
                "carelessness and pipeline artefacts, NOT a motivated adversary.",
                "Cohort-relative: a whole submission produced by one pipeline is not flagged.",
                "Groups smaller than min_group are not assessed."])
