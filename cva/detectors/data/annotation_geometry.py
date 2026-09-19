"""data.annotation_geometry — boxes shifted or shrunk while the class label stays correct.

``label_consistency`` compares classes, not coordinates, so this passes it untouched.

Method: per (contributor, category), four box features — log area fraction, log aspect ratio,
centre x, centre y (relative to the image) — compared with the same category's boxes from the
REST of the cohort: two-sample KS (p < ``p_max``) AND a median shift of at least ``effect_min``
MADs from EVERY other contributor, in the same direction (so a tampered peer cannot drag the
reference toward the contributor under test; with a single peer the two are symmetric and both
are flagged — declared). Sample z is measured against the nearest peer. For each flagged (contributor, category, feature) the individual boxes lying
``sample_z`` robust-MADs out *in the shifted direction* are flagged.

Findings are **sample-level**, per the plan: cohort comparison happens here, but cross-sample
aggregation into a contributor score is the risk engine's. One finding per sample, merging
features. Severity ``low`` for one feature, ``medium`` for two or more. ``nature`` is
``indeterminate`` (a mis-drawn box is sloppy or deliberate; the geometry cannot say).

Multiplicity: the KS p-value (the WEAKEST across peers) is Bonferroni-corrected over every
(contributor, category, feature) test that had a peer. A contributor is only anomalous if it differs
from every peer in the same direction, so a tampered peer cannot hide it.

Limitations: no generic detector is available to say where a box *should* be, so "anomalous" means
"unlike this category's boxes from other contributors"; a cohort that shares the tampering hides it;
absolute-pixel boxes on mixed-resolution images are normalised by image size.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import ks_2samp

from cva.core.capability import Capability
from cva.core.types import Nature, Severity

from ._stub_types import Dataset
from .base import (
    CheckContext,
    EvidenceStore,
    Params,
    as_ctx,
    attribution,
    finalise,
    make_finding,
    not_performed,
    register_detector,
)
from .taxonomy import ANNOTATION_GEOMETRY_TAMPER

DEFAULTS = {"min_n": 15, "p_max": 1e-3, "effect_min": 1.0, "sample_z": 1.5,
            "mad_floor": {"log_area": 0.05, "log_aspect": 0.03, "cx": 0.02, "cy": 0.02}}
_FEATURES = ("log_area", "log_aspect", "cx", "cy")
_TEXT = {"log_area": "box area", "log_aspect": "box aspect ratio", "cx": "horizontal box position",
         "cy": "vertical box position"}


def _features(sample, lb) -> np.ndarray | None:
    if lb.bbox is None or sample.width <= 0 or sample.height <= 0:
        return None
    x, y, w, h = lb.bbox
    if w <= 0 or h <= 0:
        return None
    return np.array([np.log(w * h / (sample.width * sample.height)), np.log(w / h),
                     (x + w / 2) / sample.width, (y + h / 2) / sample.height])


@register_detector
class AnnotationGeometry:
    id = "data.annotation_geometry"
    version = "1.0.0"
    requires = {Capability.DATASET_LABELS, Capability.DATASET_CONTRIBUTOR_META}
    optional: set[Capability] = set()
    attack_classes = {ANNOTATION_GEOMETRY_TAMPER}

    def detect(self, dataset: Dataset, embeddings, model, ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        self.ctx = as_ctx(ctx)
        return finalise(self._detect(dataset, embeddings, model), ctx)

    def _detect(self, dataset: Dataset, embeddings, model) -> list:
        p = self.p
        by: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
        for s in dataset.samples:
            if s.contributor is None:
                continue
            for lb in s.labels:
                f = _features(s, lb)
                if f is not None:
                    by[(s.contributor, lb.category_id)].append((s.sample_id, f))
        cats = {k[1] for k in by}
        contribs = {k[0] for k in by}
        hits: dict[str, list[dict]] = defaultdict(list)
        usable_by_cat = {cat: {c: np.stack([f for _, f in by[(c, cat)]]) for c in sorted(contribs)
                               if len(by.get((c, cat), [])) >= p["min_n"]} for cat in sorted(cats)}
        # FAMILY-WISE CORRECTION: one KS-per-peer test set per (contributor, category, feature) that
        # has at least one peer; Bonferroni over that whole family (4 features x contributors x K
        # categories — ~1,300+ at COCO K=80 with 4 contributors, invisible at K=4).
        family = sum(len(u) * len(_FEATURES) for u in usable_by_cat.values() if len(u) >= 2)
        if family == 0:
            biggest = max((len(v) for v in by.values()), default=0)
            return [not_performed(
                self.id, self.version, self.attack_classes,
                f"no (contributor, category) group has at least {p['min_n']} boxes together with a "
                f"peer group to compare against ({len(by)} group(s) found, largest has {biggest} boxes)")]
        for cat in sorted(cats):
            usable = usable_by_cat[cat]
            for c, M in usable.items():
                others = {o: A for o, A in usable.items() if o != c}
                if not others:
                    continue
                mine = by[(c, cat)]
                for j, name in enumerate(_FEATURES):
                    # Anomalous only if it differs from EVERY peer, in the same direction — so a
                    # tampered peer cannot drag the reference toward the contributor under test.
                    # Judged against the NEAREST peer (the most conservative one).
                    mm = float(np.median(M[:, j]))
                    peers = []
                    for A in others.values():
                        mo = float(np.median(A[:, j]))
                        sc = max(1.4826 * float(np.median(np.abs(A[:, j] - mo))), p["mad_floor"][name])
                        peers.append((mo, sc, float(ks_2samp(M[:, j], A[:, j]).pvalue)))
                    shifts = [(mm - mo) / sc for mo, sc, _ in peers]
                    if not (all(x >= p["effect_min"] for x in shifts) or all(x <= -p["effect_min"] for x in shifts)):
                        continue
                    if min(1.0, max(pv for _, _, pv in peers) * family) >= p["p_max"]:
                        continue
                    k = int(np.argmin(np.abs(shifts)))
                    med_r, mad_r, pv = peers[k]
                    shift = shifts[k]
                    sign = 1 if shift > 0 else -1
                    z = sign * (M[:, j] - med_r) / mad_r
                    for (sid, _), zi in zip(mine, z, strict=False):
                        if zi >= p["sample_z"]:
                            hits[sid].append({"contributor": c, "category": cat, "feature": name,
                                              "z": float(zi), "group_shift_mads": float(shift),
                                              "group_median": mm, "cohort_median": med_r,
                                              "ks_p": pv, "ks_p_adj": min(1.0, pv * family), "family": family,
                                              "n": len(mine), "peers": len(peers)})
        findings = []
        for sid, hs in sorted(hits.items()):
            s = dataset.sample(sid)
            feats = {h["feature"]: h for h in hs}
            top = max(hs, key=lambda h: h["z"])
            lines = "; ".join(
                f"{_TEXT[f]} for '{dataset.category_name(h['category'])}' is {abs(h['group_shift_mads']):.1f} "
                f"cohort-MADs {'above' if h['group_shift_mads'] > 0 else 'below'} the other contributors' "
                f"(KS p={h['ks_p_adj']:.0e} after correcting for {h['family']} tests, n={h['n']})" for f, h in feats.items())
            findings.append(make_finding(
                detector_id=self.id, version=self.version, target_type="sample", target_ref=sid,
                severity=Severity.MEDIUM if len(feats) >= 2 else Severity.LOW,
                confidence=float(min(0.7, 0.3 + 0.1 * len(feats) + 0.02 * min(top["z"], 10))),
                score_raw=top["z"], threshold=p["sample_z"],
                reason=(f"Sample {sid} has a box in the tail of a geometry shift by "
                        f"{attribution(s)}: {lines}. This box lies {top['z']:.1f} robust-MADs out in that "
                        f"direction."),
                attack_class=ANNOTATION_GEOMETRY_TAMPER, nature=Nature.INDETERMINATE,
                evidence=[self.ev.json({"hits": hs}, "geometry statistics vs cohort")],
                access_assumptions=["DATASET_LABELS (bounding boxes)", "DATASET_CONTRIBUTOR_META",
                                    "no model, no embeddings"],
                limitations=[f"Bonferroni-corrected over {hs[0]['family']} (contributor, category, feature) "
                             "tests; validated on synthetic K=4 and K=6 categories only, NOT at COCO scale "
                             "(K=80), where min_n / p_max / effect_min need re-tuning.",
                             "No detector says where a box should be: 'anomalous' means unlike the same "
                             "category's boxes from other contributors.",
                             "A cohort that shares the tampering hides it.",
                             "Cross-sample aggregation is the risk engine's, not this detector's."]))
        return findings
