"""data.negative_space — objects present in an image but deliberately left unannotated.

Invisible to every detector that only examines what IS labelled. Method: run the contributed model
over its own training images and flag confident detections that fall where no annotation exists;
the finding is the *contributor pattern* (a contributor whose rate of unannotated-but-detected
objects is far above the cohort's), not any single image.

Model contract assumed (a limitation, declared): ``model.predict(x)`` for a detector returns an
array shaped ``(N, M, 6)`` of post-NMS ``[x1, y1, x2, y2, score, class]`` in the model's INPUT
pixel space. A model that returns anything else is reported as such, not guessed at.

Flag rule: a detection with score >= ``score_min`` whose IoU with EVERY annotated box is
< ``iou_max`` (it has no corresponding annotation). A contributor is anomalous when the share of its images with >= 1 such detection is
>= ``rate_min``, >= ``ratio_min`` x the rest of the cohort's (floored), and binomially significant.
Sample-level findings are emitted for the flagged images of anomalous contributors.

Statistics: two-sample Fisher test against the pooled peers (the cohort rate is estimated, not
known), Bonferroni-corrected over contributors, and the rate must also exceed ``ratio_min`` x the
WORST single peer — so a poisoned peer cannot pull the yardstick toward itself.

Severity ``low``, ``medium`` when the contributor's rate >= 0.4. ``nature`` is ``indeterminate`` (a
missed annotation is sloppy or deliberate; the image cannot say). This is a data-question use of
the contributed model — the same circularity caveat as spectral signatures: corroborate, do not
rely on it alone. Label findings as depending on the model under test.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import fisher_exact

from cva.core.capability import Capability
from cva.core.types import Availability, Nature, Severity

from ._stub_types import Dataset
from .base import (
    CheckContext,
    EvidenceStore,
    Params,
    as_ctx,
    attribution,
    finalise,
    group_source,
    make_finding,
    not_performed,
    register_detector,
    model_input,
    preprocess_note,
    category_name,
)
from .taxonomy import NEGATIVE_SPACE_POISONING

DEFAULTS = {"score_min": 0.8, "iou_max": 0.3, "rate_min": 0.15, "ratio_min": 3.0,
            "cohort_floor": 0.02, "p_max": 1e-3, "min_flagged": 5, "min_peer_images": 10,
            "high_rate": 0.4, "batch": 32}


def parse_detections(out) -> np.ndarray | None:
    a = np.asarray(out)
    return a if a.ndim == 3 and a.shape[-1] == 6 else None


def _max_iou(det: np.ndarray, boxes: list[np.ndarray]) -> float:
    """Best IoU between one detection (xyxy) and any annotated box — 0 if there are none."""
    x1, y1, x2, y2 = det
    a = max((x2 - x1) * (y2 - y1), 1e-9)
    best = 0.0
    for b in boxes:
        iw = max(0.0, min(x2, b[2]) - max(x1, b[0]))
        ih = max(0.0, min(y2, b[3]) - max(y1, b[1]))
        inter = iw * ih
        best = max(best, inter / (a + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9))
    return best


@register_detector
class NegativeSpace:
    id = "data.negative_space"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.DATASET_LABELS,
                Capability.MODEL_PREDICT}
    # DATASET_CONTRIBUTOR_META moved from `requires` to `optional` — audit item 10, ruling
    # adopted. Hard-requiring it meant a §3 tier-5 dataset (no contributor attribution, an
    # explicitly supported case) got ZERO coverage of negative_space_poisoning. The §6.4
    # signal — a confident detection where no annotation exists — does not need contributor
    # identity; identity is only how the finding is FRAMED, and what the cohort test needs.
    # Without it the detector runs DEGRADED in the per-sample mode below.
    optional = {Capability.DATASET_CONTRIBUTOR_META}
    attack_classes = {NEGATIVE_SPACE_POISONING}

    def detect(self, dataset: Dataset, embeddings, model, ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        self.ctx = as_ctx(ctx)
        return finalise(self._detect(dataset, embeddings, model), ctx)

    def _detect(self, dataset: Dataset, embeddings, model) -> list:
        p = self.p
        if model is None:
            return [not_performed(self.id, self.version, self.attack_classes, "no model supplied")]
        samples = [s for s in dataset.samples if s.contributor is not None and s.labels]
        # No contributor attribution anywhere: a §3 tier-5 dataset. The cohort test is
        # impossible — there are no cohorts — but the underlying signal is not, so run the
        # per-sample mode rather than returning `not_performed` on a supported case.
        degraded = not samples
        if degraded:
            samples = [s for s in dataset.samples if s.labels]
            if not samples:
                return [not_performed(
                    self.id, self.version, self.attack_classes,
                    f"no sample carries any annotation ({len(dataset.samples)} samples), so "
                    "there is nothing an unannotated detection could be unannotated against")]
        H, W = model.input_shape[-2:]
        flagged: dict[str, list[dict]] = {}
        for i in range(0, len(samples), p["batch"]):
            chunk = samples[i:i + p["batch"]]
            X = np.stack([model_input(s, model) for s in chunk])
            det = parse_detections(model.predict(X))
            if det is None:
                return [not_performed(self.id, self.version, self.attack_classes,
                                      "the model's output is not a box-detector output "
                                      "((N, M, 6) [x1,y1,x2,y2,score,class])", Availability.UNAVAILABLE)]
            for s, d in zip(chunk, det, strict=False):
                sx, sy = s.width / W, s.height / H
                ann = [np.array([lb.bbox[0], lb.bbox[1], lb.bbox[0] + lb.bbox[2], lb.bbox[1] + lb.bbox[3]])
                       for lb in s.labels if lb.bbox is not None]
                for x1, y1, x2, y2, sc, cl in d:
                    if sc < p["score_min"]:
                        continue
                    box = np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                    if _max_iou(box, ann) < p["iou_max"]:
                        flagged.setdefault(s.sample_id, []).append(
                            {"bbox_xyxy": [float(v) for v in box], "score": float(sc), "class": int(cl)})
        if degraded:
            return self._per_sample(dataset, samples, flagged)
        tot, hit = Counter(), Counter()
        for s in samples:
            tot[s.contributor] += 1
            hit[s.contributor] += s.sample_id in flagged
        findings = []
        family = len(tot)                     # one test per contributor -> Bonferroni over them
        tested = 0                            # contributors that actually had peers to compare with
        for c in sorted(tot):
            k, n = hit[c], tot[c]
            peers = [o for o in tot if o != c and tot[o] >= p["min_peer_images"]]
            if not peers:
                continue
            tested += 1
            rest_k, rest_n = sum(hit[o] for o in peers), sum(tot[o] for o in peers)
            cohort = max(rest_k / rest_n, p["cohort_floor"])
            worst_peer = max(max(hit[o] / tot[o] for o in peers), p["cohort_floor"])
            rate = k / n
            # must exceed the WORST peer by the ratio, so one poisoned peer can't set the yardstick
            if k < p["min_flagged"] or rate < p["rate_min"] or rate < p["ratio_min"] * worst_peer:
                continue
            # two-sample (this contributor vs pooled peers): the cohort rate is estimated from data
            pv = float(fisher_exact([[k, n - k], [rest_k, rest_n - rest_k]], alternative="greater")[1])
            pv = min(1.0, pv * family)
            if pv >= p["p_max"]:
                continue
            src = group_source(dataset, c)
            for s in samples:
                if s.contributor != c or s.sample_id not in flagged:
                    continue
                d0 = max(flagged[s.sample_id], key=lambda d: d["score"])
                findings.append(make_finding(
                    detector_id=self.id, version=self.version, target_type="sample",
                    target_ref=s.sample_id,
                    severity=Severity.MEDIUM if rate >= p["high_rate"] else Severity.LOW,
                    confidence=float(min(0.75, 0.3 + 0.5 * min(1.0, rate))),
                    score_raw=rate, threshold=p["rate_min"],
                    reason=(f"The contributed model confidently detects an object "
                            f"(score {d0['score']:.2f}, class {category_name(dataset, d0['class'])!r}) at "
                            f"{[round(v) for v in d0['bbox_xyxy']]} in {s.sample_id} where no annotation "
                            f"exists. {k} of {n} images from {attribution(src, c)} show this ({rate:.0%}; "
                            f"rest of cohort {cohort:.0%}, worst peer {worst_peer:.0%}; p={pv:.1e} after correcting for "
                            f"{family} contributors) — a systematic pattern, not a "
                            f"single miss. Depends on the contributed model under test."),
                    attack_class=NEGATIVE_SPACE_POISONING, nature=Nature.INDETERMINATE,
                    evidence=[self.ev.json({"detections": flagged[s.sample_id], "contributor_rate": rate,
                                            "cohort_rate": cohort, "n": n}, "unannotated detections")],
                    access_assumptions=["DATASET_IMAGES", "DATASET_LABELS", "DATASET_CONTRIBUTOR_META",
                                        "MODEL_PREDICT of the contributed model (data-question use)",
                                        preprocess_note(model),
                                        "model output is (N, M, 6) [x1,y1,x2,y2,score,class] in input pixels"],
                    limitations=["A data-question use of the contributed model: corroborate, do not rely "
                                 "on it alone.",
                                 "Only as good as the model's own detections; a model that also learned "
                                 "to ignore the object hides it.",
                                 "A missed annotation is sloppy or deliberate; the image cannot say."]))
        if tested == 0:
            return [not_performed(self.id, self.version, self.attack_classes,
                                  f"no contributor has a peer with at least {p['min_peer_images']} "
                                  "images to be compared with, so no cohort rate could be estimated")]
        return findings

    def _per_sample(self, dataset: Dataset, samples: list, flagged: dict) -> list:
        """The DEGRADED mode: no contributor attribution, so no cohort and no test.

        What survives is the §6.4 signal itself — a confident detection where no annotation
        exists — reported per image. What is lost is everything the cohort gives: whether
        the rate is unusual, and whom to attribute it to. So this mode claims much less. It
        is capped at `low` severity and at a confidence that cannot reach D5's floor even
        once a calibrator exists, because "one image has an unannotated object" is ordinary
        annotation noise; only a systematic rate is evidence of poisoning.

        Emitting nothing here was the alternative, and it is worse: a tier-5 dataset would
        get a coverage row claiming `negative_space_poisoning` was assessed by a detector
        that structurally could not assess it.
        """
        p = self.p
        rate = len(flagged) / len(samples) if samples else 0.0
        if not flagged:
            return [not_performed(
                self.id, self.version, self.attack_classes,
                f"no contributor attribution in this dataset, so only the per-image signal "
                f"was available, and no image of {len(samples)} carries a confident "
                f"detection (score >= {p['score_min']}) outside every annotation",
                Availability.DEGRADED)]
        out = []
        for s in samples:
            if s.sample_id not in flagged:
                continue
            d0 = max(flagged[s.sample_id], key=lambda d: d["score"])
            out.append(make_finding(
                detector_id=self.id, version=self.version, target_type="sample",
                target_ref=s.sample_id, severity=Severity.LOW,
                confidence=0.3, score_raw=d0["score"], threshold=p["score_min"],
                reason=(f"The contributed model confidently detects an object "
                        f"(score {d0['score']:.2f}, class "
                        f"{dataset.category_name(d0['class'])!r}) at "
                        f"{[round(v) for v in d0['bbox_xyxy']]} in {s.sample_id} where no "
                        f"annotation exists. {len(flagged)} of {len(samples)} annotated "
                        f"images in this dataset show this ({rate:.0%}). This dataset "
                        "carries NO contributor attribution, so this rate cannot be "
                        "compared with a cohort and nothing here says whether it is "
                        "unusual or who produced it. Depends on the contributed model "
                        "under test."),
                attack_class=NEGATIVE_SPACE_POISONING, nature=Nature.INDETERMINATE,
                availability=Availability.DEGRADED,
                evidence=[self.ev.json(
                    {"detections": flagged[s.sample_id], "dataset_rate": rate,
                     "n_annotated_images": len(samples)}, "unannotated detections")],
                access_assumptions=[
                    "DATASET_IMAGES", "DATASET_LABELS",
                    "NO DATASET_CONTRIBUTOR_META — degraded, per-image only",
                    "MODEL_PREDICT of the contributed model (data-question use)",
                    "model output is (N, M, 6) [x1,y1,x2,y2,score,class] in input pixels"],
                limitations=[
                    "DEGRADED: no contributor attribution, so the cohort comparison that "
                    "makes this detector's finding a POISONING claim could not be run. "
                    "This is a per-image observation, not evidence of a systematic pattern.",
                    "A data-question use of the contributed model: corroborate, do not rely "
                    "on it alone.",
                    "Only as good as the model's own detections; a model that also learned "
                    "to ignore the object hides it.",
                    "A missed annotation is sloppy or deliberate; the image cannot say."]))
        return out
