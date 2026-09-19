"""data.label_consistency — label flipping / mislabelling, two independent signals.

  1. k-NN label disagreement — of a sample's k nearest neighbours in the INDEPENDENT embedding
     space, what fraction share its label? Very low agreement -> candidate.
  2. Confident learning (cleanlab ``find_label_issues``) over **out-of-fold** class probabilities
     from a disposable linear probe on the embeddings.

Both are reported separately in the evidence; a sample flagged by BOTH is higher confidence.

Why the probe must be out-of-fold: fit and predict on the same rows and the probe memorises a
flipped label, predicts it confidently, and confident learning reports no issue on precisely the
samples it exists to catch. That is the same inversion ``backend_plan.md`` §12.2 withdrew
``MODEL_PREDICT`` for — here it would be self-inflicted. The probe is a disposable internal
classifier, NOT the contributed model, and nothing is retrained (ADR-004).

No ``optional`` capability: the contributed model is never consulted (circularity, ADR-009).

Licence: cleanlab is AGPL-3.0-or-later; accepted (backend_plan.md §5.12). ``find_label_issues`` is
the single call to swap if a deployment cannot accept AGPL.

Severity: one signal -> ``low``; both -> ``medium``. ``nature`` is ``quality`` (a flip rate
consistent with human error). Limitations: an image's label is its dominant category, so
multi-object images are approximated; small or imbalanced classes make both signals noisy.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Capability
from cva.core.finding import Nature, Severity
from ._stub_types import Dataset, EmbeddingIndex
from .base import (EvidenceStore, Params, attribution, contact_sheet, dominant_category,
                   group_source, make_finding, not_performed, register_data)
from .taxonomy import LABEL_FLIPPING

DEFAULTS = {"k": 10, "knn_agree_max": 0.2, "min_neighbours": 5, "cv_folds": 5, "seed": 0,
            "probe_C": 1.0}


def out_of_fold_probs(X: np.ndarray, y: np.ndarray, folds: int, seed: int, C: float):
    """Stratified out-of-fold class probabilities from a linear probe, or None if impossible."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    counts = np.bincount(y)
    counts = counts[counts > 0]
    if len(counts) < 2 or counts.min() < 2:
        return None
    n_splits = int(min(folds, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(LogisticRegression(max_iter=2000, C=C), X, y, cv=cv,
                             method="predict_proba")


@register_data
class LabelConsistency:
    id = "data.label_consistency"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.DATASET_LABELS}
    optional: set[Capability] = set()          # settled 2026-09-19: no degraded mode exists
    attack_classes = {LABEL_FLIPPING}

    def __init__(self, params: dict | None = None, evidence_dir=None):
        self.p = Params(DEFAULTS, params)
        self.ev = EvidenceStore(evidence_dir)

    def detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None = None, model=None) -> list:
        p = self.p
        if embeddings is None:
            return [not_performed(self.id, self.version, self.attack_classes,
                                  "no embedding index (the independent backbone) was supplied")]
        rows = [(s, dominant_category(s)) for s in dataset.samples]
        rows = [(s, d) for s, d in rows if d is not None and embeddings.vector(s.sample_id) is not None]
        if len(rows) < 2 * p["k"]:
            return [not_performed(self.id, self.version, self.attack_classes,
                                  f"only {len(rows)} labelled samples with embeddings (< {2 * p['k']})")]
        ids = [s.sample_id for s, _ in rows]
        cats = sorted({d for _, d in rows})
        y = np.array([cats.index(d) for _, d in rows])
        lab = dict(zip(ids, (d for _, d in rows)))
        X = embeddings.vectors(ids)

        # --- signal 1: k-NN label disagreement -----------------------------------------
        knn_agree, knn_nb = {}, {}
        for sid in ids:
            nb = [(j, c) for j, c in embeddings.knn(sid, p["k"] + 8) if j in lab][: p["k"]]
            if len(nb) >= p["min_neighbours"]:
                knn_agree[sid] = sum(lab[j] == lab[sid] for j, _ in nb) / len(nb)
                knn_nb[sid] = [j for j, _ in nb]

        # --- signal 2: confident learning on OUT-OF-FOLD probabilities ------------------
        cl_flag, cl_score, cl_thr = {}, {}, {}
        P = out_of_fold_probs(X, y, p["cv_folds"], p["seed"], p["probe_C"])
        cl_note = None
        if P is None:
            cl_note = "confident learning skipped: a class has fewer than 2 samples or only one class exists"
        else:
            from cleanlab.filter import find_label_issues

            mask = find_label_issues(labels=y, pred_probs=P)
            t = {j: float(P[y == j, j].mean()) for j in range(len(cats))}
            for i, sid in enumerate(ids):
                cl_flag[sid] = bool(mask[i])
                cl_score[sid] = float(1.0 - P[i, y[i]])
                cl_thr[sid] = float(1.0 - t[int(y[i])])

        findings = []
        for i, sid in enumerate(ids):
            k_hit = sid in knn_agree and knn_agree[sid] <= p["knn_agree_max"]
            c_hit = cl_flag.get(sid, False)
            if not (k_hit or c_hit):
                continue
            s = dataset.sample(sid)
            name = dataset.category_name(lab[sid])
            parts, sig = [], {}
            if sid in knn_agree:
                maj = max(set(lab[j] for j in knn_nb[sid]), key=[lab[j] for j in knn_nb[sid]].count)
                sig["knn"] = {"agreement": knn_agree[sid], "neighbour_majority": dataset.category_name(maj),
                              "flagged": k_hit, "neighbours": knn_nb[sid]}
                if k_hit:
                    parts.append(f"only {knn_agree[sid]:.0%} of its {len(knn_nb[sid])} nearest neighbours "
                                 f"share that label (they are mostly '{dataset.category_name(maj)}'; "
                                 f"threshold {p['knn_agree_max']:.0%})")
            if sid in cl_flag:
                sig["confident_learning"] = {"p_given_label": 1 - cl_score[sid], "flagged": c_hit}
                if c_hit:
                    parts.append(f"an out-of-fold linear probe gives its label only "
                                 f"{1 - cl_score[sid]:.0%} probability and confident learning marks it a "
                                 f"label issue")
            both = k_hit and c_hit
            use_cl = c_hit and not k_hit
            score = cl_score[sid] if use_cl else (1 - knn_agree[sid])
            thr = cl_thr[sid] if use_cl else (1 - p["knn_agree_max"])
            who = ""
            if s.contributor is not None:
                who = f" Supplied by {attribution(group_source(dataset, s.contributor), s.contributor)}."
            sheet = None
            if self.ev.root is not None:
                sheet = self.ev.png(contact_sheet(dataset, [sid] + knn_nb.get(sid, [])[:11]),
                                    "sample (first tile) and its nearest neighbours")
            findings.append(make_finding(
                detector_id=self.id, version=self.version, target_type="sample", target_ref=sid,
                severity=Severity.MEDIUM if both else Severity.LOW,
                confidence=0.8 if both else (0.5 if c_hit else 0.45),
                score_raw=score, threshold=thr,
                reason=(f"Sample {sid} is labelled '{name}', but " + " and ".join(parts) +
                        f". Flagged by {'BOTH signals' if both else 'one signal'}; score_raw is the "
                        f"{'confident-learning' if use_cl else 'k-NN disagreement'} score.{who}"),
                attack_class=LABEL_FLIPPING, nature=Nature.QUALITY,
                evidence=[self.ev.json(sig, "label-consistency signals"), sheet],
                access_assumptions=["DATASET_IMAGES", "DATASET_LABELS",
                                    f"embeddings: {embeddings.extractor_id}@{embeddings.extractor_version}",
                                    "contributed model NOT used (circularity, ADR-009)"],
                limitations=[
                    "An image's label is its dominant category; multi-object images are approximated.",
                    "Small or imbalanced classes make both signals noisy.",
                    "Detects labels that disagree with the embedding neighbourhood; a flip into a "
                    "visually indistinguishable class is invisible."] + ([cl_note] if cl_note else [])))
        return findings
