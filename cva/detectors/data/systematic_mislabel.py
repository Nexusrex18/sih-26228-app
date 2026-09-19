"""data.systematic_mislabel — a contributor *consistently* applying a wrong class mapping.

Distinct from ``label_flip`` on purpose: every sample is internally consistent (each 'tank'-as-
'truck' looks like its neighbours), so per-sample confident learning finds no local disagreement.
It is detectable ONLY at the source level.

Method: a disposable linear probe on the independent embeddings is fit **leaving each contributor
out** (``GroupKFold`` by contributor), so a contributor's own labels never influence the probe that
judges them. Per contributor: a confusion table of declared label (a) vs probe prediction (b),
compared with the rest of the cohort. A (contributor, a->b) cell is flagged when it is large
(>= ``rate_min``), far above the cohort (>= ``gap_min``), and binomially significant.

This is one of the few Module A findings that is naturally contributor-level, so it is emitted at
``target_type="contributor"`` — one finding per contributor (all flagged pairs in one table), which
also keeps ``finding_id`` unique. Cohort statistics are computed here; the beta-binomial posterior
is the rollup's and is not implemented.

Severity: peak rate >= 0.40 -> ``high``, else ``medium``. ``nature`` is ``indeterminate`` — a bad
SOP and a deliberate steer produce the same table.

Limitations: depends on a proxy classifier; if most contributors share the mapping the cohort
normalises it; the contributed model is not consulted (circularity, ADR-009).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from cva.core.capability import Capability
from cva.core.finding import Nature, Severity
from ._stub_types import Dataset, EmbeddingIndex
from .base import (EvidenceStore, Params, attribution, dominant_category, group_source,
                   make_finding, not_performed, register_data)
from .taxonomy import SYSTEMATIC_MISLABELLING

DEFAULTS = {"min_n": 10, "rate_min": 0.25, "gap_min": 0.2, "p_max": 1e-3, "cohort_floor": 0.02,
            "high_rate": 0.4, "folds": 5, "probe_C": 1.0}


@register_data
class SystematicMislabel:
    id = "data.systematic_mislabel"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.DATASET_LABELS, Capability.DATASET_CONTRIBUTOR_META}
    optional: set[Capability] = set()
    attack_classes = {SYSTEMATIC_MISLABELLING}

    def __init__(self, params: dict | None = None, evidence_dir=None):
        self.p = Params(DEFAULTS, params)
        self.ev = EvidenceStore(evidence_dir)

    def detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None = None, model=None) -> list:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        p = self.p
        if embeddings is None:
            return [not_performed(self.id, self.version, self.attack_classes, "no embedding index supplied")]
        rows = [(s, dominant_category(s)) for s in dataset.samples if s.contributor is not None]
        rows = [(s, d) for s, d in rows if d is not None and embeddings.vector(s.sample_id) is not None]
        groups = np.array([s.contributor for s, _ in rows])
        contribs = sorted(set(groups.tolist()))
        if len(contribs) < 2:
            return [not_performed(self.id, self.version, self.attack_classes,
                                  "fewer than 2 contributors — there is no cohort to compare against")]
        X = embeddings.vectors([s.sample_id for s, _ in rows])
        y = np.array([d for _, d in rows])
        pred = np.full(len(y), -1)
        for tr, te in GroupKFold(n_splits=min(p["folds"], len(contribs))).split(X, y, groups):
            if len(set(y[tr].tolist())) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=p["probe_C"]).fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        ok = pred >= 0
        classes = sorted(set(y.tolist()))
        findings = []
        for c in contribs:
            mine = (groups == c) & ok
            other = (groups != c) & ok
            cells = []
            for a in classes:
                na = int((mine & (y == a)).sum())
                if na < p["min_n"]:
                    continue
                nb_other = int((other & (y == a)).sum())
                for b in classes:
                    if b == a:
                        continue
                    k = int((mine & (y == a) & (pred == b)).sum())
                    rate = k / na
                    k_o = int((other & (y == a) & (pred == b)).sum())
                    cohort = max(k_o / nb_other if nb_other else 0.0, p["cohort_floor"])
                    if rate >= p["rate_min"] and rate - cohort >= p["gap_min"]:
                        pv = float(binomtest(k, na, cohort, alternative="greater").pvalue)
                        if pv < p["p_max"]:
                            cells.append({"declared": a, "probe_reads": b, "k": k, "n": na,
                                          "rate": rate, "cohort_rate": cohort, "p": pv})
            if not cells:
                continue
            cells.sort(key=lambda d: -d["rate"])
            top = cells[0]
            lines = "; ".join(
                f"of {d['n']} images declared '{dataset.category_name(d['declared'])}', an independent "
                f"probe reads {d['k']} ({d['rate']:.0%}) as '{dataset.category_name(d['probe_reads'])}' "
                f"(rest of cohort: {d['cohort_rate']:.0%}, p={d['p']:.1e})" for d in cells[:3])
            findings.append(make_finding(
                detector_id=self.id, version=self.version, target_type="contributor", target_ref=c,
                severity=Severity.HIGH if top["rate"] >= p["high_rate"] else Severity.MEDIUM,
                confidence=float(min(0.9, 0.5 + 0.8 * (top["rate"] - p["rate_min"]))),
                score_raw=top["rate"], threshold=p["rate_min"],
                reason=(f"{attribution(group_source(dataset, c), c).capitalize()} shows a consistent "
                        f"class-mapping deviation: {lines}."),
                attack_class=SYSTEMATIC_MISLABELLING, nature=Nature.INDETERMINATE,
                evidence=[self.ev.json({"cells": cells,
                                        "categories": {k.category_id: k.name for k in dataset.categories}},
                                       "declared-vs-probe confusion cells vs cohort")],
                access_assumptions=["DATASET_IMAGES", "DATASET_LABELS", "DATASET_CONTRIBUTOR_META",
                                    f"embeddings: {embeddings.extractor_id}@{embeddings.extractor_version}",
                                    "probe trained leaving this contributor out; contributed model NOT used"],
                limitations=["Depends on a proxy classifier: a class it cannot separate cannot be judged.",
                             "If most contributors share the mapping the cohort normalises it.",
                             "A bad SOP and a deliberate steer are indistinguishable from this table.",
                             "An image's label is its dominant category; multi-object images are approximated."]))
        return findings
