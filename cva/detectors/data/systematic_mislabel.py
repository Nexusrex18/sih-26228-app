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

Multiplicity policy: one hypothesis test per (contributor, declared class, read-as class) cell;
family-wise Bonferroni correction over all such cells; two-sample Fisher test against the pooled
peers (the cohort rate is estimated, not known); and a cell must also exceed the WORST single peer
by ``gap_min``, so a poisoned peer cannot pull the yardstick toward itself. The price: two
contributors sharing the SAME mapping each become the other's worst peer, and neither is flagged
(pinned by a test; a median-of-peers gate would trade this for less protection with few peers).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import fisher_exact

from cva.core.capability import Capability
from cva.core.types import Nature, Severity

from ._stub_types import Dataset
from .base import (
    CheckContext,
    EvidenceStore,
    Params,
    as_ctx,
    attribution,
    dominant_category,
    finalise,
    group_source,
    make_finding,
    not_performed,
    register_detector,
    sentence_case,
)
from .taxonomy import SYSTEMATIC_MISLABELLING

DEFAULTS = {"min_n": 10, "rate_min": 0.25, "gap_min": 0.2, "p_max": 1e-3,
            "high_rate": 0.4, "folds": 5, "probe_C": 1.0}


@register_detector
class SystematicMislabel:
    id = "data.systematic_mislabel"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.DATASET_LABELS, Capability.DATASET_CONTRIBUTOR_META}
    optional: set[Capability] = set()
    attack_classes = {SYSTEMATIC_MISLABELLING}

    def detect(self, dataset: Dataset, embeddings, model, ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        self.ctx = as_ctx(ctx)
        return finalise(self._detect(dataset, embeddings, model), ctx)

    def _detect(self, dataset: Dataset, embeddings, model) -> list:
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
        # counts[c][a] = images c declares as class a;  hits[(c, a, b)] = of those, probe reads b
        counts = {c: {a: int(((groups == c) & ok & (y == a)).sum()) for a in classes} for c in contribs}
        hits = {(c, a, b): int(((groups == c) & ok & (y == a) & (pred == b)).sum())
                for c in contribs for a in classes for b in classes if b != a}

        def peers(c, a):                       # other contributors with enough images of class a
            return [o for o in contribs if o != c and counts[o][a] >= p["min_n"]]

        # FAMILY-WISE CORRECTION: every (contributor, declared a, read-as b) cell that has a cohort
        # is one hypothesis test. Bonferroni over that whole family — with K classes it is
        # ~ contributors x K x (K-1) tests, and an uncorrected 1e-3 would buy ~K^2/1000 false
        # positives per contributor by construction.
        family = sum(1 for c in contribs for a in classes
                     if counts[c][a] >= p["min_n"] and peers(c, a) for b in classes if b != a)
        findings = []
        for c in contribs:
            cells = []
            for a in classes:
                na = counts[c][a]
                pr = peers(c, a)
                if na < p["min_n"] or not pr:
                    continue
                n_o = sum(counts[o][a] for o in pr)
                for b in classes:
                    if b == a:
                        continue
                    k = hits[(c, a, b)]
                    rate = k / na
                    k_o = sum(hits[(o, a, b)] for o in pr)
                    # the WORST peer, so one poisoned peer cannot pull the yardstick toward c
                    worst_peer = max(hits[(o, a, b)] / counts[o][a] for o in pr)
                    if rate < p["rate_min"] or rate - worst_peer < p["gap_min"]:
                        continue
                    # two-sample (contributor vs pooled peers): the cohort rate is ESTIMATED from
                    # data, not a known constant, so a one-sample binomial would be anti-conservative
                    pv = float(fisher_exact([[k, na - k], [k_o, n_o - k_o]], alternative="greater")[1])
                    p_adj = min(1.0, pv * family)
                    if p_adj < p["p_max"]:
                        cells.append({"declared": a, "probe_reads": b, "k": k, "n": na, "rate": rate,
                                      "cohort_rate": k_o / n_o if n_o else 0.0,
                                      "worst_peer_rate": worst_peer, "p_adj": p_adj, "family": family})
            if not cells:
                continue
            cells.sort(key=lambda d: -d["rate"])
            top = cells[0]
            lines = "; ".join(
                f"of {d['n']} images declared '{dataset.category_name(d['declared'])}', an independent "
                f"probe reads {d['k']} ({d['rate']:.0%}) as '{dataset.category_name(d['probe_reads'])}' "
                f"(rest of cohort: {d['cohort_rate']:.0%}, worst peer {d['worst_peer_rate']:.0%}; "
                f"p={d['p_adj']:.1e} after correcting for {d['family']} tests)" for d in cells[:3])
            findings.append(make_finding(
                detector_id=self.id, version=self.version, target_type="contributor", target_ref=c,
                severity=Severity.HIGH if top["rate"] >= p["high_rate"] else Severity.MEDIUM,
                confidence=float(min(0.9, 0.5 + 0.8 * (top["rate"] - p["rate_min"]))),
                score_raw=top["rate"], threshold=p["rate_min"],
                reason=(f"{sentence_case(attribution(group_source(dataset, c), c))} shows a consistent "
                        f"class-mapping deviation: {lines}."),
                attack_class=SYSTEMATIC_MISLABELLING, nature=Nature.INDETERMINATE,
                evidence=[self.ev.json({"cells": cells,
                                        "categories": {k.category_id: k.name for k in dataset.categories}},
                                       "declared-vs-probe confusion cells vs cohort")],
                access_assumptions=["DATASET_IMAGES", "DATASET_LABELS", "DATASET_CONTRIBUTOR_META",
                                    f"embeddings: {embeddings.extractor_id}@{embeddings.extractor_version}",
                                    "probe trained leaving this contributor out; contributed model NOT used"],
                limitations=["Depends on a proxy classifier: a class it cannot separate cannot be judged.",
                             "If most contributors share the mapping the cohort normalises it. Worse, each "
                             "contributor must beat the WORST peer: if two contributors apply the SAME wrong "
                             "mapping, each is the other's worst peer and neither is flagged.",
                             "A bad SOP and a deliberate steer are indistinguishable from this table.",
                             "An image's label is its dominant category; multi-object images are approximated.",
                             f"Bonferroni-corrected over {family} (contributor, class, class) tests: power "
                             "falls as K grows. Validated on synthetic K=4 and K=6 only — NOT at COCO scale "
                             "(K=80, ~6,000 tests per contributor), where min_n and p_max need re-tuning."]))
        return findings
