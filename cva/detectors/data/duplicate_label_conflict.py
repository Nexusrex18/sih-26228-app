"""data.duplicate_label_conflict — near-duplicates whose labels disagree.

Invisible to both parents on their own: near-dup finds the pair but never looks at labels;
label-consistency finds neither sample individually wrong (each looks fine — it is the pair that
contradicts itself). The join is trivial: cluster with ``near_duplicate.find_clusters`` (the ONE
definition of clustering, recomputed here because ``detect()`` has no channel for another
detector's output), then flag any cluster whose members do not all carry the same label set.

Severity ``high`` — a contradiction between two copies of the same image is not a matter of degree —
but only when the cluster was confirmed by the embedding; a pHash-only cluster is the same weak
evidence ``near_dup`` caps at confidence 0.5, so it is ``medium`` at confidence 0.45.
``nature`` is ``quality``. One finding per member; the reason says whether that member holds the
majority or a minority label. Members with no labels are ignored (absence is not disagreement).

Limitation: inherits ``near_dup``'s clustering and its evasion weakness; when no embedding index is
available clusters are pHash-only and confidence drops accordingly.
"""
from __future__ import annotations

from collections import Counter

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
    group_source,
    make_finding,
    register_detector,
    sample_by_id,
    category_name,
)
from .near_duplicate import (  # the SAME clustering parameters as near_dup
    DEFAULTS,
    find_clusters,
)
from .taxonomy import DUPLICATE_LABEL_CONFLICT


def _labelset(sample) -> frozenset[int] | None:
    return frozenset(lb.category_id for lb in sample.labels) or None


@register_detector
class DuplicateLabelConflict:
    id = "data.duplicate_label_conflict"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.DATASET_LABELS}
    optional: set[Capability] = set()
    attack_classes = {DUPLICATE_LABEL_CONFLICT}

    def _name(self, ds: Dataset, ls: frozenset[int]) -> str:
        return "+".join(sorted(category_name(ds, c) for c in ls))

    def detect(self, dataset: Dataset, embeddings, model, ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        self.ctx = as_ctx(ctx)
        return finalise(self._detect(dataset, embeddings, model), ctx)

    def _detect(self, dataset: Dataset, embeddings, model) -> list:
        findings = []
        for cl in find_clusters(dataset, embeddings, self.p):
            members = [sample_by_id(dataset, m) for m in cl.members]
            sets = {m.sample_id: _labelset(m) for m in members if _labelset(m) is not None}
            counts = Counter(sets.values())
            if len(counts) < 2:
                continue
            ranked = counts.most_common()
            maj = ranked[0][0] if ranked[0][1] > ranked[1][1] else None      # a strict majority, or none
            tally = ", ".join(f"'{self._name(dataset, ls)}' x{n}" for ls, n in counts.most_common())
            emb = cl.confirmed_by == "phash+embedding"
            basis = (f"pHash Hamming <= {cl.min_hamming}, embedding cosine >= {cl.min_cosine:.3f}"
                     if emb else f"pHash Hamming <= {cl.min_hamming} ONLY")
            sheet = self.ev.sheet(dataset, cl.members, f"conflicting-label cluster: {tally}")
            table = self.ev.json({"labels": {k: sorted(v) for k, v in sets.items()},
                                  "categories": {c.category_id: c.name for c in dataset.categories}},
                                 "label sets per member")
            for m in members:
                ls = sets.get(m.sample_id)
                if ls is None:
                    continue
                role = ("is tied with the other labels (no strict majority)" if maj is None else
                        "holds the majority label" if ls == maj else "holds a minority label")
                who = ""
                if m.contributor is not None:
                    who = f" Supplied by {attribution(group_source(dataset, m.contributor), m.contributor)}."
                findings.append(make_finding(
                    detector_id=self.id, version=self.version, target_type="sample",
                    target_ref=m.sample_id,
                    # pHash-only clustering is the same evidence near_dup caps at 0.5 and stamps
                    # "cannot separate duplicates from similar images" — never a lone HIGH.
                    severity=Severity.HIGH if emb else Severity.MEDIUM,
                    confidence=0.85 if emb else 0.45, score_raw=float(len(counts)), threshold=2.0,
                    reason=(f"{len(members)} near-duplicate images ({basis}) carry contradictory labels: "
                            f"{tally}. Sample {m.sample_id} is labelled '{self._name(dataset, ls)}' and "
                            f"{role}.{who} Contact sheet attached."),
                    attack_class=DUPLICATE_LABEL_CONFLICT, nature=Nature.QUALITY,
                    evidence=[sheet, table],
                    access_assumptions=["DATASET_IMAGES", "DATASET_LABELS",
                                        f"embeddings: {embeddings.extractor_id}@{embeddings.extractor_version}"
                                        if embeddings is not None else "no embedding index"],
                    limitations=["Inherits near_dup's clustering and its evasion weakness.",
                                 "Images with several objects are compared on their full label SET."]
                    + ([] if emb else ["No embedding index: clusters are pHash-only."])))
        return findings
