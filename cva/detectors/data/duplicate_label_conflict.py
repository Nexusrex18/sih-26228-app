"""data.duplicate_label_conflict — near-duplicates whose labels disagree.

Invisible to both parents on their own: near-dup finds the pair but never looks at labels;
label-consistency finds neither sample individually wrong (each looks fine — it is the pair that
contradicts itself). The join is trivial: cluster with ``near_duplicate.find_clusters`` (the ONE
definition of clustering, recomputed here because ``detect()`` has no channel for another
detector's output), then flag any cluster whose members do not all carry the same label set.

Severity ``high`` — a contradiction between two copies of the same image is not a matter of degree.
``nature`` is ``quality``. One finding per member; the reason says whether that member holds the
majority or a minority label. Members with no labels are ignored (absence is not disagreement).

Limitation: inherits ``near_dup``'s clustering and its evasion weakness; when no embedding index is
available clusters are pHash-only and confidence drops accordingly.
"""
from __future__ import annotations

from collections import Counter

from cva.core.capability import Capability
from cva.core.finding import Nature, Severity
from ._stub_types import Dataset, EmbeddingIndex
from .base import (EvidenceStore, Params, attribution, contact_sheet, group_source, make_finding,
                   register_data)
from .near_duplicate import DEFAULTS as ND_DEFAULTS, find_clusters
from .taxonomy import DUPLICATE_LABEL_CONFLICT


def _labelset(sample) -> frozenset[int] | None:
    return frozenset(lb.category_id for lb in sample.labels) or None


@register_data
class DuplicateLabelConflict:
    id = "data.duplicate_label_conflict"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.DATASET_LABELS}
    optional: set[Capability] = set()
    attack_classes = {DUPLICATE_LABEL_CONFLICT}

    def __init__(self, params: dict | None = None, evidence_dir=None):
        self.p = Params(ND_DEFAULTS, params)      # same clustering parameters as near_dup
        self.ev = EvidenceStore(evidence_dir)

    def _name(self, ds: Dataset, ls: frozenset[int]) -> str:
        return "+".join(sorted(ds.category_name(c) for c in ls))

    def detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None = None, model=None) -> list:
        findings = []
        for cl in find_clusters(dataset, embeddings, self.p):
            members = [dataset.sample(m) for m in cl.members]
            sets = {m.sample_id: _labelset(m) for m in members if _labelset(m) is not None}
            counts = Counter(sets.values())
            if len(counts) < 2:
                continue
            (maj, maj_n), = counts.most_common(1)
            tally = ", ".join(f"'{self._name(dataset, ls)}' x{n}" for ls, n in counts.most_common())
            emb = cl.confirmed_by == "phash+embedding"
            basis = (f"pHash Hamming <= {cl.min_hamming}, embedding cosine >= {cl.min_cosine:.3f}"
                     if emb else f"pHash Hamming <= {cl.min_hamming} ONLY")
            sheet = self.ev.png(contact_sheet(dataset, cl.members),
                                f"conflicting-label cluster: {tally}")
            table = self.ev.json({"labels": {k: sorted(v) for k, v in sets.items()},
                                  "categories": {c.category_id: c.name for c in dataset.categories}},
                                 "label sets per member")
            for m in members:
                ls = sets.get(m.sample_id)
                if ls is None:
                    continue
                role = ("holds the majority label" if ls == maj and counts[ls] > 1 else
                        "holds a minority label" if ls != maj else "is tied on label")
                who = ""
                if m.contributor is not None:
                    who = f" Supplied by {attribution(group_source(dataset, m.contributor), m.contributor)}."
                findings.append(make_finding(
                    detector_id=self.id, version=self.version, target_type="sample",
                    target_ref=m.sample_id, severity=Severity.HIGH,
                    confidence=0.85 if emb else 0.55, score_raw=float(len(counts)), threshold=2.0,
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
