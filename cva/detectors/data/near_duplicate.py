"""data.near_dup — near-duplicate flooding (two-stage, clustered).

pHash alone misses re-encoded/rescaled/lightly-cropped duplicates; embeddings alone are O(n²)
and flag semantic neighbours that are not duplicates. So:

  1. candidates  — 64-bit pHash, Hamming distance <= ``hamming`` (default 8)
  2. confirm     — cosine similarity in the independent embedding space >= ``cosine`` (0.97)
  3. cluster     — connected components over confirmed pairs
  4. one finding per MEMBER sample, referencing its cluster, with a contact sheet

If no embedding index is available the detector still runs, degraded and stated: pHash only, a
stricter ``hamming_only`` cut, confidence capped, and the limitation stamped on every finding.

Severity (cluster size): 2 -> ``low`` (one pair is weak evidence), 3–9 -> ``medium``,
>= 10 -> ``high`` (flooding). ``nature`` is ``quality`` — flooding is usually collection
sloppiness; escalation needs corroboration, which the risk engine — not this file — performs.

Known limitation: weak against duplicates deliberately perturbed to evade BOTH pHash and the
embedding (rotation/crop/colour-shift resistant, not adversarially perturbed).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cva.core.capability import Capability
from cva.core.finding import Nature, Severity
from ._stub_types import Dataset, EmbeddingIndex
from .base import (EvidenceStore, Params, attribution, contact_sheet, dominant_category,
                   group_source, load_rgb, make_finding, not_performed, register_data)
from .taxonomy import NEAR_DUPLICATE_FLOODING

DEFAULTS = {"hamming": 8, "cosine": 0.97, "hamming_only": 4, "chunk": 512}
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int32)
    return _POP8[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1).astype(np.int32)


def phash64(sample) -> np.uint64:
    import imagehash

    bits = imagehash.phash(load_rgb(sample), hash_size=8).hash.flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


@dataclass(frozen=True)
class DupCluster:
    members: tuple[str, ...]
    min_hamming: int
    min_cosine: float | None
    mean_cosine: float | None
    confirmed_by: str            # "phash+embedding" | "phash-only"


def find_clusters(dataset: Dataset, embeddings: EmbeddingIndex | None,
                  params: Params | dict | None = None) -> list[DupCluster]:
    """Shared with ``data.duplicate_label_conflict`` — the one place clustering is defined."""
    p = params if isinstance(params, Params) else Params(DEFAULTS, params)
    n = len(dataset)
    if n < 2:
        return []
    ids = [s.sample_id for s in dataset.samples]
    h = np.array([phash64(s) for s in dataset.samples], dtype=np.uint64)
    cut = p["hamming"] if embeddings is not None else min(p["hamming"], p["hamming_only"])
    rows: list[int] = []
    cols: list[int] = []
    ham: dict[tuple[int, int], int] = {}
    cos: dict[tuple[int, int], float] = {}
    ch = p["chunk"]
    for i0 in range(0, n, ch):
        d = _popcount(h[i0:i0 + ch, None] ^ h[None, :])
        for ii, jj in zip(*np.nonzero(d <= cut)):
            i, j = i0 + int(ii), int(jj)
            if j <= i:
                continue
            if embeddings is not None:
                vi, vj = embeddings.vector(ids[i]), embeddings.vector(ids[j])
                if vi is None or vj is None:
                    continue
                c = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12))
                if c < p["cosine"]:
                    continue
                cos[(i, j)] = c
            rows.append(i)
            cols.append(j)
            ham[(i, j)] = int(d[ii, jj])
    if not rows:
        return []
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    out: list[DupCluster] = []
    for c in range(ncomp):
        idx = np.nonzero(lab == c)[0]
        if len(idx) < 2:
            continue
        s = set(int(i) for i in idx)
        hs = [v for (i, j), v in ham.items() if i in s and j in s]
        cs = [v for (i, j), v in cos.items() if i in s and j in s]
        out.append(DupCluster(tuple(sorted(ids[i] for i in idx)), min(hs),
                              min(cs) if cs else None, float(np.mean(cs)) if cs else None,
                              "phash+embedding" if embeddings is not None else "phash-only"))
    return sorted(out, key=lambda c: c.members)


def _severity(size: int) -> Severity:
    return Severity.LOW if size == 2 else Severity.MEDIUM if size < 10 else Severity.HIGH


@register_data
class NearDuplicate:
    id = "data.near_dup"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES}
    optional: set[Capability] = set()
    attack_classes = {NEAR_DUPLICATE_FLOODING}

    def __init__(self, params: dict | None = None, evidence_dir=None):
        self.p = Params(DEFAULTS, params)
        self.ev = EvidenceStore(evidence_dir)

    def detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None = None, model=None) -> list:
        p = self.p
        clusters = find_clusters(dataset, embeddings, self.p)
        findings = []
        for cl in clusters:
            members = [dataset.sample(m) for m in cl.members]
            contribs = Counter(m.contributor for m in members)
            top_c, top_n = contribs.most_common(1)[0]
            who = ""
            if top_c is not None:
                src = group_source(dataset, top_c)
                who = (f" {top_n} of {len(members)} come from "
                       f"{attribution(src, top_c)}.")
            emb = cl.confirmed_by == "phash+embedding"
            basis = (f"pHash Hamming <= {cl.min_hamming} and embedding cosine >= {cl.min_cosine:.3f} "
                     f"({embeddings.extractor_id})" if emb else
                     f"pHash Hamming <= {cl.min_hamming} ONLY (no embedding index to confirm)")
            sheet = self.ev.png(contact_sheet(dataset, cl.members), f"cluster of {len(members)}")
            table = self.ev.json({"members": [{"sample_id": m.sample_id, "contributor": m.contributor,
                                               "label": dominant_category(m)} for m in members],
                                  "min_hamming": cl.min_hamming, "min_cosine": cl.min_cosine},
                                 "duplicate cluster members")
            if emb:
                score, thr = float(cl.mean_cosine), p["cosine"]
                conf = min(0.95, 0.5 + 0.45 * (score - thr) / max(1e-6, 1 - thr) + 0.02 * min(len(members), 10))
                limits = []
            else:
                score, thr = 1 - cl.min_hamming / 64.0, 1 - p["hamming_only"] / 64.0
                conf = min(0.5, 0.3 + 0.02 * len(members))
                limits = ["Embedding confirmation was unavailable: pHash-only evidence, which cannot "
                          "separate true duplicates from visually similar distinct images."]
            for m in members:
                others = [x.sample_id for x in members if x.sample_id != m.sample_id][:4]
                findings.append(make_finding(
                    detector_id=self.id, version=self.version, target_type="sample",
                    target_ref=m.sample_id, severity=_severity(len(members)), confidence=conf,
                    score_raw=score, threshold=thr,
                    reason=(f"Sample {m.sample_id} sits in a cluster of {len(members)} near-duplicate "
                            f"images ({basis}); nearest siblings: {', '.join(others)}.{who} "
                            f"Contact sheet attached."),
                    attack_class=NEAR_DUPLICATE_FLOODING, nature=Nature.QUALITY,
                    evidence=[sheet, table],
                    access_assumptions=["DATASET_IMAGES",
                                        f"embeddings: {embeddings.extractor_id}@{embeddings.extractor_version}"
                                        if embeddings is not None else "no embedding index"],
                    limitations=limits + [
                        "Weak against duplicates deliberately perturbed to evade both pHash and the "
                        "embedding (adversarial rotation/crop/colour-shift).",
                        "One pair is weak evidence; escalation needs corroboration (risk engine)."]))
        return findings
