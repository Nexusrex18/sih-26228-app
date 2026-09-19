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

Complexity: pair search is pigeonhole multi-index hashing (see ``candidate_pairs``), not an n x n
matrix; there is still no declared dataset-size budget — a flood of near-identical images is compared
pairwise inside one bucket.

Known limitation: weak against duplicates deliberately perturbed to evade BOTH pHash and the
embedding (rotation/crop/colour-shift resistant, not adversarially perturbed).
"""
from __future__ import annotations

import weakref
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cva.core.capability import Capability
from cva.core.types import Nature, Severity

from ._stub_types import Dataset, EmbeddingIndex
from .base import (
    CheckContext,
    EvidenceStore,
    Params,
    as_ctx,
    attribution,
    dominant_category,
    finalise,
    group_source,
    load_rgb,
    make_finding,
    register_detector,
    DATASET_CACHE,
    sample_by_id,
    n_samples,
)
from .taxonomy import NEAR_DUPLICATE_FLOODING

DEFAULTS = {"hamming": 8, "cosine": 0.97, "hamming_only": 4}
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


def candidate_pairs(h: np.ndarray, cut: int) -> dict[tuple[int, int], int]:
    """All index pairs (i < j) with Hamming(h[i], h[j]) <= cut -> their distance.

    Pigeonhole multi-index hashing instead of an n x n distance matrix: split the 64 bits into
    ``cut + 1`` contiguous chunks; two hashes within ``cut`` differing bits MUST agree exactly on
    at least one chunk, so bucketing by each chunk's value finds every true pair while only ever
    comparing within a bucket. Exact (identical result to brute force — tested), and roughly
    n * bucket_size instead of n^2 for ordinary data. A flood of near-identical images still
    lands in one bucket and is compared pairwise — inherent to reporting every pair."""
    n = len(h)
    m = cut + 1
    bounds = np.linspace(0, 64, m + 1).astype(int)
    out: dict[tuple[int, int], int] = {}
    for a in range(m):
        lo, hi = int(bounds[a]), int(bounds[a + 1])
        key = (h >> np.uint64(lo)) & np.uint64((1 << (hi - lo)) - 1)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        ends = np.r_[starts[1:], n]
        for s0, e0 in zip(starts, ends, strict=False):
            if e0 - s0 < 2:
                continue
            idx = order[s0:e0]
            d = _popcount(h[idx][:, None] ^ h[idx][None, :])
            for x, y in zip(*np.nonzero(np.triu(d <= cut, 1)), strict=False):
                i, j = sorted((int(idx[x]), int(idx[y])))
                out[(i, j)] = int(d[x, y])
    return out


# Clusters computed for a dataset, so ``near_dup`` and ``duplicate_label_conflict`` — which cluster
# with the same parameters — hash and compare the images ONCE per scan, not twice. Memoised per
# dataset object (see ``base.DATASET_CACHE``) and validated against the embedding object by identity.
def find_clusters(dataset: Dataset, embeddings: EmbeddingIndex | None,
                  params: Params | dict | None = None) -> list[DupCluster]:
    """Shared with ``data.duplicate_label_conflict`` — the one place clustering is defined.
    Memoised per (dataset, embeddings, parameters)."""
    p = params if isinstance(params, Params) else Params(DEFAULTS, params)
    key = ("clusters", tuple(sorted(p.as_dict().items())), id(embeddings))
    ref, clusters = DATASET_CACHE.get(
        dataset, key,
        lambda: (weakref.ref(embeddings) if embeddings is not None else None,
                 tuple(_find_clusters(dataset, embeddings, p))))
    if (ref() if ref is not None else None) is embeddings:
        return list(clusters)
    # id() collision with a dead embedding object: recompute rather than trust a stale entry
    return _find_clusters(dataset, embeddings, p)


def _find_clusters(dataset: Dataset, embeddings: EmbeddingIndex | None, p: Params) -> list[DupCluster]:
    n = n_samples(dataset)
    if n < 2:
        return []
    ids = [s.sample_id for s in dataset.samples]
    h = np.array([phash64(s) for s in dataset.samples], dtype=np.uint64)
    cut = p["hamming"] if embeddings is not None else min(p["hamming"], p["hamming_only"])
    rows: list[int] = []
    cols: list[int] = []
    ham: dict[tuple[int, int], int] = {}
    cos: dict[tuple[int, int], float] = {}
    for (i, j), dist in candidate_pairs(h, cut).items():
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
        ham[(i, j)] = dist
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


@register_detector
class NearDuplicate:
    id = "data.near_dup"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES}
    optional: set[Capability] = set()
    attack_classes = {NEAR_DUPLICATE_FLOODING}

    def detect(self, dataset: Dataset, embeddings, model, ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        self.ctx = as_ctx(ctx)
        return finalise(self._detect(dataset, embeddings, model), ctx)

    def _detect(self, dataset: Dataset, embeddings, model) -> list:
        p = self.p
        clusters = find_clusters(dataset, embeddings, self.p)
        findings = []
        for cl in clusters:
            members = [sample_by_id(dataset, m) for m in cl.members]
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
            sheet = self.ev.sheet(dataset, cl.members, f"cluster of {len(members)}")
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
