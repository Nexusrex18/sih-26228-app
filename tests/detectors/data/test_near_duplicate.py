from __future__ import annotations

import numpy as np

from attacklab.near_dup_ood_attack import inject_near_duplicates
from cva.detectors.data._stub_types import ArrayEmbeddingIndex, stub_embeddings
from cva.detectors.data.near_duplicate import NearDuplicate, find_clusters
from tests.detectors.data.helpers import detect, resolve


def test_registry_row():
    assert {str(c) for c in NearDuplicate.requires} == {"DATASET_IMAGES"}


def test_clean_zero_findings(clean, clean_emb):
    assert resolve(NearDuplicate, clean).runnable
    assert detect(NearDuplicate, clean, clean_emb, None) == []


def test_flooding_found_against_ground_truth(clean, workdir):
    ds, m = inject_near_duplicates(clean, workdir / "nd", seed=21, n_sources=5,
                                   variants_per_source=8, target_contributor="B")
    emb = stub_embeddings(ds)
    fs = detect(NearDuplicate, ds, emb, None)
    flagged = {f.target_ref for f in fs}
    injected, sources = set(m["injected"]), set(m["sources"])
    # precision: nothing outside injected-or-source is flagged
    assert flagged <= injected | sources, flagged - injected - sources
    # recall: nearly every injected variant is caught
    assert len(flagged & injected) >= 0.85 * len(injected), (len(flagged & injected), len(injected))
    # one cluster per source, source included
    cl = find_clusters(ds, emb)
    assert len(cl) == 5 and all(set(c.members) & sources for c in cl)
    f = next(x for x in fs if x.target_ref in injected)
    assert f.attack_class == "near_duplicate_flooding" and f.nature.value == "quality"
    assert f.severity.value in ("medium", "high")
    assert "near-duplicate" in f.reason and "cosine" in f.reason and "'B'" in f.reason
    assert f.score_raw >= f.threshold and f.threshold == 0.97


def test_embedding_confirmation_rejects_phash_false_candidates(clean, workdir):
    ds, m = inject_near_duplicates(clean, workdir / "nd2", seed=22, n_sources=3, variants_per_source=4)
    rng = np.random.default_rng(0)
    orthogonal = ArrayEmbeddingIndex([s.sample_id for s in ds.samples],
                                     np.eye(len(ds.samples), 64 * 4)[:, :256] + rng.normal(0, 1e-3, (len(ds.samples), 256)))
    assert detect(NearDuplicate, ds, orthogonal, None) == []


def test_degrades_to_phash_only_and_says_so(clean, workdir):
    ds, m = inject_near_duplicates(clean, workdir / "nd3", seed=23, n_sources=3, variants_per_source=4)
    fs = detect(NearDuplicate, ds, None, None)
    assert fs, "pHash-only should still catch some exact-ish duplicates"
    assert all(f.confidence <= 0.5 for f in fs)
    assert all(any("pHash-only" in l for l in f.limitations) for f in fs)
    assert all("ONLY" in f.reason for f in fs)


def test_evidence_is_content_addressed(clean, workdir, tmp_path):
    ds, _ = inject_near_duplicates(clean, workdir / "nd4", seed=24, n_sources=2, variants_per_source=4)
    emb = stub_embeddings(ds)
    a = detect(NearDuplicate, ds, emb, None, out_dir=tmp_path / "e1")
    b = detect(NearDuplicate, ds, emb, None, out_dir=tmp_path / "e2")
    pa = sorted(e.path for f in a for e in f.evidence if e.path)
    pb = sorted(e.path for f in b for e in f.evidence if e.path)
    assert pa == pb and pa and all(p.startswith("evidence/") and len(p.split("/")[1]) > 60 for p in pa)


def test_multi_index_hashing_finds_exactly_what_brute_force_finds():
    """candidate_pairs() must be EXACT (pigeonhole), not approximate — same pairs, same distances."""
    from cva.detectors.data.near_duplicate import _popcount, candidate_pairs
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, size=300, dtype=np.uint64)
    h = base.copy()
    for i in range(0, 300, 3):                       # plant close pairs at assorted distances
        flips = rng.choice(64, size=int(rng.integers(0, 9)), replace=False)
        v = int(base[i])
        for b in flips:
            v ^= 1 << int(b)
        h[(i + 1) % 300] = np.uint64(v)
    for cut in (4, 8):
        brute = {}
        D = _popcount(h[:, None] ^ h[None, :])
        for i in range(300):
            for j in range(i + 1, 300):
                if D[i, j] <= cut:
                    brute[(i, j)] = int(D[i, j])
        assert candidate_pairs(h, cut) == brute
        assert brute, "fixture must actually contain pairs"


def test_zero_arg_construction_like_the_orchestrator(clean, clean_emb):
    from tests.detectors.data.helpers import make_ctx
    f = NearDuplicate().detect(clean, clean_emb, None, make_ctx(scan_id="abc"))
    assert f == []                      # constructed with NO arguments, configured only by ctx


def test_clusters_are_computed_once_and_shared(clean, workdir, monkeypatch):
    """A13: near_dup and duplicate_label_conflict cluster with the same parameters; the pHash pass
    (the expensive part) must run once per scan, not twice."""
    from attacklab.near_dup_ood_attack import inject_near_duplicates
    from cva.detectors.data import near_duplicate as nd
    from cva.detectors.data.duplicate_label_conflict import DuplicateLabelConflict
    ds, _ = inject_near_duplicates(clean, workdir / "cache", seed=25, n_sources=3, variants_per_source=4,
                                   conflict_rate=0.3)
    emb = stub_embeddings(ds)
    calls = {"n": 0}
    real = nd.phash64
    monkeypatch.setattr(nd, "phash64", lambda s: (calls.__setitem__("n", calls["n"] + 1), real(s))[1])
    a = detect(NearDuplicate, ds, emb)
    first = calls["n"]
    b = detect(DuplicateLabelConflict, ds, emb)
    assert first == len(ds.samples) and calls["n"] == first, "second detector must reuse the clusters"
    assert a and b
    # a different parameter set, or different embeddings, must NOT reuse a stale result
    detect(NearDuplicate, ds, emb, params={"cosine": 0.99})
    assert calls["n"] == 2 * len(ds.samples)
    detect(NearDuplicate, ds, None)
    assert calls["n"] == 3 * len(ds.samples)
