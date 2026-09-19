from __future__ import annotations

import numpy as np

from attacklab.near_dup_ood_attack import inject_near_duplicates
from cva.detectors.data._stub_types import ArrayEmbeddingIndex, stub_embeddings
from cva.detectors.data.near_duplicate import NearDuplicate, find_clusters
from conftest import resolve


def test_registry_row():
    assert {str(c) for c in NearDuplicate.requires} == {"DATASET_IMAGES"}


def test_clean_zero_findings(clean, clean_emb):
    assert resolve(NearDuplicate, clean).runnable
    assert NearDuplicate().detect(clean, clean_emb, None) == []


def test_flooding_found_against_ground_truth(clean, workdir):
    ds, m = inject_near_duplicates(clean, workdir / "nd", seed=21, n_sources=5,
                                   variants_per_source=8, target_contributor="B")
    emb = stub_embeddings(ds)
    fs = NearDuplicate().detect(ds, emb, None)
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
                                     np.eye(len(ds), 64 * 4)[:, :256] + rng.normal(0, 1e-3, (len(ds), 256)))
    assert NearDuplicate().detect(ds, orthogonal, None) == []


def test_degrades_to_phash_only_and_says_so(clean, workdir):
    ds, m = inject_near_duplicates(clean, workdir / "nd3", seed=23, n_sources=3, variants_per_source=4)
    fs = NearDuplicate().detect(ds, None, None)
    assert fs, "pHash-only should still catch some exact-ish duplicates"
    assert all(f.confidence <= 0.5 for f in fs)
    assert all(any("pHash-only" in l for l in f.limitations) for f in fs)
    assert all("ONLY" in f.reason for f in fs)


def test_evidence_is_content_addressed(clean, workdir, tmp_path):
    ds, _ = inject_near_duplicates(clean, workdir / "nd4", seed=24, n_sources=2, variants_per_source=4)
    emb = stub_embeddings(ds)
    a = NearDuplicate(evidence_dir=tmp_path / "e1").detect(ds, emb, None)
    b = NearDuplicate(evidence_dir=tmp_path / "e2").detect(ds, emb, None)
    pa = sorted(e.path for f in a for e in f.evidence if e.path)
    pb = sorted(e.path for f in b for e in f.evidence if e.path)
    assert pa == pb and pa and all(p.startswith("evidence/") and len(p.split("/")[1]) > 60 for p in pa)
