from __future__ import annotations

import numpy as np
import pytest

from attacklab.near_dup_ood_attack import inject_ood
from attacklab.synth_dataset import make_clean_dataset
from cva.core.capability import Capability
from cva.detectors.data._stub_types import stub_embeddings
from cva.detectors.data.trigger_ood import (
    OutOfDistribution,
    TriggerArtifact,
    cluster_flags,
    spectral_flags,
)
from tests.detectors.data.helpers import detect, make_ctx, reference_matrix, resolve


def test_registry_rows():
    assert {str(c) for c in TriggerArtifact.requires} == {"DATASET_IMAGES"}
    assert {str(c) for c in TriggerArtifact.optional} == {"MODEL_ACTIVATIONS", "MODEL_PREDICT"}
    assert {str(c) for c in OutOfDistribution.requires} == {"DATASET_IMAGES", "REFERENCE_CLEAN_SET"}


def test_state_machine_degrades_without_a_model(clean):
    r = resolve(TriggerArtifact, clean)
    assert r.state.value == "DEGRADED" and r.runnable            # model-free (a) still runs
    r2 = resolve(TriggerArtifact, clean, extra={Capability.MODEL_ACTIVATIONS, Capability.MODEL_PREDICT})
    assert r2.state.value == "OK"
    assert resolve(OutOfDistribution, clean).state.value == "UNAVAILABLE"      # no reference


# ------------------------------------------------------------------ (a) model-free
def test_clean_zero_findings_model_free(clean):
    assert detect(TriggerArtifact, clean, None, None) == []


def test_frequency_residue_finds_the_trigger_and_names_where(poisoned):
    ds, trig = poisoned
    fs = detect(TriggerArtifact, ds, None, None)
    flagged = {f.target_ref for f in fs}
    assert flagged <= trig, flagged - trig                        # precision: nothing clean flagged
    assert len(flagged) >= 0.9 * len(trig)                        # recall
    f = fs[0]
    assert f.attack_class == "trigger_injection" and f.nature.value == "adversarial"
    assert f.severity.value == "medium"                           # one model-free corroborator
    assert "bottom-right" in f.reason and "recurs in" in f.reason and "'B'" in f.reason
    assert any("no model" in n.lower() or "absent" in n for n in f.limitations)   # says what was not run


def test_isolated_anomalies_do_not_count(clean, workdir):
    """One heavily-textured image is natural; a pasted trigger repeats at one position."""
    from tests.detectors.data.triggers import inject_trigger
    ds, ids = inject_trigger(clean, workdir / "one", seed=5, n=2, target_class=0, contributor="B")
    assert detect(TriggerArtifact, ds, None, None) == []          # 2 < max(5, 1%)


# ------------------------------------------------------------------ (d) patch saliency
def test_patch_saliency_corroborates_and_raises_severity(poisoned, poisoned_model):
    ds, trig = poisoned
    fs = detect(TriggerArtifact, ds, None, poisoned_model)
    hi = [f for f in fs if f.severity.value == "high"]
    assert hi and {f.target_ref for f in hi} <= trig
    f = hi[0]
    assert "occluding that region changes" in f.reason and "control regions" in f.reason
    assert "depend on the contributed model" in f.reason
    assert any("MODEL_PREDICT" in a for a in f.access_assumptions)


def test_clean_dataset_clean_model_zero_findings(clean, clean_model):
    assert detect(TriggerArtifact, clean, None, clean_model) == []


def test_sweep_on_named_candidates_finds_region_without_frequency_residue(poisoned, poisoned_model):
    ds, trig = poisoned
    cand = sorted(trig)[:12]
    # (a) disabled -> only the sweep; the candidates arrive through ctx.profile, not a constructor
    fs = detect(TriggerArtifact, ds, None, poisoned_model,
                params={"z_thr": 1e9, "candidate_samples": cand})
    assert fs and {f.target_ref for f in fs} <= set(cand)
    assert all("frequency" not in f.reason.lower().split("signals:")[1].split(";")[0] for f in fs)


# ------------------------------------------------------------------ (b)/(c) mechanics + never alone
def _planted(n_clean=120, n_poison=8, d=32, shift=9.0, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (n_clean + n_poison, d))
    F[n_clean:, 0] += shift                  # a separated minority cluster in activation space
    return F, np.arange(len(F)) >= n_clean


def test_spectral_and_clustering_flag_a_planted_minority_and_not_null_data():
    F, planted = _planted()
    sf, _ = spectral_flags(F, 6.0)
    cf, info = cluster_flags(F, 0.2, 0.5, 10, seed=0)
    assert sf[planted].all() and not sf[~planted].any()
    assert cf[planted].all() and not cf[~planted].any()
    null = np.random.default_rng(1).normal(0, 1, (128, 32))
    assert not spectral_flags(null, 6.0)[0].any() and not cluster_flags(null, 0.2, 0.5, 10, 0)[0].any()


def test_model_based_methods_alone_never_exceed_low(clean):
    from cva.detectors.data.base import EvidenceStore
    det = TriggerArtifact()
    det.ev = EvidenceStore(None)                 # state detect() would have set up
    sid = clean.samples[0].sample_id
    recs = [{"method": "spectral_signature", "raw": 9.0, "thr": 6.0},
            {"method": "activation_clustering", "raw": 0.9, "thr": 0.5, "small": 0.1, "silhouette": 0.9}]
    f = det._finding(clean, sid, recs, [], 8, None)
    assert f.severity.value == "low" and f.confidence <= 0.35
    assert "capped at low" in f.reason and "depend on the contributed model" in f.reason


def test_activation_methods_stay_silent_on_clean_model(clean, clean_model):
    fs = detect(TriggerArtifact, clean, None, clean_model)
    assert not [f for f in fs if "spectral" in f.reason or "activation cluster" in f.reason]


# ------------------------------------------------------------------ data.ood
@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    d = make_clean_dataset(tmp_path_factory.mktemp("ref") / "r", seed=99, n=200)
    return d, stub_embeddings(d)


def test_ood_unavailable_without_reference_embeddings(clean, clean_emb):
    fs = detect(OutOfDistribution, clean, clean_emb, None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
    assert "reference embeddings" in fs[0].reason and "probes_x" in fs[0].reason


def test_ood_reference_arrives_through_ctx_not_a_constructor(clean, clean_emb, reference):
    rds, remb = reference
    assert OutOfDistribution().detect(clean, clean_emb, None,
        make_ctx(profile={"reference_embeddings": reference_matrix(rds, remb)})) is not None


def test_ood_clean_near_zero(clean, clean_emb, reference):
    rds, remb = reference
    fs = detect(OutOfDistribution, clean, clean_emb, None,
                profile={"reference_embeddings": reference_matrix(rds, remb)})
    assert len(fs) <= 0.02 * len(clean), [f.reason for f in fs]        # stated tolerance: <= 2%


def test_ood_mechanism_with_explicit_vectors():
    """Tests the k-NN / leave-one-out-threshold LOGIC with hand-built vectors — nothing fitted to a
    fixture. A point on the reference manifold is not flagged; one orthogonal to it is."""
    from pathlib import Path

    from cva.detectors.data._stub_types import ArrayEmbeddingIndex, Dataset, Sample
    rng = np.random.default_rng(0)
    d = 16
    axis = np.zeros(d); axis[0] = 1.0
    ref = axis + rng.normal(0, 0.05, (120, d))                       # a tight in-distribution cloud
    inlier = axis + rng.normal(0, 0.05, d)
    outlier = np.zeros(d); outlier[7] = 1.0                          # orthogonal to everything in ref
    ids = ["in", "out"]
    ds = Dataset([Sample(i, "x", Path("/nonexistent"), 8, 8) for i in ids], [])
    emb = ArrayEmbeddingIndex(ids, np.stack([inlier, outlier]))
    fs = detect(OutOfDistribution, ds, emb, None, profile={"reference_embeddings": ref})
    assert [f.target_ref for f in fs] == ["out"]
    assert fs[0].score_raw > fs[0].threshold and fs[0].attack_class == "out_of_distribution"
    assert "leave-one-out" in fs[0].reason


@pytest.mark.xfail(strict=False, reason=(
    "CIRCULAR FIXTURE, not evidence. The stub embedding cannot separate the synthetic OOD "
    "textures on its own: an earlier version passed only because a hard-coded statistics block "
    "(_STATS_MU/_STATS_SD) had been fitted to this very attack's images, which is why that block "
    "was removed. Kept as a marker for the real test: OOD detection needs the real backbone and "
    "a real reference set. test_ood_mechanism_with_explicit_vectors covers the logic."))
def test_ood_insertion_found(clean, reference, workdir):
    rds, remb = reference
    ds, m = inject_ood(clean, workdir / "ood1", seed=41, n=30, target_contributor="B")
    fs = detect(OutOfDistribution, ds, stub_embeddings(ds), None,
                profile={"reference_embeddings": reference_matrix(rds, remb)})
    flagged, inj = {f.target_ref for f in fs}, set(m["injected"])
    assert len(flagged & inj) >= 0.8 * len(inj) and len(flagged - inj) <= 0.02 * len(clean)


def test_tiny_dataset_without_a_model_is_reported_not_silently_empty(clean):
    small = type(clean)(clean.samples[:12], clean.categories)
    fs = detect(TriggerArtifact, small, None, None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "baseline" in fs[0].reason
