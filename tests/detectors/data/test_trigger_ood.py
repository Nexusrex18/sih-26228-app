from __future__ import annotations

import numpy as np
import pytest

from attacklab.near_dup_ood_attack import inject_ood
from attacklab.synth_dataset import make_clean_dataset
from cva.core.capability import Capability, CapabilitySet
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
    assert len(fs) <= 0.02 * len(clean.samples), [f.reason for f in fs]        # stated tolerance: <= 2%


def test_ood_mechanism_with_explicit_vectors():
    """Tests the k-NN / leave-one-out-threshold LOGIC with hand-built vectors — nothing fitted to a
    fixture. A point on the reference manifold is not flagged; one orthogonal to it is."""
    from pathlib import Path

    from attacklab._types import Dataset, Sample
    from cva.detectors.data._stub_types import ArrayEmbeddingIndex
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


@pytest.mark.xfail(strict=True, reason=(
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
    assert len(flagged & inj) >= 0.8 * len(inj) and len(flagged - inj) <= 0.02 * len(clean.samples)


def test_tiny_dataset_without_a_model_is_reported_not_silently_empty(clean):
    small = type(clean)(clean.samples[:12], clean.categories)
    fs = detect(TriggerArtifact, small, None, None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "baseline" in fs[0].reason


# ------------------------------------------------------------------ determinism (regression)
class _PlantedActivations:
    """A model whose penultimate activations are fixed per image (looked up by content): class
    clusters, plus a planted, well-separated minority inside class 0 — what activation clustering
    is built to find. No torch, no training: only the seeding of the clustering is under test."""
    model_id, fmt, num_classes, input_shape = "planted", "callable", 4, (3, 64, 64)

    def __init__(self, ds, planted):
        import hashlib
        from cva.detectors.data.base import dominant_category, to_model_input
        rng = np.random.default_rng(3)
        self.table = {}
        for s in ds.samples:
            c = dominant_category(s)
            v = rng.normal(0, 1, 32) + 3.0 * np.eye(32)[c]
            if s.sample_id in planted:
                v[5] += 12.0
            self.table[hashlib.sha1(to_model_input(s, self.input_shape).tobytes()).hexdigest()] = v
        self._h = hashlib

    def activations(self, x):
        F = np.stack([self.table[self._h.sha1(a.tobytes()).hexdigest()] for a in x]).astype(np.float32)
        return {"fc1": F, "logits": F[:, :4]}

    def capabilities(self):
        return CapabilitySet(frozenset({Capability.MODEL_ACTIVATIONS}))


def _signature(findings):
    return sorted((f.target_ref, f.finding_id, round(f.score_raw, 12), f.severity.value,
                   round(f.confidence, 12), f.reason) for f in findings)


def test_activation_clustering_is_deterministic_across_scans(clean):
    """Regression: the seed fix routed ctx.rng_seed into patch saliency, but activation clustering
    kept passing DEFAULTS['seed'] (None) to np.random.seed -> OS entropy -> a different silhouette
    (the finding's score_raw, and the value that gates the flag) on every scan of the same data.

    Needs an AMBIGUOUS clustering to be sensitive: well-separated clusters converge to the same
    answer whatever the seed. So: noisy activations, gates loosened so findings exist."""
    model = _PlantedActivations(clean, set())                        # no planted structure at all
    loose = {"ac_silhouette_min": 0.02, "ac_size_max": 0.5}
    runs = [detect(TriggerArtifact, clean, None, model, params=loose, seed=7) for _ in range(6)]
    assert runs[0], "loosened gates must produce activation-clustering findings, or this tests nothing"
    assert any("activation cluster" in f.reason for f in runs[0])       # the path under test ran
    assert all(_signature(r) == _signature(runs[0]) for r in runs[1:])


def test_activation_methods_run_end_to_end_on_a_planted_cluster(clean):
    """Regression: when spectral signatures actually flagged a sample, float(z[j]) raised
    TypeError (ART returns a 2-D array) — a crash no earlier test reached, because no test's model
    ever produced a flag. The planted minority must be found and reported, not crash the detector."""
    from cva.detectors.data.base import dominant_category
    planted = {s.sample_id for s in clean.samples if dominant_category(s) == 0}
    planted = set(sorted(planted)[:8])
    model = _PlantedActivations(clean, planted)
    fs = detect(TriggerArtifact, clean, None, model, seed=7)
    got = {f.target_ref for f in fs}
    assert planted <= got, f"planted samples not reported: {sorted(planted - got)}"
    assert any("spectral-signature outlier" in f.reason for f in fs)   # the crashing path ran
    assert all(f.severity.value == "low" and f.confidence <= 0.35 for f in fs)   # never alone


def test_tiny_dataset_WITH_a_model_is_reported_not_silently_empty(clean):
    """The other half of the never-a-silent-skip rule: with a model present but too little data for
    ANY sub-method (fewer images than the baseline needs, classes smaller than the clustering
    minimum, no region to occlude), an empty list would read as 'checked, nothing wrong'."""
    small = type(clean)(clean.samples[:12], clean.categories)
    fs = detect(TriggerArtifact, small, None, _PlantedActivations(small, set()))
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
    assert "no sub-method could run" in fs[0].reason and "baseline" in fs[0].reason


def test_a_model_whose_capabilities_raise_is_an_error_not_a_quiet_skip(clean):
    """Defects must surface (the orchestrator turns a raise into ERROR); a blanket except that turned
    them into 'capability absent' once hid a NameError in a test fixture."""
    class Broken:
        model_id, fmt, num_classes, input_shape = "broken", "callable", 4, (3, 64, 64)
        def capabilities(self):
            raise RuntimeError("probe blew up")
    with pytest.raises(RuntimeError, match="probe blew up"):
        detect(TriggerArtifact, clean, None, Broken())


def test_per_scan_state_is_per_instance_not_shared(clean):
    a, b = TriggerArtifact(), TriggerArtifact()
    assert a._by_block is not b._by_block and a._ran is not b._ran
