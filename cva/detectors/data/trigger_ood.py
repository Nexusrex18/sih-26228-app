"""Trigger artifacts and out-of-distribution insertion — ``trigger_ood.py`` registers TWO ids.

``data.trigger_artifact``  (attack_class ``trigger_injection``, nature ``adversarial``)
    requires DATASET_IMAGES; optional MODEL_ACTIVATIONS, MODEL_PREDICT -> DEGRADED to (a) only.
    Registry shape (backend_plan.md §9.4) rather than the plan's §3 prose: a single id whose
    model-free method (a) must survive the absence of a model cannot put MODEL_* in ``requires``.

    (a) frequency residue  model-free. Squared high-pass residual (pixel minus its 3x3 mean) at
                           NATIVE resolution — a resize would blur a 1-px checkerboard away —
                           area-pooled to a grid x grid block map, log-transformed, robust-z
                           against the dataset's own per-block median/MAD. Only
                           anomalies that repeat at the SAME block across >= max(5, 1%) samples
                           count — a pasted trigger sits where it was pasted; natural texture
                           does not repeat.
    (d) patch saliency     needs MODEL_PREDICT. Occlude the candidate region and measure how many
                           predictions change, against control occlusions of other regions.
                           Targeted escalation: it tests regions (a) located (or
                           ``candidate_samples`` supplied), never the full ~6.5M-pass sweep.
    (b) spectral signatures / (c) activation clustering
                           need MODEL_ACTIVATIONS. ART's scorer / clusterer on the contributed
                           model's own activations — the explicit ADR-009 exception, labelled as
                           depending on the model under test. Circularity caveat: a subtle
                           enough backdoor can suppress its own spectral signature.

    One finding per sample, merging sub-methods (so ``finding_id`` cannot collide). Severity:
    (b)/(c) alone -> ``low`` and confidence <= 0.35 — they NEVER carry a finding by themselves;
    any of (a)/(d) -> ``medium``; two corroborators, or one plus (b)/(c) -> ``high``.

    Honest status: on this repo's synthetic corpus neither ART method separated poisoned from
    clean activations (see tests); their gates are set so they stay silent on clean data, and
    their mechanics are unit-tested on planted activation clusters, not on a model-level
    detection claim.

``data.ood``  (attack_class ``out_of_distribution``, nature ``quality``)
    requires DATASET_IMAGES, REFERENCE_CLEAN_SET. Mean cosine distance to the k nearest
    reference embeddings, thresholded at the reference set's own leave-one-out distribution.
    ``Detector.detect()`` cannot receive the reference set, so it is constructor-injected
    (``reference_set`` + ``reference_embeddings``) — a contract question logged for Backend.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

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
    not_performed,
    register_detector,
    seed_of,
    to_model_input,
)
from .taxonomy import OUT_OF_DISTRIBUTION, TRIGGER_INJECTION

DEFAULTS = {
    "grid": 8, "hp_window": 3, "mad_floor": 0.1, "z_thr": 3.8, "min_baseline": 30,
    "min_group_abs": 5, "min_group_frac": 0.01,
    "flip_min": 0.5, "flip_gap_min": 0.3, "target_conc_min": 0.6, "max_group_samples": 100,
    "control_regions": 3, "sweep_window_cells": 2, "sweep_max_candidates": 20,
    "ss_z": 6.0, "ss_min_class": 20,
    "ac_size_max": 0.2, "ac_silhouette_min": 0.5, "ac_min_class": 20, "ac_dims": 10,
    "seed": None,                 # None -> ctx.rng_seed
    "candidate_samples": None,    # sample_ids to seed the targeted patch-saliency sweep (§6.3.1)
    "activation_layer": None,     # None -> the model's penultimate activation (see limitations)
}
_CORROBORATORS = {"frequency_residue", "patch_saliency"}
_METHOD_CONF = {"frequency_residue": 0.5, "patch_saliency": 0.6,
                "spectral_signature": 0.25, "activation_clustering": 0.25}


# ---------------------------------------------------------------- (a) frequency residue
def hf_energy_map(sample, grid: int, window: int = 3) -> np.ndarray:
    """log mean squared high-pass residual per block, shape (grid, grid). Measured at native
    resolution, THEN pooled: resizing first would destroy a pixel-scale trigger."""
    from PIL import Image
    from scipy.ndimage import uniform_filter

    g = np.asarray(load_rgb(sample).convert("L"), dtype=np.float32)
    r = (g - uniform_filter(g, window, mode="reflect")) ** 2
    pooled = np.asarray(Image.fromarray(r, mode="F").resize((grid, grid), Image.BOX))
    return np.log(pooled + 1.0)


def frequency_residue(dataset: Dataset, p: Params) -> dict[str, dict]:
    """sample_id -> {block:(r,c), z, group_size}, only for anomalies repeating at one block."""
    n = len(dataset)
    if n < p["min_baseline"]:
        return {}
    maps = np.stack([hf_energy_map(s, p["grid"], p["hp_window"]) for s in dataset.samples])
    med = np.median(maps, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(maps - med), axis=0), p["mad_floor"])
    z = (maps - med) / mad
    flat = z.reshape(n, -1)
    arg = flat.argmax(axis=1)
    top = flat[np.arange(n), arg]
    b = maps.shape[1]
    hit = [(i, divmod(int(arg[i]), b), float(top[i])) for i in range(n) if top[i] >= p["z_thr"]]
    by_block: dict[tuple[int, int], list] = defaultdict(list)
    for i, rc, zz in hit:
        by_block[rc].append((i, zz))
    need = max(p["min_group_abs"], math.ceil(p["min_group_frac"] * n))
    out = {}
    for rc, members in by_block.items():
        if len(members) >= need:
            for i, zz in members:
                out[dataset.samples[i].sample_id] = {"block": rc, "z": zz, "group_size": len(members)}
    return out


def _where(rc: tuple[int, int], grid: int) -> str:
    r, c = rc
    v = "top" if r < grid / 3 else "bottom" if r >= 2 * grid / 3 else "middle"
    h = "left" if c < grid / 3 else "right" if c >= 2 * grid / 3 else "centre"
    return "centre" if (v, h) == ("middle", "centre") else f"{v}-{h}".replace("middle-", "")


# ---------------------------------------------------------------- (d) patch saliency
def _occlude(X: np.ndarray, region: tuple[int, int, int, int], grid: int) -> np.ndarray:
    r0, c0, r1, c1 = region
    H, W = X.shape[-2:]
    y0, y1, x0, x1 = r0 * H // grid, r1 * H // grid, c0 * W // grid, c1 * W // grid
    X2 = X.copy()
    X2[:, :, y0:y1, x0:x1] = X.mean(axis=(2, 3), keepdims=True)           # per-image mean colour
    return X2


def occlusion_test(model, X: np.ndarray, region, grid: int, controls: list, seed: int = 0) -> dict:
    base = model.predict(X).argmax(1)
    after = model.predict(_occlude(X, region, grid)).argmax(1)
    flipped = after != base
    ctrl = [float((model.predict(_occlude(X, r, grid)).argmax(1) != base).mean()) for r in controls]
    trans = Counter(zip(base[flipped].tolist(), after[flipped].tolist(), strict=False))
    modal_base, modal_share = Counter(base.tolist()).most_common(1)[0]
    return {"flip_rate": float(flipped.mean()), "control_flip_rate": float(np.mean(ctrl)) if ctrl else 0.0,
            "base_class": int(modal_base), "base_class_share": modal_share / len(base),
            "new_class": (Counter(after[flipped].tolist()).most_common(1)[0][0] if flipped.any() else None),
            "transitions": {f"{a}->{b}": c for (a, b), c in trans.most_common(3)}}


def _control_regions(region, grid: int, k: int, seed: int) -> list:
    r0, c0, r1, c1 = region
    h, w = r1 - r0, c1 - c0
    cands = [(r, c, r + h, c + w) for r in range(0, grid - h + 1) for c in range(0, grid - w + 1)
             if r + h <= r0 or r >= r1 or c + w <= c0 or c >= c1]
    if not cands:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
    return [cands[int(i)] for i in sorted(idx)]


# ---------------------------------------------------------------- (b)/(c) via ART
def spectral_flags(F: np.ndarray, z_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """ART's spectral-signature scores on one class's centred activations -> (flags, robust z)."""
    from art.defences.detector.poison import SpectralSignatureDefense

    sc = SpectralSignatureDefense.spectral_signature_scores(F - F.mean(axis=0))
    med = np.median(sc)
    z = (sc - med) / (1.4826 * np.median(np.abs(sc - med)) + 1e-12)
    return z >= z_thr, z


def cluster_flags(F: np.ndarray, size_max: float, sil_min: float, dims: int, seed: int
                  ) -> tuple[np.ndarray, dict]:
    """ART's activation clustering (2-means on reduced activations) on one class. Flags the
    smaller cluster only if it is small AND well separated — ART's own 'smaller' rule flags
    something in every class of clean data, which is why it is gated here."""
    from art.defences.detector.poison.activation_defence import cluster_activations
    from sklearn.metrics import silhouette_score

    state = np.random.get_state()                    # ART's KMeans reads the global RNG
    np.random.seed(seed)
    try:
        cl, red = cluster_activations([F], nb_clusters=2, nb_dims=dims, reduce="PCA")
    finally:
        np.random.set_state(state)
    lab = np.asarray(cl[0])
    if len(set(lab.tolist())) < 2:
        return np.zeros(len(F), bool), {"small": 0.0, "silhouette": 0.0}
    small_id = int(np.argmin(np.bincount(lab, minlength=2)))
    frac = float((lab == small_id).mean())
    sil = float(silhouette_score(red[0], lab))
    flags = (lab == small_id) if (frac <= size_max and sil >= sil_min) else np.zeros(len(F), bool)
    return flags, {"small": frac, "silhouette": sil}


# ---------------------------------------------------------------- the detector
@register_detector
class TriggerArtifact:
    id = "data.trigger_artifact"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES}
    optional = {Capability.MODEL_ACTIVATIONS, Capability.MODEL_PREDICT}
    attack_classes = {TRIGGER_INJECTION}
    _layer_used = None
    _by_block: dict = {}

    def _has(self, model, cap: Capability) -> bool:
        if model is None:
            return False
        try:
            return cap in model.capabilities()
        except Exception:
            return False

    def detect(self, dataset: Dataset, embeddings, model, ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        self.ctx = as_ctx(ctx)
        cs = self.p["candidate_samples"]
        self.candidates = list(cs) if cs else None
        self._layer_used = None
        return finalise(self._detect(dataset, embeddings, model), ctx)

    def _detect(self, dataset: Dataset, embeddings, model) -> list:
        p = self.p
        grid = p["grid"]
        rec: dict[str, list[dict]] = defaultdict(list)
        notes: list[str] = []

        if len(dataset) < p["min_baseline"]:
            notes.append(f"frequency residue not run: {len(dataset)} images is fewer than the "
                         f"{p['min_baseline']} needed for a per-block baseline")
            if not (self._has(model, Capability.MODEL_PREDICT) or self._has(model, Capability.MODEL_ACTIVATIONS)):
                # nothing model-free could run and no model was available: say so, don't return []
                return [not_performed(self.id, self.version, self.attack_classes, notes[0]
                                      + ", and no model was supplied for the model-based methods")]
        fr = frequency_residue(dataset, p)
        self._by_block = {sid: v["block"] for sid, v in fr.items()}
        for sid, v in fr.items():
            rec[sid].append({"method": "frequency_residue", "raw": v["z"], "thr": p["z_thr"], **v})

        if self._has(model, Capability.MODEL_PREDICT):
            self._saliency(dataset, model, fr, rec, notes, grid)
        else:
            notes.append("patch saliency not run: MODEL_PREDICT absent")
        if self._has(model, Capability.MODEL_ACTIVATIONS):
            self._activation_methods(dataset, model, rec, notes)
        else:
            notes.append("spectral signatures / activation clustering not run: MODEL_ACTIVATIONS absent")

        return [self._finding(dataset, sid, ms, notes, grid, embeddings) for sid, ms in sorted(rec.items())]

    # ---- (d)
    def _saliency(self, dataset, model, fr, rec, notes, grid):
        p = self.p
        groups: dict[Any, list[str]] = defaultdict(list)
        for sid, v in fr.items():
            r, c = v["block"]
            groups[(r, c, r + 1, c + 1)].append(sid)
        if self.candidates:                                             # coarse sweep on named candidates
            region = self._sweep(dataset, model, grid)
            if region is not None:
                have = set(groups[region])                        # NOT the list being appended to
                groups[region].extend(x for x in self.candidates if x not in have)
        for region, ids in groups.items():
            ids = sorted(ids)[: p["max_group_samples"]]
            try:
                X = np.stack([to_model_input(dataset.sample(i), model.input_shape) for i in ids])
                res = occlusion_test(model, X, region, grid, _control_regions(region, grid, p["control_regions"], seed_of(p, self.ctx)))
            except Exception as e:                                       # a bad model output is not a crash
                notes.append(f"patch saliency failed on region {region}: {type(e).__name__}: {e}")
                continue
            res["region"] = region
            if (res["flip_rate"] >= p["flip_min"] and res["flip_rate"] - res["control_flip_rate"] >= p["flip_gap_min"]
                    and res["base_class_share"] >= p["target_conc_min"]):
                for i in ids:
                    rec[i].append({"method": "patch_saliency", "raw": res["flip_rate"], "thr": p["flip_min"], **res})
            else:
                notes.append(f"patch saliency on region {region}: flip {res['flip_rate']:.0%} vs control "
                             f"{res['control_flip_rate']:.0%} — not significant")

    def _sweep(self, dataset, model, grid):
        p, w = self.p, self.p["sweep_window_cells"]
        votes: Counter = Counter()
        for sid in self.candidates[: p["sweep_max_candidates"]]:
            x = to_model_input(dataset.sample(sid), model.input_shape)[None]
            base = model.predict(x)[0]
            b = int(base.argmax())
            best, best_drop = None, 0.0
            for r in range(grid - w + 1):
                for c in range(grid - w + 1):
                    pr = model.predict(_occlude(x, (r, c, r + w, c + w), grid))[0]
                    if int(pr.argmax()) != b and base[b] - pr[b] > best_drop:
                        best, best_drop = (r, c, r + w, c + w), float(base[b] - pr[b])
            if best:
                votes[best] += 1
        return votes.most_common(1)[0][0] if votes else None

    # ---- (b)/(c)
    def _activation_methods(self, dataset, model, rec, notes):
        p = self.p
        keep = [(s, dominant_category(s)) for s in dataset.samples]
        keep = [(s, d) for s, d in keep if d is not None and d < model.num_classes]
        if not keep:
            return
        y = np.array([d for _, d in keep])
        # Batched: images are decoded 64 at a time and only the (N x D) activation matrix is kept —
        # never the whole image tensor (100k x 3x64x64 float32 would be ~4.9 GB).
        chunks, layer = [], None
        for i in range(0, len(keep), 64):
            Xb = np.stack([to_model_input(s, model.input_shape) for s, _ in keep[i:i + 64]])
            a = model.activations(Xb)
            if not a:
                notes.append("activations not extractable: spectral/clustering not run")
                return
            if layer is None:
                names = list(a.keys())
                layer = p["activation_layer"] or (names[-2] if len(names) >= 2 else names[-1])
                if layer not in a:
                    notes.append(f"activation layer {layer!r} not in the model's activations {names}")
                    return
                self._layer_used = layer
            chunks.append(np.asarray(a[layer]).reshape(len(Xb), -1))
        acts = np.concatenate(chunks)
        for cls in sorted(set(y.tolist())):
            idx = np.nonzero(y == cls)[0]
            F = acts[idx]
            if len(idx) >= p["ss_min_class"]:
                flags, z = spectral_flags(F, p["ss_z"])
                for j in np.nonzero(flags)[0]:
                    rec[keep[idx[j]][0].sample_id].append(
                        {"method": "spectral_signature", "raw": float(z[j]), "thr": p["ss_z"], "class": int(cls)})
            if len(idx) >= p["ac_min_class"]:
                flags, info = cluster_flags(F, p["ac_size_max"], p["ac_silhouette_min"], p["ac_dims"], p["seed"])
                for j in np.nonzero(flags)[0]:
                    rec[keep[idx[j]][0].sample_id].append(
                        {"method": "activation_clustering", "raw": info["silhouette"],
                         "thr": p["ac_silhouette_min"], "class": int(cls), **info})

    # ---- assemble
    def _finding(self, dataset, sid, ms, notes, grid, embeddings):
        s = dataset.sample(sid)
        methods = {m["method"]: m for m in ms}
        corr = len(_CORROBORATORS & set(methods))
        model_based = {"patch_saliency", "spectral_signature", "activation_clustering"} & set(methods)
        weak = len({"spectral_signature", "activation_clustering"} & set(methods))
        sev = (Severity.LOW if corr == 0 else Severity.HIGH if (corr >= 2 or weak) else Severity.MEDIUM)
        conf = 1 - np.prod([1 - _METHOD_CONF[m] for m in methods])
        conf = float(min(0.95, conf) if corr else min(0.35, conf))
        primary = methods.get("frequency_residue") or methods.get("patch_saliency") or ms[0]
        parts = []
        if "frequency_residue" in methods:
            f = methods["frequency_residue"]
            parts.append(f"a high-frequency anomaly in the {_where(f['block'], grid)} cell of the image "
                         f"(block {f['block']}, {f['z']:.1f} robust-z above the dataset norm) recurs in "
                         f"{f['group_size']} samples at the same position")
        if "patch_saliency" in methods:
            f = methods["patch_saliency"]
            parts.append(f"occluding that region changes {f['flip_rate']:.0%} of the group's predictions "
                         f"(control regions: {f['control_flip_rate']:.0%}) away from class "
                         f"{dataset.category_name(f['base_class'])!r}"
                         + (f" toward {dataset.category_name(f['new_class'])!r}" if f["new_class"] is not None else ""))
        if "spectral_signature" in methods:
            parts.append(f"a spectral-signature outlier within its class (z={methods['spectral_signature']['raw']:.1f})")
        if "activation_clustering" in methods:
            f = methods["activation_clustering"]
            parts.append(f"it falls in a small, well-separated activation cluster ({f['small']:.0%} of its "
                         f"class, silhouette {f['silhouette']:.2f})")
        who = ""
        if s.contributor is not None:
            who = f" Supplied by {attribution(group_source(dataset, s.contributor), s.contributor)}."
        dep = (" Findings from patch saliency / spectral signatures / activation clustering depend on the "
               "contributed model under test." if model_based else "")
        sheet = None
        if "frequency_residue" in methods:
            blk = methods["frequency_residue"]["block"]
            mates = sorted(k for k, v in self._by_block.items() if v == blk)
            sheet = self.ev.sheet(dataset, mates, f"samples sharing the anomaly at block {blk}")
        return make_finding(
            detector_id=self.id, version=self.version, target_type="sample", target_ref=sid,
            severity=sev, confidence=conf, score_raw=float(primary["raw"]), threshold=float(primary["thr"]),
            reason=(f"Sample {sid}: " + "; ".join(parts) + f". Signals: {', '.join(sorted(methods))}; "
                    f"score_raw is the {primary['method'].replace('_', ' ')} score.{who}{dep}"
                    + ("" if corr else " No model-free corroboration, so severity is capped at low.")),
            attack_class=TRIGGER_INJECTION, nature=Nature.ADVERSARIAL,
            evidence=[self.ev.json({"signals": ms, "notes": notes}, "trigger-artifact signals"), sheet],
            access_assumptions=["DATASET_IMAGES"] + (
                ["MODEL_PREDICT / MODEL_ACTIVATIONS of the contributed model (data-question use)"] if model_based else []),
            limitations=[
                "Defeated by smoothed/blended triggers (a) and by adaptive attackers who blend the trigger's "
                "activation pattern with clean data (b)/(c).",
                "Spectral signatures / activation clustering carry a circularity caveat: a subtle backdoor can "
                "suppress its own signature; they never carry a finding alone.",
                "Assumes category_id == the model's class index.",
                *([f"Spectral/clustering used activation layer {self._layer_used!r}: by default the "
                   "PENULTIMATE entry of the model's activation dict, which assumes that dict is in "
                   "topological order — set activation_layer explicitly for a model where it is not."]
                  if self._layer_used else [])] + notes)


# ---------------------------------------------------------------- data.ood
OOD_DEFAULTS = {"k": 5, "quantile": 0.99, "margin": 1.15, "min_reference": 50,
                "contributor_share": 0.2, "contributor_min": 10}


@register_detector
class OutOfDistribution:
    """Reference embeddings arrive as ``ctx.profile["reference_embeddings"]`` (an ``(M, d)`` array in
    the SAME space as ``embeddings``). ``ctx.probes_x`` is NOT usable for this: it holds raw
    images, and comparing them with embeddings needs an extractor this detector does not (and must
    not) own — an open contract question for Backend, see the module README."""

    id = "data.ood"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.REFERENCE_CLEAN_SET}
    optional: set[Capability] = set()
    attack_classes = {OUT_OF_DISTRIBUTION}

    def detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None, model,
               ctx: CheckContext | None = None) -> list:
        self.p = Params.from_ctx(self.id, OOD_DEFAULTS, ctx)
        self.ev = EvidenceStore(as_ctx(ctx).out_dir)
        return finalise(self._detect(dataset, embeddings, as_ctx(ctx)), ctx)

    def _detect(self, dataset: Dataset, embeddings: EmbeddingIndex | None, ctx: CheckContext) -> list:
        p = self.p
        ref = ctx.profile.get("reference_embeddings")
        if ref is None:
            return [not_performed(
                self.id, self.version, self.attack_classes,
                "no reference embeddings were supplied (ctx.profile['reference_embeddings']); "
                "ctx.probes_x carries raw images and cannot be compared with embeddings without "
                "an extractor")]
        if embeddings is None:
            return [not_performed(self.id, self.version, self.attack_classes, "no embedding index supplied")]
        R = np.asarray(ref, dtype=np.float32)
        if len(R) < p["min_reference"]:
            return [not_performed(self.id, self.version, self.attack_classes,
                                  f"reference set has {len(R)} images (< {p['min_reference']}); "
                                  "too small for a stable threshold")]
        k, share, share_min = p["k"], p["contributor_share"], p["contributor_min"]
        R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        k = min(k, len(R) - 1)
        S = R @ R.T
        np.fill_diagonal(S, -np.inf)
        d_ref = 1 - np.sort(S, axis=1)[:, -k:].mean(axis=1)
        tau = float(np.quantile(d_ref, p["quantile"]) * p["margin"])
        ids = [s.sample_id for s in dataset.samples if embeddings.vector(s.sample_id) is not None]
        V = embeddings.vectors(ids)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        d = 1 - np.sort(V @ R.T, axis=1)[:, -k:].mean(axis=1)
        flagged = {sid: float(dv) for sid, dv in zip(ids, d, strict=False) if dv > tau}
        per_c, tot_c = Counter(), Counter()
        for s in dataset.samples:
            if s.contributor is not None:
                tot_c[s.contributor] += 1
                per_c[s.contributor] += s.sample_id in flagged
        out = []
        for sid, dv in sorted(flagged.items()):
            s = dataset.sample(sid)
            c = s.contributor
            heavy = c is not None and per_c[c] >= share_min and per_c[c] / tot_c[c] >= share
            who = ""
            if c is not None:
                who = (f" {per_c[c]} of {tot_c[c]} samples from {attribution(group_source(dataset, c), c)} "
                       f"are out-of-distribution.")
            out.append(make_finding(
                detector_id=self.id, version=self.version, target_type="sample", target_ref=sid,
                severity=Severity.MEDIUM if heavy else Severity.LOW,
                confidence=float(min(0.9, 0.4 + 0.5 * min(1.0, (dv - tau) / max(tau, 1e-6)))),
                score_raw=dv, threshold=tau,
                reason=(f"Sample {sid} lies {dv:.3f} (mean cosine distance to its {k} nearest reference "
                        f"images) from the clean reference set, beyond the reference's own "
                        f"leave-one-out threshold {tau:.3f} ({len(R)} reference images).{who}"),
                attack_class=OUT_OF_DISTRIBUTION, nature=Nature.QUALITY,
                evidence=[self.ev.json({"distance": dv, "threshold": tau, "k": k,
                                        "reference_n": len(R)}, "OOD distance")],
                access_assumptions=["DATASET_IMAGES", "REFERENCE_CLEAN_SET",
                                    f"embeddings: {embeddings.extractor_id}@{embeddings.extractor_version}"],
                limitations=["Only as good as the reference clean set: a generated reference is "
                             "reproducible, not independently verified clean.",
                             "The threshold is estimated from the reference's own leave-one-out tail; "
                             "with a few hundred images that tail is unstable across draws.",
                             "Legitimately novel but benign imagery is also out-of-distribution."]))
        return out
