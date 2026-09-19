"""model.strip — STRIP, and the only backdoor check that runs on a bare ONNX model.

Superimpose clean images on an input and measure the entropy of the prediction. A clean
model's prediction is destroyed by the superposition, so entropy rises. A backdoored
model's trigger dominates the superposition, so it stays pathologically confident and
entropy collapses.

Needs query access only. Since ONNX exposes no backward pass, Neural Cleanse cannot run
there — on a plausible majority of real deliveries STRIP is the ONLY backdoor check
available, which is why it is benchmarked as a primary method rather than a fallback.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Capability
from cva.core.finding import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register


def entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log2(p)).sum(axis=-1)


def strip_entropies(model, x: np.ndarray, overlay_pool: np.ndarray,
                    n_overlay: int = 16, alpha: float = 0.5,
                    seed: int = 0) -> np.ndarray:
    """Mean perturbed-prediction entropy per input."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(x), dtype=np.float64)
    for i, xi in enumerate(x):
        idx = rng.choice(len(overlay_pool), n_overlay, replace=False)
        blended = np.clip((1 - alpha) * xi[None, ...] + alpha * overlay_pool[idx], 0, 1)
        out[i] = entropy(np.asarray(model.predict(blended.astype(np.float32)))).mean()
    return out


@register
class StripCheck:
    id = "model.strip"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET}
    optional: set = set()
    attack_classes = {"model.backdoor_patch", "model.backdoor_blended"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x = ctx.probes_x
        if x is None or len(x) < 32:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "STRIP needs a clean probe set to superimpose; fewer than 32 images were "
                "available.", (Capability.REFERENCE_CLEAN_SET,), "model.backdoor_patch")]

        n_test = int(ctx.opt("strip_samples", 96))
        n_ovl = int(ctx.opt("strip_overlays", 16))
        rng = np.random.default_rng(ctx.rng_seed)
        sel = rng.choice(len(x), min(n_test, len(x)), replace=False)

        # Baseline: the model's own entropy distribution on clean, unperturbed inputs.
        clean_ent = entropy(np.asarray(model.predict(x[sel].astype(np.float32))))
        pert_ent = strip_entropies(model, x[sel], x, n_ovl, seed=ctx.rng_seed)

        # A clean model's entropy RISES under superposition. Ratio near or below 1 means
        # the model is ignoring the superimposed content — the backdoor signature.
        ratio = float(pert_ent.mean() / max(clean_ent.mean(), 1e-6))
        # Reference-free: judge the LOW tail, where trigger-carrying inputs concentrate.
        p05 = float(np.percentile(pert_ent, 5))
        max_ent = float(np.log2(model.num_classes)) if model.num_classes else 1.0
        norm_low = p05 / max_ent if max_ent else 1.0

        thr = float(ctx.opt("strip_threshold", 0.35))
        plot = _plot(ctx, model.model_id, clean_ent, pert_ent)
        ev = [Evidence("table", "STRIP entropy statistics", data={
            "mean_clean_entropy": round(float(clean_ent.mean()), 4),
            "mean_perturbed_entropy": round(float(pert_ent.mean()), 4),
            "perturbed_p05": round(p05, 4),
            "normalised_low_tail": round(norm_low, 4),
            "max_entropy(log2 K)": round(max_ent, 4),
            "threshold": thr, "samples": len(sel), "overlays_each": n_ovl})]
        if plot:
            ev.append(Evidence("plot", "Clean vs superimposed prediction entropy", path=plot))

        suspicious = norm_low < thr
        conf = float(np.clip((thr - norm_low) / thr, 0, 1) * 0.85 + 0.1) if suspicious else \
            float(np.clip((norm_low - thr) / (1 - thr), 0, 1) * 0.5)
        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.HIGH if suspicious else Severity.INFO,
            confidence=conf, score_raw=norm_low, threshold=thr,
            reason=(
                f"Prediction entropy under superposition collapses to {p05:.3f} at the 5th "
                f"percentile ({norm_low:.3f} of the {max_ent:.2f}-bit maximum), below the "
                f"{thr} threshold. The model stays confident when its input has been "
                "destroyed — the signature of a trigger-conditioned backdoor."
                if suspicious else
                f"Prediction entropy rises normally under superposition (5th percentile "
                f"{p05:.3f} = {norm_low:.3f} of maximum, above the {thr} threshold). No "
                "entropy-collapse signature."),
            attack_class="model.backdoor_patch", evidence=ev,
            access_assumptions=["query access only — runs where gradients are unavailable, "
                                "including every ONNX model"],
            limitations=[
                "Detects triggers that dominate a superposition. Large or global triggers, "
                "and entropy-aware adaptive attacks, are outside its reach.",
                "Threshold is reference-free and therefore conservative; a clean model "
                "with an unusually strong class attractor can score low."],
            disposition=Disposition.QUARANTINE if suspicious else Disposition.ACCEPT,
            disposition_rule="strip.entropy_collapse" if suspicious else "strip.normal",
            nature=Nature.ADVERSARIAL if suspicious else Nature.INDETERMINATE,
        )]


def _plot(ctx, mid, clean, pert) -> str | None:
    if ctx.out_dir is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        d = Path(ctx.out_dir) / "evidence"
        d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=130)
        ax.hist(clean, bins=24, alpha=0.65, label="clean input", color="#5b8def")
        ax.hist(pert, bins=24, alpha=0.65, label="superimposed", color="#e0544c")
        ax.set_xlabel("prediction entropy (bits)"); ax.set_ylabel("count")
        ax.set_title(f"STRIP — {mid}", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout()
        p = d / f"strip_{mid}.png"
        fig.savefig(p); plt.close(fig)
        return f"evidence/{p.name}"
    except Exception:
        return None
