"""model.strip — an INPUT-level trigger detector, correctly scoped.

CORRECTED 2026-09-19. This was specified, in the plan and in the first implementation, as a
model-level backdoor check and as "the only backdoor check that runs on ONNX". Both are
wrong, and measurement is what showed it.

STRIP asks: does THIS INPUT carry a trigger? It superimposes clean images on the input and
watches whether the prediction survives. A triggered input keeps its prediction, because
the trigger dominates the superposition; a clean input does not.

From a clean probe set there is no trigger to find, so the statistic collapses onto "how
confident is this model in general" — a training artifact, not a backdoor signature.
Measured on a matched pair, the backdoored model scored 0.049 and the clean model 0.133;
both sit far below any fixed threshold, so the check had a 100% false-alarm rate.

It therefore REQUIRES a suspect input set and reports UNAVAILABLE without one. The
model-level, reference-free, black-box question belongs to model.universal_margin.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Capability
from cva.core.types import (Disposition, Evidence, Finding, Nature, Severity,
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
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET,
                Capability.SUSPECT_INPUTS}
    optional: set = set()
    attack_classes = {"backdoor_trigger"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x = ctx.probes_x
        suspect = getattr(ctx, "suspect_x", None)
        if suspect is None or len(suspect) == 0:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "STRIP is an input-level trigger detector and needs inputs that MAY carry "
                "a trigger. Only a clean probe set was supplied, on which the statistic "
                "measures the model's general confidence rather than a backdoor. The "
                "model-level question is covered by model.universal_margin.",
                (Capability.SUSPECT_INPUTS,), "backdoor_trigger")]
        if x is None or len(x) < 32:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "needs at least 32 clean images to superimpose",
                (Capability.REFERENCE_CLEAN_SET,), "backdoor_trigger")]

        n_test = int(ctx.opt("strip_samples", 96))
        n_ovl = int(ctx.opt("strip_overlays", 16))
        rng = np.random.default_rng(ctx.rng_seed)
        sel = rng.choice(len(x), min(n_test, len(x)), replace=False)

        # Baseline from KNOWN-CLEAN inputs, then score the suspect set against it. The
        # comparison must be between two input populations through one model, never
        # between one population and an absolute constant.
        clean_ent = strip_entropies(model, x[sel], x, n_ovl, seed=ctx.rng_seed)
        pert_ent = strip_entropies(model, suspect[:n_test], x, n_ovl, seed=ctx.rng_seed)

        # A clean model's entropy RISES under superposition. Ratio near or below 1 means
        # the model is ignoring the superimposed content — the backdoor signature.
        # Fraction of suspect inputs whose superimposed entropy falls below the 1st
        # percentile of the clean baseline. A relative test, so it transfers across models
        # and tasks; the absolute threshold in the first implementation did not.
        floor = float(np.percentile(clean_ent, 1))
        norm_low = float((pert_ent < floor).mean())
        p05 = float(np.percentile(pert_ent, 5))
        max_ent = float(np.log2(model.num_classes)) if model.num_classes else 1.0

        thr = float(ctx.opt("strip_flagged_fraction", 0.05))
        plot = _plot(ctx, model.model_id, clean_ent, pert_ent)
        ev = [Evidence("table", "STRIP entropy statistics", data={
            "clean_baseline_mean": round(float(clean_ent.mean()), 4),
            "suspect_mean": round(float(pert_ent.mean()), 4),
            "clean_p01_floor": round(floor, 4),
            "suspect_p05": round(p05, 4),
            "fraction_below_floor": round(norm_low, 4),
            "max_entropy(log2 K)": round(max_ent, 4),
            "threshold": thr, "samples": len(sel), "overlays_each": n_ovl})]
        if plot:
            ev.append(Evidence("plot", "Clean vs superimposed prediction entropy", path=plot))

        suspicious = norm_low > thr
        conf = (float(np.clip((norm_low - thr) / max(1 - thr, 1e-6), 0, 1) * 0.8 + 0.15)
                if suspicious else 0.0)
        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.HIGH if suspicious else Severity.INFO,
            confidence=conf, score_raw=norm_low, threshold=thr,
            reason=(
                f"{norm_low*100:.1f}% of the suspect inputs keep their prediction under "
                f"superposition, falling below the clean baseline's 1st percentile "
                f"({floor:.3f} bits) — above the {thr*100:.0f}% threshold. Those inputs "
                "carry something the superposition cannot destroy."
                if suspicious else
                f"Suspect inputs behave like the clean baseline under superposition "
                f"({norm_low*100:.1f}% below its 1st percentile, threshold "
                f"{thr*100:.0f}%). No trigger-carrying inputs identified."),
            attack_class="backdoor_trigger", evidence=ev,
            access_assumptions=["query access only",
                                "a suspect input set was supplied alongside clean probes"],
            limitations=[
                "Identifies trigger-carrying INPUTS, not a backdoored model. A clean "
                "result means these inputs look clean, not that the model is.",
                "Detects triggers that dominate a superposition. Large or global triggers, "
                "and entropy-aware adaptive attacks, are outside its reach."],
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
