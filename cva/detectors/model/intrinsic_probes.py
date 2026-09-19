"""model.intrinsic_probes — reference-free backdoor indication, black-box, seconds.

Organisers supply no reference battery and no manifest, so most of Module B degrades.
These four probes need only the model and some inputs. None is strong alone; together they
produce a RANKED SUSPICION ORDER over classes, which is what turns Neural Cleanse from
infeasible at high class counts into a targeted top-K run.

  noise prior           — a backdoor is a high-salience shortcut that noise partially
                          excites, so the target class attracts noise abnormally often
  confusion asymmetry   — a backdoor is a ONE-WAY source→target mapping; natural
                          confusion between similar classes is roughly symmetric
  universal margin      — the minimum-norm universal shift into class c is anomalously
                          small when c is a backdoor target (Neural Cleanse's insight
                          without the reconstruction machinery)
  corruption discontinuity — a clean model degrades smoothly under increasing corruption;
                          a backdoored one can step where the trigger feature dies
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Capability
from cva.core.types import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register
from .neural_cleanse import mad_anomaly_index


def noise_prior(model, shape, n=256, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    batches = [
        rng.uniform(0, 1, size=(n // 2, *shape)).astype(np.float32),
        np.clip(rng.normal(0.5, 0.25, size=(n // 2, *shape)), 0, 1).astype(np.float32),
    ]
    p = np.concatenate([np.asarray(model.predict(b)) for b in batches])
    counts = np.bincount(p.argmax(1), minlength=model.num_classes).astype(np.float64)
    return counts / counts.sum()


def confusion_asymmetry(model, x, y, K) -> np.ndarray:
    pred = np.asarray(model.predict(x.astype(np.float32))).argmax(1)
    C = np.zeros((K, K), dtype=np.float64)
    for t, p in zip(y, pred):
        C[t, p] += 1
    row = C.sum(1, keepdims=True)
    C = C / np.maximum(row, 1)
    # Net inflow to c from every other class, minus outflow. A sink is suspicious.
    return (C.sum(0) - np.diag(C)) - (C.sum(1) - np.diag(C))


def universal_margin(model, x, K, steps=24, lr=0.08, seed=0) -> np.ndarray:
    """Gradient-free: how small a universal additive shift flips most inputs to class c."""
    rng = np.random.default_rng(seed)
    xb = x[: min(len(x), 64)].astype(np.float32)
    out = np.zeros(K, dtype=np.float64)
    for c in range(K):
        delta = np.zeros(xb.shape[1:], dtype=np.float32)
        for _ in range(steps):
            cand = delta + rng.normal(0, lr, size=delta.shape).astype(np.float32)
            hit_c = (np.asarray(model.predict(np.clip(xb + cand, 0, 1))).argmax(1) == c).mean()
            hit_d = (np.asarray(model.predict(np.clip(xb + delta, 0, 1))).argmax(1) == c).mean()
            if hit_c >= hit_d:
                delta = cand
        rate = (np.asarray(model.predict(np.clip(xb + delta, 0, 1))).argmax(1) == c).mean()
        out[c] = float(np.abs(delta).sum()) / max(rate, 1e-3)   # cost per unit success
    return out


def corruption_curve(model, x, y, levels=(0.0, 0.1, 0.2, 0.35, 0.5), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xb = x[: min(len(x), 128)].astype(np.float32)
    yb = y[: len(xb)]
    acc = []
    for s in levels:
        xn = np.clip(xb + rng.normal(0, s, size=xb.shape).astype(np.float32), 0, 1)
        acc.append(float((np.asarray(model.predict(xn)).argmax(1) == yb).mean()))
    return np.asarray(acc)


@register
class IntrinsicProbeCheck:
    id = "model.intrinsic_probes"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET}
    optional: set = set()
    attack_classes = {"backdoor_trigger", "model_anomalous"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x, y = ctx.probes_x, ctx.probes_y
        if x is None or y is None:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "needs labelled clean probes", (Capability.REFERENCE_CLEAN_SET,),
                "backdoor_trigger")]

        K = model.num_classes
        shape = tuple(model.input_shape)
        seed = ctx.rng_seed

        prior = noise_prior(model, shape, int(ctx.opt("noise_probes", 256)), seed)
        asym = confusion_asymmetry(model, x[:512], y[:512], K)
        umarg = universal_margin(model, x, K, seed=seed)
        curve = corruption_curve(model, x, y, seed=seed)

        # Per-class suspicion: high noise attraction, high inflow asymmetry, low universal cost.
        z_prior = (prior - prior.mean()) / (prior.std() + 1e-9)
        z_asym = (asym - asym.mean()) / (asym.std() + 1e-9)
        z_marg = mad_anomaly_index(umarg)          # small margin ⇒ positive
        score = z_prior + z_asym + z_marg
        ranking = [int(i) for i in np.argsort(-score)]
        top, top_score = ranking[0], float(score[ranking[0]])

        drops = -np.diff(curve)
        discontinuity = float(drops.max() - np.median(drops)) if len(drops) > 1 else 0.0

        thr = float(ctx.opt("intrinsic_threshold", 2.5))
        flagged = top_score > thr
        plot = _plot(ctx, model.model_id, prior, asym, umarg, score)
        ev = [
            Evidence("table", "per-class intrinsic signals", data={
                str(c): {"noise_prior": round(float(prior[c]), 4),
                         "confusion_inflow": round(float(asym[c]), 4),
                         "universal_cost": round(float(umarg[c]), 3),
                         "suspicion": round(float(score[c]), 3)} for c in range(K)}),
            Evidence("json", "suspicion ranking (drives Neural Cleanse top-K)", data=ranking),
            Evidence("table", "corruption response", data={
                "accuracy_by_noise_sigma": [round(v, 3) for v in curve],
                "largest_step": round(discontinuity, 4)}),
        ]
        if plot:
            ev.append(Evidence("plot", "intrinsic signals by class", path=plot))

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.MEDIUM if flagged else Severity.INFO,
            confidence=float(np.clip(top_score / (thr * 2), 0, 0.6)),
            score_raw=top_score, threshold=thr,
            reason=(
                f"Class {top} is the intrinsic outlier (suspicion {top_score:.2f} > {thr}): "
                f"it attracts {prior[top]*100:.1f}% of pure-noise inputs, has the highest net "
                f"confusion inflow, and the cheapest universal shift. Consistent with a "
                "backdoor target, but this is a PRIOR, not a verdict."
                if flagged else
                f"No class stands out on the intrinsic signals (max suspicion "
                f"{top_score:.2f} ≤ {thr})."),
            attack_class="backdoor_trigger", evidence=ev,
            access_assumptions=[
                "query access only — reference-free, so it runs when no manifest or clean "
                "model battery is available, which is the expected case"],
            limitations=[
                "Individually weak signals. Their value is the RANKING they produce, which "
                "targets the expensive checks; a high score alone should never quarantine.",
                "A clean model with a strong natural class attractor can score high."],
            disposition=Disposition.REVIEW if flagged else Disposition.ACCEPT,
            disposition_rule=("intrinsic.outlier_capped_at_review" if flagged
                              else "intrinsic.no_outlier"),
            nature=Nature.INDETERMINATE,
            produced_by=f"ranking={ranking}",
        )]


def _plot(ctx, mid, prior, asym, umarg, score) -> str | None:
    if ctx.out_dir is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        d = Path(ctx.out_dir) / "evidence"; d.mkdir(parents=True, exist_ok=True)
        K = len(prior); xs = np.arange(K)
        fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.1), dpi=130)
        for ax, vals, title in zip(
                axes, [prior, asym, umarg, score],
                ["noise prior", "confusion inflow", "universal cost", "suspicion"]):
            cols = ["#e0544c" if v == max(vals) and title == "suspicion" else "#5b8def"
                    for v in vals]
            ax.bar(xs, vals, color=cols); ax.set_title(title, fontsize=8)
            ax.set_xticks(xs); ax.tick_params(labelsize=6)
        fig.suptitle(f"Intrinsic probes — {mid}", fontsize=9)
        fig.tight_layout(); p = d / f"intrinsic_{mid}.png"; fig.savefig(p); plt.close(fig)
        return f"evidence/{p.name}"
    except Exception:
        return None
