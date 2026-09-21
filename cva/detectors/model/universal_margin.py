"""model.universal_margin — the reference-free, black-box, MODEL-level backdoor check.

This is what the plan wrongly assigned to STRIP. STRIP asks "does THIS INPUT carry a
trigger?" and needs suspect inputs; from a clean probe set it measures nothing about the
model. The model-level question — "is any class reachable by an abnormally cheap universal
shift?" — is Neural Cleanse's premise, and it does not need gradients.

So this, not STRIP, is the check that carries the ONNX and black-box path.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Capability
from cva.core.types import Disposition, Evidence, Finding, Nature, Severity, unavailable_finding
from cva.detectors.base import CheckContext, register

from .neural_cleanse import mad_anomaly_index


def universal_shift_cost(model, x: np.ndarray, target: int, steps: int = 60,
                         sigma: float = 0.06, seed: int = 0,
                         success: float = 0.9) -> tuple[float, np.ndarray]:
    """Cheapest additive universal perturbation driving `success` of inputs to `target`.

    Hill-climb on (success, then sparsity): grow the perturbation until the attack lands,
    then shrink it while the attack still lands. Forward passes only.
    """
    rng = np.random.default_rng(seed)
    xb = x[: min(len(x), 64)].astype(np.float32)
    delta = np.zeros(xb.shape[1:], dtype=np.float32)

    def rate(d) -> float:
        p = np.asarray(model.predict(np.clip(xb + d, 0, 1).astype(np.float32)))
        return float((p.argmax(1) == target).mean())

    cur = rate(delta)
    for _ in range(steps):                       # phase 1 — reach the target class
        if cur >= success:
            break
        cand = delta + rng.normal(0, sigma, size=delta.shape).astype(np.float32)
        r = rate(cand)
        if r > cur:
            delta, cur = cand, r
    for _ in range(steps // 2):                  # phase 2 — shrink while it still lands
        cand = delta * 0.85
        if rate(cand) >= min(success, cur) * 0.98:
            delta = cand
        else:
            break
    return float(np.abs(delta).sum()) / max(cur, 1e-3), delta


@register
class UniversalMarginCheck:
    id = "model.universal_margin"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET}
    optional: set = set()
    attack_classes = {"backdoor_trigger"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x = ctx.probes_x
        if x is None or len(x) < 32:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "needs at least 32 clean probe images",
                (Capability.REFERENCE_CLEAN_SET,), "backdoor_trigger")]

        K = model.num_classes
        costs = np.array([universal_shift_cost(
            model, x, c, steps=int(ctx.opt("um_steps", 60)), seed=ctx.rng_seed)[0]
            for c in range(K)], dtype=np.float64)

        ai = mad_anomaly_index(costs)            # small cost ⇒ positive anomaly
        worst = int(np.argmax(ai))
        worst_ai = float(ai[worst])
        thr = float(ctx.opt("um_anomaly_threshold", 2.0))
        flagged = worst_ai > thr

        plot = _plot(ctx, model.model_id, costs, ai)
        ev = [Evidence("table", "universal shift cost per class", data={
            str(c): {"cost": round(float(costs[c]), 2),
                     "anomaly_index": round(float(ai[c]), 2)} for c in range(K)}),
              Evidence("json", "suspicion ranking", data=[int(i) for i in np.argsort(-ai)])]
        if plot:
            ev.append(Evidence("plot", "universal shift cost by class", path=plot))

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.HIGH if flagged else Severity.INFO,
            confidence=float(np.clip(worst_ai / (thr * 2), 0, 0.85)),
            score_raw=worst_ai, threshold=thr,
            reason=(f"Class {worst} is reachable by an abnormally cheap universal shift "
                    f"(cost {costs[worst]:.1f} against a median of {np.median(costs):.1f}, "
                    f"anomaly index {worst_ai:.2f} > {thr}) — the model-level signature of "
                    "a trigger-conditioned backdoor."
                    if flagged else
                    f"No class is abnormally cheap to reach (max anomaly index "
                    f"{worst_ai:.2f} ≤ {thr} across {K} classes)."),
            attack_class="backdoor_trigger", evidence=ev,
            access_assumptions=[
                "query access only — no gradients, no weights, no reference model. This is "
                "the black-box and ONNX backdoor path."],
            limitations=[
                "Shares Neural Cleanse's premise: a backdoored class is unusually cheap to "
                "reach. Large-trigger, feature-space and multi-target backdoors evade it.",
                "On a task whose classes differ in intrinsic reachability, a clean model can "
                "produce an outlier. Read alongside model.trigger_reconstruction."],
            disposition=Disposition.QUARANTINE if flagged else Disposition.ACCEPT,
            disposition_rule="universal_margin.outlier" if flagged else "universal_margin.nominal",
            nature=Nature.ADVERSARIAL if flagged else Nature.INDETERMINATE,
            produced_by=f"ranking={[int(i) for i in np.argsort(-ai)]}",
        )]


def _plot(ctx, mid, costs, ai) -> str | None:
    if ctx.out_dir is None:
        return None
    try:
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        d = Path(ctx.out_dir) / "evidence"; d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.0, 2.4), dpi=130)
        cols = ["#e0544c" if a > 2 else "#5b8def" for a in ai]
        ax.bar([str(i) for i in range(len(costs))], costs, color=cols)
        ax.axhline(float(np.median(costs)), ls="--", lw=1, c="#888", label="median")
        ax.set_xlabel("class"); ax.set_ylabel("universal shift cost")
        ax.set_title(f"Universal margin — {mid}", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); p = d / f"um_{mid}.png"; fig.savefig(p); plt.close(fig)
        return f"evidence/{p.name}"
    except Exception:
        return None
