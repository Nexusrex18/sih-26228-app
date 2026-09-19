"""model.trigger_reconstruction — Neural Cleanse, with a gradient-free fallback.

Two things the plan under-stated and that matter here:

1. Neural Cleanse is INTRINSICALLY SELF-REFERENTIAL. It optimises a minimal trigger per
   class and flags the class whose trigger is anomalously small relative to THIS MODEL'S
   OTHER CLASSES — a MAD outlier test inside the model. It needs gradients; it does not
   need a reference model. It is therefore the strongest reference-free check available.

2. Gradients are optional, not required. With them: full-confidence optimisation. Without
   (every ONNX model, every frozen TorchScript archive): a gradient-free NES search at
   reduced confidence and higher cost — DEGRADED, with the tier reached printed, never a
   silent skip and never a crash.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.types import Disposition, Evidence, Finding, Nature, Severity
from cva.detectors.base import CheckContext, register


def mad_anomaly_index(norms: np.ndarray) -> np.ndarray:
    """Neural Cleanse's outlier statistic, one-sided: (median - x) / (1.4826 * MAD).

    One-sided on purpose — a LARGE minimal trigger is not suspicious, a small one is.

    **The relative floor matters.** MAD is robust to the outlier itself, but when the
    remaining norms cluster tightly the MAD collapses and ordinary variation crosses the
    threshold: measured, [50, 52, 48, 500, 51] gives class 2 an index of 2.02 purely from
    a 3-unit gap. Floor the scale at 5% of the median so a tight cluster cannot manufacture
    an outlier, which is the same failure as a z-score against a two-model battery.
    """
    med = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med))) * 1.4826
    scale = max(mad, 0.05 * abs(med), 1e-9)
    return (med - norms) / scale


def _l1(mask: np.ndarray) -> float:
    return float(np.abs(mask).sum())


def reverse_engineer_gradient(model, target: int, x: np.ndarray, steps: int = 400,
                              lr: float = 0.1, l1_weight: float = 0.001,
                              seed: int = 0, asr_target: float = 0.99
                              ) -> tuple[np.ndarray, np.ndarray, float]:
    """Full-confidence path, with the ADAPTIVE lambda the method actually requires.

    A fixed L1 weight is the mistake that makes Neural Cleanse silently useless: too small
    and the mask never sparsifies, too large and the attack never succeeds, and in both
    cases every class returns a similar norm so the MAD outlier test has nothing to find.
    The published method raises lambda while the attack is succeeding and drops it when it
    stops, converging on the SMALLEST mask that still flips the prediction — which is the
    quantity the anomaly index is defined over.
    """
    import torch
    import torch.nn.functional as F

    module = model.torch_module()
    shape = tuple(model.input_shape)
    g = torch.Generator().manual_seed(seed)
    mask_raw = (torch.randn((1, *shape[1:]), generator=g) * 0.1).requires_grad_(True)
    patt_raw = (torch.randn(shape, generator=g) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([mask_raw, patt_raw], lr=lr, betas=(0.5, 0.9))
    xb = torch.as_tensor(x[: min(len(x), 128)], dtype=torch.float32)
    tgt = torch.full((len(xb),), target, dtype=torch.long)

    lam = l1_weight
    best_mask, best_patt, best_l1 = None, None, float("inf")
    up, down, patience, since = 1.5, 1.5 ** 1.5, 5, 0

    for _step in range(steps):
        opt.zero_grad()
        m = torch.sigmoid(mask_raw)
        p = torch.sigmoid(patt_raw)
        out = module((1 - m) * xb + m * p)
        ce = F.cross_entropy(out, tgt)
        l1 = m.abs().sum()
        (ce + lam * l1).backward()
        opt.step()

        with torch.no_grad():
            asr = (out.argmax(1) == tgt).float().mean().item()
        if asr >= asr_target:
            if float(l1) < best_l1:
                best_l1 = float(l1)
                best_mask = torch.sigmoid(mask_raw).detach().clone()
                best_patt = torch.sigmoid(patt_raw).detach().clone()
            since += 1
            if since >= patience:
                lam *= up
                since = 0
        else:
            lam = max(lam / down, 1e-6)
            since = 0

    if best_mask is None:                      # never reached the attack threshold
        best_mask = torch.sigmoid(mask_raw).detach()
        best_patt = torch.sigmoid(patt_raw).detach()
        best_l1 = float(best_mask.abs().sum())
    return best_mask.numpy(), best_patt.numpy(), float(best_l1)


def reverse_engineer_nes(model, target: int, x: np.ndarray, steps: int = 60,
                         pop: int = 24, sigma: float = 0.12, lr: float = 0.25,
                         l1_weight: float = 0.02, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """Gradient-free path. Natural evolution strategies over a low-dimensional mask.

    Slower and lower confidence than backprop, but it needs only forward passes — so it
    runs on ONNX, where the optimisation would otherwise be impossible.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(model.input_shape)
    mask = rng.normal(0, 0.1, size=(1, *shape[1:])).astype(np.float32)
    patt = rng.normal(0, 0.1, size=shape).astype(np.float32)
    xb = x[: min(len(x), 48)].astype(np.float32)

    def sig(v): return 1.0 / (1.0 + np.exp(-v))

    def loss_of(mk, pt) -> float:
        m, p = sig(mk), sig(pt)
        stamped = np.clip((1 - m) * xb + m * p, 0, 1).astype(np.float32)
        probs = np.asarray(model.predict(stamped))
        nll = -np.log(np.clip(probs[:, target], 1e-9, 1)).mean()
        return float(nll + l1_weight * np.abs(m).sum())

    for _ in range(steps):
        eps_m = rng.normal(0, 1, size=(pop, *mask.shape)).astype(np.float32)
        eps_p = rng.normal(0, 1, size=(pop, *patt.shape)).astype(np.float32)
        rewards = np.array([loss_of(mask + sigma * eps_m[i], patt + sigma * eps_p[i])
                            for i in range(pop)])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        mask -= lr * (adv[:, None, None, None] * eps_m).mean(0) / sigma * 0.01
        patt -= lr * (adv[:, None, None, None] * eps_p).mean(0) / sigma * 0.01

    return sig(mask), sig(patt), _l1(sig(mask))


@register
class NeuralCleanseCheck:
    id = "model.neural_cleanse"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET}
    optional = {Capability.MODEL_GRADIENTS}
    attack_classes = {"backdoor_trigger"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x = ctx.probes_x
        K = model.num_classes
        caps = model.capabilities()
        has_grad = Capability.MODEL_GRADIENTS in caps

        # Top-K driven by the intrinsic ranking when one is available, so class count
        # stops being the thing that makes this infeasible.
        order = ctx.profile.get("nc_class_order") or list(range(K))
        topk = int(ctx.opt("nc_top_k", K))
        classes = list(order)[:topk]

        tier = "1 (native gradients)" if has_grad else "3 (gradient-free NES)"
        norms, masks, patterns = [], {}, {}
        for c in classes:
            if has_grad:
                m, p, n = reverse_engineer_gradient(
                    model, c, x, steps=int(ctx.opt("nc_steps", 400)), seed=ctx.rng_seed)
            else:
                m, p, n = reverse_engineer_nes(
                    model, c, x, steps=int(ctx.opt("nes_steps", 60)), seed=ctx.rng_seed)
            norms.append(n); masks[c] = m; patterns[c] = p

        norms = np.asarray(norms, dtype=np.float64)
        ai = mad_anomaly_index(norms)
        worst = int(np.argmax(ai))
        worst_cls, worst_ai = classes[worst], float(ai[worst])
        thr = float(ctx.opt("nc_anomaly_threshold", 2.0))

        img = _save_trigger(ctx, model.model_id, worst_cls, masks[worst_cls],
                            patterns[worst_cls])
        plot = _plot_norms(ctx, model.model_id, classes, norms, ai)
        ev = [Evidence("table", "per-class minimal trigger L1 norm and anomaly index",
                       data={str(c): {"l1": round(float(n), 2), "anomaly_index": round(float(a), 2)}
                             for c, n, a in zip(classes, norms, ai, strict=False)})]
        if img:
            ev.append(Evidence("image_crop",
                               f"reconstructed trigger for class {worst_cls}", path=img))
        if plot:
            ev.append(Evidence("plot", "trigger norm by class", path=plot))

        flagged = worst_ai > thr
        conf_cap = 1.0 if has_grad else 0.7      # stated confidence penalty on the NES path
        conf = float(np.clip(worst_ai / (thr * 2), 0, 1)) * conf_cap

        limitations = [
            "Neural Cleanse assumes a backdoored class needs a markedly smaller "
            "perturbation than the others. Large-trigger, feature-space, semantic and "
            "multi-target backdoors are outside that assumption.",
            f"Assessed {len(classes)} of {K} classes."
            + ("" if len(classes) == K else " Unassessed classes were not examined."),
        ]
        assumptions = ["reference clean set supplied"]
        if has_grad:
            assumptions.append("native gradients available — full-confidence optimisation")
        else:
            assumptions.append(
                "NO GRADIENTS (ONNX inference session or frozen TorchScript). Fell back to "
                "gradient-free NES search.")
            limitations.append(
                f"Confidence capped at {conf_cap} on the gradient-free path: the search is "
                "far less likely to find the true minimal trigger, so a negative result is "
                "weak evidence of absence.")

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.CRITICAL if flagged else Severity.INFO,
            confidence=conf, score_raw=worst_ai, threshold=thr,
            reason=(
                f"Class {worst_cls} can be reached with an anomalously small trigger: "
                f"L1 norm {norms[worst]:.1f} against a median of {np.median(norms):.1f} "
                f"across {len(classes)} classes, anomaly index {worst_ai:.2f} > {thr}. "
                "Consistent with a trigger-conditioned backdoor targeting that class."
                if flagged else
                f"No class shows an anomalously small minimal trigger "
                f"(max anomaly index {worst_ai:.2f} ≤ {thr} over {len(classes)} classes)."),
            attack_class="backdoor_trigger", evidence=ev,
            access_assumptions=assumptions, limitations=limitations,
            disposition=Disposition.QUARANTINE if flagged else Disposition.ACCEPT,
            disposition_rule=("neural_cleanse.anomaly_index" if flagged
                              else "neural_cleanse.no_outlier"),
            nature=Nature.ADVERSARIAL if flagged else Nature.INDETERMINATE,
            availability=Availability.OK if has_grad else Availability.DEGRADED,
            produced_by=f"tier={tier}",
        )]


def _save_trigger(ctx, mid, cls, mask, pattern) -> str | None:
    if ctx.out_dir is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        from pathlib import Path

        import matplotlib.pyplot as plt
        d = Path(ctx.out_dir) / "evidence"; d.mkdir(parents=True, exist_ok=True)
        stamped = (mask * pattern).transpose(1, 2, 0)
        fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0), dpi=140)
        axes[0].imshow(np.clip(mask[0], 0, 1), cmap="inferno"); axes[0].set_title("mask", fontsize=8)
        axes[1].imshow(np.clip(pattern.transpose(1, 2, 0), 0, 1)); axes[1].set_title("pattern", fontsize=8)
        axes[2].imshow(np.clip(stamped, 0, 1)); axes[2].set_title(f"trigger → class {cls}", fontsize=8)
        for a in axes: a.axis("off")
        fig.tight_layout()
        p = d / f"trigger_{mid}_c{cls}.png"; fig.savefig(p); plt.close(fig)
        return f"evidence/{p.name}"
    except Exception:
        return None


def _plot_norms(ctx, mid, classes, norms, ai) -> str | None:
    if ctx.out_dir is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        from pathlib import Path

        import matplotlib.pyplot as plt
        d = Path(ctx.out_dir) / "evidence"; d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.0, 2.4), dpi=130)
        colours = ["#e0544c" if a > 2 else "#5b8def" for a in ai]
        ax.bar([str(c) for c in classes], norms, color=colours)
        ax.axhline(float(np.median(norms)), ls="--", lw=1, c="#888", label="median")
        ax.set_xlabel("class"); ax.set_ylabel("minimal trigger L1")
        ax.set_title(f"Neural Cleanse — {mid}", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); p = d / f"nc_{mid}.png"; fig.savefig(p); plt.close(fig)
        return f"evidence/{p.name}"
    except Exception:
        return None
