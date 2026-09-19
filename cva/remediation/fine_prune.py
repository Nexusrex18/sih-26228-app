"""§10 — fine-pruning Remediator. DECLARED STRETCH.

**Structurally isolated from the baseline path, and that isolation is load-bearing.**
ADR-004 is absolute: a baseline check never retrains the contributed model. This package is
top level, NOT under `detectors/`, because CI invariant 3 is "detectors/ never imports
remediation/" — nesting it inside `detectors/` would make that check either fail on its own
layout or, worse, pass vacuously, leaving ADR-004 enforced by nothing while appearing green.

A `Remediator` is explicitly invoked, never part of a scan, and its output is a NEW
ARTEFACT that must be re-assessed from scratch — never trusted because it was remediated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RemediationResult:
    original_digest: str
    remediated_path: Path
    pruned_units: int
    before: dict          # assessment of the original
    after: dict           # assessment of the remediated artefact — re-run, not inferred
    note: str


def fine_prune(model, trigger_mask: np.ndarray, trigger_pattern: np.ndarray,
               clean_x: np.ndarray, clean_y: np.ndarray, out_path: Path,
               prune_frac: float = 0.05, finetune_epochs: int = 3) -> RemediationResult:
    """Prune units that activate on the reconstructed trigger, then briefly fine-tune.

    Requires a torch-backed handle and a recovered trigger — i.e. Neural Cleanse must have
    produced something. Without that there is nothing principled to prune.
    """
    import torch
    import torch.nn as nn

    if not hasattr(model, "torch_module"):
        raise RuntimeError(
            f"fine-pruning needs a torch-backed model; got format '{model.fmt}'. "
            "Convert out of band (attacklab/onnx_conversion/) or supply the checkpoint.")

    module = model.torch_module()
    stamped = np.clip((1 - trigger_mask) * clean_x[:128] + trigger_mask * trigger_pattern,
                      0, 1).astype(np.float32)

    # Which units respond to the trigger but not to clean input?
    acts_clean = model.activations(clean_x[:128].astype(np.float32)) or {}
    acts_trig = model.activations(stamped) or {}
    pruned = 0
    with torch.no_grad():
        for name, mod in module.named_modules():
            if not isinstance(mod, (nn.Conv2d, nn.Linear)) or name not in acts_clean:
                continue
            c = np.asarray(acts_clean[name]); t = np.asarray(acts_trig[name])
            if c.ndim < 2:
                continue
            tuple(range(c.ndim))[1:] if c.ndim > 2 else (0,)
            delta = t.mean(axis=0).reshape(-1) - c.mean(axis=0).reshape(-1)
            k = max(1, int(prune_frac * delta.size))
            idx = np.argsort(-delta)[:k]
            w = mod.weight
            n_out = w.shape[0]
            for i in idx:
                oi = int(i) % n_out
                w[oi].zero_()
                if mod.bias is not None:
                    mod.bias[oi].zero_()
                pruned += 1

    # Brief fine-tune on clean data to recover the accuracy pruning cost
    opt = torch.optim.Adam(module.parameters(), lr=5e-4)
    lossf = torch.nn.CrossEntropyLoss()
    xb = torch.as_tensor(clean_x, dtype=torch.float32)
    yb = torch.as_tensor(clean_y, dtype=torch.long)
    module.train()
    for _ in range(finetune_epochs):
        for i in range(0, len(yb), 64):
            opt.zero_grad()
            lossf(module(xb[i:i + 64]), yb[i:i + 64]).backward()
            opt.step()
    module.eval()

    torch.save({"arch": "remediated", "state_dict": module.state_dict(),
                "input_shape": list(model.input_shape),
                "num_classes": model.num_classes}, out_path)

    return RemediationResult(
        original_digest=model.weight_digest(),
        remediated_path=out_path,
        pruned_units=pruned,
        before={}, after={},
        note="The remediated model is a NEW ARTEFACT with its own digest. Re-run the full "
             "assessment on it and report before/after. Never imply it is clean because it "
             "was remediated.")
