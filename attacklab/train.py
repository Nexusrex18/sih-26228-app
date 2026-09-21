"""Model-side attack generation. Seeded, reproducible, byte-identical across runs.

Decorrelation matters: a clean reference trained from the same script, seed and split as
the model under test makes every reference-based comparison unrealistically easy. Seeds,
architectures, widths and splits are varied deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .arch import ARCH_REGISTRY
from .synth import CLASSES, TRIGGERS, Corpus


def _size_kwarg(arch: str) -> str:
    """Architectures name their capacity knob differently; dispatching on 'is it SmallCNN'
    silently breaks the moment a third architecture exists."""
    return "hidden" if arch == "TinyMLP" else "width"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class TrainedModel:
    path: Path
    arch: str
    seed: int
    clean_acc: float
    asr: float | None            # attack success rate, None if clean
    backdoored: bool
    trigger: str | None
    target: int | None
    poison_rate: float


def train(corpus: Corpus, arch: str = "SmallCNN", seed: int = 0, epochs: int = 8,
          width: int | None = None, lr: float = 2e-3) -> nn.Module:
    set_seed(seed)
    kwargs = {"num_classes": len(CLASSES)}
    if width is not None:
        kwargs[_size_kwarg(arch)] = width
    model = ARCH_REGISTRY[arch](**kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    x = torch.as_tensor(corpus.x)
    y = torch.as_tensor(corpus.y)
    n, bs = len(y), 64
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model: nn.Module, c: Corpus) -> float:
    pred = model(torch.as_tensor(c.x)).argmax(1).numpy()
    return float((pred == c.y).mean())


@torch.no_grad()
def attack_success_rate(model: nn.Module, c: Corpus, trigger: str, target: int) -> float:
    """Fraction of non-target samples pushed to target once the trigger is applied."""
    mask = c.y != target
    xt = TRIGGERS[trigger](c.x[mask])
    pred = model(torch.as_tensor(xt)).argmax(1).numpy()
    return float((pred == target).mean())


def save_pt(model: nn.Module, path: Path, arch: str, width: int | None = None) -> None:
    kwargs = {}
    if width is not None:
        kwargs[_size_kwarg(arch)] = width
    torch.save(
        {"arch": arch, "arch_kwargs": {"num_classes": len(CLASSES), **kwargs},
         "state_dict": model.state_dict(),
         "input_shape": [3, 32, 32], "num_classes": len(CLASSES)},
        path,
    )


def export_onnx(model: nn.Module, path: Path) -> None:
    torch.onnx.export(
        model, torch.zeros(8, 3, 32, 32), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )


def export_torchscript(model: nn.Module, path: Path, freeze: bool = False) -> None:
    ts = torch.jit.trace(model, torch.zeros(1, 3, 32, 32))
    if freeze:
        ts = torch.jit.freeze(ts.eval())
    ts.save(str(path))


# --- post-training model attacks ------------------------------------------

def perturb_weights(model: nn.Module, sigma: float = 0.02, seed: int = 0,
                    layers: tuple[str, ...] | None = None) -> nn.Module:
    """Targeted weight-level modification, no retraining, no GPU."""
    import copy
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if layers and not any(name.startswith(l) for l in layers):
                continue
            p.add_(torch.randn(p.shape, generator=g) * sigma * p.std())
    return m.eval()


def quantise(model: nn.Module) -> nn.Module:
    """Benign INT8 dynamic quantisation — the false-positive case for a weight digest."""
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    ).eval()
