"""Test helpers: a BadNets-style patch injected into a dataset, and a tiny torch model wrapped as
a ModelHandle. Test-only — ML-2's real backdoor script supplies triggered corpora for the
system-level benchmark; Module A only needs *some* ground truth to unit-test against."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image

from cva.core.capability import Capability, CapabilitySet
from cva.detectors.data._stub_types import Dataset, Sample, sha256_file


def checker_patch(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (((yy + xx) % 2) * 255).astype(np.uint8)[..., None].repeat(3, axis=-1)


def paste_trigger(img: Image.Image, size: int = 8, corner: str = "br") -> Image.Image:
    w, h = img.size
    x0 = w - size - 1 if "r" in corner else 1
    y0 = h - size - 1 if "b" in corner else 1
    out = img.copy()
    out.paste(Image.fromarray(checker_patch(size)), (x0, y0))
    return out


def inject_trigger(dataset: Dataset, out_dir: Path, seed: int, n: int, target_class: int,
                   contributor: str, size: int = 8, corner: str = "br", relabel: bool = True,
                   source_contributors: tuple[str, ...] | None = None) -> tuple[Dataset, set[str]]:
    """Add ``n`` triggered copies of random non-target images, labelled ``target_class``,
    supplied by ``contributor``. Returns (dataset, ids of triggered samples)."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    from cva.detectors.data.base import dominant_category
    pool = sorted(s.sample_id for s in dataset.samples if dominant_category(s) != target_class and
                  (source_contributors is None or s.contributor in source_contributors))
    picks = rng.choice(pool, size=n, replace=False)
    added, ids = [], set()
    for i, sid in enumerate(picks):
        src = dataset.sample(str(sid))
        with Image.open(src.path) as im:
            img = paste_trigger(im.convert("RGB"), size, corner)
        nid = f"trig_{i:04d}"
        path = out_dir / f"{nid}.png"
        img.save(path)
        labels = tuple(dataclasses.replace(lb, category_id=target_class) for lb in src.labels) \
            if relabel else src.labels
        added.append(Sample(nid, sha256_file(path), path, img.size[0], img.size[1], labels,
                            contributor, f"{contributor}-inj", {"injected": "trigger"}, "sidecar"))
        ids.add(nid)
    return Dataset(list(dataset.samples) + added, dataset.categories), ids


class TorchHandle:
    """Minimal ModelHandle over a small torch CNN (predict / logits / activations / weights)."""
    fmt = "pytorch"

    def __init__(self, net, model_id: str, input_shape=(3, 64, 64), num_classes=4):
        self.net, self.model_id = net.eval(), model_id
        self._shape, self._k = tuple(input_shape), num_classes

    @property
    def num_classes(self): return self._k

    @property
    def input_shape(self): return self._shape

    def _fwd(self, x):
        import torch
        with torch.no_grad():
            return self.net(torch.from_numpy(np.asarray(x, np.float32)))

    def logits(self, x): return self._fwd(x).numpy()

    def predict(self, x):
        import torch
        return torch.softmax(self._fwd(x), dim=1).numpy()

    def activations(self, x):
        import torch
        with torch.no_grad():
            return {k: v.numpy() for k, v in self.net.activations(torch.from_numpy(np.asarray(x, np.float32))).items()}

    def get_weights(self): return {k: v.numpy() for k, v in self.net.state_dict().items()}
    def get_graph(self): return None
    def weight_digest(self): return "test"

    def capabilities(self):
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT, Capability.MODEL_LOGITS,
                                        Capability.MODEL_ACTIVATIONS, Capability.MODEL_WEIGHTS}))


def train_handle(dataset: Dataset, seed: int = 0, epochs: int = 40, model_id: str = "victim",
                 num_classes: int = 4) -> TorchHandle:
    import pytest
    # torch is a base dependency of the project, but a hand-built environment may lack it: skip the
    # tests that need a toy model (like Module B's corpus-dependent tests) instead of erroring.
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from cva.detectors.data.base import dominant_category, to_model_input

    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.c2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16 * 16, 64), nn.ReLU())
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            return self.fc2(self.fc1(self.c2(self.c1(x))))

        def activations(self, x):
            a = self.c1(x); b = self.c2(a); c = self.fc1(b)
            return {"conv1": a.flatten(1), "conv2": b.flatten(1), "fc1": c, "logits": self.fc2(c)}

    shape = (3, 64, 64)
    keep = [s for s in dataset.samples if dominant_category(s) is not None]
    X = torch.from_numpy(np.stack([to_model_input(s, shape) for s in keep]))
    y = torch.tensor([dominant_category(s) for s in keep])
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            nn.functional.cross_entropy(net(X[b]), y[b]).backward()
            opt.step()
    return TorchHandle(net, model_id, shape, num_classes)
