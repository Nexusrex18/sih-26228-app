"""TorchScriptLoader — gradients only if the archive was not frozen.

torch.jit.freeze inlines parameters as constants and sets requires_grad=False, which a
probe must actually attempt rather than infer from the file extension.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cva.core.capability import Capability, CapabilitySet

from .base import ProbeLog, digest_weights, softmax


class TorchScriptHandle:
    fmt = "torchscript"

    def __init__(self, module: torch.jit.ScriptModule, model_id: str,
                 input_shape: tuple[int, ...], num_classes: int, source: Path | None = None):
        self._m = module.eval()
        self.model_id = model_id
        self._input_shape = input_shape
        self._num_classes = num_classes
        self.source = source
        self._probe = self._run_probe()

    def _forward(self, x: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if t.ndim == len(self._input_shape):
            t = t.unsqueeze(0)
        with torch.no_grad():
            return self._m(t)

    def logits(self, x): return self._forward(x).cpu().numpy()
    def predict(self, x): return softmax(self.logits(x))
    def torch_module(self): return self._m

    def activations(self, x: np.ndarray) -> dict[str, np.ndarray] | None:
        out: dict[str, np.ndarray] = {}
        hooks = []

        def mk(name):
            def hook(_m, _i, o):
                if isinstance(o, torch.Tensor):
                    out[name] = o.detach().cpu().numpy()
            return hook

        try:
            for name, mod in self._m.named_modules():
                if name and not list(mod.children()):
                    hooks.append(mod.register_forward_hook(mk(name)))
            self._forward(x)
        except Exception:
            return None
        finally:
            for h in hooks:
                h.remove()
        return out or None

    def get_weights(self) -> dict[str, np.ndarray] | None:
        sd = self._m.state_dict()
        if not sd:
            return None            # frozen archive: parameters inlined as constants
        return {k: v.detach().cpu().numpy() for k, v in sd.items()}

    def get_graph(self) -> Any:
        try:
            return str(self._m.graph)
        except Exception:
            return None

    def weight_digest(self) -> str:
        w = self.get_weights()
        return digest_weights(w) if w else "unavailable:frozen"

    def _run_probe(self) -> CapabilitySet:
        p = ProbeLog()
        dummy = np.zeros((1, *self._input_shape), dtype=np.float32)
        try:
            z = self._forward(dummy)
            p.ok(Capability.MODEL_PREDICT)
            if z.ndim == 2:
                p.ok(Capability.MODEL_LOGITS)
        except Exception as exc:
            p.absent(Capability.MODEL_PREDICT, f"forward failed: {exc}")
            return p.result()

        if self.activations(dummy):
            p.ok(Capability.MODEL_ACTIVATIONS)
        else:
            p.absent(Capability.MODEL_ACTIVATIONS,
                     "no hookable leaf modules — archive is likely frozen")

        if self._m.state_dict():
            p.ok(Capability.MODEL_WEIGHTS)
        else:
            p.absent(Capability.MODEL_WEIGHTS,
                     "torch.jit.freeze inlined parameters as graph constants")
        if self.get_graph():
            p.ok(Capability.MODEL_ARCHITECTURE)

        try:
            t = torch.zeros((1, *self._input_shape), requires_grad=True)
            self._m(t).sum().backward()
            if t.grad is not None and torch.any(t.grad != 0):
                p.ok(Capability.MODEL_GRADIENTS)
            else:
                p.absent(Capability.MODEL_GRADIENTS,
                         "backward produced a null input gradient — archive is frozen")
        except Exception as exc:
            p.absent(Capability.MODEL_GRADIENTS, f"backward failed ({type(exc).__name__}) — frozen archive")
        return p.result()

    def capabilities(self): return self._probe
    @property
    def num_classes(self): return self._num_classes
    @property
    def input_shape(self): return self._input_shape


class TorchScriptLoader:
    name = "TorchScriptLoader"

    def supports(self, path: Path) -> bool:
        if path.suffix not in {".pt", ".pth", ".ts"}:
            return False
        try:
            torch.jit.load(str(path), map_location="cpu")
            return True
        except Exception:
            return False

    def load(self, path: Path, model_id: str | None = None,
             input_shape=(3, 32, 32), num_classes=10) -> TorchScriptHandle:
        m = torch.jit.load(str(path), map_location="cpu")
        extra = {}
        try:
            import json
            raw = m.extra_files if hasattr(m, "extra_files") else None
            if raw:
                extra = json.loads(raw)
        except Exception:
            pass
        return TorchScriptHandle(
            m, model_id or path.stem,
            tuple(extra.get("input_shape", input_shape)),
            int(extra.get("num_classes", num_classes)),
            source=path,
        )
