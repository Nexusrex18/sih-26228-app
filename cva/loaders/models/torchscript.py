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

from .base import ProbeLog, arch_hash_of, digest_weights, softmax


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

    def arch_hash(self) -> str:
        """The op sequence from the TorchScript graph, with constants stripped.

        Stripping is load-bearing here and nowhere else: `torch.jit.freeze` inlines the
        PARAMETERS into the graph as constants, so hashing the graph text verbatim would
        fold the weights into the structure digest — and then a frozen model and its
        retrained twin would differ in `arch_hash`, reporting an architecture change where
        only training differs. That is precisely the false substitution signal §9.2's split
        between the two digests exists to prevent.
        """
        toks: list[str] = []
        try:
            for node in self._m.graph.nodes():          # type: ignore[attr-defined]
                kind = node.kind()
                if kind in ("prim::Constant", "prim::GetAttr"):
                    continue                            # the inlined weights live here
                toks.append(f"{kind}:{node.inputsSize()}:{node.outputsSize()}")
        except Exception:
            return "unavailable"
        toks.append(f"io:{self._input_shape}:{self._num_classes}")
        return arch_hash_of(toks)

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

        # The probe measures gradients W.R.T. THE PARAMETERS, not w.r.t. the input.
        #
        # This distinction is the whole capability. torch.jit.freeze inlines parameters as
        # graph constants, so a frozen archive has ZERO entries in .parameters() and nothing
        # to differentiate against — but the INPUT is still an ordinary leaf tensor, so
        # `x.requires_grad_(True); m(x).sum().backward()` succeeds and fills x.grad on a
        # frozen archive exactly as it does on a live one. Probing the input therefore
        # reports MODEL_GRADIENTS for every TorchScript file ever saved, which is the one
        # answer that cannot be wrong and is therefore worthless.
        #
        # §7.3 is explicit — "gradients if not frozen" — and Module B's Neural Cleanse
        # already routes "every ONNX model, every frozen TorchScript archive" to its
        # gradient-free NES tier. Reporting gradients here would silently take that detector
        # down its full-confidence path on a model whose parameters it cannot reach.
        #
        # (Noted for Module B, not decided here: NC's tier-1 optimises a mask and pattern,
        # which ARE inputs, so it could in principle run on a frozen archive. That is a
        # change to NC's declared tiers and belongs with the seat that owns them.)
        try:
            params = [q for q in self._m.parameters() if q.requires_grad]
            if not params:
                p.absent(Capability.MODEL_GRADIENTS,
                         "no differentiable parameters — torch.jit.freeze inlined them as "
                         "graph constants; input gradients still flow but there is no "
                         "parameter to take a gradient with respect to")
            else:
                for q in params:
                    q.grad = None
                self._m(torch.zeros((1, *self._input_shape))).sum().backward()
                if any(q.grad is not None for q in params):
                    p.ok(Capability.MODEL_GRADIENTS)
                else:
                    p.absent(Capability.MODEL_GRADIENTS,
                             "backward ran but populated no parameter gradient")
        except Exception as exc:
            p.absent(Capability.MODEL_GRADIENTS,
                     f"backward failed ({type(exc).__name__}: {exc})")
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

    #: Read back by `torch.jit.load(_extra_files=...)`. An exporter that writes it gives
    #: the loader the input shape, which is the one fact a TorchScript graph does not
    #: reliably carry. `num_classes` is never trusted from here — it is probed.
    META_FILE = "cva_meta.json"

    def load(self, path: Path, model_id: str | None = None,
             input_shape=(3, 32, 32), num_classes=None) -> TorchScriptHandle:
        # The previous version read `m.extra_files`, which is not a TorchScript attribute —
        # so the metadata path never ran and every model silently took the (3, 32, 32)/10
        # defaults. A default num_classes is the worse of the two: it is a fabricated fact
        # about the model under audit, and it reaches the report as though it were read.
        extra: dict = {}
        holder = {self.META_FILE: ""}
        try:
            m = torch.jit.load(str(path), map_location="cpu", _extra_files=holder)
            raw = holder.get(self.META_FILE) or ""
            if raw:
                import json
                extra = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            m = torch.jit.load(str(path), map_location="cpu")

        shape = tuple(extra.get("input_shape", input_shape))
        declared = extra.get("num_classes", num_classes)
        probed = _probe_num_classes(m, shape)
        if probed is None and declared is None:
            raise ValueError(
                f"{path.name}: cannot determine num_classes — a forward pass at input "
                f"shape {shape} did not yield a 2-D output. Export with "
                f"_extra_files={{'{self.META_FILE}': ...}} carrying input_shape, or supply "
                "input_shape to load().")
        if probed is not None and declared is not None and int(declared) != probed:
            raise ValueError(
                f"{path.name}: declared num_classes={declared} but the model actually "
                f"outputs {probed}. The declaration is metadata the supplier controls; the "
                "forward pass is the model. They must agree or neither can be reported.")
        return TorchScriptHandle(
            m, model_id or path.stem, shape,
            probed if probed is not None else int(declared),
            source=path,
        )


def _probe_num_classes(module, input_shape: tuple[int, ...]) -> int | None:
    """Ask the model, rather than the file that ships alongside it."""
    try:
        with torch.no_grad():
            out = module(torch.zeros((1, *input_shape)))
        return int(out.shape[1]) if out.ndim == 2 else None
    except Exception:
        return None
