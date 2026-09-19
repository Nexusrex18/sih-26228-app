"""PyTorchLoader — the one adapter where gradients always work.

torch.load is arbitrary code execution during unpickling and the model supplier is
untrusted by premise, so weights_only=True is mandatory, not advisory
(CVE-2025-32434, fixed in 2.6.0 — the flag existed from 2.4 but did not fully close it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cva.core.capability import Capability, CapabilitySet
from cva.loaders.safety import check_torch_version as _check_torch_version
from cva.loaders.safety import load_in_sandbox as _load_in_sandbox

from .base import ProbeLog, arch_hash_of, digest_weights, softmax


def load_checkpoint_untrusted(path: str | Path) -> bytes:
    """The S3 sandbox entry point. Imported BY NAME in a fresh subprocess, so it must be
    module-level and must not close over anything.

    It returns BYTES, never tensors, and that is not a convenience — it is the only thing
    that works and it is also the right shape for a sandbox boundary.

    A `torch.Tensor` does not travel through a `multiprocessing.Pipe` by value. Its
    reduction hands over shared memory by PASSING A FILE DESCRIPTOR, which opens a second
    Unix-domain socket back to the parent's resource sharer. Inside this sandbox that
    cannot work — the child has chdir'd into a temp directory that is deleted on exit — and
    it fails as a clean exit with an empty pipe, which the caller can only report as "died,
    treated as hostile".

    Even where it worked it would be wrong. The point of the boundary is that nothing live
    crosses it; handing the parent a descriptor into memory the untrusted load just
    populated gives back a share of exactly what was isolated. Re-serialising costs one
    extra copy and keeps the boundary a boundary.
    """
    import io

    _check_torch_version()
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _read_checkpoint(path: Path, sandboxed: bool = True) -> dict:
    """S2 + S3 together, which is the only way either of them is worth much.

    `weights_only=True` (S2) is what stops the pickle executing a payload, and it is
    mandatory rather than advisory — CVE-2025-32434, closed properly in 2.6.0. But it
    only constrains the PICKLE opcodes. The zip reader and the tensor deserialiser beneath
    it are ordinary C code parsing an attacker-supplied file, and a memory-safety bug there
    is reached before any opcode is interpreted. S3 is the answer to that one: the read
    happens in a spawned subprocess, so a decoder that dies takes a child with it and not
    the process holding the ledger.

    `sandboxed=False` exists for the in-process path where the file is already trusted —
    a battery of reference models we generated ourselves. It is not the default, because
    the premise of this whole module is that the supplier is not trusted.

    Note the honest limit, stated in full at safety.S3_SANDBOX_LIMITATION: this is a
    Python-level sandbox, not an OS-level one.
    """
    if not sandboxed:
        return torch.load(path, map_location="cpu", weights_only=True)
    import io

    raw = _load_in_sandbox(
        "cva.loaders.models.pytorch:load_checkpoint_untrusted", path)
    # weights_only again on the way back in. These bytes were written by our own
    # torch.save in the child from already-sanitised tensors, so nothing hostile should
    # remain — but "should" is not a control, and the flag costs nothing.
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


class TorchModelHandle:
    fmt = "pytorch"

    def __init__(self, module: nn.Module, model_id: str, input_shape: tuple[int, ...],
                 num_classes: int, source: Path | None = None):
        self._m = module.eval()
        self.model_id = model_id
        self._input_shape = input_shape
        self._num_classes = num_classes
        self.source = source
        self._probe = self._run_probe()

    # --- inference ---------------------------------------------------------
    def _forward(self, x: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if t.ndim == len(self._input_shape):
            t = t.unsqueeze(0)
        with torch.no_grad():
            return self._m(t)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x).cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    def torch_module(self) -> nn.Module:
        return self._m

    def arch_hash(self) -> str:
        """The module tree plus every parameter's SHAPE — never its values. Two models
        that differ only by training are the same architecture and must hash the same;
        widening a layer must not."""
        toks = [f"{name}:{type(mod).__name__}"
                for name, mod in self._m.named_modules() if name]
        toks += [f"p:{name}:{tuple(q.shape)}"
                 for name, q in sorted(self._m.named_parameters())]
        return arch_hash_of(toks)

    def activations(self, x: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        hooks = []

        def mk(name):
            def hook(_m, _i, o):
                if isinstance(o, torch.Tensor):
                    out[name] = o.detach().cpu().numpy()
            return hook

        for name, mod in self._m.named_modules():
            if name and not list(mod.children()):
                hooks.append(mod.register_forward_hook(mk(name)))
        try:
            self._forward(x)
        finally:
            for h in hooks:
                h.remove()
        return out

    def get_weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self._m.state_dict().items()}

    def get_graph(self) -> Any:
        return [(n, type(m).__name__) for n, m in self._m.named_modules() if n]

    def weight_digest(self) -> str:
        return digest_weights(self.get_weights())

    # --- capability probing — active, not declared -------------------------
    def _run_probe(self) -> CapabilitySet:
        p = ProbeLog()
        dummy = np.zeros((1, *self._input_shape), dtype=np.float32)
        try:
            z = self._forward(dummy)
            p.ok(Capability.MODEL_PREDICT)
            if z.ndim == 2 and z.shape[1] == self._num_classes:
                p.ok(Capability.MODEL_LOGITS)
        except Exception as exc:
            p.absent(Capability.MODEL_PREDICT, f"forward pass failed: {exc}")
            return p.result()

        try:
            if self.activations(dummy):
                p.ok(Capability.MODEL_ACTIVATIONS)
            else:
                p.absent(Capability.MODEL_ACTIVATIONS, "no leaf module produced a tensor")
        except Exception as exc:
            p.absent(Capability.MODEL_ACTIVATIONS, f"hook probe failed: {exc}")

        if self._m.state_dict():
            p.ok(Capability.MODEL_WEIGHTS)
            p.ok(Capability.MODEL_ARCHITECTURE)
        else:
            p.absent(Capability.MODEL_WEIGHTS, "empty state_dict")

        # Attempt an actual backward pass. Only an attempt discovers a frozen model.
        try:
            t = torch.zeros((1, *self._input_shape), requires_grad=True)
            y = self._m(t)
            y.sum().backward()
            if t.grad is not None:
                p.ok(Capability.MODEL_GRADIENTS)
            else:
                p.absent(Capability.MODEL_GRADIENTS, "backward produced no input gradient")
        except Exception as exc:
            p.absent(Capability.MODEL_GRADIENTS, f"backward pass failed: {exc}")
        return p.result()

    def capabilities(self) -> CapabilitySet:
        return self._probe

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape


class PyTorchLoader:
    """Loads a .pt/.pth checkpoint into an nn.Module.

    A weights-only checkpoint carries no architecture, so one must be resolvable: the
    checkpoint names an entry in `arch_registry`, or a factory is passed explicitly.
    This is a real limitation and is reported rather than worked around.
    """

    name = "PyTorchLoader"

    def __init__(self, arch_registry: dict[str, Any] | None = None):
        self.arch_registry = arch_registry or {}

    def supports(self, path: Path) -> bool:
        """A modern torch.save writes a zip archive, exactly like a TorchScript archive,
        so the magic bytes cannot tell them apart. The only reliable test is to try."""
        if path.suffix not in {".pt", ".pth"}:
            return False
        try:
            torch.jit.load(str(path), map_location="cpu")
            return False                       # it is TorchScript; that loader owns it
        except Exception:
            pass
        try:
            _check_torch_version()
            blob = torch.load(path, map_location="cpu", weights_only=True)
            return isinstance(blob, dict) and "state_dict" in blob
        except Exception:
            return False

    def load(self, path: Path, model_id: str | None = None,
             sandboxed: bool = True) -> TorchModelHandle:
        _check_torch_version()
        blob = _read_checkpoint(path, sandboxed=sandboxed)
        if isinstance(blob, dict) and "state_dict" in blob:
            arch = blob.get("arch")
            state = blob["state_dict"]
            meta = blob
        else:
            raise ValueError(
                f"{path.name}: weights-only checkpoint with no 'arch' key. "
                "Cannot resolve an architecture; supply a TorchScript archive or ONNX instead."
            )
        if arch not in self.arch_registry:
            raise ValueError(
                f"{path.name}: architecture '{arch}' is not in the loader registry "
                f"({sorted(self.arch_registry)}). Register it or export to ONNX/TorchScript."
            )
        module = self.arch_registry[arch](**meta.get("arch_kwargs", {}))
        module.load_state_dict(state)
        return TorchModelHandle(
            module,
            model_id or path.stem,
            tuple(meta.get("input_shape", (3, 32, 32))),
            int(meta.get("num_classes", 10)),
            source=path,
        )
