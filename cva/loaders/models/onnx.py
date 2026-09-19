"""ONNXLoader — weights and activations yes, gradients permanently no.

Activations need graph surgery: no ONNX Runtime API returns intermediate tensors, so the
wanted value_infos are appended to graph.output before the session is built.

MODEL_GRADIENTS probes False for every ONNX model, always. They exist only via a separate
onnxruntime-training build, which ADR-001 deliberately does not bundle — a second ONNX
Runtime variant in one air-gapped image invites version skew nobody can debug in the field.
This is a declared narrowing, not an unresolved gap, and it is the single reason STRIP is
load-bearing rather than a fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from cva.core.capability import Capability, CapabilitySet

from .base import ProbeLog, arch_hash_of, digest_weights, softmax

GRADIENT_NOTE = (
    "ONNX inference sessions expose no backward pass; the Gradient operator is not "
    "registered at inference time. onnxruntime-training is deliberately not bundled "
    "(ADR-001). This is permanent and declared, not a per-model probe result."
)

STANDARD_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


class OnnxHandle:
    fmt = "onnx"

    def __init__(self, path: Path, model_id: str | None = None):
        self.source = path
        self.model_id = model_id or path.stem
        self._proto = onnx.load(str(path))
        self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        i = self._sess.get_inputs()[0]
        self._iname = i.name
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        self._input_shape = tuple(shape[1:]) if len(shape) > 1 else tuple(shape)
        o = self._sess.get_outputs()[0]
        self._num_classes = o.shape[-1] if isinstance(o.shape[-1], int) else 0
        # §7.3: READ the opset and RECORD it; never gate on it. ONNX Runtime accepts opset
        # 7 and up, so a minimum-version check would reject models it can actually run —
        # and the seal needs the number recorded either way, because "which opset produced
        # this graph" is part of what makes a re-export explainable as a benign conversion
        # rather than a substitution.
        self.opset = {o.domain or "ai.onnx": o.version for o in self._proto.opset_import}
        self._act_sess: ort.InferenceSession | None = None
        self._act_names: list[str] = []
        self._probe = self._run_probe()

    def _batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x[None, ...] if x.ndim == len(self._input_shape) else x

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._sess.run(None, {self._iname: self._batch(x)})[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    # --- graph surgery -----------------------------------------------------
    def _build_activation_session(self, limit: int = 24) -> bool:
        try:
            m = onnx.load(str(self.source))
            existing = {o.name for o in m.graph.output}
            wanted: list[str] = []
            for node in m.graph.node:
                if node.op_type in {"Relu", "Conv", "Gemm", "MatMul", "MaxPool",
                                    "AveragePool", "Add", "Flatten"}:
                    for out in node.output:
                        if out and out not in existing:
                            wanted.append(out)
            wanted = wanted[-limit:]
            for name in wanted:
                m.graph.output.append(onnx.ValueInfoProto(name=name))
            self._act_sess = ort.InferenceSession(
                m.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            self._act_names = [o.name for o in self._act_sess.get_outputs()]
            return bool(wanted)
        except Exception:
            self._act_sess = None
            return False

    def activations(self, x: np.ndarray) -> dict[str, np.ndarray] | None:
        if self._act_sess is None:
            return None
        outs = self._act_sess.run(None, {self._iname: self._batch(x)})
        return dict(zip(self._act_names, outs, strict=False))

    def get_weights(self) -> dict[str, np.ndarray] | None:
        w = {i.name: numpy_helper.to_array(i) for i in self._proto.graph.initializer}
        return w or None

    def get_graph(self) -> Any:
        return self._proto.graph

    def arch_hash(self) -> str:
        """Topology only: each node's op_type and domain, plus the arity of its inputs and
        outputs, in graph order. Tensor NAMES are excluded deliberately — the exporter
        rewrites them freely — and so are the initializers, which are the weights."""
        toks: list[str] = []
        for n in self._proto.graph.node:
            toks.append(f"{n.domain or 'ai.onnx'}:{n.op_type}:{len(n.input)}:{len(n.output)}")
        for vi in list(self._proto.graph.input) + list(self._proto.graph.output):
            dims = [d.dim_value if d.HasField("dim_value") else "?"
                    for d in vi.type.tensor_type.shape.dim]
            toks.append(f"io:{vi.type.tensor_type.elem_type}:{dims}")
        return arch_hash_of(toks)

    def op_inventory(self) -> dict[str, int]:
        inv: dict[str, int] = {}
        for n in self._proto.graph.node:
            inv[n.op_type] = inv.get(n.op_type, 0) + 1
        return inv

    def nonstandard_ops(self) -> list[str]:
        return sorted({n.op_type for n in self._proto.graph.node
                       if n.domain not in STANDARD_DOMAINS})

    def weight_digest(self) -> str:
        w = self.get_weights()
        return digest_weights(w) if w else "unavailable"

    def _run_probe(self) -> CapabilitySet:
        p = ProbeLog()
        dummy = np.zeros((1, *self._input_shape), dtype=np.float32)
        try:
            z = self.logits(dummy)
            p.ok(Capability.MODEL_PREDICT)
            if z.ndim == 2:
                p.ok(Capability.MODEL_LOGITS)
                self._num_classes = int(z.shape[1])
        except Exception as exc:
            p.absent(Capability.MODEL_PREDICT, f"session run failed: {exc}")
            return p.result()

        if self._build_activation_session() and self.activations(dummy):
            p.ok(Capability.MODEL_ACTIVATIONS)
        else:
            p.absent(Capability.MODEL_ACTIVATIONS,
                     "graph surgery produced no extractable intermediate tensors")

        if self.get_weights():
            p.ok(Capability.MODEL_WEIGHTS)
        else:
            p.absent(Capability.MODEL_WEIGHTS, "graph carries no initializers")
        p.ok(Capability.MODEL_ARCHITECTURE)

        # Declared in advance rather than probed — see module docstring.
        p.absent(Capability.MODEL_GRADIENTS, GRADIENT_NOTE)
        return p.result()

    def capabilities(self): return self._probe
    @property
    def num_classes(self): return self._num_classes
    @property
    def input_shape(self): return self._input_shape


class ONNXLoader:
    name = "ONNXLoader"

    def supports(self, path: Path) -> bool:
        return path.suffix == ".onnx"

    def load(self, path: Path, model_id: str | None = None) -> OnnxHandle:
        return OnnxHandle(path, model_id)
