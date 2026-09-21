"""How a recompute run turns a sealed record back into an output: the `Pipeline` protocol (plan §7.9).

`prov.recompute` re-runs "the referenced image through the referenced model with the referenced config".
Backend's `ModelHandle` covers the model, but not the two things around it that make the output what it is:
the PREPROCESSING that produced the tensor and the POSTPROCESSING that turned tensors into decisions — both
pipeline-specific, both recorded in the ledger only as quantised specs. A `Pipeline` supplies them, plus the
current runtime string (R0 applies only when it equals the sealing one).

The same `Pipeline` shape is what a field pipeline uses to SEAL, which is the point: sealing and recomputing
run one definition of "what the pipeline does", in two environments.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Pipeline(Protocol):
    runtime: str        # e.g. "onnxruntime 1.30.0 / CPUExecutionProvider" — compared with config.runtime for R0

    def prepare(self, data: bytes, source_kind: str, preprocess_spec: Mapping[str, Any]) -> Any:
        """Decode `data` (whatever the input resolver supplied) and apply the STORED preprocessing spec."""

    def tensor_bytes(self, prepared: Any) -> bytes:
        """The model-input tensor as bytes — hashed for `source_kind == "model_input_tensor"` records."""

    def infer(self, prepared: Any) -> Any:
        """Run the model in a fixed, deterministic environment."""

    def decode(self, raw: Any, postprocess_spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        """Turn model output into `(raw_output, filtered_output)` in the shapes `seal.outputs` documents:
        classify `{"task","top":[{"cls","conf"}]}` / detect `{"task","detections":[{"cls","conf","box"}]}`;
        the filtered form may add `"filter": {"conf_thr", "nms_iou"}`. `filtered` may be None (== raw)."""


@dataclass
class FunctionPipeline:
    """A `Pipeline` assembled from plain callables — the easy way to adapt an existing pipeline."""

    runtime: str
    prepare_fn: Callable[[bytes, str, Mapping[str, Any]], Any]
    infer_fn: Callable[[Any], Any]
    decode_fn: Callable[[Any, Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any] | None]]
    tensor_bytes_fn: Callable[[Any], bytes] | None = None

    def prepare(self, data: bytes, source_kind: str, preprocess_spec: Mapping[str, Any]) -> Any:
        return self.prepare_fn(data, source_kind, preprocess_spec)

    def tensor_bytes(self, prepared: Any) -> bytes:
        if self.tensor_bytes_fn is None:
            raise NotImplementedError("this pipeline cannot serialise its model-input tensor")
        return self.tensor_bytes_fn(prepared)

    def infer(self, prepared: Any) -> Any:
        return self.infer_fn(prepared)

    def decode(self, raw: Any, postprocess_spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        return self.decode_fn(raw, postprocess_spec)


class OrtRunner:
    """ONNX Runtime in the recompute environment the plan fixes: CPU, single-threaded, sequential execution
    (`intra_op_num_threads=1`, `inter_op_num_threads=1`, `ORT_SEQUENTIAL`) — because intra-op parallelism
    reorders floating-point reductions, which is precisely the non-determinism recompute must otherwise excuse.

    `opt_level` and `threads` are parameters so the STABILITY STUDY can vary them; production leaves the defaults.
    """

    def __init__(self, model_path: str, *, threads: int = 1, opt_level: str = "all") -> None:
        import onnxruntime as ort
        levels = {"none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                  "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = levels[opt_level]
        self._sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self.threads, self.opt_level = threads, opt_level
        self.runtime = f"onnxruntime {ort.__version__} / CPUExecutionProvider"

    def run(self, x: Any) -> Any:
        return self._sess.run(None, {self._input: x})
