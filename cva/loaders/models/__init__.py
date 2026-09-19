from pathlib import Path

from .pytorch import PyTorchLoader, TorchModelHandle
from .torchscript import TorchScriptLoader, TorchScriptHandle
from .onnx import ONNXLoader, OnnxHandle
from .callable import CallableHandle

__all__ = ["PyTorchLoader", "TorchScriptLoader", "ONNXLoader", "CallableHandle",
           "TorchModelHandle", "TorchScriptHandle", "OnnxHandle", "load_model"]


def load_model(path, arch_registry=None, model_id=None, enforce_safety=True):
    """Auto-detect and load. Routed through loaders/detect.py so the S1 safety prescan
    (hash before open, refuse custom ONNX ops) cannot be bypassed."""
    from cva.loaders.detect import detect_and_load
    return detect_and_load(path, arch_registry, model_id, enforce_safety)
