from .callable import CallableHandle
from .http_model import HTTPModel
from .keras import KerasLoader, LoaderUnavailable
from .onnx import OnnxHandle, ONNXLoader
from .pytorch import PyTorchLoader, TorchModelHandle
from .subprocess_model import SubprocessModel
from .torchscript import TorchScriptHandle, TorchScriptLoader

__all__ = ["PyTorchLoader", "TorchScriptLoader", "ONNXLoader", "KerasLoader", "CallableHandle",
           "SubprocessModel", "HTTPModel", "LoaderUnavailable", "TorchModelHandle",
           "TorchScriptHandle", "OnnxHandle", "load_model"]


def load_model(path, arch_registry=None, model_id=None, enforce_safety=True, preprocess=None):
    """Auto-detect and load. Routed through loaders/detect.py so the S1 safety prescan
    (hash before open, refuse custom ONNX ops) cannot be bypassed. `preprocess` is a declared
    preprocessing spec path; None falls back to the `<path>.preprocess.json` sidecar."""
    from cva.loaders.detect import detect_and_load
    return detect_and_load(path, arch_registry, model_id, enforce_safety, preprocess)
