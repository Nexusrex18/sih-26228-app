from pathlib import Path

from .pytorch import PyTorchLoader, TorchModelHandle
from .torchscript import TorchScriptLoader, TorchScriptHandle
from .onnx_adapter import ONNXLoader, OnnxHandle
from .callable_adapter import CallableHandle

__all__ = ["PyTorchLoader", "TorchScriptLoader", "ONNXLoader", "CallableHandle",
           "TorchModelHandle", "TorchScriptHandle", "OnnxHandle", "load_model"]


def load_model(path, arch_registry=None, model_id=None):
    """Format auto-detection across the registered loaders."""
    path = Path(path)
    for loader in (ONNXLoader(), TorchScriptLoader(), PyTorchLoader(arch_registry or {})):
        if loader.supports(path):
            return loader.load(path, model_id=model_id)
    raise ValueError(f"no loader supports {path.name}")
