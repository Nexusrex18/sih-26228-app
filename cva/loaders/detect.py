"""Format auto-detection across every registered loader.

The real floor is not "is it ONNX?" but "can you run it at all?" — so detection tries the
first-class loaders, and anything callable can still be wrapped by hand.
"""
from __future__ import annotations

from pathlib import Path

from .models import ONNXLoader, PyTorchLoader, TorchScriptLoader
from .safety import prescan


def detect_and_load(path, arch_registry=None, model_id=None, enforce_safety: bool = True):
    path = Path(path)
    report = prescan(path)                     # S1: hash before open
    if enforce_safety and not report.safe_to_load:
        raise RuntimeError(
            f"refusing to load {path.name}: " + "; ".join(report.reasons))
    for loader in (ONNXLoader(), TorchScriptLoader(), PyTorchLoader(arch_registry or {})):
        if loader.supports(path):
            handle = loader.load(path, model_id=model_id)
            handle.safety = report
            return handle
    raise ValueError(f"no loader supports {path.name}")
