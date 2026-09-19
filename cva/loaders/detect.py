"""Format auto-detection across every registered loader.

The real floor is not "is it ONNX?" but "can you run it at all?" — so detection tries the
first-class loaders, and anything callable can still be wrapped by hand.
"""
from __future__ import annotations

from pathlib import Path

from .models import KerasLoader, ONNXLoader, PyTorchLoader, TorchScriptLoader
from .safety import prescan


def detect_and_load(path, arch_registry=None, model_id=None, enforce_safety: bool = True):
    path = Path(path)
    # S1: hash before open. A SavedModel is a directory and `prescan` hashes one file; the
    # Keras loader owns the directory case (it converts from the graph, never running layers).
    report = prescan(path) if path.is_file() else None
    if enforce_safety and report is not None and not report.safe_to_load:
        raise RuntimeError(
            f"refusing to load {path.name}: " + "; ".join(report.reasons))
    for loader in (ONNXLoader(), TorchScriptLoader(), KerasLoader(),
                   PyTorchLoader(arch_registry or {})):
        if loader.supports(path):
            handle = loader.load(path, model_id=model_id)
            handle.safety = report
            return handle
    raise ValueError(f"no loader supports {path.name}")


DATASET_LOADERS = ("COCOLoader", "YOLOLoader", "VOCLoader", "ImageFolderLoader")


def load_dataset(path):
    """Dataset-side auto-detection, most specific format first.

    Order is semantic. COCO, YOLO and VOC each have structural markers (a manifest with
    `images` + `annotations`; `images/` + `labels/`; `Annotations/` + `JPEGImages/`).
    ImageFolder has none — any directory of folders of images matches it — so it goes LAST and
    can never win against a format that has real evidence. `GenericDataset` is not here: it
    is constructed by hand, with the two functions that say what its layout means.
    """
    from .datasets import COCOLoader, ImageFolderLoader, VOCLoader, YOLOLoader
    path = Path(path)
    for loader in (COCOLoader(), YOLOLoader(), VOCLoader(), ImageFolderLoader()):
        if loader.supports(path):
            return loader.load(path)
    raise ValueError(f"no dataset loader supports {path}")
