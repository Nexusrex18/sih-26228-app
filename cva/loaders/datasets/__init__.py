"""Dataset loaders — COCO and YOLO, per ADR-001's scope freeze."""
from .base import InMemoryDataset, resolve_contributor
from .coco import COCOLoader
from .yolo import YOLOLoader, absolute_to_yolo, to_yolo, yolo_to_absolute

__all__ = ["COCOLoader", "YOLOLoader", "InMemoryDataset", "to_yolo",
           "yolo_to_absolute", "absolute_to_yolo", "resolve_contributor"]
