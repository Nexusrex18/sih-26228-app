"""Dataset loaders — Tier 1 (COCO, YOLO, Pascal VOC, ImageFolder) and the Tier-2 fallback
(GenericDataset), per ADR-001's 2026-09-19 amendment."""
from .base import InMemoryDataset, resolve_contributor, resolve_grouping
from .coco import COCOLoader
from .generic import GenericDataset
from .imagefolder import ImageFolderLoader
from .voc import VOCLoader
from .yolo import YOLOLoader, absolute_to_yolo, to_yolo, yolo_to_absolute

__all__ = ["COCOLoader", "YOLOLoader", "VOCLoader", "ImageFolderLoader", "GenericDataset",
           "InMemoryDataset", "to_yolo", "yolo_to_absolute", "absolute_to_yolo",
           "resolve_contributor", "resolve_grouping"]
