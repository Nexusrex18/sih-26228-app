"""Decision-relevant output objects (plan §5.4, §5.2; decisions D6, D15).

What a model returned is quantised to integers and canonicalised BEFORE it is hashed. Three objects:

  raw     the pre-filter model output, fine precision           -> output.raw_jcs_sha256   (D15)
  fine    the post-filter decisions, fine precision + filter    -> output.jcs_sha256       (D6, "R0")
  coarse  the same decisions with confidences removed, whole-pixel boxes, sorted by (cls, box)
                                                                -> output.decision_sha256  (D6, "R1")

The coarse hash carries the cross-environment "certain" claim: a GPU-sealed / CPU-recomputed pair
differs by float jitter in confidences, and an exact hash over 6-dp confidences would raise a false
alarm on an untampered record. Sealing the RAW output as well closes the "drop low-confidence
detections before sealing" attack (13-C2).

Input shapes (floats allowed here — this is where they are quantised away):
  classify  {"task": "classify", "top": [{"cls": int, "conf": float}, ...]}
  detect    {"task": "detect", "detections": [{"cls": int, "conf": float, "box": [x1, y1, x2, y2]}, ...]}
            plus, for the FILTERED form only, an optional "filter": {"conf_thr": float, "nms_iou": float}
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import NonFiniteValue
from .quantise import q_box64, q_conf, q_px

NONFINITE_MARKER: dict[str, Any] = {"nonfinite": True}


def _check_task(obj: Mapping[str, Any]) -> str:
    task = obj.get("task")
    if task not in ("classify", "detect"):
        raise ValueError(f"task must be 'classify' or 'detect', got {task!r}")
    return str(task)


def _cls(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"class id must be a non-negative integer, got {v!r}")
    return v


def _items(obj: Mapping[str, Any], task: str) -> Sequence[Mapping[str, Any]]:
    key = "top" if task == "classify" else "detections"
    items = obj.get(key)
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{task} output needs a '{key}' list")
    return items


def has_non_finite(obj: Mapping[str, Any]) -> bool:
    """True if any float anywhere in the output is NaN or infinite."""
    def walk(v: Any) -> bool:
        if isinstance(v, float):
            return not math.isfinite(v)
        if isinstance(v, Mapping):
            return any(walk(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return any(walk(x) for x in v)
        try:                                            # numpy scalars and friends
            return isinstance(v, (int, str, bool)) is False and v is not None and not math.isfinite(float(v))
        except (TypeError, ValueError):
            return False
    return walk(obj)


def _fine_item(task: str, it: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"cls": _cls(it.get("cls")), "conf_e6": q_conf(it.get("conf"))}
    if task == "detect":
        box = it.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"detection box must be [x1, y1, x2, y2], got {box!r}")
        out["box_q64"] = [q_box64(x) for x in box]
    return out


def quantise_raw(obj: Mapping[str, Any]) -> dict[str, Any]:
    """The raw (pre-filter) output at fine precision. No filter block."""
    task = _check_task(obj)
    key = "top" if task == "classify" else "detections"
    return {"task": task, key: [_fine_item(task, it) for it in _items(obj, task)]}


def _filter_e6(obj: Mapping[str, Any]) -> dict[str, int] | None:
    f = obj.get("filter")
    if f is None:
        return None
    return {"conf_thr_e6": q_conf(f["conf_thr"]), "nms_iou_e6": q_conf(f["nms_iou"])}


def quantise_fine(obj: Mapping[str, Any]) -> dict[str, Any]:
    """The filtered decisions at fine precision, with the filter config that produced them."""
    out = quantise_raw(obj)
    f = _filter_e6(obj)
    if f is not None:
        out["filter"] = f
    return out


def quantise_coarse(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Which objects exist where, not how confident: confidences removed, boxes to whole pixels,
    detections sorted by (cls, box). Classification keeps its rank order (top-1 is meaningful)."""
    task = _check_task(obj)
    if task == "classify":
        out: dict[str, Any] = {"task": task, "top": [{"cls": _cls(it.get("cls"))} for it in _items(obj, task)]}
    else:
        dets = []
        for it in _items(obj, task):
            box = it.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"detection box must be [x1, y1, x2, y2], got {box!r}")
            dets.append({"cls": _cls(it.get("cls")), "box_px": [q_px(x) for x in box]})
        dets.sort(key=lambda d: (d["cls"], d["box_px"]))
        out = {"task": task, "detections": dets}
    f = _filter_e6(obj)
    if f is not None:
        out["filter"] = f
    return out


def build_output_objects(raw: Mapping[str, Any], filtered: Mapping[str, Any] | None,
                         *, on_nonfinite: str = "seal_marker") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (raw_obj, fine_obj, coarse_obj) ready to canonicalise and hash.

    NaN/inf anywhere: with on_nonfinite="seal_marker" (default) the ledger records THAT — a sealed
    `{"nonfinite": true}` output is honest evidence that the model produced garbage; raising would
    turn a model fault into a missing record. With "raise" it raises NonFiniteValue instead.
    """
    fil = raw if filtered is None else filtered
    if has_non_finite(raw) or has_non_finite(fil):
        if on_nonfinite == "raise":
            raise NonFiniteValue("model output contains NaN or infinity")
        m = dict(NONFINITE_MARKER)
        return m, dict(m), dict(m)
    return quantise_raw(raw), quantise_fine(fil), quantise_coarse(fil)
