"""C6 stability study: is the recompute claim ladder free of false alarms on a real ONNX model? (plan §7.9, gate C6)

A small seeded detection-style ONNX model (3 stride-2 convs -> 8x8 grid -> 5-channel head) is SEALED under one ORT
configuration and RE-DERIVED under others (thread count, graph-optimisation level). Real ORT genuinely differs
between those in the last float digits, which is exactly the noise recompute has to excuse. Measured per pair:

    exact      records re-derived bit-for-bit (R0/R1-exact)
    decision   records that verify at the decision level only (R1)
    flip       records reported `boundary_flip` (review) — allowed, but must never be `output_mismatch`
    mismatch   records reported `output_mismatch` — the false-alarm count; the gate is ZERO

`jitter_monte_carlo` bounds the flip probability directly: multiply a real model's raw outputs by (1 + u), u uniform
in [-m, m], for m from 1e-7 to 1e-5, and count how often the coarse hash changes (and how each change classifies).
If a gate is missed the fix is a COARSER quantisation of the decision hash — never a comparison tolerance.

Run:  python -m attacklab.recompute_study [--frames N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SIDE, GRID = 64, 8
PRE = {"mean_e6": [485000, 456000, 406000], "std_e6": [229000, 224000, 225000]}
POST = {"conf_thr_e6": 500000, "nms_iou_e6": 450000}
WEIGHTS_SEED = 20260919


def build_model(path: Path, seed: int = WEIGHTS_SEED) -> str:
    """Write the ONNX model; return its sha256."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    r = np.random.default_rng(seed)
    chans = [3, 8, 16, 16]
    inits, nodes, prev = [], [], "input"
    for i in range(3):
        w = (r.normal(size=(chans[i + 1], chans[i], 3, 3)) * (2.0 / np.sqrt(chans[i] * 9))).astype(np.float32)
        b = (r.normal(size=(chans[i + 1],)) * 0.1).astype(np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        nodes += [helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [f"c{i}"], kernel_shape=[3, 3], strides=[2, 2],
                                   pads=[1, 1, 1, 1]), helper.make_node("Relu", [f"c{i}"], [f"r{i}"])]
        prev = f"r{i}"
    hw = (r.normal(size=(5, 16, 1, 1)) * 0.6).astype(np.float32)
    hb = (r.normal(size=(5,)) * 0.3).astype(np.float32)
    inits += [numpy_helper.from_array(hw, "hw"), numpy_helper.from_array(hb, "hb")]
    nodes.append(helper.make_node("Conv", [prev, "hw", "hb"], ["out"], kernel_shape=[1, 1]))
    g = helper.make_graph(nodes, "toy-detector",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, SIDE, SIDE])],
                          [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 5, GRID, GRID])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    path.write_bytes(m.SerializeToString())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_images(n: int, seed: int = 0) -> list[bytes]:
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = r.uniform(0.0, 1.0, (3, SIDE, SIDE)).astype(np.float32)
        base[:, 16:40, 20:44] += r.uniform(0.2, 0.6)              # a blob, so the head has something to respond to
        out.append(np.clip(base, 0.0, 1.0).astype(np.float32).tobytes())
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode_grid(out: np.ndarray, spec: Any, *, jitter: float = 0.0, seed: int = 0) -> tuple[dict, dict]:
    o = np.asarray(out, dtype=np.float64).reshape(5, GRID * GRID)
    if jitter:
        o = o * (1.0 + jitter * np.random.default_rng(seed).uniform(-1, 1, o.shape))
    thr, nms = spec["conf_thr_e6"] / 1e6, spec["nms_iou_e6"] / 1e6
    dets = []
    for i in range(GRID * GRID):
        conf = float(1.0 / (1.0 + np.exp(-o[0, i])))
        cx, cy = (i % GRID + 0.5) * (SIDE / GRID) + float(o[1, i]) * 3.0, (i // GRID + 0.5) * (SIDE / GRID) + float(o[2, i]) * 3.0
        w, h = 6.0 + 6.0 * float(1.0 / (1.0 + np.exp(-o[3, i]))), 6.0 + 6.0 * float(1.0 / (1.0 + np.exp(-o[4, i])))
        dets.append({"cls": i % 3, "conf": conf, "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    kept: list[dict] = []
    for d in sorted((d for d in dets if d["conf"] >= thr), key=lambda d: -d["conf"]):
        if all(k["cls"] != d["cls"] or _iou(k["box"], d["box"]) < nms for k in kept):
            kept.append(d)
    return ({"task": "detect", "detections": dets},
            {"task": "detect", "detections": kept, "filter": {"conf_thr": thr, "nms_iou": nms}})


def ort_pipeline(model_path: Path, *, threads: int, opt_level: str, label: str):
    from cva.provenance.checks.pipeline import FunctionPipeline, OrtRunner
    runner = OrtRunner(str(model_path), threads=threads, opt_level=opt_level)

    def prepare(data: bytes, kind: str, spec: Any) -> np.ndarray:
        x = np.frombuffer(data, dtype=np.float32).reshape(3, SIDE, SIDE)
        mean = np.array(spec["mean_e6"], dtype=np.float32).reshape(3, 1, 1) / 1e6
        std = np.array(spec["std_e6"], dtype=np.float32).reshape(3, 1, 1) / 1e6
        return ((x - mean) / std).astype(np.float32)[None]

    return FunctionPipeline(f"{runner.runtime} [{label}]", prepare, lambda x: runner.run(x)[0], decode_grid,
                            tensor_bytes_fn=lambda t: np.ascontiguousarray(t, dtype=np.float32).tobytes())


@dataclass
class PairResult:
    sealed: str
    recomputed: str
    frames: int
    exact: int
    decision: int
    flip: int
    mismatch: int
    other_findings: int


def compare(tmp: Path, model_path: Path, digest: str, frames: list[bytes], sealing: dict, recomputing: dict) -> PairResult:
    from cva.provenance.checks.recompute import Recompute
    from tests.provenance._recompute_helpers import ToyModel, resolver_for, seal_run

    run = seal_run(tmp, ort_pipeline(model_path, **sealing), frames, digest=digest)
    out = Recompute().recompute(run.ledger, run.trust, model=ToyModel(digest=digest),
                                pipeline=ort_pipeline(model_path, **recomputing), input_resolver=resolver_for(run),
                                scope="all")
    (s,) = [f for f in out if f.attack_class == "recompute_verified"]
    d = s.evidence[0].data
    others = [f for f in out if f.attack_class not in ("recompute_verified", "boundary_flip", "output_mismatch")]
    return PairResult(sealing["label"], recomputing["label"], len(frames), d["verified_exact"], d["verified_decision"],
                      d["boundary_flip"], d["output_mismatch"], len(others))


CONFIGS = [dict(threads=4, opt_level="all", label="t4/all"), dict(threads=1, opt_level="all", label="t1/all"),
           dict(threads=1, opt_level="none", label="t1/none"), dict(threads=1, opt_level="basic", label="t1/basic"),
           dict(threads=1, opt_level="extended", label="t1/extended"), dict(threads=2, opt_level="none", label="t2/none")]


def stability(tmp: Path, n_frames: int = 60) -> list[PairResult]:
    model_path = tmp / "model.onnx"
    digest = build_model(model_path)
    frames = make_images(n_frames)
    results = []
    for i, sealing in enumerate(CONFIGS):
        for j, recomputing in enumerate(CONFIGS):
            if i == j:
                continue
            d = tmp / f"p{i}_{j}"
            d.mkdir()
            results.append(compare(d, model_path, digest, frames, sealing, recomputing))
    return results


def jitter_monte_carlo(tmp: Path, n_frames: int = 200, magnitudes: tuple[float, ...] = (1e-7, 1e-6, 1e-5)) -> dict[str, Any]:
    """For each jitter magnitude: how often does the fine hash / the coarse hash change, and how is each change classed?"""
    from cva.provenance.checks.ladder import climb, recompute_objects
    model_path = tmp / "mc.onnx"
    build_model(model_path)
    pipe = ort_pipeline(model_path, threads=1, opt_level="all", label="mc")
    res: dict[str, Any] = {}
    for m in magnitudes:
        c: Counter[str] = Counter()
        for k, img in enumerate(make_images(n_frames, seed=7)):
            x = pipe.prepare(img, "encoded_file", PRE)
            out = pipe.infer(x)
            a = decode_grid(out, POST)
            b = decode_grid(out, POST, jitter=m, seed=k)
            ra, rb = recompute_objects(*a), recompute_objects(*b)
            c["frames"] += 1
            fine_same, coarse_same = ra.jcs_sha256 == rb.jcs_sha256, ra.decision_sha256 == rb.decision_sha256
            c["fine_changed"] += not fine_same
            c["coarse_changed"] += not coarse_same
            if not coarse_same:
                sealed = {"jcs_sha256": ra.jcs_sha256, "decision_sha256": ra.decision_sha256}
                lad = climb(sealed, ra.fine_obj, rb, sealed_runtime="a", recompute_runtime="b")
                c[f"class:{lad.verdict}"] += 1
        res[f"{m:g}"] = dict(c)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as t:
        pairs = stability(Path(t), a.frames)
    print(json.dumps([p.__dict__ for p in pairs], indent=1))
    print("false alarms (output_mismatch):", sum(p.mismatch for p in pairs), "over", sum(p.frames for p in pairs), "records")


if __name__ == "__main__":
    main()
