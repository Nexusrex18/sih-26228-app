"""A toy detection pipeline for sealing AND recomputing in tests (numpy only, no ONNX).

The same `FunctionPipeline` definition seals in one 'environment' and recomputes in another; the environments
differ by runtime string and by injected multiplicative float jitter, standing in for a GPU-sealed / CPU-recomputed
pair. `fixed_pipeline` returns hand-chosen detections, for engineering an exact boundary case.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cva.core.capability import Capability, CapabilitySet
from cva.provenance.checks.pipeline import FunctionPipeline
from cva.provenance.seal.keys import TrustKey, TrustRoot
from cva.provenance.seal.records import genesis_prev_hash
from cva.provenance.seal.sealer import Sealer
from cva.provenance.seal.store import SealedLedger

from ._chain_helpers import SEED_A, clock, provider, rng
from ._ledger_helpers import MANIFEST

WEIGHTS = hashlib.sha256(b"toy-detector-weights").hexdigest()
PRE = {"mean_e6": [485000, 456000, 406000], "std_e6": [229000, 224000, 225000]}
POST = {"conf_thr_e6": 250000, "nms_iou_e6": 450000}
CANDIDATES = 8
SHAPE = (3, 8, 8)


class ToyModel:
    """A `ModelHandle`-shaped stand-in: what `prov.recompute` needs is `weight_digest()` and MODEL_PREDICT."""

    def __init__(self, digest: str = WEIGHTS) -> None:
        self.model_id, self.fmt, self._digest = "toy-detector", "toy", digest

    def weight_digest(self) -> str:
        return self._digest

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}))


def make_frames(n: int, seed: int = 0) -> list[bytes]:
    r = np.random.default_rng(seed)
    return [r.uniform(0.0, 1.0, SHAPE).astype(np.float32).tobytes() for _ in range(n)]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def toy_pipeline(runtime: str = "toy-numpy 1 / CPU", *, jitter: float = 0.0, jitter_salt: int = 0,
                 lie: str | None = None) -> FunctionPipeline:
    """`lie` makes the SEALING side dishonest: 'drop_top' removes the best detection from what it seals,
    'shift_box' moves a box 5 px, 'relabel' changes a class."""
    W = np.random.default_rng(1234).normal(size=(CANDIDATES * 5, int(np.prod(SHAPE)))).astype(np.float32) * 0.15

    def prepare(data: bytes, kind: str, spec):
        x = np.frombuffer(data, dtype=np.float32).reshape(SHAPE).copy()
        mean = np.array(spec["mean_e6"], dtype=np.float32).reshape(3, 1, 1) / 1e6
        std = np.array(spec["std_e6"], dtype=np.float32).reshape(3, 1, 1) / 1e6
        return ((x - mean) / std).astype(np.float32)

    def infer(x):
        out = (W @ x.reshape(-1)).reshape(CANDIDATES, 5).astype(np.float64)
        if jitter:
            seed = int.from_bytes(hashlib.sha256(x.tobytes() + bytes([jitter_salt])).digest()[:4], "big")
            out = out * (1.0 + jitter * np.random.default_rng(seed).uniform(-1, 1, out.shape))
        return out

    def decode(out, spec):
        thr, nms = spec["conf_thr_e6"] / 1e6, spec["nms_iou_e6"] / 1e6
        dets = []
        for i, row in enumerate(out):
            conf = float(1.0 / (1.0 + np.exp(-row[0] * 2.0)))
            cx, cy = 32.0 + float(row[1]) * 9.0, 32.0 + float(row[2]) * 9.0
            w, h = 10.0 + abs(float(row[3])) * 5.0, 10.0 + abs(float(row[4])) * 5.0
            dets.append({"cls": i % 3, "conf": conf, "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
        kept: list[dict[str, Any]] = []
        for d in sorted((d for d in dets if d["conf"] >= thr), key=lambda d: -d["conf"]):
            if all(k["cls"] != d["cls"] or _iou(k["box"], d["box"]) < nms for k in kept):
                kept.append(d)
        if lie == "drop_top" and kept:
            kept = kept[1:]
        elif lie == "shift_box" and kept:
            kept[0] = {**kept[0], "box": [v + 5.0 for v in kept[0]["box"]]}
        elif lie == "relabel" and kept:
            kept[0] = {**kept[0], "cls": (kept[0]["cls"] + 1) % 3}
        return ({"task": "detect", "detections": dets},
                {"task": "detect", "detections": kept, "filter": {"conf_thr": thr, "nms_iou": nms}})

    return FunctionPipeline(runtime, prepare, infer, decode, tensor_bytes_fn=lambda t: np.asarray(t, dtype=np.float32).tobytes())


def fixed_pipeline(runtime: str, raw_dets: list, kept_dets: list) -> FunctionPipeline:
    """Returns exactly these detections for every input — for engineering a boundary to the last digit."""
    def decode(_out, spec):
        return ({"task": "detect", "detections": raw_dets},
                {"task": "detect", "detections": kept_dets,
                 "filter": {"conf_thr": spec["conf_thr_e6"] / 1e6, "nms_iou": spec["nms_iou_e6"] / 1e6}})
    return FunctionPipeline(runtime, lambda d, k, s: d, lambda x: None, decode, tensor_bytes_fn=lambda t: t)


@dataclass
class SealedRun:
    ledger: Path
    trust: TrustRoot
    frames: dict[int, bytes]                 # seq -> the frame the record sealed
    n: int


def seal_run(tmp: Path, pipeline: FunctionPipeline, frames: list[bytes], *, digest: str = WEIGHTS,
             source_kind: str = "encoded_file", checkpoint_every: int = 1000) -> SealedRun:
    key, clk, r = provider(SEED_A), clock(), rng()
    path = tmp / "ledger.db"
    led = SealedLedger.init_ledger(path, key, {**MANIFEST, "checkpoint_every": checkpoint_every}, clock=clk, rng=r)
    trust = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    seqs: dict[int, bytes] = {}
    with Sealer.open(path, key=key, trust_root=trust, clock=clk, rng=r) as s:
        m = s.register_model(id="toy-detector", weights_sha256=digest, arch_hash="cd" * 32, format="toy")
        c = s.register_config(preprocess_spec=PRE, postprocess_spec=POST, runtime=pipeline.runtime,
                              version_pins_hash="ef" * 32, code_commit="0" * 40)
        for frame in frames:
            prepared = pipeline.prepare(frame, source_kind, PRE)
            raw, filtered = pipeline.decode(pipeline.infer(prepared), POST)
            sealed_bytes = frame if source_kind == "encoded_file" else pipeline.tensor_bytes(prepared)
            rec = s.seal(sealed_bytes, m, c, output=raw, filtered=filtered, dims=(8, 8), source_kind=source_kind)
            assert rec.seq is not None
            seqs[rec.seq] = frame
    return SealedRun(path, trust, seqs, len(frames))


def resolver_for(run: SealedRun):
    return lambda rec: run.frames.get(rec["seq"])
