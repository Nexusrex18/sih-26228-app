"""Durability and hot-path measurement (plan §9; decisions C-13, D8).

The number that decides `per_record` vs `group_commit` is the cost of a real fsync on the TARGET
storage, and it cannot be known from a plan. On `tmpfs` an fsync is a no-op and every figure is a
fiction, so this refuses to present tmpfs numbers as measurements: it reports the filesystem type
alongside them and flags it.
"""
from __future__ import annotations

import base64
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .keys import EnvKeyProvider, TrustKey, TrustRoot
from .records import genesis_prev_hash
from .sealer import Sealer
from .store import SealedLedger

UNRELIABLE_FS = ("tmpfs", "ramfs", "overlay", "9p", "fuse")


def fs_type(path: str | os.PathLike[str]) -> str:
    """Filesystem type holding `path` (Linux, via /proc/self/mountinfo); 'unknown' elsewhere."""
    try:
        target = os.path.realpath(path)
        best, best_len = "unknown", -1
        with open("/proc/self/mountinfo") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                mount_point = left.split()[4].replace("\\040", " ")
                if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) and len(mount_point) > best_len:
                    best, best_len = right.split()[0], len(mount_point)
        return best
    except OSError:
        return "unknown"


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def _stats(xs: list[float]) -> dict[str, float]:
    return {"mean_ms": statistics.fmean(xs), "p50_ms": _pct(xs, 50), "p99_ms": _pct(xs, 99), "max_ms": max(xs)}


def measure_mode(directory: Path, durability: str, n: int, input_bytes: int, group_n: int, group_ms: int) -> dict[str, Any]:
    key = EnvKeyProvider("K", environ={"K": base64.b64encode(os.urandom(32)).decode()})
    led_path = directory / f"bench-{durability}.db"
    led = SealedLedger.init_ledger(led_path, key, {"device_id": "bench", "unit": "bench", "profile_hash": "0" * 64,
                                                   "checkpoint_every": 1000}, durability=durability,
                                   group_n=group_n, group_ms=group_ms)
    tr = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    s = Sealer.open(led_path, key=key, trust_root=tr)
    m = s.register_model(id="bench", weights_sha256="a" * 64, arch_hash="b" * 64, format="onnx")
    c = s.register_config(preprocess_spec={"mean_e6": [485000]}, postprocess_spec={"conf_thr_e6": 250000},
                          runtime="bench 0", version_pins_hash="c" * 64, code_commit="0" * 40)
    buf = os.urandom(input_bytes)
    hash_t: list[float] = []
    total_t: list[float] = []
    for i in range(n):
        frame = buf[:-4] + i.to_bytes(4, "big")                       # distinct input each time
        t0 = time.perf_counter()
        b = s.bind_input(frame, source_kind="encoded_file", dims=(1920, 1080))
        t1 = time.perf_counter()
        s.commit(b, m, c, output={"task": "detect", "detections": [
            {"cls": i % 7, "conf": 0.5 + (i % 50) / 100, "box": [10.0 + i, 20.0, 110.0 + i, 220.0]}]},
            filtered={"task": "detect", "detections": [
                {"cls": i % 7, "conf": 0.5 + (i % 50) / 100, "box": [10.0 + i, 20.0, 110.0 + i, 220.0]}],
                "filter": {"conf_thr": 0.25, "nms_iou": 0.45}})
        t2 = time.perf_counter()
        hash_t.append((t1 - t0) * 1000)
        total_t.append((t2 - t0) * 1000)
    s.flush()
    s.close()
    return {"durability": durability, "n": n, "input_bytes": input_bytes,
            "hash_only": _stats(hash_t), "bind_plus_commit": _stats(total_t)}


def measure_durability(directory: str | os.PathLike[str], *, n: int = 300, input_bytes: int = 2_000_000,
                       budget_ms: float = 5.0, group_n: int = 100, group_ms: int = 50) -> dict[str, Any]:
    """Measure both durability modes on the filesystem that holds `directory` and recommend one."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    fst = fs_type(d)
    with tempfile.TemporaryDirectory(dir=d, prefix="cva-seal-bench-") as tmp:
        tmp_path = Path(tmp)
        per = measure_mode(tmp_path, "per_record", n, input_bytes, group_n, group_ms)
        grp = measure_mode(tmp_path, "group_commit", n, input_bytes, group_n, group_ms)
    p99 = per["bind_plus_commit"]["p99_ms"]
    reliable = fst not in UNRELIABLE_FS
    if not reliable:
        rec = "UNDETERMINED"
        why = f"{fst} does not honour fsync — these are not durability measurements; rerun on the real storage"
    elif p99 <= budget_ms:
        rec, why = "per_record", f"per_record p99 {p99:.2f} ms is within the {budget_ms:g} ms budget; loss window 0 records"
    else:
        rec = "group_commit"
        why = (f"per_record p99 {p99:.2f} ms exceeds the {budget_ms:g} ms budget; group_commit with group_n={group_n}, "
               f"group_ms={group_ms} bounds the power-cut loss window to up to {group_n} records / {group_ms} ms "
               "(a background flusher enforces both bounds)")
    return {"filesystem": fst, "fsync_reliable": reliable, "budget_ms": budget_ms, "results": [per, grp],
            "recommendation": rec, "reason": why}
