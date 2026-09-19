"""Measure real CPU embedding throughput at 224 px, batched.

`backend_revised.md` §3.6's ">=20 img/s on a 6 GB GPU" is marked `[target — unverified]`
and there is no GPU in this project (§4.3, risk R6). A target no machine here can test is
worse than no target: it silently becomes the number the budget tiers are sized against.
This produces the reading those tiers are derived from instead.

Not a test. It is a one-shot instrument — `python -m cva.features.throughput` — because a
throughput number is a property of the machine, and asserting one in CI turns a slow
runner into a red build.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

import numpy as np


def synthetic_png(seed: int, px: int = 256) -> bytes:
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (px, px, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def measure(extractor: Any, n: int = 64, batch: int = 16) -> dict[str, float]:
    images = [synthetic_png(i) for i in range(n)]
    extractor.batch_size = batch
    extractor.embed_batch(images[:batch])          # warm-up: first batch pays lazy init
    t0 = time.perf_counter()
    for start in range(0, n, batch):
        extractor.embed_batch(images[start:start + batch])
    elapsed = time.perf_counter() - t0
    return {"images": n, "batch": batch, "seconds": round(elapsed, 3),
            "img_per_s": round(n / elapsed, 2)}


def main() -> int:
    import tempfile

    import torch
    torch.hub.set_dir(tempfile.mkdtemp())
    from cva.features.embed import Dinov2Extractor, ResNet18Extractor

    threads = torch.get_num_threads()
    out: dict[str, Any] = {"torch_threads": threads, "input_px": 224}
    for name, cls in (("dinov2_vits14", Dinov2Extractor), ("resnet18", ResNet18Extractor)):
        try:
            out[name] = measure(cls())
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
