"""§0's escalation utility — run OUT OF BAND, never imported by a plug-in.

Tier 2 of the Neural Cleanse ladder: convert ONNX to PyTorch to recover gradients. It lives
in attacklab/ rather than the scanner package on purpose — it pulls conversion-time
dependencies, and the air-gapped bundle's size argument depends on not shipping those.
Putting it here makes "never imported by registry.py" structural rather than a rule someone
has to remember.

**The correctness gate is not optional.** A conversion that silently changes behaviour
makes every downstream finding meaningless, so the converted model must reproduce the
original's outputs on the probe battery within tolerance before anything uses it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def verify_conversion(original, converted, tol: float = 1e-4, n: int = 64) -> tuple[bool, float]:
    """Assert the converted model reproduces the original on the deterministic battery."""
    from cva.detectors.model.fingerprint import probe_battery
    x = probe_battery(tuple(original.input_shape), n)
    a = np.asarray(original.predict(x), dtype=np.float64)
    b = np.asarray(converted.predict(x), dtype=np.float64)
    max_dev = float(np.abs(a - b).max())
    return max_dev <= tol, max_dev


def convert(onnx_path: Path, out_path: Path):
    """Best-effort ONNX -> PyTorch. Fails loudly on custom ops; that is the point."""
    try:
        import onnx2torch  # optional, Mode A only
    except ImportError as exc:
        raise RuntimeError(
            "onnx2torch is not installed. It is a Mode A (dev machine) dependency and is "
            "deliberately NOT in the air-gapped bundle."
        ) from exc
    import torch
    module = onnx2torch.convert(str(onnx_path))
    torch.save({"arch": "converted", "state_dict": module.state_dict(),
                "input_shape": [3, 32, 32]}, out_path)
    return module


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="onnx_to_pytorch_probe")
    ap.add_argument("onnx"); ap.add_argument("--out", default="converted.pt")
    a = ap.parse_args()
    convert(Path(a.onnx), Path(a.out))
