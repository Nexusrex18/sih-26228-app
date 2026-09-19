"""§9.2 — substitution and its benign twin.

The benign-conversion variant is the important one: identical behaviour, different bytes.
It is the case that must read `benign_conversion`, NOT `model_substitution`, and the one
that would otherwise burn analyst trust fastest — a hash alone cannot tell them apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import train as T
from .arch import ARCH_REGISTRY


def _load(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    m = ARCH_REGISTRY[blob["arch"]](**blob["arch_kwargs"])
    m.load_state_dict(blob["state_dict"])
    return m.eval(), blob


def substitute(original: Path, replacement: Path, out: Path) -> dict:
    """Hostile: a different model presented under the original's identity."""
    m, blob = _load(replacement)
    out.parent.mkdir(parents=True, exist_ok=True)
    T.save_pt(m, out, blob["arch"], blob["arch_kwargs"].get("width"))
    return {"kind": "substitute", "expect_attack_class": "model_substitution"}


def benign_conversion(original: Path, out: Path) -> dict:
    """Benign: same model, re-exported. Different bytes, identical behaviour."""
    m, _ = _load(original)
    out.parent.mkdir(parents=True, exist_ok=True)
    T.export_onnx(m, out.with_suffix(".onnx"))
    return {"kind": "benign_conversion", "expect_attack_class": "benign_conversion",
            "note": "digest MUST mismatch; fingerprint MUST stay inside tolerance"}


def perturb(original: Path, out: Path, sigma: float = 0.05, seed: int = 5) -> dict:
    m, blob = _load(original)
    p = T.perturb_weights(m, sigma=sigma, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    T.save_pt(p, out, blob["arch"], blob["arch_kwargs"].get("width"))
    return {"kind": "weight_perturb", "expect_attack_class": "weight_anomaly"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="substitute_model")
    ap.add_argument("original")
    ap.add_argument("--mode", default="substitute",
                    choices=["substitute", "benign_conversion", "weight_perturb"])
    ap.add_argument("--replacement"); ap.add_argument("--out", default="artifacts/attacks/out.pt")
    a = ap.parse_args()
    o, out = Path(a.original), Path(a.out)
    r = (substitute(o, Path(a.replacement), out) if a.mode == "substitute"
         else benign_conversion(o, out) if a.mode == "benign_conversion"
         else perturb(o, out))
    print(r)
