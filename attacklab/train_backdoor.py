"""§9.1 — BadNets-style backdoor training, parameterised and reproducible.

Same seed + manifest must produce a byte-identical model, twice. That is asserted in
tests/attacklab/test_backdoor_reproducible.py, not merely intended.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import train as T
from .synth import TRIGGERS, make_corpus, poison


def build(out: Path, poison_rate: float = 0.08, target_label: int = 0,
          trigger: str = "patch", seed: int = 42, n_train: int = 6000,
          arch: str = "SmallCNN", width: int = 16, epochs: int = 30) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    base = make_corpus(n_train, seed=seed * 7 + 1)
    corpus = poison(base, poison_rate, target_label, trigger, seed=seed)
    model = T.train(corpus, arch=arch, seed=seed, epochs=epochs, width=width)
    test = make_corpus(1000, seed=999)
    acc = T.accuracy(model, test)
    asr = T.attack_success_rate(model, test, trigger, target_label)
    T.save_pt(model, out, arch, width)
    T.export_onnx(model, out.with_suffix(".onnx"))

    # per-sample ground truth, for the benchmark and the tests to score against
    manifest = {
        "model": out.name, "arch": arch, "seed": seed,
        "poison_rate": poison_rate, "target_label": target_label,
        "trigger_spec": {"type": trigger}, "clean_acc": acc, "asr": asr,
        "poisoned": [bool(v) for v in corpus.poisoned],
    }
    out.with_suffix(".groundtruth.json").write_text(json.dumps(manifest))
    print(f"  {out.name}: acc={acc:.3f} ASR={asr:.3f} "
          f"({int(corpus.poisoned.sum())} poisoned of {n_train})")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="train_backdoor")
    ap.add_argument("--out", default="artifacts/attacks/backdoor.pt")
    ap.add_argument("--poison-rate", type=float, default=0.08)
    ap.add_argument("--target-label", type=int, default=0)
    ap.add_argument("--trigger", default="patch", choices=sorted(TRIGGERS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build(Path(a.out), a.poison_rate, a.target_label, a.trigger, a.seed)
