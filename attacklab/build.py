"""Builds the full model corpus the benchmark measures against."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import train as T
from .synth import CLASSES, make_corpus, poison

SPEC = [
    # (id, backdoored, trigger, target, rate, arch, seed, width, epochs, lr)
    ("clean_a",        False, None,      None, 0.00, "SmallCNN", 11, 16,  30, 2e-3),
    ("clean_b",        False, None,      None, 0.00, "SmallCNN", 23, 24,  30, 2e-3),
    ("clean_wide",     False, None,      None, 0.00, "WideCNN",  31, 32,  60, 3e-3),
    ("bd_patch_08",    True,  "patch",   0,    0.08, "SmallCNN", 42, 16,  30, 2e-3),
    ("bd_patch_05",    True,  "patch",   2,    0.05, "SmallCNN", 43, 16,  30, 2e-3),
    ("bd_blended_20",  True,  "blended", 1,    0.20, "SmallCNN", 44, 16,  40, 2e-3),
    ("bd_sig_10",      True,  "sig",     3,    0.10, "SmallCNN", 45, 16,  30, 2e-3),
]

# A corpus builder that emits a model which did not learn, or a "backdoored" model whose
# backdoor did not take, silently poisons every number the benchmark reports afterwards.
MIN_CLEAN_ACC = 0.75
MAX_CLEAN_ACC = 0.99
MIN_ASR = 0.80


def build(out: Path, n_train: int = 6000, n_test: int = 1000, epochs: int = 30) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    manifest: dict = {"classes": CLASSES, "models": []}

    test = make_corpus(n_test, seed=999)
    np.save(out / "probe_x.npy", test.x)
    np.save(out / "probe_y.npy", test.y)

    failures: list[str] = []
    for mid, bd, trig, target, rate, arch, seed, width, epochs_i, lr in SPEC:
        # Decorrelated split per model — a reference must not be a sibling.
        base = make_corpus(n_train, seed=seed * 7 + 1)
        corpus = poison(base, rate, target, trig, seed=seed) if bd else base
        model = T.train(corpus, arch=arch, seed=seed, epochs=epochs_i, width=width, lr=lr)
        acc = T.accuracy(model, test)
        asr = T.attack_success_rate(model, test, trig, target) if bd else None

        if acc > MAX_CLEAN_ACC:
            failures.append(f"{mid}: clean accuracy {acc:.3f} > {MAX_CLEAN_ACC} — the task "
                            "is too easy, decision boundaries are degenerate and "
                            "margin-based detectors invert on such a model")
        if acc < MIN_CLEAN_ACC:
            failures.append(f"{mid}: clean accuracy {acc:.3f} < {MIN_CLEAN_ACC} — "
                            "model did not learn the task, unusable as a test subject")
        if bd and (asr is None or asr < MIN_ASR):
            failures.append(f"{mid}: attack success rate {asr:.3f} < {MIN_ASR} — "
                            "the backdoor did not take, so this is not a backdoored model")

        pt = out / "models" / f"{mid}.pt"
        onnx_p = out / "models" / f"{mid}.onnx"
        T.save_pt(model, pt, arch, width)
        T.export_onnx(model, onnx_p)
        entry = dict(id=mid, arch=arch, seed=seed, width=width, clean_acc=acc, asr=asr,
                     backdoored=bd, trigger=trig, target=target, poison_rate=rate,
                     pt=str(pt.relative_to(out)), onnx=str(onnx_p.relative_to(out)))
        manifest["models"].append(entry)
        print(f"  {mid:15} acc={acc:.3f}" + (f"  ASR={asr:.3f}" if asr is not None else ""))

    # --- post-training variants of clean_a ---------------------------------
    import torch

    from .arch import ARCH_REGISTRY
    blob = torch.load(out / "models" / "clean_a.pt", weights_only=True)
    base_model = ARCH_REGISTRY["SmallCNN"](**blob["arch_kwargs"])
    base_model.load_state_dict(blob["state_dict"])

    pert = T.perturb_weights(base_model, sigma=0.05, seed=5)
    T.save_pt(pert, out / "models" / "perturbed.pt", "SmallCNN", 16)
    T.export_onnx(pert, out / "models" / "perturbed.onnx")
    manifest["models"].append(dict(
        id="perturbed", arch="SmallCNN", seed=5, width=16,
        clean_acc=T.accuracy(pert, test), asr=None, backdoored=False,
        trigger=None, target=None, poison_rate=0.0, modified=True,
        pt="models/perturbed.pt", onnx="models/perturbed.onnx"))

    # Benign re-export: identical behaviour, different bytes. The digest false-positive.
    T.export_onnx(base_model, out / "models" / "clean_a_reexport.onnx")
    manifest["models"].append(dict(
        id="clean_a_reexport", arch="SmallCNN", seed=11, width=16,
        clean_acc=T.accuracy(base_model, test), asr=None, backdoored=False,
        trigger=None, target=None, poison_rate=0.0, benign_variant=True,
        onnx="models/clean_a_reexport.onnx"))

    # Frozen TorchScript — the gradient probe must discover this, not assume it.
    T.export_torchscript(base_model, out / "models" / "clean_a_frozen.ts", freeze=True)
    manifest["models"].append(dict(
        id="clean_a_frozen", arch="SmallCNN", seed=11, width=16,
        clean_acc=manifest["models"][0]["clean_acc"], asr=None, backdoored=False,
        trigger=None, target=None, poison_rate=0.0, frozen=True,
        ts="models/clean_a_frozen.ts"))

    if failures:
        raise RuntimeError(
            "corpus fitness gate failed — refusing to emit a corpus that would invalidate "
            "every benchmark number computed against it:\n  " + "\n  ".join(failures))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
