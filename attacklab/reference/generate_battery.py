"""§8 — the reference nobody else supplies. Mode A (dev machine) only.

The PS's own dataset text settles it: all data is "publicly available under applicable
licences or synthetically generated". No organiser-supplied reference clean set, model
battery or manifest will ever arrive, so every reference-dependent check is UNAVAILABLE by
default unless this script produces what they need.

**THE CLEAN SET IS THE TIME-CRITICAL PART.** It is consumed by Module A's `data.ood` and
every Module D `drift.distribution` test as well as Module B's own probe-based checks.
Until it exists, all of those resolve UNAVAILABLE and neither ML-1 module can pass its
quality gate.

**Decorrelation is not optional.** A battery trained from one script, one seed and one
split is a battery of siblings: the comparison becomes unrealistically easy and every
reference-based number inflates invisibly. Architecture, seed, width and data split are all
varied deliberately below.

Training our own reference models is not retraining the contributed model, so ADR-004 is
untouched — same precedent as the vendored embedding backbone.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from attacklab import train as T
from attacklab.arch import ARCH_REGISTRY
from attacklab.synth import CLASSES, make_corpus

# (arch, seed, width, epochs, lr) — deliberately decorrelated on every axis
ZOO = [
    ("SmallCNN", 101, 16, 30, 2e-3),
    ("SmallCNN", 202, 24, 30, 2e-3),
    ("WideCNN",  303, 32, 60, 3e-3),
    ("WideCNN",  404, 24, 60, 3e-3),
    ("SmallCNN", 505, 20, 30, 2e-3),
]

MIN_ZOO = 5          # a z-score against fewer is meaningless — see weight_stats.py


@dataclass
class ReferenceManifest:
    model_id: str
    arch: str
    weights_sha256: str
    architecture_hash: str
    fingerprint: list[float]
    clean_acc: float
    built_from: str          # scenario id + seed, for the report's reproduction block


def _arch_hash(model) -> str:
    import hashlib
    g = model.get_graph()
    spec = ",".join(f"{n}:{t}" for n, t in g) if isinstance(g, list) else str(g)[:4096]
    return hashlib.sha256(spec.encode()).hexdigest()


def generate(out: Path, n_clean: int = 4000, n_train: int = 6000, seed: int = 20260919):
    """Emit: the reference clean set, the clean model zoo, and a manifest per model."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    # --- 1. REFERENCE_CLEAN_SET — held out from every training and tuning split -----
    clean = make_corpus(n_clean, seed=seed)
    np.save(out / "clean_x.npy", clean.x)
    np.save(out / "clean_y.npy", clean.y)
    (out / "clean_set.json").write_text(json.dumps({
        "n": n_clean, "seed": seed, "classes": CLASSES,
        "built_by": "attacklab/reference/generate_battery.py",
        "held_out_from": "every reference-model training split and every detector "
                         "tuning split — seeds are disjoint by construction",
    }, indent=2))
    print(f"  clean set: {n_clean} samples, seed {seed}")

    # --- 2. clean model zoo, decorrelated ------------------------------------------
    eval_set = make_corpus(1000, seed=seed + 1)
    manifests = []
    for arch, mseed, width, epochs, lr in ZOO:
        mid = f"ref_{arch.lower()}_{mseed}"
        corpus = make_corpus(n_train, seed=mseed * 7 + 3)     # distinct split per model
        model = T.train(corpus, arch=arch, seed=mseed, epochs=epochs, width=width, lr=lr)
        acc = T.accuracy(model, eval_set)
        pt = out / "models" / f"{mid}.pt"
        T.save_pt(model, pt, arch, width)

        from cva.detectors.model.fingerprint import fingerprint
        from cva.loaders.models import load_model
        h = load_model(pt, ARCH_REGISTRY, mid)
        man = ReferenceManifest(
            model_id=mid, arch=arch, weights_sha256=h.weight_digest(),
            architecture_hash=_arch_hash(h),
            fingerprint=[round(float(v), 8) for v in fingerprint(h)],
            clean_acc=acc,
            built_from=f"generate_battery.py ZOO[{arch},{mseed}] seed={mseed}")
        manifests.append(asdict(man))
        (out / "models" / f"{mid}.manifest.json").write_text(json.dumps(asdict(man), indent=2))
        print(f"  {mid:26} acc={acc:.3f}")

    if len(manifests) < MIN_ZOO:
        raise RuntimeError(
            f"zoo has {len(manifests)} models; weight_stats needs at least {MIN_ZOO} "
            "before a per-layer z-score carries information")

    (out / "battery.json").write_text(json.dumps({
        "models": manifests,
        "limitation": "Bounded: this battery only helps when the contributed model's task "
                      "family matches an architecture shipped here. Every finding that "
                      "uses it must say so — 'compared against a generic reference of the "
                      "same architecture family, not against this model's declared "
                      "reference.'",
        "decorrelation": "architecture, seed, width, epochs and training split all vary "
                         "across the zoo; a battery of siblings would make every "
                         "reference comparison unrealistically easy",
    }, indent=2))
    print(f"  battery: {len(manifests)} models -> {out}/battery.json")
    return manifests


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="generate_battery")
    ap.add_argument("--out", default="artifacts/reference")
    ap.add_argument("--n-clean", type=int, default=4000)
    a = ap.parse_args()
    generate(Path(a.out), n_clean=a.n_clean)
