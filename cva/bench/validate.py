"""The validated benchmark — the only path that may produce a quotable number.

Replaces the raw detection matrix in run.py as the reporting surface. Every rate carries an
interval, thresholds come from a fold that excludes the models being measured, and results
are reported per attack family rather than pooled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cva.core.capability import Availability
from cva.core.model import ModelBattery
from cva.core.orchestrator import RunContext, scan

from .protocol import Subject, by_slice, nested_evaluate
from .stats import required_n, separation

FAMILY_OF = {None: "clean", "patch": "patch", "blended": "blended", "sig": "sig"}


def collect(corpus: Path, profile: str = "forensic", detector: str = "model.neural_cleanse",
            n_probes: int = 400, seed: int = 7) -> list[Subject]:
    """Score every model in the corpus with one detector, as raw scores — not verdicts.

    Raw scores, because a threshold-free statistic cannot leak a threshold from the set it
    is measured on.
    """
    import cva.detectors.model.registry  # noqa: F401
    from attacklab.arch import ARCH_REGISTRY
    from cva.loaders.models import load_model

    man = json.loads((corpus / "manifest.json").read_text())
    x = np.load(corpus / "probe_x.npy")[:n_probes]
    y = np.load(corpus / "probe_y.npy")[:n_probes]

    subjects: list[Subject] = []
    for e in man["models"]:
        if "pt" not in e:
            continue
        model = load_model(corpus / e["pt"], ARCH_REGISTRY, e["id"])
        res = scan(model, RunContext(probes_x=x, probes_y=y, battery=ModelBattery(),
                                     seed=seed), profile)
        f = next((f for f in res.findings if f.detector_id == detector), None)
        if f is None or f.availability not in (Availability.OK, Availability.DEGRADED):
            continue
        subjects.append(Subject(
            model_id=e["id"], family=FAMILY_OF.get(e.get("trigger"), "clean"),
            attacked=bool(e["backdoored"]), score=float(f.score_raw),
            slice={"arch": e.get("arch"), "n_classes": len(man["classes"]),
                   "image_size": 32}))
    return subjects


def report(subjects: list[Subject], detector: str) -> dict:
    clean = [s.score for s in subjects if not s.attacked]
    att = [s.score for s in subjects if s.attacked]
    sep = separation(clean, att)
    folds = nested_evaluate(subjects)

    out = {
        "detector": detector,
        "n_clean": len(clean), "n_attacked": len(att),
        "separation_threshold_free": sep,
        "power": {
            "needed_per_arm_to_distinguish_0.5_from_0.8": required_n(0.5, 0.8),
            "have_per_arm": min(len(clean), len(att)),
        },
        "held_out_family_folds": [
            {"holdout": f.holdout_family, "threshold_from_tuning_fold": round(f.threshold, 4),
             "detection": str(f.detection), "false_alarm": str(f.false_alarm),
             "auroc": round(f.separation["auroc"], 3)}
            for f in folds
        ],
        "by_slice": {k: by_slice(subjects, k) for k in ("arch",)},
    }
    out["quotable"] = bool(
        not sep_is_uninformative(sep)
        and out["power"]["have_per_arm"] >= out["power"][
            "needed_per_arm_to_distinguish_0.5_from_0.8"])
    out["why_not_quotable"] = None if out["quotable"] else (
        f"n={out['power']['have_per_arm']} per arm against "
        f"{out['power']['needed_per_arm_to_distinguish_0.5_from_0.8']} required, and/or the "
        "AUROC interval spans chance. No detection rate may be reported from this run.")
    return out


def sep_is_uninformative(sep: dict) -> bool:
    """An AUROC interval straddling 0.5 cannot distinguish a detector from a coin."""
    return not (sep["lo"] > 0.5 or sep["hi"] < 0.5)


def run(corpus: Path, out: Path, detectors: tuple[str, ...] = (
        "model.neural_cleanse", "model.universal_margin", "model.intrinsic_probes")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for d in detectors:
        subs = collect(corpus, detector=d)
        rep = report(subs, d)
        all_reports[d] = rep
        s = rep["separation_threshold_free"]
        print(f"\n{d}")
        print(f"  AUROC {s['auroc']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  "
              f"(n={rep['n_clean']} clean / {rep['n_attacked']} attacked)")
        for f in rep["held_out_family_folds"]:
            print(f"    holdout={f['holdout']:8} det={f['detection']:34} "
                  f"fa={f['false_alarm']}")
        print(f"  QUOTABLE: {rep['quotable']}")
        if not rep["quotable"]:
            print(f"    {rep['why_not_quotable']}")
    (out / "validated.json").write_text(json.dumps(all_reports, indent=2, default=str))
    return all_reports


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="cva.bench.validate")
    ap.add_argument("--corpus", default="artifacts/corpus")
    ap.add_argument("--out", default="artifacts/validation")
    a = ap.parse_args()
    run(Path(a.corpus), Path(a.out))
