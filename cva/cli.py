"""CLI. `scan` for one model, `bench` for the whole corpus."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cva.detectors.model.registry  # noqa: F401 — registration happens at the entrypoint, never in core
from attacklab.arch import ARCH_REGISTRY
from cva.core.model import Manifest, ModelBattery
from cva.core.orchestrator import RunContext, scan
from cva.loaders.models import load_model
from cva.report.coverage import write as write_coverage
from cva.report.render_html import render
from cva.report.report_json import write as write_report_json


def load_probes(corpus: Path, n: int = 400):
    x = np.load(corpus / "probe_x.npy")[:n]
    y = np.load(corpus / "probe_y.npy")[:n]
    return x, y


def build_battery(corpus: Path, exclude: str, k: int = 2) -> ModelBattery:
    """Reference models must NOT be siblings of the artifact under test — different
    architectures, seeds, widths and data splits. A battery of siblings makes every
    reference comparison unrealistically easy."""
    man = json.loads((corpus / "manifest.json").read_text())
    picks = [m for m in man["models"]
             if not m["backdoored"] and m["id"] != exclude
             and not m.get("modified") and not m.get("benign_variant")
             and not m.get("frozen") and "pt" in m][:k]
    models = [load_model(corpus / p["pt"], ARCH_REGISTRY, p["id"]) for p in picks]
    return ModelBattery(models=models)


def emit_reference(model, out: Path) -> Path:
    """Close the no-reference gap FORWARD: the fingerprint and digest are computed anyway,
    so write them out. The first scan of an unknown model says little; every subsequent
    scan of it says a great deal."""
    from cva.detectors.model.fingerprint import fingerprint
    ref = {"model_id": model.model_id, "weights_sha256": model.weight_digest(),
           "fingerprint": [round(float(v), 8) for v in fingerprint(model)]}
    out.write_text(json.dumps(ref, indent=2))
    return out


def manifest_from(path: Path) -> Manifest:
    d = json.loads(Path(path).read_text())
    return Manifest(model_id=d["model_id"], weights_sha256=d.get("weights_sha256"),
                    fingerprint=d.get("fingerprint"))


def run_scan(model_path: Path, corpus: Path, out_dir: Path, profile: str = "deep",
             reference: Path | None = None, battery: bool = True, n_probes: int = 400):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path, ARCH_REGISTRY)
    x, y = load_probes(corpus, n_probes)
    bat = build_battery(corpus, model.model_id) if battery else ModelBattery()
    if reference and Path(reference).exists():
        bat.manifest = manifest_from(reference)
    ctx = RunContext(probes_x=x, probes_y=y, battery=bat, out_dir=out_dir, seed=7)
    res = scan(model, ctx, profile)
    emit_reference(model, out_dir / f"{model.model_id}.reference.json")
    (out_dir / f"{model.model_id}.findings.json").write_text(
        json.dumps([f.to_dict() for f in res.findings], indent=2))
    write_report_json(res, out_dir / f"{model.model_id}.report.json")
    write_coverage(res, out_dir / f"{model.model_id}.coverage.md")
    render([res], out_dir / f"{model.model_id}.report.html",
           f"CV Assurance — Module B — {model.model_id}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="cva")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("model"); s.add_argument("--corpus", default="artifacts/corpus")
    s.add_argument("--out", default="artifacts/reports"); s.add_argument("--profile", default="deep")
    s.add_argument("--reference", default=None)
    s.add_argument("--no-battery", action="store_true")
    b = sub.add_parser("bench")
    b.add_argument("--corpus", default="artifacts/corpus")
    b.add_argument("--out", default="artifacts/bench"); b.add_argument("--profile", default="deep")
    a = ap.parse_args()
    if a.cmd == "scan":
        r = run_scan(Path(a.model), Path(a.corpus), Path(a.out), a.profile,
                     Path(a.reference) if a.reference else None, not a.no_battery)
        print(f"{r.model_id}: {r.verdict}  ->  {a.out}/{r.model_id}.report.html")
    else:
        from cva.bench.run import run_bench
        run_bench(Path(a.corpus), Path(a.out), a.profile)
