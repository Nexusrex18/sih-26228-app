"""CLI. `scan` for one model, `bench` for the whole corpus."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

import cva.detectors.model.registry  # noqa: F401 — registration happens at the entrypoint, never in core
from attacklab.arch import ARCH_REGISTRY
from cva.core.model import Manifest, ModelBattery
from cva.core.orchestrator import RunContext, scan
from cva.core.scanid import scan_out_dir
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


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_dataset(path: Path | None):
    if path is None:
        return None
    from cva.loaders.detect import load_dataset as _load
    return _load(Path(path))


def repro_command(a) -> str:
    """The command that reproduces this scan.

    Deliberately WITHOUT `--out`: V10 runs the same scan into two roots and diffs the
    reports, and an out path inside `reproduction.command` would make that diff fail on a
    correct run. Where the output went is not an input to the scan.
    """
    argv = ["python", "-m", "cva.cli", "scan"]
    if a.model:
        argv.append(a.model)
    if a.dataset:
        argv += ["--dataset", a.dataset]
    argv += ["--profile", a.profile]
    if a.budget_tier:
        argv += ["--budget-tier", a.budget_tier]
    argv += ["--seed", str(a.seed)]
    return shlex.join(argv)


def cmd_scan(a) -> int:
    """V7: the plan prints BEFORE any work, and --dry-run stops right after it."""
    model = load_model(Path(a.model), ARCH_REGISTRY) if a.model else None
    ds = load_dataset(Path(a.dataset) if a.dataset else None)
    ctx = RunContext(out_dir=Path(a.out), seed=a.seed, dataset=ds,
                     code_commit=code_commit())
    res = scan(model, ctx, a.profile, dry_run=a.dry_run, plan_sink=print,
               budget_tier=a.budget_tier)
    if a.dry_run:
        print(f"\nDRY RUN — nothing executed. scan_id would be {res.scan_id}")
        return 0
    out = scan_out_dir(Path(a.out), res.scan_id)
    command = repro_command(a)
    write_report_json(res, out / "report.json", command)
    write_coverage(res, out / "coverage.md")
    # The report sits in <out>/<scan_id>/ and the shared evidence store in <out>/evidence/.
    render([res], out / "report.html", f"CV Assurance — {res.model_id}",
           evidence_root=Path(a.out) / "evidence", command=command)
    print(f"{res.model_id}: {res.verdict}  ->  {out}/report.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cva")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("model", nargs="?", default=None)
    s.add_argument("--model", dest="model", default=None)
    s.add_argument("--dataset", default=None)
    s.add_argument("--corpus", default=None)
    s.add_argument("--out", default="artifacts/reports")
    s.add_argument("--profile", default="baseline")
    s.add_argument("--budget-tier", default=None)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--reference", default=None)
    s.add_argument("--no-battery", action="store_true")

    b = sub.add_parser("bench")
    b.add_argument("--corpus", default="artifacts/corpus")
    b.add_argument("--out", default="artifacts/bench")
    b.add_argument("--profile", default="deep")

    a = ap.parse_args(argv)
    if a.cmd == "bench":
        from cva.bench.run import run_bench
        run_bench(Path(a.corpus), Path(a.out), a.profile)
        return 0

    # The Module B path: a corpus supplies probes and a reference battery.
    if a.corpus and not a.dry_run:
        r = run_scan(Path(a.model), Path(a.corpus), Path(a.out), a.profile,
                     Path(a.reference) if a.reference else None, not a.no_battery)
        print(f"{r.model_id}: {r.verdict}  ->  {a.out}/{r.model_id}.report.html")
        return 0
    return cmd_scan(a)


if __name__ == "__main__":
    raise SystemExit(main())
