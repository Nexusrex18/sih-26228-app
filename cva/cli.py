"""CLI. `scan` for one model, `bench` for the whole corpus."""
from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

import cva.detectors.data.registry  # noqa: F401 — Module A's detectors; without this, no data.* check is planned
import cva.detectors.model.registry  # noqa: F401 — registration happens at the entrypoint, never in core
from attacklab.arch import ARCH_REGISTRY
from cva.core import determinism
from cva.core.egress import egress_guard, run_canaries
from cva.core.model import Manifest, ModelBattery
from cva.core.orchestrator import (
    RunContext,
    ScanResult,
    resolve_profile,
    scan,
    seal_report,
    validate_profile_ids,
)
from cva.core.scanid import scan_out_dir
from cva.loaders.models import load_model
from cva.loaders.models.http_model import HTTPModel, require_loopback
from cva.loaders.models.subprocess_model import SubprocessModel
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
    report_path = write_report_json(res, out_dir / f"{model.model_id}.report.json")
    seal_report(res, ctx, report_path)     # AFTER the report is on disk, BEFORE the HTML shows the seq
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
    # `--model-cmd` is ONE string and shlex.join quotes it back into one argument, so the
    # command round-trips through a shell as the list it was parsed into.
    for flag, attr in (("--model-cmd", "model_cmd"), ("--model-url", "model_url"),
                       ("--input-shape", "input_shape"), ("--num-classes", "num_classes"),
                       ("--model-id", "model_id")):
        if _opt(a, attr) is not None:
            argv += [flag, str(_opt(a, attr))]
    if a.dataset:
        argv += ["--dataset", a.dataset]
    argv += ["--profile", a.profile]
    if a.budget_tier:
        argv += ["--budget-tier", a.budget_tier]
    if _opt(a, "calibration"):
        argv += ["--calibration", str(a.calibration)]
    argv += ["--seed", str(a.seed)]
    return shlex.join(argv)


def _opt(a, name: str):
    """`selftest` builds its own Namespace without the model-adapter flags."""
    return getattr(a, name, None)


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(p) for p in str(text).split(","))
    except ValueError:
        dims = ()
    if not dims or any(d < 1 for d in dims):
        raise ValueError(f"--input-shape must be positive comma-separated integers, "
                         f"e.g. 3,64,64; got {text!r}")
    return dims


def model_arg_error(a) -> str | None:
    """Why the model arguments are unusable, or None. One model source, and the two
    query-only adapters cannot infer what a model path's loader reads from the file."""
    cmd, url, path = _opt(a, "model_cmd"), _opt(a, "model_url"), _opt(a, "model")
    shape, n_cls, mid = _opt(a, "input_shape"), _opt(a, "num_classes"), _opt(a, "model_id")
    if sum(bool(x) for x in (path, cmd, url)) > 1:
        return "give exactly ONE model source: a model path, --model-cmd, or --model-url"
    if not (cmd or url):
        stray = [f for f, v in (("--input-shape", shape), ("--num-classes", n_cls),
                                ("--model-id", mid)) if v is not None]
        return (f"{', '.join(stray)} only apply with --model-cmd or --model-url"
                if stray else None)
    which = "--model-cmd" if cmd else "--model-url"
    if shape is None:
        return f"--input-shape (e.g. 3,64,64) is required with {which}"
    if n_cls is None:
        return f"--num-classes N is required with {which}"
    try:
        parse_shape(shape)
        if n_cls < 1:
            return "--num-classes must be at least 1"
        if cmd and not shlex.split(cmd):
            return "--model-cmd is empty"
        if url:
            require_loopback(url)          # localhost only: the audited data must not leave
    except ValueError as exc:               # (also shlex's "No closing quotation")
        return str(exc)
    return None


def build_model(a):
    """The model handle for `a`: a file loader, a Tier-2 adapter, or None (dataset-only).
    `model_arg_error` has already vetted the arguments."""
    cmd, url = _opt(a, "model_cmd"), _opt(a, "model_url")
    if cmd:
        argv = shlex.split(cmd)              # a list, never a shell
        return SubprocessModel(argv, _opt(a, "model_id") or f"subprocess-{Path(argv[-1]).stem}",
                               parse_shape(a.input_shape), a.num_classes)
    if url:
        return HTTPModel(url, _opt(a, "model_id") or f"http-{urlsplit(url).netloc}",
                         parse_shape(a.input_shape), a.num_classes)
    return load_model(Path(a.model), ARCH_REGISTRY) if a.model else None


def execute_scan(a, command: str | None = None) -> tuple[ScanResult, Path | None]:
    """Everything `cva scan` does, minus the printing. Shared with `cva selftest`.

    A profile that sets `egress_guard` arms the process-level guard around model loading
    AND the scan, because the backbone load is where an egress is most likely to hide.
    A profile that pins the clock also arms the reproducibility harness, and does so
    BEFORE any model is loaded so the ONNX session options are pinned too.
    """
    err = model_arg_error(a)
    if err:
        raise ValueError(err)
    prof = resolve_profile(a.profile, a.budget_tier)     # unknown profile: a load error, now
    # Here and not in scan(): this is where every registry is imported. The zero-detector gate
    # calls scan() with empty registries and a real profile, and that must keep working.
    validate_profile_ids(prof)
    if prof.get("pin_clock"):
        determinism.arm(a.seed)
    guard = egress_guard() if prof.get("egress_guard") else contextlib.nullcontext()
    calibration = None
    if _opt(a, "calibration"):
        from cva.risk.calibration import load_calibration  # lazy: only a scan that asks
        calibration = load_calibration(Path(a.calibration))
    with guard:
        model = build_model(a)
        ds = load_dataset(Path(a.dataset) if a.dataset else None)
        ctx = RunContext(out_dir=Path(a.out), seed=a.seed, dataset=ds,
                         code_commit=code_commit(), calibration=calibration)
        res = scan(model, ctx, a.profile, dry_run=a.dry_run, plan_sink=print,
                   budget_tier=a.budget_tier)
    if a.dry_run:
        return res, None
    out = scan_out_dir(Path(a.out), res.scan_id)
    command = command or repro_command(a)
    report_path = write_report_json(res, out / "report.json", command)
    # Seal AFTER report.json is final and BEFORE the HTML, which may show the seq. The seq is
    # never written into report.json: the ledger holds that file's digest (plan §7.9).
    seal_report(res, ctx, report_path)
    write_coverage(res, out / "coverage.md")
    # The report sits in <out>/<scan_id>/ and the shared evidence store in <out>/evidence/.
    render([res], out / "report.html", f"CV Assurance — {res.model_id}",
           evidence_root=Path(a.out) / "evidence", command=command)
    return res, out


def cmd_scan(a) -> int:
    """V7: the plan prints BEFORE any work, and --dry-run stops right after it."""
    res, out = execute_scan(a)
    if out is None:
        print(f"\nDRY RUN — nothing executed. scan_id would be {res.scan_id}")
        return 0
    print(f"{res.model_id}: {res.verdict}  ->  {out}/report.json")
    return 0


def cmd_selftest(a) -> int:
    """The product proving its own claims, offline: arm the process-level guard, fire one
    canary of each kind (V13), then scan a generated corpus at `standard` tier so the
    backbone load — where an egress would hide — really executes (§5.6).

    Exit 0 only if every canary fired, the scan ran, its report validates against the
    published schema, and the embedding index was actually built. `unshare -rn cva
    selftest` (V11) adds the OS-level proof that no route existed at all.
    """
    determinism.ensure_hashseed(a.seed)          # may re-exec: the hash seed is read at start-up
    out_root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="cva-selftest-"))
    failures: list[str] = []

    with egress_guard():
        print("Egress guard armed (process level, loopback allowed).")
        for c in run_canaries():
            print(f"  canary {'PASS' if c.ok else 'FAIL'}  {c.name}: {c.detail}")
            if not c.ok:
                failures.append(f"canary '{c.name}'")
        if failures:
            print("\nselftest FAILED: the guard is not doing what the report would claim.")
            return 1

        from cva.fixtures import build as build_fixtures
        fx = build_fixtures(out_root / "fixtures", a.seed)
        args = argparse.Namespace(
            model=str(fx.model), dataset=str(fx.dataset), out=str(out_root / "reports"),
            profile="selftest", budget_tier=a.budget_tier, seed=a.seed, dry_run=False)
        # The reproducing command is `cva selftest`, not a scan of temp-dir fixtures: those
        # paths differ per run and would make V10's diff fail on a correct run.
        res, out = execute_scan(args, shlex.join(
            ["python", "-m", "cva.cli", "selftest", "--seed", str(a.seed)]
            + (["--budget-tier", a.budget_tier] if a.budget_tier else [])))

    assert out is not None
    print(f"\n{res.model_id}: {res.verdict}  ->  {out}/report.json")
    from jsonschema import Draft202012Validator
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas"
                         / "report.schema.json").read_text())
    errs = list(Draft202012Validator(schema).iter_errors(
        json.loads((out / "report.json").read_text())))
    if errs:
        failures.append(f"report.json fails its own schema ({len(errs)} errors, "
                        f"first: {errs[0].message[:120]})")
    if any("No embedding index was available" in lim
           for f in res.findings for lim in f.limitations):
        failures.append("the embedding backbone did not load, so the load most likely to "
                        "egress was NOT exercised")
    if not any(r.resolution.runnable and r.check_id.startswith("data.") for r in res.plan):
        failures.append("no data.* check was runnable, so no embedding was ever requested")
    errored = [f.detector_id for f in res.findings if f.availability.value == "ERROR"]
    if errored:
        failures.append(f"checks raised: {', '.join(sorted(set(errored)))}")

    if failures:
        print("\nselftest FAILED:\n" + "\n".join(f"  - {m}" for m in failures))
        return 1
    print("selftest PASSED — canaries fired, scan ran offline at "
          f"'{res.profile.get('budget_tier')}' tier, report is schema-valid.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cva")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    # Positional AND flag share a destination. A `nargs="?"` positional with a real default
    # overwrites the flag's value with that default when it matches nothing, so
    # `cva scan --model m.onnx` silently scanned NO model (V7's exact call, and `make demo`'s).
    # SUPPRESS makes the positional leave the attribute alone; the flag's default supplies None.
    s.add_argument("model", nargs="?", default=argparse.SUPPRESS)
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
    # Tier-2 model adapters (query-only). Mutually exclusive with a model path.
    s.add_argument("--model-cmd", default=None,
                   help="command that reads a float32 .npy on stdin and writes a .npy of "
                        "logits on stdout; parsed with shlex into a list, never run in a shell")
    s.add_argument("--model-url", default=None,
                   help="loopback-only HTTP endpoint: POST {inputs} -> {outputs}")
    s.add_argument("--input-shape", default=None, help="per-sample shape, e.g. 3,64,64 "
                   "(required with --model-cmd/--model-url)")
    s.add_argument("--num-classes", type=int, default=None,
                   help="required with --model-cmd/--model-url")
    s.add_argument("--model-id", default=None, help="label for a --model-cmd/--model-url model")
    s.add_argument("--calibration", default=None,
                   help="a calibration set saved by the benchmark; without it the report "
                        "says calibration: null")

    t = sub.add_parser("selftest")
    t.add_argument("--out", default=None, help="default: a fresh temp dir")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--budget-tier", default=None)

    b = sub.add_parser("bench")
    b.add_argument("--corpus", default="artifacts/corpus")
    b.add_argument("--out", default="artifacts/bench")
    b.add_argument("--profile", default="deep")

    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return cmd_selftest(a)
    if a.cmd == "bench":
        from cva.bench.run import run_bench
        run_bench(Path(a.corpus), Path(a.out), a.profile)
        return 0

    err = model_arg_error(a)
    if err:
        ap.error(err)
    if a.corpus and not a.model and not a.dry_run:
        ap.error("--corpus scans need a model path (a --model-cmd/--model-url model has no "
                 "reference battery to build)")

    # The Module B path: a corpus supplies probes and a reference battery.
    if a.corpus and not a.dry_run:
        r = run_scan(Path(a.model), Path(a.corpus), Path(a.out), a.profile,
                     Path(a.reference) if a.reference else None, not a.no_battery)
        print(f"{r.model_id}: {r.verdict}  ->  {a.out}/{r.model_id}.report.html")
        return 0
    return cmd_scan(a)


if __name__ == "__main__":
    raise SystemExit(main())
