"""CLI. `scan` for one model, `bench` for the whole corpus."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from cva.core import determinism
from cva.core.capability import Capability
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
from cva.core.scanid import EvidenceStore, scan_out_dir
from cva.loaders.preprocess import attach_preprocess, preprocess_store_name
from cva.report.coverage import write as write_coverage
from cva.report.render_html import render
from cva.report.report_json import verdict_line
from cva.report.report_json import write as write_report_json


def _register_analysis():
    import cva.detectors.data.registry  # noqa: F401
    import cva.detectors.model.registry  # noqa: F401


def _arch_registry():
    from attacklab.arch import ARCH_REGISTRY
    return ARCH_REGISTRY


def load_model(*args, **kwargs):
    from cva.loaders.models import load_model as loader
    return loader(*args, **kwargs)


def require_loopback(url):
    from cva.loaders.models.http_model import require_loopback as validate
    return validate(url)


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
    models = [load_model(corpus / p["pt"], _arch_registry(), p["id"]) for p in picks]
    return ModelBattery(models=models)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def reference_manifest(model, *, with_fingerprint: bool) -> dict[str, Any]:
    """What to register about `model` at acceptance. A key is ABSENT when its fact is unknown,
    never an empty string and never a placeholder: `weights_sha256` only when the digest is
    really 64 lowercase hex (a frozen TorchScript archive or a query-only endpoint returns
    "unavailable:..." and registering that would make every later scan "match" it), and
    `fingerprint` only when the caller asked for it and the model answers."""
    ref: dict[str, Any] = {"model_id": getattr(model, "model_id", "-")}
    try:
        digest = model.weight_digest()
    except Exception:
        digest = None
    if isinstance(digest, str) and _HEX64.match(digest):
        ref["weights_sha256"] = digest
    try:
        arch = getattr(model, "arch_hash", None)
        arch = arch() if callable(arch) else arch
    except Exception:
        arch = None
    if isinstance(arch, str) and _HEX64.match(arch):
        ref["arch_hash"] = arch
    pre_hash = getattr(model, "preprocess_hash", None)
    pre_ref = getattr(model, "preprocess_ref", None)
    if isinstance(pre_hash, str) and _HEX64.match(pre_hash):
        ref["preprocess_hash"] = pre_hash
    if isinstance(pre_ref, str) and pre_ref:
        ref["preprocess_ref"] = pre_ref
    if with_fingerprint:
        from cva.detectors.model.fingerprint import fingerprint
        try:
            ref["fingerprint"] = [round(float(v), 8) for v in fingerprint(model)]
        except Exception:
            pass                             # an unanswerable model has no fingerprint to register
    return ref


def store_preprocess(model, out_root: Path) -> None:
    """Put the declared spec's canonical bytes in `<out_root>/evidence/`, so that the
    `preprocess_ref` in the report resolves to bytes that exist. The store hashes the same bytes
    the loader hashed, so the returned name must be the handle's `preprocess_ref`; if it is not,
    something altered the bytes in between and the record would point at the wrong spec."""
    blob = getattr(model, "preprocess_spec_bytes", None)
    if blob is None:
        return
    name = EvidenceStore(out_root).put_bytes(blob, ".json")
    expected = preprocess_store_name(getattr(model, "preprocess_hash", ""))
    if name != expected:
        raise RuntimeError(
            f"the stored preprocessing spec is named {name}, but the model handle's "
            f"preprocess_hash implies {expected}")


def write_reference(model, path: Path, evidence_root: Path, *, with_fingerprint: bool) -> Path:
    """Write the reference manifest, storing the preprocessing spec it points at."""
    store_preprocess(model, evidence_root)
    path.write_text(json.dumps(reference_manifest(model, with_fingerprint=with_fingerprint),
                               indent=2, sort_keys=True))
    return path


def emit_reference(model, out: Path) -> Path:
    """Close the no-reference gap FORWARD: the fingerprint and digest are computed anyway,
    so write them out. The first scan of an unknown model says little; every subsequent
    scan of it says a great deal."""
    return write_reference(model, out, out.parent, with_fingerprint=True)


def manifest_from(path: Path) -> Manifest:
    d = json.loads(Path(path).read_text())
    return Manifest(model_id=d["model_id"], weights_sha256=d.get("weights_sha256"),
                    fingerprint=d.get("fingerprint"))


def run_scan(model_path: Path, corpus: Path, out_dir: Path, profile: str = "deep",
             reference: Path | None = None, battery: bool = True, n_probes: int = 400,
             preprocess: Path | None = None):
    _register_analysis()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path, _arch_registry(), preprocess=preprocess)
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
    write_seal_sidecar(res, out_dir / f"{model.model_id}.seal.json", report_path)
    warn_unsealed(res)
    write_coverage(res, out_dir / f"{model.model_id}.coverage.md")
    render([res], out_dir / f"{model.model_id}.report.html",
           f"CV Assurance — Module B — {model.model_id}")
    return res


def _ledgers(a) -> dict[str, Any]:
    """Construct the two ledger slots from `--inference-ledger` / `--audit-ledger`.

    Item 7, Backend half. The capability slot has existed since `core/capability.py:37` and
    `core/ledger.py`, but nothing on the CLI ever filled it, so every scan ran with
    `NullAuditLedger`, reported "not sealed", and left all 24 of Module C's attack classes
    uncovered with no way for an operator to change that.

    Both stubs probe ACTIVELY — `JsonlInferenceLedger` reports `INFERENCE_LEDGER` only if
    the file opened and its first record parsed — so a path that exists but holds garbage
    does not present itself as a ledger. Nothing here asserts a capability on the strength
    of a flag having been passed.

    Module C's half is independent and not done here: `PROV_CHECKS` is not registered with
    `core.registry`, and `LedgerVerify` exposes `resolve(caps)` but no `check(model, ctx)`,
    so the orchestrator has nothing to call. `_prov_gap` says so in the report rather than
    letting the capability appear while the rows do not.
    """
    from cva.core.ledger import JsonlAuditLedger, JsonlInferenceLedger

    out: dict[str, Any] = {}
    if _opt(a, "inference_ledger"):
        out["inference_ledger"] = JsonlInferenceLedger(Path(a.inference_ledger))
    if _opt(a, "audit_ledger"):
        out["audit_ledger"] = JsonlAuditLedger(Path(a.audit_ledger))
    return out


def write_seal_sidecar(res, path: Path, report_path: Path) -> Path:
    """Write `seal.json` — the sealing OUTCOME, beside the report and never inside it.

    Audit item 19 asked for the reason a scan was not sealed to reach the report's provenance
    section, and it cannot: `report.json` is written, fsynced and hashed BEFORE `seal_report`
    runs, and plan §7.9 forbids writing back into it — *"putting it into report.json would
    change the file after it was hashed, and the ledger would then hold a digest of a file
    that no longer exists, reporting 'differs from sealed digest' on every clean scan."*

    So the outcome goes in a sidecar, the same way the reference manifest does: next to the
    report, never counted by it, never hashed by the seal. That keeps §7.9 intact and still
    gives Module E something MACHINE-READABLE — which the HTML and the stdout warning are
    not. Without it a consumer of `report.json` cannot tell "no ledger was configured" from
    "the ledger raised", and those are very different facts about a scan.

    `report.json` does not reference this file, deliberately: a hashed artefact that points
    at an unhashed one invites the reader to treat the second as sealed too.
    """
    payload = {
        "scan_id": res.scan_id,
        "report_file": report_path.name,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "sealed": res.ledger_seq is not None and res.ledger_error is None,
        "ledger_seq": res.ledger_seq,
        "ledger_error": res.ledger_error,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def warn_unsealed(res) -> None:
    """Say out loud that the scan record could not be sealed, and why.

    `append_scan_record` used to swallow the exception entirely, so a report whose
    tamper-evidence was missing was indistinguishable at the terminal from one that was
    sealed. The report still gets written — the scan's findings are not in doubt — but the
    operator has to be told which of the two they are holding.
    """
    if getattr(res, "ledger_error", None):
        print(f"WARNING: the scan record was NOT sealed — the audit ledger raised: "
              f"{res.ledger_error}")


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
                       ("--model-id", "model_id"), ("--preprocess", "preprocess")):
        if _opt(a, attr) is not None:
            argv += [flag, str(_opt(a, attr))]
    if a.dataset:
        argv += ["--dataset", a.dataset]
    if _opt(a, "reference_dataset"):
        argv += ["--reference-dataset", str(a.reference_dataset)]
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
    if _opt(a, "preprocess") is not None and not (path or cmd or url):
        return ("--preprocess declares how a MODEL is fed: give a model path, --model-cmd "
                "or --model-url with it")
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
    from cva.loaders.models.http_model import HTTPModel
    from cva.loaders.models.subprocess_model import SubprocessModel
    cmd, url = _opt(a, "model_cmd"), _opt(a, "model_url")
    pre = _opt(a, "preprocess")
    if cmd:
        argv = shlex.split(cmd)              # a list, never a shell
        sub = SubprocessModel(argv, _opt(a, "model_id") or f"subprocess-{Path(argv[-1]).stem}",
                              parse_shape(a.input_shape), a.num_classes)
        attach_preprocess(sub, None, pre)    # a query-only model has no sidecar; `--preprocess` only
        return sub
    if url:
        http = HTTPModel(url, _opt(a, "model_id") or f"http-{urlsplit(url).netloc}",
                         parse_shape(a.input_shape), a.num_classes)
        attach_preprocess(http, None, pre)
        return http
    return load_model(Path(a.model), _arch_registry(), preprocess=pre) if a.model else None


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
    _register_analysis()
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
        ref_ds = load_dataset(Path(_opt(a, "reference_dataset"))
                              if _opt(a, "reference_dataset") else None)
        ctx = RunContext(out_dir=Path(a.out), seed=a.seed, dataset=ds,
                         code_commit=code_commit(), calibration=calibration,
                         reference_dataset=ref_ds,
                         **_ledgers(a))
        res = scan(model, ctx, a.profile, dry_run=a.dry_run, plan_sink=print,
                   budget_tier=a.budget_tier)
    if a.dry_run:
        return res, None
    out = scan_out_dir(Path(a.out), res.scan_id)
    if model is not None:
        # The reference manifest, next to the report and NOT inside it: report.json never
        # counts it and the seal never hashes it. The fingerprint runs its own deterministic
        # battery, so it needs a model that ANSWERS, not a probe set; gating it on probes meant
        # the real scan path never registered the one thing that closes the no-reference gap.
        write_reference(model, out / "reference.json", Path(a.out),
                        with_fingerprint=Capability.MODEL_PREDICT in model.capabilities())
    command = command or repro_command(a)
    report_path = write_report_json(res, out / "report.json", command)
    # Seal AFTER report.json is final and BEFORE the HTML, which may show the seq. The seq is
    # never written into report.json: the ledger holds that file's digest (plan §7.9).
    seal_report(res, ctx, report_path)
    write_seal_sidecar(res, out / "seal.json", report_path)
    warn_unsealed(res)
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
    if res.contributor_baseline_unavailable:
        print(f"note: the reference dataset was NOT used: {res.contributor_baseline_unavailable}")
    print(f"{res.model_id}: {verdict_line(res)}  ->  {out}/report.json")
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
    print(f"\n{res.model_id}: {verdict_line(res)}  ->  {out}/report.json")
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
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == 'drift':
        from cva.detectors.drift.command import main as drift_main
        drift_main(args[1:])
        return 0
    _register_analysis()
    ap = argparse.ArgumentParser(prog="cva")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("drift", help="compare declared reference and incoming imagery", add_help=False)

    s = sub.add_parser("scan")
    # Positional AND flag share a destination. A `nargs="?"` positional with a real default
    # overwrites the flag's value with that default when it matches nothing, so
    # `cva scan --model m.onnx` silently scanned NO model (V7's exact call, and `make demo`'s).
    # SUPPRESS makes the positional leave the attribute alone; the flag's default supplies None.
    s.add_argument("model", nargs="?", default=argparse.SUPPRESS)
    s.add_argument("--model", dest="model", default=None)
    s.add_argument("--dataset", default=None)
    s.add_argument("--reference-dataset", default=None,
                   help="a second dataset, in any supported layout, that YOU know to be clean: "
                        "the same data checks run over it and its flag rate is the absolute "
                        "baseline that a contaminated cohort is compared against")
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
    s.add_argument("--preprocess", default=None,
                   help="the DECLARED preprocessing spec of the model (JSON: mean, std, layout, "
                        "dtype, optional value_range/input_shape). Default: the sidecar "
                        "<model>.preprocess.json if it exists. Without a spec the report says "
                        "prov.recompute cannot be performed for this model")
    s.add_argument("--calibration", default=None,
                   help="a calibration set saved by the benchmark; without it the report "
                        "says calibration: null")
    # --- Module C seam (item 7, Backend half) -------------------------------------------
    s.add_argument("--inference-ledger", default=None,
                   help="the FIELD ledger to audit (JSONL): the artefact prov.* checks "
                        "verify. Supplying it grants INFERENCE_LEDGER. Without it every "
                        "prov.* row resolves UNAVAILABLE with that as the stated reason")
    s.add_argument("--audit-ledger", default=None,
                   help="where this scan's own scan_record is appended. The default is a "
                        "Null ledger and the report says plainly that the record was not "
                        "sealed; the built-in JSONL ledger is a sha256 chain with NO key, "
                        "so it still reports no SIGNING_KEY and the report still says "
                        "not sealed. Module C's signed store replaces it")

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
                     Path(a.reference) if a.reference else None, not a.no_battery,
                     preprocess=Path(a.preprocess) if a.preprocess else None)
        print(f"{r.model_id}: {verdict_line(r)}  ->  {a.out}/{r.model_id}.report.html")
        return 0
    return cmd_scan(a)


if __name__ == "__main__":
    raise SystemExit(main())
