"""`cva` command line — argparse regressions, the reproducing command, and `selftest` end to end.

The argparse tests never load a model: `cmd_scan` (or `build_model`) is replaced, so what is
asserted is what `main()` PARSED and passed on. `selftest` cannot run in-process at all — its
`ensure_hashseed` re-executes the interpreter (`os.execv`) when PYTHONHASHSEED is unset, which
would replace the pytest process — so it runs as a subprocess and is marked slow.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cva import cli
from cva.core.capability import Availability, Capability, CapabilitySet
from cva.core.orchestrator import UnknownProfile
from cva.features import vendor
from cva.risk.calibration import CalibrationSet, save_calibration

REPO = Path(__file__).resolve().parents[2]


class FakeModel:
    model_id = "fake-001"
    fmt = "onnx"
    opset = 17

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                             ((Capability.MODEL_WEIGHTS, "probed: no weights readable"),))


@pytest.fixture
def parsed(monkeypatch):
    """Run `main()` up to, but not into, the scan; hand back every Namespace it built."""
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_scan", lambda a: (seen.append(a), 0)[1])
    return seen


def _error_text(capsys) -> str:
    return capsys.readouterr().err


# --- the shared positional/flag destination ------------------------------------------------

def test_the_model_flag_is_recorded_not_overwritten_by_the_absent_positional(parsed):
    """`cva scan --model X --dry-run` used to scan NO model: the `nargs='?'` positional shares
    the destination and overwrote the flag with its default when it matched nothing."""
    assert cli.main(["scan", "--model", "X.onnx", "--dry-run"]) == 0
    (a,) = parsed
    assert a.model == "X.onnx"
    assert a.dry_run is True


def test_the_positional_model_still_works(parsed):
    cli.main(["scan", "X.onnx", "--dry-run"])
    assert parsed[0].model == "X.onnx"
    cli.main(["scan", "--dry-run", "Y.onnx"])
    assert parsed[1].model == "Y.onnx"


def test_no_model_at_all_is_a_none_not_a_missing_attribute(parsed):
    cli.main(["scan", "--dry-run"])
    (a,) = parsed
    assert a.model is None


def test_dry_run_scans_the_model_the_flag_named(monkeypatch, tmp_path, capsys):
    """Behaviour, not just the namespace: the model that reaches `build_model` is X."""
    asked: list[str | None] = []
    monkeypatch.setattr(cli, "build_model", lambda a: (asked.append(a.model), FakeModel())[1])
    results = []
    real = cli.execute_scan
    monkeypatch.setattr(cli, "execute_scan",
                        lambda a, command=None: results.append(real(a, command)) or results[-1])

    assert cli.main(["scan", "--model", "X.onnx", "--dry-run", "--out", str(tmp_path)]) == 0

    assert asked == ["X.onnx"]
    (res, out) = results[0]
    assert out is None, "a dry run writes nothing"
    assert res.model_id == "fake-001" and res.verdict == "DRY-RUN"
    printed = capsys.readouterr().out
    assert "Plan" in printed, "V7: the plan prints before anything else"
    assert "DRY RUN" in printed
    assert not list(tmp_path.iterdir()), "a dry run must not create output"


def test_dry_run_with_no_model_makes_every_model_check_unavailable(monkeypatch, tmp_path, capsys):
    results = []
    real = cli.execute_scan
    monkeypatch.setattr(cli, "execute_scan",
                        lambda a, command=None: results.append(real(a, command)) or results[-1])
    assert cli.main(["scan", "--dry-run", "--out", str(tmp_path)]) == 0
    res, _ = results[0]
    assert res.model_id == "-"
    model_rows = [r for r in res.plan if r.check_id.startswith("model.")]
    assert model_rows, "premise: the real model registry is imported by the entrypoint"
    for row in model_rows:
        assert row.resolution.state is Availability.UNAVAILABLE, row.check_id
        assert row.resolution.exclusion_reason == "capability", row.check_id


# --- the query-only adapters ----------------------------------------------------------------

ADAPTER_OK = ["--input-shape", "3,64,64", "--num-classes", "2"]


def test_model_cmd_without_input_shape_or_num_classes_is_an_argparse_error(parsed, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scan", "--model-cmd", "python m.py", "--num-classes", "2", "--dry-run"])
    assert exc.value.code == 2
    assert "--input-shape" in _error_text(capsys)

    with pytest.raises(SystemExit):
        cli.main(["scan", "--model-cmd", "python m.py", "--input-shape", "3,64,64",
                  "--dry-run"])
    assert "--num-classes" in _error_text(capsys)
    assert parsed == [], "an invalid invocation must not reach the scan"


def test_model_url_without_input_shape_or_num_classes_is_an_argparse_error(parsed, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scan", "--model-url", "http://127.0.0.1:1", "--dry-run"])
    assert exc.value.code == 2
    err = _error_text(capsys)
    assert "--input-shape" in err
    assert parsed == []


@pytest.mark.parametrize("flags", [
    ["--model-cmd", "python m.py"],
    ["--model-url", "http://127.0.0.1:1"],
], ids=["cmd", "url"])
def test_an_adapter_conflicts_with_a_model_path(parsed, capsys, flags):
    for path_args in (["X.onnx"], ["--model", "X.onnx"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(["scan", *path_args, *flags, *ADAPTER_OK, "--dry-run"])
        assert exc.value.code == 2
        assert "one model source" in _error_text(capsys).lower()
    assert parsed == []


def test_the_two_adapters_conflict_with_each_other(parsed, capsys):
    with pytest.raises(SystemExit):
        cli.main(["scan", "--model-cmd", "python m.py", "--model-url", "http://127.0.0.1:1",
                  *ADAPTER_OK, "--dry-run"])
    assert "one model source" in _error_text(capsys).lower()


@pytest.mark.parametrize("bad, needle", [
    (["--input-shape", "3,x,64", "--num-classes", "2"], "--input-shape"),
    (["--input-shape", "3,0,64", "--num-classes", "2"], "--input-shape"),
    (["--input-shape", "3,64,64", "--num-classes", "0"], "--num-classes"),
], ids=["non-integer-shape", "zero-dim", "zero-classes"])
def test_a_malformed_adapter_argument_is_an_argparse_error(parsed, capsys, bad, needle):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scan", "--model-cmd", "python m.py", *bad, "--dry-run"])
    assert exc.value.code == 2
    assert needle in _error_text(capsys)


def test_adapter_only_flags_without_an_adapter_are_an_error_not_ignored(parsed, capsys):
    with pytest.raises(SystemExit):
        cli.main(["scan", "X.onnx", "--input-shape", "3,64,64", "--dry-run"])
    assert "--input-shape" in _error_text(capsys)


def test_a_non_loopback_model_url_is_refused_before_any_connection(parsed, capsys):
    with pytest.raises(SystemExit):
        cli.main(["scan", "--model-url", "http://example.com/predict", *ADAPTER_OK,
                  "--dry-run"])
    assert "loopback" in _error_text(capsys).lower() or "localhost" in _error_text(capsys).lower()
    assert parsed == []


def test_valid_adapter_arguments_reach_the_scan(parsed):
    cli.main(["scan", "--model-cmd", "python /tmp/m.py --fast", *ADAPTER_OK,
              "--model-id", "mine", "--dry-run"])
    cli.main(["scan", "--model-url", "http://127.0.0.1:1", *ADAPTER_OK, "--dry-run"])
    cmd_args, url_args = parsed
    assert cmd_args.model is None
    assert cmd_args.model_cmd == "python /tmp/m.py --fast"
    assert cmd_args.input_shape == "3,64,64" and cmd_args.num_classes == 2
    assert cmd_args.model_id == "mine"
    assert url_args.model_url == "http://127.0.0.1:1"


def test_a_corpus_scan_needs_a_model_path(parsed, capsys):
    with pytest.raises(SystemExit):
        cli.main(["scan", "--corpus", "some/corpus", "--model-cmd", "python m.py",
                  *ADAPTER_OK])
    assert "--corpus" in _error_text(capsys)


def test_build_model_makes_the_adapter_handles_without_running_anything():
    from cva.loaders.models.http_model import HTTPModel
    from cva.loaders.models.subprocess_model import SubprocessModel

    a = argparse.Namespace(model=None, model_cmd="python /tmp/m.py", model_url=None,
                           input_shape="3,8,8", num_classes=2, model_id=None)
    m = cli.build_model(a)
    assert isinstance(m, SubprocessModel) and m.model_id == "subprocess-m"

    b = argparse.Namespace(model=None, model_cmd=None, model_url="http://127.0.0.1:1/x",
                           input_shape="3,8,8", num_classes=2, model_id=None)
    h = cli.build_model(b)
    assert isinstance(h, HTTPModel) and h.model_id == "http-127.0.0.1:1"

    assert cli.build_model(argparse.Namespace(model=None)) is None


# --- --calibration ----------------------------------------------------------------------------

def test_the_calibration_flag_is_accepted_and_carried(parsed, tmp_path):
    cal = tmp_path / "cal.json"
    cli.main(["scan", "--dry-run", "--calibration", str(cal)])
    assert parsed[0].calibration == str(cal)
    cli.main(["scan", "--dry-run"])
    assert parsed[1].calibration is None


def test_a_calibration_file_is_loaded_by_the_scan(tmp_path, capsys):
    cal = save_calibration(CalibrationSet(), tmp_path / "cal.json")
    assert cli.main(["scan", "--dry-run", "--calibration", str(cal),
                     "--out", str(tmp_path / "out")]) == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_an_unreadable_calibration_file_is_a_load_error_not_a_silent_uncalibrated_scan(
        tmp_path):
    with pytest.raises(ValueError):
        cli.main(["scan", "--dry-run", "--calibration", str(tmp_path / "missing.json"),
                  "--out", str(tmp_path / "out")])


def test_an_unknown_profile_is_a_load_error_not_the_default(tmp_path):
    with pytest.raises(UnknownProfile):
        cli.main(["scan", "--dry-run", "--profile", "stirct", "--out", str(tmp_path)])


# --- the reproducing command ---------------------------------------------------------------------

def _repro(monkeypatch, *argv: str) -> tuple[argparse.Namespace, list[str]]:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_scan", lambda a: (seen.append(a), 0)[1])
    cli.main(["scan", *argv])
    return seen[0], shlex.split(cli.repro_command(seen[0]))


def test_repro_command_never_contains_out(monkeypatch):
    a, tokens = _repro(monkeypatch, "X.onnx", "--out", "/somewhere/else", "--dataset", "d",
                       "--dry-run")
    assert a.out == "/somewhere/else"
    assert "--out" not in tokens
    assert "/somewhere/else" not in tokens
    assert tokens[:4] == ["python", "-m", "cva.cli", "scan"]


def test_repro_command_includes_the_model_source_and_every_new_flag(monkeypatch, tmp_path):
    cal = str(tmp_path / "cal.json")
    _, tokens = _repro(monkeypatch, "--model-cmd", "python /tmp/m.py --fast",
                       "--input-shape", "3,64,64", "--num-classes", "2", "--model-id", "mine",
                       "--dataset", "data/coco", "--profile", "blackbox",
                       "--budget-tier", "deep", "--calibration", cal, "--seed", "11")
    pairs = dict(zip(tokens, tokens[1:], strict=False))
    # The command is ONE argument after round-tripping through a shell, not three.
    assert pairs["--model-cmd"] == "python /tmp/m.py --fast"
    assert pairs["--input-shape"] == "3,64,64"
    assert pairs["--num-classes"] == "2"
    assert pairs["--model-id"] == "mine"
    assert pairs["--dataset"] == "data/coco"
    assert pairs["--profile"] == "blackbox"
    assert pairs["--budget-tier"] == "deep"
    assert pairs["--calibration"] == cal
    assert pairs["--seed"] == "11"
    assert "--dry-run" not in tokens


def test_repro_command_for_a_url_model_and_a_positional_model(monkeypatch):
    _, url = _repro(monkeypatch, "--model-url", "http://127.0.0.1:8080/p",
                    "--input-shape", "3,8,8", "--num-classes", "4")
    assert url[url.index("--model-url") + 1] == "http://127.0.0.1:8080/p"
    _, path = _repro(monkeypatch, "X.onnx")
    assert path[4] == "X.onnx"
    assert "--model-cmd" not in path and "--calibration" not in path


def test_repro_command_omits_what_was_not_given(monkeypatch):
    _, tokens = _repro(monkeypatch)
    assert "--budget-tier" not in tokens and "--calibration" not in tokens
    assert "--dataset" not in tokens and "--model-cmd" not in tokens
    assert tokens[-4:] == ["--profile", "baseline", "--seed", "7"]


def test_repro_command_works_for_selftests_namespace_without_the_adapter_flags():
    a = argparse.Namespace(model="m.onnx", dataset="d", out="/tmp/x", profile="selftest",
                           budget_tier=None, seed=42, dry_run=False)
    tokens = shlex.split(cli.repro_command(a))
    assert "--out" not in tokens
    assert tokens[-4:] == ["--profile", "selftest", "--seed", "42"]


# --- --preprocess: the declared spec, and the reference manifest ---------------------------------------

HEX64 = "ab" * 32
SPEC = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "layout": "CHW", "dtype": "float32",
        "value_range": [0.0, 1.0], "input_shape": [3, 8, 8]}


def _spec_file(tmp_path: Path, spec: dict | None = None, name: str = "given.preprocess.json"):
    p = tmp_path / name
    p.write_text(json.dumps(SPEC if spec is None else spec))
    return p


class RefModel:
    """What a file loader hands over, minus the file: digests, a structure digest, predict."""
    fmt = "onnx"
    opset = 17
    model_id = "ref-001"
    input_shape = (3, 8, 8)
    num_classes = 2
    digest = HEX64
    arch = "cd" * 32

    def weight_digest(self) -> str:
        return self.digest

    def arch_hash(self) -> str:
        return self.arch

    def predict(self, x):
        return np.full((len(x), 2), 0.5, dtype=np.float32)

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}))


@pytest.fixture
def offline_scan(monkeypatch):
    """`execute_scan` for real, with EMPTY detector registries: what is under test is what the
    CLI writes around a scan, not what the detectors find."""
    real = cli.scan
    monkeypatch.setattr(cli, "scan", lambda model, ctx, profile, **kw: real(
        model, ctx, profile, registries=({}, {}), **kw))


def _scan_args(tmp_path: Path, **extra) -> argparse.Namespace:
    return argparse.Namespace(model="X.onnx", dataset=None, out=str(tmp_path / "out"),
                              profile="baseline", budget_tier=None, seed=7, dry_run=False,
                              **extra)


def _run(monkeypatch, model, args) -> Path:
    monkeypatch.setattr(cli, "build_model", lambda a: model)
    _res, out = cli.execute_scan(args)
    assert out is not None
    return out


def test_the_preprocess_flag_is_accepted_and_carried(parsed, tmp_path):
    p = str(tmp_path / "spec.json")
    cli.main(["scan", "X.onnx", "--preprocess", p, "--dry-run"])
    cli.main(["scan", "X.onnx", "--dry-run"])
    assert parsed[0].preprocess == p
    assert parsed[1].preprocess is None


def test_repro_command_carries_preprocess_but_never_out(monkeypatch, tmp_path):
    p = str(tmp_path / "spec.json")
    _, tokens = _repro(monkeypatch, "X.onnx", "--preprocess", p, "--out", "/somewhere/else")
    assert tokens[tokens.index("--preprocess") + 1] == p
    assert "--out" not in tokens
    _, plain = _repro(monkeypatch, "X.onnx")
    assert "--preprocess" not in plain


def test_preprocess_without_a_model_is_an_argparse_error(parsed, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scan", "--preprocess", "spec.json", "--dry-run"])
    assert exc.value.code == 2
    assert "--preprocess" in _error_text(capsys)
    assert parsed == []


def test_preprocess_is_accepted_with_a_query_only_adapter(parsed):
    cli.main(["scan", "--model-cmd", "python m.py", *ADAPTER_OK, "--preprocess", "s.json",
              "--dry-run"])
    assert parsed[0].preprocess == "s.json"


def test_build_model_attaches_an_explicit_spec_to_a_query_only_handle(tmp_path):
    from cva.loaders.preprocess import preprocess_digest

    a = argparse.Namespace(model=None, model_cmd="python /tmp/m.py", model_url=None,
                           input_shape="3,8,8", num_classes=2, model_id=None,
                           preprocess=str(_spec_file(tmp_path)))
    h = cli.build_model(a)
    digest, blob = preprocess_digest(SPEC)
    assert (h.preprocess_hash, h.preprocess_ref, h.preprocess_spec_bytes) \
        == (digest, f"{digest}.json", blob)
    a.preprocess = None
    assert not hasattr(cli.build_model(a), "preprocess_hash")


def test_execute_scan_writes_reference_json_with_every_known_fact(
        monkeypatch, tmp_path, offline_scan):
    from cva.loaders.preprocess import attach_preprocess, preprocess_digest

    model = RefModel()
    attach_preprocess(model, None, _spec_file(tmp_path))
    digest, blob = preprocess_digest(SPEC)
    out = _run(monkeypatch, model, _scan_args(tmp_path))

    ref = json.loads((out / "reference.json").read_text())
    # `cva scan` builds no probes, but the fingerprint runs its own battery, so a model that
    # ANSWERS gets one registered: it is what closes the no-reference gap on every later scan.
    fingerprint = ref.pop("fingerprint")
    assert fingerprint and all(isinstance(v, float) for v in fingerprint)
    assert ref == {"model_id": "ref-001", "weights_sha256": HEX64, "arch_hash": "cd" * 32,
                   "preprocess_hash": digest, "preprocess_ref": f"{digest}.json"}
    # The ref resolves: the spec bytes sit in the shared evidence store, under that name.
    stored = tmp_path / "out" / "evidence" / f"{digest}.json"
    assert stored.read_bytes() == blob
    # ...and the report carries the same three facts in `target`, but does not list the file.
    rep = json.loads((out / "report.json").read_text())
    assert rep["target"]["preprocess_hash"] == digest
    assert rep["target"]["preprocess_ref"] == f"{digest}.json"
    assert rep["target"]["arch_hash"] == "cd" * 32
    assert "reference.json" not in (out / "report.json").read_text()


@pytest.mark.parametrize("digest", ["unavailable:frozen", "unavailable:black-box", "", "AB" * 32,
                                    "ab" * 31, None])
def test_the_reference_never_carries_a_weights_digest_that_is_not_64_lowercase_hex(
        monkeypatch, tmp_path, offline_scan, digest):
    class Model(RefModel):
        pass

    Model.digest = digest                   # type: ignore[assignment]
    out = _run(monkeypatch, Model(), _scan_args(tmp_path))
    ref = json.loads((out / "reference.json").read_text())
    assert "weights_sha256" not in ref
    assert ref["model_id"] == "ref-001"


def test_a_model_with_no_spec_and_no_structure_digest_gets_neither_key(
        monkeypatch, tmp_path, offline_scan):
    class Bare:
        fmt = "subprocess"
        model_id = "bare-001"
        input_shape = (3, 8, 8)
        num_classes = 2

        def weight_digest(self) -> str:
            return "unavailable:black-box"

        def capabilities(self) -> CapabilitySet:
            return CapabilitySet(frozenset({Capability.MODEL_PREDICT}))

    out = _run(monkeypatch, Bare(), _scan_args(tmp_path))
    ref = json.loads((out / "reference.json").read_text())
    assert ref == {"model_id": "bare-001"}
    assert not (tmp_path / "out" / "evidence").exists() or not list(
        (tmp_path / "out" / "evidence").glob("*.json")), "no spec, nothing stored"
    rep = json.loads((out / "report.json").read_text())
    assert not {"preprocess_hash", "preprocess_ref", "arch_hash"} & set(rep["target"])


def test_no_key_of_the_reference_is_ever_an_empty_string(monkeypatch, tmp_path, offline_scan):
    out = _run(monkeypatch, RefModel(), _scan_args(tmp_path))
    ref = json.loads((out / "reference.json").read_text())
    assert all(v not in ("", None) for v in ref.values())
    assert "preprocess_hash" not in ref and "preprocess_ref" not in ref


def test_a_dataset_only_scan_writes_no_reference(monkeypatch, tmp_path, offline_scan):
    out = _run(monkeypatch, None, _scan_args(tmp_path))
    assert not (out / "reference.json").exists()


def test_the_fingerprint_is_written_only_when_the_caller_has_probes():
    m = RefModel()
    with_fp = cli.reference_manifest(m, with_fingerprint=True)
    assert isinstance(with_fp["fingerprint"], list) and with_fp["fingerprint"]
    assert "fingerprint" not in cli.reference_manifest(m, with_fingerprint=False)

    class Unanswerable(RefModel):
        def predict(self, x):
            raise RuntimeError("no")

    # Unknown means absent, not an error and not an empty list.
    assert "fingerprint" not in cli.reference_manifest(Unanswerable(), with_fingerprint=True)


def test_execute_scan_asks_for_a_fingerprint_when_the_model_can_be_queried(
        monkeypatch, tmp_path, offline_scan):
    """The fingerprint needs a model that answers, not a probe set: gating it on probes meant
    the real `cva scan` path never registered the one thing that closes the no-reference gap."""
    asked: list[bool] = []
    real = cli.write_reference
    monkeypatch.setattr(cli, "write_reference", lambda *a, with_fingerprint, **k: (
        asked.append(with_fingerprint), real(*a, with_fingerprint=with_fingerprint, **k))[1])
    _run(monkeypatch, RefModel(), _scan_args(tmp_path))
    assert asked == [True]


def test_emit_reference_stores_the_spec_beside_the_manifest_it_points_at(tmp_path):
    from cva.loaders.preprocess import attach_preprocess, preprocess_digest

    model = RefModel()
    attach_preprocess(model, None, _spec_file(tmp_path))
    dest = tmp_path / "corpus_out" / "ref.reference.json"
    dest.parent.mkdir()
    cli.emit_reference(model, dest)
    ref = json.loads(dest.read_text())
    assert ref["preprocess_ref"] == f"{preprocess_digest(SPEC)[0]}.json"
    assert (dest.parent / "evidence" / ref["preprocess_ref"]).is_file()
    assert ref["fingerprint"], "the corpus path has probes, so it registers a fingerprint"
    # ...and the emitted manifest is still what `manifest_from` reads back.
    assert cli.manifest_from(dest).weights_sha256 == HEX64


def test_a_stored_spec_whose_name_differs_from_the_handles_ref_is_an_error(tmp_path):
    model = RefModel()
    model.preprocess_ref = "0" * 64 + ".json"           # type: ignore[attr-defined]
    model.preprocess_spec_bytes = b"{}"                 # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="stored preprocessing spec"):
        cli.store_preprocess(model, tmp_path)


# --- selftest end to end (slow) --------------------------------------------------------------------

needs_backbone = pytest.mark.skipif(
    not (vendor.available("resnet18") or vendor.available("dinov2_vits14")),
    reason="no vendored backbone on disk — Mode A runs `make vendor`")


def _selftest(out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "cva.cli", "selftest", "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=900, check=False)


def _report_of(out: Path) -> Path:
    found = list((out / "reports").glob("s-*/report.json"))
    assert len(found) == 1, f"expected exactly one report under {out}/reports, got {found}"
    return found[0]


@pytest.fixture(scope="module")
def first_selftest(tmp_path_factory):
    out = tmp_path_factory.mktemp("selftest-a")
    return out, _selftest(out)


@pytest.mark.slow
@needs_backbone
def test_selftest_passes_end_to_end(first_selftest):
    out, proc = first_selftest
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "selftest PASSED" in proc.stdout
    assert "selftest FAILED" not in proc.stdout
    assert _report_of(out).is_file()


@pytest.mark.slow
@needs_backbone
def test_v10_two_selftest_runs_write_byte_identical_reports(first_selftest, tmp_path):
    """Two roots, because two selftest runs derive the same scan_id by design and a scan is
    never overwritten: one root would compare a file with itself."""
    out_a, proc_a = first_selftest
    assert proc_a.returncode == 0, proc_a.stdout[-2000:] + proc_a.stderr[-2000:]
    out_b = tmp_path / "selftest-b"
    proc_b = _selftest(out_b)
    assert proc_b.returncode == 0, proc_b.stdout[-2000:] + proc_b.stderr[-2000:]

    a, b = _report_of(out_a), _report_of(out_b)
    assert a.parent.name == b.parent.name, "the scan_id is derived from the seed"
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.slow
@needs_backbone
def test_selftest_exercises_the_preprocess_hash_path(first_selftest):
    """The fixture model ships a sidecar, so selftest reports the hash, stores the spec it
    points at, writes the reference manifest, and does NOT claim `prov.recompute` is
    impossible."""
    out, proc = first_selftest
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    report = _report_of(out)
    rep = json.loads(report.read_text())
    target = rep["target"]
    assert len(target["preprocess_hash"]) == 64 and len(target["arch_hash"]) == 64
    assert target["preprocess_ref"] == f"{target['preprocess_hash']}.json"
    assert (out / "reports" / "evidence" / target["preprocess_ref"]).is_file()
    assert not any("No preprocessing spec was declared" in s
                   for s in rep["coverage"]["standing_limitations"])
    ref = json.loads((report.parent / "reference.json").read_text())
    assert ref["preprocess_hash"] == target["preprocess_hash"]
    assert ref["arch_hash"] == target["arch_hash"]
    assert len(ref["weights_sha256"]) == 64 and ref["fingerprint"]
