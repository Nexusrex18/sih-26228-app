"""Plan V7 and the frozen-TorchScript path, end to end.

A frozen archive (`torch.jit.freeze`) inlines its parameters as graph constants, so a real
supplier's exported model has no named weights, no parameter gradients and no weight digest.
Until now nothing ran that path through the product: the conformance suite probes the handle,
but no report had ever been produced from one.
"""
from __future__ import annotations

import json
import re

import pytest

from cva import fixtures
from cva.cli import main
from cva.core.capability import Capability
from cva.loaders.models import load_model

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def frozen_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("frozen")
    return fixtures.build_frozen_model(d / "demo_frozen.pt")


def test_the_handle_reports_no_gradients_no_weights_and_no_digest(frozen_model):
    h = load_model(frozen_model)
    caps = h.capabilities()
    assert h.fmt == "torchscript"
    assert Capability.MODEL_GRADIENTS not in caps
    assert Capability.MODEL_WEIGHTS not in caps
    assert "freeze" in (caps.note_for(Capability.MODEL_GRADIENTS) or "")
    assert Capability.MODEL_PREDICT in caps
    assert h.weight_digest() == "unavailable:frozen"
    arch = h.arch_hash() if callable(h.arch_hash) else h.arch_hash
    assert _HEX64.match(arch)


def test_v7_the_plan_prints_before_any_work_and_gradients_are_unavailable(
        frozen_model, tmp_path, capsys):
    ds = fixtures.build_dataset(tmp_path)
    rc = main(["scan", "--dataset", str(ds), "--model", str(frozen_model),
               "--profile", "baseline", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    grad = next(line for line in out.splitlines() if line.strip().startswith("Gradients"))
    assert "NOT AVAILABLE" in grad and "freeze" in grad
    # The plan is printed, and the dry run says nothing executed.
    assert out.index("Plan") < out.index("DRY RUN")
    assert "torchscript" in out
    # `missing[]` is printed against the checks that could not run.
    assert re.search(r"model\.weight_stats\s+UNAVAILABLE\s+requires MODEL_WEIGHTS", out)


def test_a_real_scan_of_a_frozen_model_produces_an_honest_report(frozen_model, tmp_path):
    out = tmp_path / "out"
    assert main(["scan", str(frozen_model), "--profile", "baseline", "--out", str(out)]) == 0
    report_path = next(out.glob("s-*/report.json"))
    report = json.loads(report_path.read_text())

    assert report["target"]["model_format"] == "torchscript"
    assert "model_sha256" not in report["target"]        # omitted, never "unavailable:frozen"
    assert _HEX64.match(report["target"]["arch_hash"])
    assert not [f for f in report["findings"] if f["availability"] == "ERROR"]

    digest = [f for f in report["findings"] if f["detector_id"] == "model.weight_digest"]
    assert digest and all(f["availability"] == "DEGRADED" for f in digest)

    limits = " ".join(report["coverage"]["standing_limitations"])
    assert "frozen TorchScript" in limits


def test_the_reference_manifest_registers_the_fingerprint_but_never_a_fake_digest(
        frozen_model, tmp_path):
    out = tmp_path / "out"
    assert main(["scan", str(frozen_model), "--profile", "baseline", "--out", str(out)]) == 0
    ref = json.loads(next(out.glob("s-*/reference.json")).read_text())
    assert "weights_sha256" not in ref                   # registering "unavailable:frozen" would
    assert _HEX64.match(ref["arch_hash"])                # make every later scan "match" it
    assert ref["fingerprint"] and all(isinstance(v, float) for v in ref["fingerprint"])
    assert "" not in ref.values()


def test_the_frozen_fixture_is_deterministic(tmp_path):
    a = fixtures.build_frozen_model(tmp_path / "a.pt")
    b = fixtures.build_frozen_model(tmp_path / "b.pt")
    assert load_model(a).arch_hash() == load_model(b).arch_hash()
