"""INTEGRATION tests — these need a built corpus and are skipped without one.

The per-check contract tests live in tests/detectors/model/ against synthetic fixtures.

Property and contract tests. The invariants example-based tests miss are the ones that
matter here — a capability model that silently lies is worse than no capability model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cva.adapters.models import load_model
from attacklab.arch import ARCH_REGISTRY
from cva.core.capability import Availability, Capability, CapabilitySet
from cva.core.finding import Severity
from cva.core.model import ModelBattery
from cva.detectors.base import REGISTRY
from cva.orchestrator import RunContext, scan

CORPUS = Path("artifacts/corpus")
pytestmark = pytest.mark.skipif(not (CORPUS / "manifest.json").exists(),
                                reason="corpus not built")


@pytest.fixture(scope="module")
def man():
    return json.loads((CORPUS / "manifest.json").read_text())


@pytest.fixture(scope="module")
def probes():
    return (np.load(CORPUS / "probe_x.npy")[:200], np.load(CORPUS / "probe_y.npy")[:200])


def test_onnx_never_reports_gradients(man):
    """The single defect that motivated the whole capability model."""
    for e in man["models"]:
        if "onnx" not in e:
            continue
        m = load_model(CORPUS / e["onnx"], ARCH_REGISTRY)
        caps = m.capabilities()
        assert Capability.MODEL_GRADIENTS not in caps, f"{e['id']}: ONNX claimed gradients"
        assert caps.note_for(Capability.MODEL_GRADIENTS), "absence must carry a reason"
        assert Capability.MODEL_WEIGHTS in caps, "ONNX must still expose weights"


def test_pytorch_reports_gradients(man):
    e = next(x for x in man["models"] if "pt" in x)
    m = load_model(CORPUS / e["pt"], ARCH_REGISTRY)
    assert Capability.MODEL_GRADIENTS in m.capabilities()


def test_frozen_torchscript_probed_not_assumed(man):
    """A frozen archive must be DISCOVERED by attempting a backward pass."""
    e = next((x for x in man["models"] if x.get("frozen")), None)
    if e is None:
        pytest.skip("no frozen archive in corpus")
    m = load_model(CORPUS / e["ts"], ARCH_REGISTRY)
    caps = m.capabilities()
    assert Capability.MODEL_PREDICT in caps
    if Capability.MODEL_GRADIENTS not in caps:
        assert "frozen" in (caps.note_for(Capability.MODEL_GRADIENTS) or "").lower()


def test_unavailable_is_never_a_silent_skip(man, probes):
    """Every registered check must appear in the plan and produce a Finding, whatever
    its resolution. A shorter report must never look like a cleaner result."""
    e = next(x for x in man["models"] if "onnx" in x)
    m = load_model(CORPUS / e["onnx"], ARCH_REGISTRY)
    ctx = RunContext(probes_x=probes[0], probes_y=probes[1],
                     battery=ModelBattery(), seed=1)
    res = scan(m, ctx, "deep")
    assert {r.check_id for r in res.plan} == set(REGISTRY)
    covered = {f.detector_id for f in res.findings}
    assert covered == set(REGISTRY), f"checks vanished: {set(REGISTRY) - covered}"


def test_error_is_not_degraded(man, probes):
    """A raising check must surface as ERROR, never folded into DEGRADED."""
    from cva.detectors.base import register

    class Exploding:
        id = "model.test_exploding"
        version = "0"
        requires: set = set()
        optional: set = set()
        attack_classes = {"tool.error"}

        def check(self, model, ctx):
            raise RuntimeError("deliberate")

    REGISTRY[Exploding.id] = Exploding
    try:
        e = next(x for x in man["models"] if "pt" in x)
        m = load_model(CORPUS / e["pt"], ARCH_REGISTRY)
        res = scan(m, RunContext(probes_x=probes[0], probes_y=probes[1],
                                 battery=ModelBattery(), seed=1), "deep")
        f = next(f for f in res.findings if f.detector_id == Exploding.id)
        assert f.availability == Availability.ERROR
    finally:
        REGISTRY.pop(Exploding.id, None)


def test_capability_resolution_states():
    caps = CapabilitySet(frozenset({Capability.MODEL_PREDICT}),
                         ((Capability.MODEL_GRADIENTS, "no backward pass"),))
    assert caps.resolve({Capability.MODEL_PREDICT}, set()).state == Availability.OK
    r = caps.resolve({Capability.MODEL_PREDICT}, {Capability.MODEL_GRADIENTS})
    assert r.state == Availability.DEGRADED and "no backward pass" in r.reason
    r2 = caps.resolve({Capability.MODEL_WEIGHTS}, set())
    assert r2.state == Availability.UNAVAILABLE and Capability.MODEL_WEIGHTS in r2.missing


def test_fingerprint_is_deterministic(man):
    from cva.detectors.model.fingerprint import fingerprint
    e = next(x for x in man["models"] if "pt" in x)
    m = load_model(CORPUS / e["pt"], ARCH_REGISTRY)
    a, b = fingerprint(m), fingerprint(m)
    assert np.allclose(a, b), "probe battery must be deterministic"


def test_benign_reexport_is_not_called_substitution(man, probes):
    """The digest false-positive case: identical behaviour, different bytes.
    The digest may fire; the fingerprint must NOT call it hostile."""
    from cva.detectors.model.fingerprint import FingerprintCheck, fingerprint
    from cva.core.model import Manifest
    from cva.detectors.base import CheckContext

    base = next(x for x in man["models"] if x["id"] == "clean_a")
    var = next((x for x in man["models"] if x.get("benign_variant")), None)
    if var is None:
        pytest.skip("no benign variant")
    m_base = load_model(CORPUS / base["pt"], ARCH_REGISTRY)
    m_var = load_model(CORPUS / var["onnx"], ARCH_REGISTRY)
    manifest = Manifest(model_id="clean_a", weights_sha256=m_base.weight_digest(),
                        fingerprint=[float(v) for v in fingerprint(m_base)])
    ctx = CheckContext(probes_x=probes[0], probes_y=probes[1],
                       battery=ModelBattery(manifest=manifest), scan_id="t")
    f = FingerprintCheck().check(m_var, ctx)[0]
    assert f.severity.rank < Severity.HIGH.rank, \
        f"benign re-export flagged as hostile: {f.reason}"
