"""Item 17 — `cva/core/interfaces.py` must describe the plug-ins that actually exist.

Two of the three Protocols had drifted: `Detector.detect` was declared 3-arg while every
detector runs 4-arg with `ctx`, and `ModelCheck.check` still took `reference_probes` /
`reference_battery`. Nothing caught it because nothing imported the file — the Protocols are
`@runtime_checkable` and no `isinstance` check existed anywhere in the tree, so a file that
describes the contract was inert exactly where it was wrong and read exactly where it was
stale. `core/drift_scan.py` is the first real consumer.

This is the missing caller. It compares the SIGNATURE, not just `hasattr`: a
`@runtime_checkable` Protocol's `isinstance` only checks method NAMES, so it would have
passed throughout the period the parameters were wrong.
"""
from __future__ import annotations

import inspect

import pytest

import cva.detectors.data.registry  # noqa: F401 — registration happens at the entrypoint
import cva.detectors.model.registry  # noqa: F401
from cva.core.interfaces import Detector, ModelCheck, _PlugIn
from cva.core.registry import DETECTOR_REGISTRY, REGISTRY


def _params(fn) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


@pytest.mark.parametrize("cid", sorted(REGISTRY))
def test_every_registered_model_check_matches_the_ModelCheck_protocol(cid):
    cls = REGISTRY[cid]
    assert isinstance(cls(), ModelCheck), f"{cid} does not satisfy ModelCheck"
    assert _params(cls.check) == _params(ModelCheck.check), (
        f"{cid}.check{inspect.signature(cls.check)} does not match the Protocol "
        f"{inspect.signature(ModelCheck.check)}")


@pytest.mark.parametrize("cid", sorted(DETECTOR_REGISTRY))
def test_every_registered_detector_matches_the_Detector_protocol(cid):
    cls = DETECTOR_REGISTRY[cid]
    assert isinstance(cls(), Detector), f"{cid} does not satisfy Detector"
    got, want = _params(cls.detect), _params(Detector.detect)
    assert got == want, (
        f"{cid}.detect{inspect.signature(cls.detect)} does not match the Protocol "
        f"{inspect.signature(Detector.detect)}")


@pytest.mark.parametrize("cid", sorted(REGISTRY) + sorted(DETECTOR_REGISTRY))
def test_every_registered_plugin_declares_the_plugin_fields(cid):
    # `_PlugIn` is not `@runtime_checkable` (a non-method-member Protocol cannot be), so the
    # fields are asserted directly rather than through isinstance.
    cls = REGISTRY.get(cid) or DETECTOR_REGISTRY[cid]
    assert set(_PlugIn.__annotations__) <= {"id", "version", "requires", "optional",
                                            "attack_classes"}
    assert isinstance(cls.id, str) and cls.id
    assert isinstance(cls.version, str) and cls.version
    for field in ("requires", "optional", "attack_classes"):
        assert isinstance(getattr(cls, field), (frozenset, set, tuple)), (cid, field)


def test_the_protocols_are_actually_imported_by_something():
    """The finding was not "the Protocols are wrong" but "the Protocols are wrong AND
    nothing reads them". This test is the caller; asserting the import keeps it honest if
    the parametrised tests above are ever narrowed to nothing."""
    assert Detector.__module__ == "cva.core.interfaces"
    assert ModelCheck.__module__ == "cva.core.interfaces"
