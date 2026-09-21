"""Item 22 — `supports()` runs OUTSIDE the sandbox, so it may not deserialise.

`load()` is sandboxed and `safety.S3_SANDBOX_LIMITATION` describes accurately what an
attacker can do inside that boundary. `supports()` is on the other side of it, and it runs
for every registered loader against every candidate file — the widest-reach path in the
loader stack. It used to call `torch.jit.load` and `torch.load` there.

The guard is asserted by DENYING the deserialisers: if any `supports()` reaches one, the
stub raises and the test fails. That is the shape that fails on revert — a test that only
checked the return values would keep passing if the calls came back.
"""
from __future__ import annotations

import zipfile

import pytest
import torch
import torch.nn as nn

from cva.loaders.models.pytorch import PyTorchLoader
from cva.loaders.models.torchscript import TorchScriptLoader


@pytest.fixture
def no_deserialising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every torch entry point that unpickles becomes a hard failure."""
    def forbidden(*a, **k):
        raise AssertionError(
            "supports() deserialised untrusted bytes outside the load sandbox")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.jit, "load", forbidden)


@pytest.fixture
def scripted(tmp_path):
    p = tmp_path / "m.pt"
    torch.jit.save(torch.jit.script(nn.Linear(4, 3)), str(p))
    return p


@pytest.fixture
def checkpoint(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"arch": "linear", "state_dict": nn.Linear(4, 3).state_dict()}, str(p))
    return p


def test_torchscript_supports_identifies_its_own_archive_without_loading(
        scripted, checkpoint, no_deserialising):
    loader = TorchScriptLoader()
    assert loader.supports(scripted) is True
    assert loader.supports(checkpoint) is False


def test_pytorch_supports_identifies_a_checkpoint_without_loading(
        scripted, checkpoint, no_deserialising):
    loader = PyTorchLoader({"linear": lambda: nn.Linear(4, 3)})
    assert loader.supports(checkpoint) is True
    assert loader.supports(scripted) is False, "TorchScript belongs to the other loader"


def test_exactly_one_loader_claims_each_archive(scripted, checkpoint, no_deserialising):
    """The two sniffs share `_is_torchscript_archive`, so they cannot disagree about a .pt.
    Both claiming it, or neither, is the failure this pins."""
    ts, pt = TorchScriptLoader(), PyTorchLoader({"linear": lambda: nn.Linear(4, 3)})
    for path in (scripted, checkpoint):
        assert [ts.supports(path), pt.supports(path)].count(True) == 1, path


def test_supports_does_not_deserialise_a_hostile_archive(tmp_path, no_deserialising):
    """A zip that is neither — the case an attacker controls. It must be answered from the
    directory, not by trying to unpickle it, and it must not raise."""
    p = tmp_path / "hostile.pt"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x04\x95 not a real pickle")
    assert TorchScriptLoader().supports(p) is False
    # The pytorch loader may still claim it: it is a zip with no TorchScript graph, which is
    # the shape of a checkpoint. It is then rejected by `load()`, INSIDE the sandbox, which
    # is where deserialising an unknown archive is supposed to happen.
    PyTorchLoader().supports(p)


def test_supports_rejects_a_non_archive_without_raising(tmp_path, no_deserialising):
    p = tmp_path / "junk.pt"
    p.write_bytes(b"not a zip at all")
    assert TorchScriptLoader().supports(p) is False
    assert PyTorchLoader().supports(p) is False


def test_supports_rejects_a_wrong_suffix_before_touching_the_file(tmp_path, no_deserialising):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"")
    assert TorchScriptLoader().supports(p) is False
    assert PyTorchLoader().supports(p) is False
