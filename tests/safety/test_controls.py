"""Verification row V6 — one crafted failing input per control S1-S8.

backend_plan.md §5.14 is explicit that every control ships with a failing-input test, and
§8 gate B2a is worded as "every S1-S8 failing-input test rejects its input". The distinction
matters: a test that feeds a control a hostile file and asserts the process survived proves
only that nothing crashed, which is exactly what a decorative control also produces. Each
test below therefore asserts a *rejection* — a raised `UnsafeArtifact` naming its control,
or an explicit unsafe verdict — and every malicious input is built in `tmp_path` at test
time so the repository never carries a weaponised fixture.
"""
from __future__ import annotations

import hashlib
import os
import socket

import pytest

from cva.loaders.safety import (
    UnsafeArtifact,
    check_json_safety,
    check_pixel_budget,
    check_torch_version,
    decoder_pins,
    load_in_sandbox,
    prescan,
    safe_join,
)

# --- S1 ---------------------------------------------------------------------------------

def test_s1_digest_is_recorded_before_the_file_is_opened(tmp_path):
    """The digest must exist even for an artefact whose parse fails.

    The crafted input is an unparseable .onnx: prescan is forced down its failure path, and
    the assertion is that the hash still came back correct. That is the ordering guarantee
    S1 buys — the ledger entry survives a load that kills the process — and it is only
    observable on an artefact that does not load.
    """
    artefact = tmp_path / "broken.onnx"
    artefact.write_bytes(b"\x08\x07this is not a serialised ONNX graph")
    expected = hashlib.sha256(artefact.read_bytes()).hexdigest()

    report = prescan(artefact)

    assert report.sha256 == expected
    assert report.size_bytes == artefact.stat().st_size
    assert report.safe_to_load is False


# --- S2 ---------------------------------------------------------------------------------

def test_s2_torch_below_the_cve_floor_is_rejected():
    """2.5.1 is the interesting case, not 1.x.

    `weights_only=True` already existed at 2.5, so a 2.5 environment passes every
    surface-level check and still carries CVE-2025-32434. The local-version suffix is
    included because release wheels carry one and a naive parse trips over it.
    """
    with pytest.raises(RuntimeError, match="CVE-2025-32434"):
        check_torch_version("2.5.1")
    with pytest.raises(RuntimeError, match="CVE-2025-32434"):
        check_torch_version("2.5.1+cpu")

    # The guard against the failure mode that fails open: string comparison would rank
    # '2.14.0' below '2.6.0' and reject a perfectly safe torch.
    check_torch_version("2.14.0+cpu")
    check_torch_version()


# --- S3 ---------------------------------------------------------------------------------

def probe_that_attempts_egress(path):
    """Stands in for a checkpoint whose unpickling phones home. Imported by the S3 child."""
    socket.create_connection(("127.0.0.1", 9), timeout=1)
    return "egress succeeded"


def probe_that_behaves(path):
    return path.name


def test_s3_a_load_attempting_network_egress_is_blocked_in_the_sandbox(tmp_path):
    """Assert the sandbox's own sentinel, not merely that the connection failed.

    An unsandboxed connection to a closed port fails too, so "it raised" would be
    indistinguishable from the control being absent. The reason string has to show that the
    stub installed by the sandbox is what stopped it.
    """
    artefact = tmp_path / "checkpoint.pt"
    artefact.write_bytes(b"\x00")

    with pytest.raises(UnsafeArtifact) as excinfo:
        load_in_sandbox(
            "tests.safety.test_controls:probe_that_attempts_egress", artefact, timeout_s=60
        )

    assert excinfo.value.control == "S3"
    assert "network egress blocked" in excinfo.value.reason

    # The sandbox has to stay usable for honest loads, or callers will route around it.
    assert load_in_sandbox(
        "tests.safety.test_controls:probe_that_behaves", artefact, timeout_s=60
    ) == "checkpoint.pt"


# --- S4 ---------------------------------------------------------------------------------

def test_s4_an_onnx_model_with_a_custom_domain_op_is_marked_unsafe(tmp_path):
    """A custom op is a request to dlopen a supplier-chosen shared library.

    onnxruntime resolves custom operators at session-creation time, so refusing at prescan
    is the last point where the decision is still ours. The model is deliberately not run
    through `onnx.checker` — the artefact under audit will not have been either.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    node = helper.make_node("Exfiltrate", ["x"], ["y"], domain="com.attacker")
    graph = helper.make_graph(
        [node],
        "custom-op-graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.attacker", 1)],
    )
    artefact = tmp_path / "custom_op.onnx"
    onnx.save(model, str(artefact))

    report = prescan(artefact)

    assert report.safe_to_load is False
    assert any("Exfiltrate" in r for r in report.reasons)


# --- S5 ---------------------------------------------------------------------------------

def test_s5_deeply_nested_and_oversized_json_are_both_rejected(tmp_path):
    """Two separate caps, two separate crafted files.

    The nested file is 2000 levels deep, which is past CPython's recursion limit: `json.load`
    on it raises `RecursionError` rather than returning, so the check genuinely has to run
    before the parser and has to be iterative itself.
    """
    nested = tmp_path / "nested.json"
    nested.write_text("[" * 2000 + "]" * 2000)

    with pytest.raises(UnsafeArtifact) as deep:
        check_json_safety(nested, max_depth=64)
    assert deep.value.control == "S5"
    assert "depth" in deep.value.reason

    oversized = tmp_path / "oversized.json"
    oversized.write_text("[" + ",".join("0" for _ in range(20000)) + "]")

    with pytest.raises(UnsafeArtifact) as big:
        check_json_safety(oversized, max_bytes=1024)
    assert big.value.control == "S5"
    assert "size cap" in big.value.reason

    # An annotation file of an ordinary shape must still pass, or COCO loading is dead.
    benign = tmp_path / "benign.json"
    benign.write_text('{"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": []}')
    check_json_safety(benign, max_depth=64)


# --- S6 ---------------------------------------------------------------------------------

def test_s6_file_name_escaping_the_dataset_root_is_rejected(tmp_path):
    """Three spellings of one escape: traversal, absolute, and a symlink out of the root.

    The symlink case is the one a string check misses entirely — the `file_name` is an
    innocent bare name and the escape lives in the filesystem, planted inside the dataset
    the auditor was handed.
    """
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "images" / "ok.jpg").write_bytes(b"jpeg")

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("not yours")
    os.symlink(secret, root / "escape.jpg")

    for hostile in ("../../etc/passwd", "/etc/passwd", "escape.jpg"):
        with pytest.raises(UnsafeArtifact) as excinfo:
            safe_join(root, hostile)
        assert excinfo.value.control == "S6"

    assert safe_join(root, "images/ok.jpg") == (root / "images" / "ok.jpg").resolve()


# --- S7 ---------------------------------------------------------------------------------

def test_s7_an_image_over_the_pixel_budget_is_rejected_from_its_header(tmp_path, monkeypatch):
    """A low cap and a small image, deliberately.

    Materialising a real 20000x20000 bomb to prove a comparison would make the gate slow for
    no extra assurance: the control reads `.size` from the header, so what is being verified
    is that the refusal happens against the *declared* dimensions with no decode. A 40x40
    image against a 1000-pixel budget exercises exactly that path in milliseconds.
    """
    Image = pytest.importorskip("PIL.Image")
    # The control tightens a process-global Pillow setting by design, so the test has to put
    # it back: a 1000-pixel ceiling left behind would make every later test that opens an
    # ordinary image fail, and the failure would look like a bug in that other test.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)

    bomb = tmp_path / "bomb.png"
    Image.new("L", (40, 40)).save(bomb)

    with pytest.raises(UnsafeArtifact) as excinfo:
        check_pixel_budget(bomb, max_pixels=1000)
    assert excinfo.value.control == "S7"
    assert "decode budget" in excinfo.value.reason

    small = tmp_path / "small.png"
    Image.new("L", (10, 10)).save(small)
    assert check_pixel_budget(small, max_pixels=1000) == (10, 10)


# --- S8 ---------------------------------------------------------------------------------

def test_s8_the_pins_name_the_decoders_actually_imported():
    """The report's access-assumptions block claims these versions ran, so compare to what
    the test itself imports rather than to distribution metadata, which can disagree with
    the loaded module in an environment that was installed over."""
    import onnx
    import onnxruntime
    import PIL
    import torch

    pins = decoder_pins()

    assert pins["pillow"] == PIL.__version__
    assert pins["onnx"] == onnx.__version__
    assert pins["onnxruntime"] == onnxruntime.__version__
    assert pins["torch"] == torch.__version__
    assert "not-installed" not in pins.values()
