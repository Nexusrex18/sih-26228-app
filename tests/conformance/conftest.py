"""Artefacts for the shared loader conformance suite, all built at run time.

Nothing here is a committed binary fixture. A checked-in `.onnx` or `.pt` is a blob nobody
re-reads, drifting from the exporter that made it until the day it fails for a reason the
suite cannot explain — and §5.14 is a module about not trusting opaque artefacts, so
shipping our own would be an odd place to start. Every file below is exported from the one
module defined here, which means the suite also proves the export path still works.

The model is deliberately tiny (3x8x8 in, 2 classes) and its weights are fixed: the suite
asserts digest STABILITY across two loads, and a randomly initialised module would make
that assertion about the seed rather than about the loader.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

INPUT_SHAPE = (3, 8, 8)
NUM_CLASSES = 2


def _build_module():
    """A conv + linear net. Conv matters: ONNX activation surgery looks for real op types,
    and a pure-Linear model would let a broken surgery pass by finding nothing to fail on."""
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self, num_classes: int = NUM_CLASSES) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, num_classes)

        def forward(self, x):
            x = self.pool(self.relu(self.conv(x)))
            return self.fc(x.flatten(1))

    torch.manual_seed(0)       # fixed weights — see the module docstring
    m = Tiny()
    return m.eval()


ARCH_REGISTRY = {"tiny": lambda **kw: _build_module()}


@pytest.fixture(scope="session")
def artefacts(tmp_path_factory) -> dict[str, Path]:
    """The four model files the suite loads, plus the two dataset trees."""
    import torch

    d = tmp_path_factory.mktemp("artefacts")
    m = _build_module()

    torch.save(
        {"arch": "tiny", "state_dict": m.state_dict(),
         "input_shape": list(INPUT_SHAPE), "num_classes": NUM_CLASSES},
        d / "state_dict.pt",
    )

    # _extra_files is how an exporter hands the loader the input shape — the one fact a
    # TorchScript graph does not reliably carry. num_classes is deliberately NOT written:
    # the loader probes it, and the fixture must not hand it the answer.
    import json
    meta = {"cva_meta.json": json.dumps({"input_shape": list(INPUT_SHAPE)})}
    scripted = torch.jit.script(m).eval()
    scripted.save(str(d / "scripted.pt"), _extra_files=dict(meta))

    # freeze() needs a ScriptModule in eval mode, never the nn.Module — it inlines the
    # parameters as graph constants, which is exactly the condition the gradient probe
    # has to discover by ATTEMPTING a backward pass rather than by reading the suffix.
    torch.jit.freeze(torch.jit.script(m).eval()).save(
        str(d / "frozen.pt"), _extra_files=dict(meta))

    # dynamo=False is the house call (attacklab/train.py): torch 2.14's default exporter
    # pulls in onnxscript, and adding a dependency so that a TEST can build a fixture would
    # put a package in the bundle that the product never uses.
    torch.onnx.export(
        m, torch.zeros(1, *INPUT_SHAPE), str(d / "model.onnx"),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    return {
        "state_dict": d / "state_dict.pt",
        "scripted": d / "scripted.pt",
        "frozen": d / "frozen.pt",
        "onnx": d / "model.onnx",
    }


def write_png(path: Path, w: int, h: int, seed: int = 0) -> None:
    """A real PNG, because the DATASET_IMAGES probe must actually decode a header."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# Three images at DIFFERENT sizes. Equal sizes would let a normalise/denormalise pair that
# silently swaps width and height round-trip perfectly, which is the single most likely
# bbox bug in the COCO/YOLO conversion and the one a square fixture cannot catch.
IMAGE_SPECS = [("a.png", 40, 20), ("b.png", 25, 50), ("c.png", 33, 17)]

# Non-contiguous, non-zero-based COCO category ids, on purpose: COCO ids are arbitrary and
# YOLO indices are dense and 0-based, so a loader that passes the id straight through
# instead of building a mapping passes a contiguous fixture and fails on real data.
COCO_CATEGORIES = [
    {"id": 17, "name": "tank"},
    {"id": 3, "name": "truck"},
    {"id": 88, "name": "apc"},
]


@pytest.fixture(scope="session")
def retrained(tmp_path_factory) -> dict[str, Path]:
    """The same architecture with different numbers in it.

    This is the fixture `arch_hash` is actually tested against: a structure digest that
    cannot be distinguished from a weight digest is not a second signal, and the
    substitution check needs two independent answers to separate a re-export from a swap.
    """
    import torch

    d = tmp_path_factory.mktemp("retrained")
    m = _build_module()
    with torch.no_grad():
        g = torch.Generator().manual_seed(99)
        for q in m.parameters():
            q.add_(torch.randn(q.shape, generator=g))

    torch.save(
        {"arch": "tiny", "state_dict": m.state_dict(),
         "input_shape": list(INPUT_SHAPE), "num_classes": NUM_CLASSES},
        d / "state_dict.pt",
    )
    # The same _extra_files as the baseline. Without it the loader falls back to a
    # different default input_shape, arch_hash moves for that reason alone, and the test
    # would "prove" structure-vs-weight separation by comparing two different structures.
    import json
    meta = {"cva_meta.json": json.dumps({"input_shape": list(INPUT_SHAPE)})}
    torch.jit.script(m).eval().save(str(d / "scripted.pt"), _extra_files=dict(meta))
    torch.jit.freeze(torch.jit.script(m).eval()).save(
        str(d / "frozen.pt"), _extra_files=dict(meta))
    torch.onnx.export(
        m, torch.zeros(1, *INPUT_SHAPE), str(d / "model.onnx"),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    return {"state_dict": d / "state_dict.pt", "scripted": d / "scripted.pt",
            "frozen": d / "frozen.pt", "onnx": d / "model.onnx"}


def _write_coco(d: Path, with_contributor: bool = False) -> Path:
    """A 3-image COCO tree with non-contiguous category ids."""
    import json

    root = d / "coco"
    (root / "images").mkdir(parents=True, exist_ok=True)
    images, annotations, ann_id = [], [], 1
    for i, (name, w, h) in enumerate(IMAGE_SPECS, start=1):
        write_png(root / "images" / name, w, h, seed=i)
        rec = {"id": i, "file_name": name, "width": w, "height": h}
        if with_contributor:
            rec["source"] = f"unit-{i}"
        images.append(rec)
        for j, cat in enumerate(COCO_CATEGORIES):
            # A box well inside the image, different per (image, category) so a swapped
            # width/height or a transposed pair cannot round-trip by coincidence.
            bw, bh = w / 4 + j, h / 5 + j
            annotations.append({"id": ann_id, "image_id": i, "category_id": cat["id"],
                                "bbox": [2 + j, 3 + j, bw, bh], "iscrowd": 0})
            ann_id += 1
    (root / "instances.json").write_text(json.dumps({
        "images": images, "annotations": annotations, "categories": COCO_CATEGORIES}))
    return root


@pytest.fixture(scope="session")
def coco_tree(tmp_path_factory) -> Path:
    return _write_coco(tmp_path_factory.mktemp("coco_ds"))


@pytest.fixture(scope="session")
def yolo_tree(tmp_path_factory) -> Path:
    """The same three images as a YOLO tree, written through the real writer.

    Built by exporting the COCO dataset rather than by hand: a hand-written fixture would
    encode my belief about the normalisation, and then the round-trip test would compare
    that belief against itself.
    """
    from cva.loaders.datasets import COCOLoader, to_yolo

    d = tmp_path_factory.mktemp("yolo_ds")
    ds = COCOLoader().load(_write_coco(d))
    return to_yolo(ds, d / "yolo")


# --------------------------------------------------------------------------
# B2b: the Tier 1 / Tier 2 dataset trees and the two query-only model adapters
# --------------------------------------------------------------------------
VOC_CLASSES = ["tank", "truck", "apc"]      # deliberately NOT sorted: the loader must sort


def voc_xml(filename: str, w: int, h: int,
            objects: list[tuple[str, int, int, int, int]]) -> str:
    """One Pascal VOC annotation. `objects` are `(name, xmin, ymin, xmax, ymax)` exactly as
    VOC writes them: 1-based and inclusive, so the tests exercise the `- 1` shift rather than
    re-deriving the loader's arithmetic."""
    body = "".join(
        f"<object><name>{name}</name><pose>Unspecified</pose><difficult>0</difficult>"
        f"<bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
        f"</bndbox></object>"
        for name, x0, y0, x1, y1 in objects)
    return (f"<annotation><filename>{filename}</filename>"
            f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>{body}"
            f"</annotation>")


def write_voc_entry(root: Path, stem: str, w: int, h: int,
                    objects: list[tuple[str, int, int, int, int]],
                    filename: str | None = None, with_image: bool = True) -> None:
    """One `Annotations/<stem>.xml` and (unless `with_image` is False) its picture."""
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    fname = filename if filename is not None else f"{stem}.png"
    if with_image:
        write_png(root / "JPEGImages" / fname, w, h, seed=len(stem))
    (root / "Annotations" / f"{stem}.xml").write_text(voc_xml(fname, w, h, objects))


@pytest.fixture(scope="session")
def voc_tree(tmp_path_factory) -> Path:
    """The three standard images with three boxes each, one per class in VOC_CLASSES."""
    root = tmp_path_factory.mktemp("voc_ds") / "voc"
    for name, w, h in IMAGE_SPECS:
        objects = []
        for j, cls in enumerate(VOC_CLASSES):
            # Sized off the image (as the COCO fixture is) so a swapped width/height cannot
            # round-trip; every box stays inside the smallest image (33x17).
            xmin, ymin = 3 + j, 4 + j
            objects.append((cls, xmin, ymin, xmin + w // 4 + j - 1, ymin + h // 5 + j - 1))
        write_voc_entry(root, Path(name).stem, w, h, objects)
    return root


@pytest.fixture(scope="session")
def imagefolder_tree(tmp_path_factory) -> Path:
    """`root/<class>/*.png` with two classes of unequal size — and no contributor anywhere."""
    root = tmp_path_factory.mktemp("imagefolder_ds") / "folders"
    for i, (cls, name, w, h) in enumerate([
            ("cats", "c1.png", 40, 20), ("cats", "c2.png", 25, 50), ("dogs", "d1.png", 33, 17)]):
        write_png(root / cls / name, w, h, seed=i)
    return root


@pytest.fixture(scope="session")
def generic_tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("generic_ds") / "flat"
    for i, (name, w, h) in enumerate(IMAGE_SPECS):
        write_png(root / name, w, h, seed=i)
    return root


def generic_label_fn(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """A `GenericDataset` label_fn returning `(class, bbox)` pairs, box inside every image."""
    return [("tank" if path.stem == "a" else "truck", (2.0, 3.0, 10.0, 8.0))]


# The child every SubprocessModel test runs. It slurps stdin into a BytesIO before np.load
# and writes through a BytesIO, because a pipe is not seekable and NumPy raises on one.
CHILD_SCRIPT = """\
import io
import sys

import numpy as np

x = np.load(io.BytesIO(sys.stdin.buffer.read()), allow_pickle=False)
m = x.reshape(len(x), -1).mean(axis=1)
out = io.BytesIO()
np.save(out, np.stack([m, -m], axis=1).astype(np.float32), allow_pickle=False)
sys.stdout.buffer.write(out.getvalue())
"""


@pytest.fixture(scope="session")
def child_command(tmp_path_factory) -> list[str]:
    import sys

    script = tmp_path_factory.mktemp("child") / "child_model.py"
    script.write_text(CHILD_SCRIPT)
    return [sys.executable, str(script)]


class QuietHandler(BaseHTTPRequestHandler):
    """Silent, and always drains the request body before answering: replying to a POST whose
    body is unread makes the client see a connection reset instead of the response."""

    def log_message(self, format: str, *args) -> None:
        pass

    def read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def reply(self, status: int, body: bytes = b"", ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LogitsHandler(QuietHandler):
    """The endpoint side of HTTPModel's protocol, computing what CHILD_SCRIPT computes."""

    def do_POST(self) -> None:
        import json

        x = np.asarray(json.loads(self.read_body())["inputs"], dtype=np.float32)
        m = x.reshape(len(x), -1).mean(axis=1)
        self.reply(200, json.dumps({"outputs": np.stack([m, -m], axis=1).tolist()}).encode())


@pytest.fixture
def serve():
    """`serve(handler_cls) -> port`: a loopback server on an OS-chosen port, in a daemon
    thread, shut down at teardown so no test leaves a listener behind."""
    servers: list[ThreadingHTTPServer] = []

    def start(handler_cls: type[BaseHTTPRequestHandler]) -> int:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        # A short poll interval: shutdown() waits up to one interval, and the default half
        # second per server would dominate the runtime of these tests.
        threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
        servers.append(srv)
        return int(srv.server_address[1])

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()
