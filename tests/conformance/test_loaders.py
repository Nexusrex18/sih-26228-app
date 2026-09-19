"""V5 — the shared loader conformance suite. One suite, five implementations.

This is the file that makes B2b cheap. Every loader added later inherits these assertions
by adding one row to `MODEL_LOADERS` or `DATASET_LOADERS`, so the cost of a new adapter is
the adapter, not the adapter plus a fresh set of tests nobody remembers to write.

The parametrisation ids are load-bearing and must stay exactly `coco`, `yolo`, `onnx`,
`torchscript` and `pytorch`: §11's V5 row runs this suite as
`pytest tests/conformance/ -k "coco or yolo or onnx or torchscript or pytorch"`, and an id
that stops matching silently drops that loader from the gate while the command still
reports success.

What is deliberately NOT asserted here: numerical accuracy. A loader's job is to open a
file and report honestly what it can do with it. Whether the model is any good is the
benchmark's question, and folding it in here would make every adapter's gate depend on a
corpus none of them own.
"""
from __future__ import annotations

import numpy as np
import pytest

from cva.core.capability import DATASET_CAPS, MODEL_CAPS, Capability

from .conftest import INPUT_SHAPE, NUM_CLASSES


# --------------------------------------------------------------------------
# model-side
# --------------------------------------------------------------------------
def _load(kind: str, artefacts):
    from cva.loaders.models import ONNXLoader, PyTorchLoader, TorchScriptLoader

    from .conftest import ARCH_REGISTRY

    if kind == "onnx":
        return ONNXLoader(), ONNXLoader().load(artefacts["onnx"])
    if kind == "torchscript":
        return TorchScriptLoader(), TorchScriptLoader().load(artefacts["scripted"])
    if kind == "pytorch":
        ldr = PyTorchLoader(ARCH_REGISTRY)
        return ldr, ldr.load(artefacts["state_dict"])
    raise AssertionError(kind)


MODEL_KINDS = ["onnx", "torchscript", "pytorch"]

#: B2b's query-only adapters. Kept OUT of `MODEL_KINDS` on purpose: test_gate_b2a pins that
#: list to the five B2a loaders, and these two have no file to load, no structure to hash and
#: no weights to digest — so the tests that need those are told to expect that, below.
BLACKBOX_MODEL_KINDS = ["subprocess", "http"]


def _load_blackbox(kind: str, request):
    from cva.loaders.models import HTTPModel, SubprocessModel

    from .conftest import LogitsHandler

    if kind == "subprocess":
        return SubprocessModel(request.getfixturevalue("child_command"), "child",
                               INPUT_SHAPE, NUM_CLASSES)
    port = request.getfixturevalue("serve")(LogitsHandler)
    return HTTPModel(f"http://127.0.0.1:{port}/predict", "endpoint", INPUT_SHAPE, NUM_CLASSES)


@pytest.fixture(params=MODEL_KINDS + BLACKBOX_MODEL_KINDS)
def model(request, artefacts):
    if request.param in BLACKBOX_MODEL_KINDS:
        return request.param, None, _load_blackbox(request.param, request)
    loader, handle = _load(request.param, artefacts)
    return request.param, loader, handle


def test_model_loader_reports_its_format_and_identity(model):
    kind, _loader, h = model
    assert h.fmt and isinstance(h.fmt, str)
    assert h.model_id
    assert h.num_classes == NUM_CLASSES
    assert tuple(h.input_shape) == INPUT_SHAPE


def test_model_capabilities_are_model_scoped_only(model):
    """`CapabilitySet` is assembled by the orchestrator from three sources. A loader that
    reported a DATASET_* or a REFERENCE_* capability would be answering for a source it
    cannot observe — the precise failure the split assembly rule exists to prevent."""
    _kind, _loader, h = model
    assert h.capabilities().caps <= MODEL_CAPS


def test_every_model_loader_can_at_least_predict(model):
    _kind, _loader, h = model
    assert Capability.MODEL_PREDICT in h.capabilities().caps
    out = h.predict(np.zeros(INPUT_SHAPE, dtype=np.float32))
    assert out.shape[-1] == NUM_CLASSES
    row = np.asarray(out).reshape(-1, NUM_CLASSES)[0]
    assert np.isclose(row.sum(), 1.0, atol=1e-4), "predict() must return probabilities"


def test_absent_capabilities_carry_a_reason(model):
    """`UNAVAILABLE` is not a skip — the report has to name what was missing and why. A
    probe that records absence without a note leaves the renderer with nothing to print."""
    _kind, _loader, h = model
    caps = h.capabilities()
    for cap in MODEL_CAPS - caps.caps:
        assert caps.note_for(cap), f"{cap} is absent with no recorded reason"


def test_digests_are_stable_across_two_independent_loads(model, artefacts):
    """Computed once at load and cached, per §9.2 — and identical between loads, or the
    seal binds a value that changes when nothing changed."""
    kind, _loader, h = model
    if kind in BLACKBOX_MODEL_KINDS:
        # Nothing to hash: a query-only endpoint has no weights and no graph, and says so
        # with a constant rather than a digest of nothing.
        assert h.weight_digest() == h.weight_digest() == "unavailable:black-box"
        return
    _loader2, h2 = _load(kind, artefacts)
    assert h.weight_digest() == h2.weight_digest()
    assert h.arch_hash() == h2.arch_hash()


def test_arch_hash_is_structure_not_weights(model, artefacts, retrained):
    """§9.2's two digests answer different questions. `weight_digest` asks "are these the
    same numbers?", `arch_hash` asks "is this the same shape of model?" — and the
    substitution check needs both to tell a re-export from a swap. If `arch_hash` moved
    with the weights it could only ever say "something changed", which starts an
    investigation without narrowing it."""
    kind, _loader, h = model
    if kind in BLACKBOX_MODEL_KINDS:
        pytest.skip("a query-only endpoint has no structure to hash")
    _l, h_retrained = _load(kind, retrained)
    assert h.arch_hash() == h_retrained.arch_hash(), (
        "same architecture, different weights must share an arch_hash")


def test_supports_accepts_its_own_format_and_declines_the_others(artefacts):
    """`supports()` is how the router picks an adapter, so an over-eager one silently
    steals another's files. The .pt collision is the real case: a modern torch.save and a
    TorchScript archive are both zip files and the magic bytes cannot separate them."""
    from cva.loaders.models import ONNXLoader, PyTorchLoader, TorchScriptLoader

    from .conftest import ARCH_REGISTRY

    onnx, ts, pt = ONNXLoader(), TorchScriptLoader(), PyTorchLoader(ARCH_REGISTRY)
    assert onnx.supports(artefacts["onnx"])
    assert not onnx.supports(artefacts["scripted"])

    assert ts.supports(artefacts["scripted"]) and ts.supports(artefacts["frozen"])
    assert not ts.supports(artefacts["onnx"])
    assert not ts.supports(artefacts["state_dict"]), (
        "a weights-only checkpoint is not a TorchScript archive")

    assert pt.supports(artefacts["state_dict"])
    assert not pt.supports(artefacts["scripted"]), (
        "TorchScriptLoader owns that file; both claiming it makes the router order-dependent")


# --- the two rows the gate names explicitly --------------------------------
def test_a_frozen_torchscript_model_reports_no_gradients(artefacts):
    """Gate B2a, and V7's proof that probing is ACTIVE rather than declarative.

    The trap is that the obvious probe passes on both. `torch.jit.freeze` inlines the
    parameters as graph constants, so a frozen archive has no differentiable parameters —
    but the INPUT is still an ordinary leaf tensor, so a backward pass through it succeeds
    and fills `x.grad` exactly as on a live model. Probing the input therefore reports
    gradients for every TorchScript file ever saved: the one answer that cannot be wrong,
    and therefore worth nothing.
    """
    from cva.loaders.models import TorchScriptLoader

    live = TorchScriptLoader().load(artefacts["scripted"])
    frozen = TorchScriptLoader().load(artefacts["frozen"])

    assert Capability.MODEL_GRADIENTS in live.capabilities().caps
    assert Capability.MODEL_GRADIENTS not in frozen.capabilities().caps
    assert frozen.capabilities().note_for(Capability.MODEL_GRADIENTS)

    # Same file type, same extension — so the difference cannot have been read off the name.
    assert artefacts["scripted"].suffix == artefacts["frozen"].suffix


def test_onnx_never_reports_gradients_and_records_its_opset(artefacts):
    """Two §7.3 rules in one place, because they are the same decision seen twice.

    Gradients are declared absent in advance rather than probed: they exist only through a
    separate `onnxruntime-training` build that ADR-001 deliberately does not bundle. That
    is a declared narrowing, not an unresolved gap.

    The opset is the mirror image — read and RECORDED, never gated. ONNX Runtime accepts
    opset 7 and up, so a minimum-version check would reject models it can happily run.
    """
    from cva.loaders.models import ONNXLoader

    h = ONNXLoader().load(artefacts["onnx"])
    assert Capability.MODEL_GRADIENTS not in h.capabilities().caps
    assert h.capabilities().note_for(Capability.MODEL_GRADIENTS)

    assert h.opset, "the opset must be recorded — the seal explains a re-export with it"
    assert all(isinstance(v, int) and v >= 7 for v in h.opset.values())


# --------------------------------------------------------------------------
# dataset-side
# --------------------------------------------------------------------------
DATASET_KINDS = ["coco", "yolo"]

#: B2b's loaders. Separate from `DATASET_KINDS` for the same reason as the black-box models:
#: test_gate_b2a pins that list to the B2a gate's five ids. They run through every test below.
B2B_DATASET_KINDS = ["voc", "imagefolder", "generic"]

#: Classification-only layouts. They carry no boxes, and the box test says so rather than
#: passing vacuously — `seen == 0` is asserted, so a loader that starts inventing boxes fails.
BOXLESS_KINDS = {"imagefolder"}


@pytest.fixture(params=DATASET_KINDS + B2B_DATASET_KINDS)
def dataset(request, coco_tree, yolo_tree):
    from cva.loaders.datasets import (
        COCOLoader,
        GenericDataset,
        ImageFolderLoader,
        VOCLoader,
        YOLOLoader,
    )

    from .conftest import generic_label_fn

    kind = request.param
    if kind == "coco":
        return kind, COCOLoader(), COCOLoader().load(coco_tree)
    if kind == "yolo":
        return kind, YOLOLoader(), YOLOLoader().load(yolo_tree)
    if kind == "voc":
        return kind, VOCLoader(), VOCLoader().load(request.getfixturevalue("voc_tree"))
    if kind == "imagefolder":
        return (kind, ImageFolderLoader(),
                ImageFolderLoader().load(request.getfixturevalue("imagefolder_tree")))
    gen = GenericDataset(request.getfixturevalue("generic_tree"), label_fn=generic_label_fn)
    return kind, gen, gen.load()


def test_dataset_capabilities_are_dataset_scoped_only(dataset):
    _kind, _loader, ds = dataset
    assert ds.capabilities().caps <= DATASET_CAPS


def test_dataset_images_is_probed_not_asserted(dataset):
    """`DATASET_IMAGES` must mean "a header actually decoded", not "the manifest mentions
    images". A manifest is the attacker-supplied half of the input."""
    _kind, _loader, ds = dataset
    assert Capability.DATASET_IMAGES in ds.capabilities().caps
    assert all(s.width > 0 and s.height > 0 for s in ds.samples)


def test_samples_carry_a_content_digest_and_real_dimensions(dataset):
    _kind, _loader, ds = dataset
    assert ds.samples
    for s in ds.samples:
        assert len(s.content_sha256) == 64
        assert s.width > 0 and s.height > 0


def test_boxes_are_absolute_pixels_inside_the_image(dataset):
    """§7.2 stores COCO-style absolute pixels internally whatever the source format.
    Normalising at ingest throws away the pixel grid, and the pixel grid is exactly what
    `Evidence(kind="image_crop")` needs to cut a crop."""
    kind, _loader, ds = dataset
    seen = 0
    for s in ds.samples:
        for lb in s.labels:
            if lb.bbox is None:
                continue
            x, y, w, h = lb.bbox
            assert w > 0 and h > 0
            assert 0 <= x <= s.width and 0 <= y <= s.height
            assert x + w <= s.width + 1e-6 and y + h <= s.height + 1e-6
            assert max(w, h) > 1.5, "a box in [0,1] means normalised units leaked through"
            seen += 1
    if kind in BOXLESS_KINDS:
        assert seen == 0, "a classification-only layout has no boxes to report"
    else:
        assert seen, "the fixture must carry boxes or this asserts nothing"


def test_categories_are_dense_and_zero_based_but_remember_the_source_id(dataset):
    """The second failure mode of COCO/YOLO conversion is not arithmetic, it is category
    mapping. COCO ids are arbitrary and need not be contiguous; YOLO indices are dense and
    0-based. The loader PRODUCES that mapping, so it is data — and it has to survive into
    the report, because "class 3" means nothing if two loaders disagree about what 3 is."""
    _kind, _loader, ds = dataset
    ids = [c.category_id for c in ds.categories]
    assert ids == list(range(len(ds.categories))), "canonical ids must be dense and 0-based"
    assert all(c.source_id is not None and c.name for c in ds.categories)
    for s in ds.samples:
        for lb in s.labels:
            assert 0 <= lb.category_id < len(ds.categories)


def test_annotations_are_a_derived_view_not_a_second_store(dataset):
    """Two stores can disagree; a store and a projection cannot. `annotations()` must be
    computed from `Sample.labels` on every call, so mutating labels moves the view."""
    _kind, _loader, ds = dataset
    before = len(ds.annotations())
    assert before == sum(len(s.labels) for s in ds.samples)
    victim = next(s for s in ds.samples if s.labels)
    victim.labels = victim.labels[:-1]
    assert len(ds.annotations()) == before - 1, "annotations() is stored, not derived"


def test_contributor_is_none_when_the_tree_says_nothing(dataset):
    """`None` rather than a default is load-bearing: a default would let the contributor
    rollup produce a confident-looking single-contributor report from a dataset carrying
    no contributor data at all."""
    from cva.core.types import ContributorSource

    _kind, _loader, ds = dataset
    assert all(s.contributor is None for s in ds.samples)
    assert all(s.contributor_source in (None, ContributorSource.NONE) for s in ds.samples)
    assert Capability.DATASET_CONTRIBUTOR_META not in ds.capabilities().caps


def test_dataset_supports_accepts_its_own_layout_only(coco_tree, yolo_tree):
    from cva.loaders.datasets import COCOLoader, YOLOLoader

    assert COCOLoader().supports(coco_tree)
    assert not COCOLoader().supports(yolo_tree)
    assert YOLOLoader().supports(yolo_tree)
    assert not YOLOLoader().supports(coco_tree)
