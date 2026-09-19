"""Gate B2b — every Tier 1 and Tier 2 adapter, beyond what the shared suite can say.

The gate's sentence is "every Tier 1 and Tier 2 adapter passes the same conformance suite".
That half lives in test_loaders.py, where VOC, ImageFolder, GenericDataset, SubprocessModel and
HTTPModel are extra rows of the SAME parametrised fixtures. This file holds what a shared
suite cannot: each adapter's own contract, the §5.14 controls that apply to its input (S5 on
VOC's XML, S6 on its `<filename>`), the §5.8 loopback-only rule for HTTPModel, the Keras
refusals, and the auto-detection order.

Everything is built in `tmp_path` or from the session fixtures in conftest; nothing is a
committed binary, and nothing leaves 127.0.0.1.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cva.loaders.detect as detect_mod
from cva.core.capability import Capability
from cva.core.types import ContributorSource
from cva.loaders.datasets import GenericDataset, ImageFolderLoader, VOCLoader
from cva.loaders.datasets.voc import XML_MAX_BYTES
from cva.loaders.detect import detect_and_load, load_dataset
from cva.loaders.models import (
    HTTPModel,
    KerasLoader,
    LoaderUnavailable,
    ONNXLoader,
    SubprocessModel,
)
from cva.loaders.models.http_model import require_loopback
from cva.loaders.safety import UnsafeArtifact

from .conftest import (
    INPUT_SHAPE,
    NUM_CLASSES,
    LogitsHandler,
    QuietHandler,
    write_png,
    write_voc_entry,
)


def _fmt(ds) -> set[str]:
    return {s.source_meta["format"] for s in ds.samples}


# ==========================================================================
# VOCLoader
# ==========================================================================
def test_voc_box_is_converted_from_one_based_inclusive_to_absolute_xywh(tmp_path):
    """VOC writes `xmin..xmax` 1-based and inclusive; the internal form is 0-based
    `(x, y, w, h)`. A tool that skips the `- 1` shifts every box by a pixel — invisible on a
    fixture, a real offset on every crop the report cuts."""
    root = tmp_path / "voc"
    write_voc_entry(root, "img1", 40, 50, [("tank", 11, 21, 30, 40), ("tank", 5, 5, 5, 5)])
    (sample,) = VOCLoader().load(root).samples
    big, single = sample.labels
    assert big.bbox == (10.0, 20.0, 20.0, 20.0)
    assert single.bbox == (4.0, 4.0, 1.0, 1.0), "xmin == xmax is a one-pixel box, not zero"


def test_voc_categories_are_dense_sorted_by_name_and_the_name_is_the_source_id(tmp_path):
    root = tmp_path / "voc"
    write_voc_entry(root, "a", 40, 50, [("truck", 2, 2, 10, 10), ("apc", 3, 3, 12, 12)])
    write_voc_entry(root, "b", 40, 50, [("tank", 2, 2, 10, 10)])
    ds = VOCLoader().load(root)

    assert [(c.category_id, c.name, c.source_id) for c in ds.categories] == [
        (0, "apc", "apc"), (1, "tank", "tank"), (2, "truck", "truck")]
    assert [ds.category_name(lb.category_id) for lb in ds.sample("a").labels] == ["truck", "apc"]


def test_voc_two_objects_on_one_image_are_two_labels_with_distinct_ids(tmp_path):
    root = tmp_path / "voc"
    write_voc_entry(root, "a", 40, 50, [("tank", 2, 2, 10, 10), ("tank", 12, 12, 20, 20)])
    labels = VOCLoader().load(root).sample("a").labels
    assert len(labels) == 2
    assert len({lb.label_id for lb in labels}) == 2, "evidence refs need distinct label ids"
    assert labels[0].bbox != labels[1].bbox


def test_voc_a_missing_image_is_skipped_not_fatal(tmp_path):
    """One annotation whose picture never arrived must not take the other 10,000 down."""
    root = tmp_path / "voc"
    write_voc_entry(root, "gone", 40, 50, [("tank", 2, 2, 10, 10)], with_image=False)
    write_voc_entry(root, "here", 40, 50, [("tank", 2, 2, 10, 10)])
    ds = VOCLoader().load(root)
    assert [s.sample_id for s in ds.samples] == ["here"]


def test_voc_filename_defaults_to_the_stem_and_unusable_objects_are_dropped(tmp_path):
    root = tmp_path / "voc"
    box = "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
    (root / "Annotations").mkdir(parents=True)
    write_png(root / "JPEGImages" / "a.jpg", 40, 50)
    (root / "Annotations" / "a.xml").write_text(
        "<annotation>"                                                   # no <filename>
        f"<object><name>tank</name>{box}</object>"
        "<object><name>tank</name><bndbox><xmin></xmin><ymin>1</ymin><xmax>5</xmax>"
        "<ymax>5</ymax></bndbox></object>"                               # empty coordinate
        f"<object><name/>{box}</object>"                                 # no class
        "<object><name>tank</name></object>"                             # no box
        "</annotation>")
    (sample,) = VOCLoader().load(root).samples
    assert sample.source_meta["file_name"] == "a.jpg"
    assert len(sample.labels) == 1


def test_voc_supports_needs_both_directories(tmp_path, voc_tree, coco_tree, imagefolder_tree):
    assert VOCLoader().supports(voc_tree)
    assert not VOCLoader().supports(coco_tree)
    assert not VOCLoader().supports(imagefolder_tree)
    (tmp_path / "Annotations").mkdir()
    assert not VOCLoader().supports(tmp_path), "Annotations/ without JPEGImages/ is not VOC"


def test_voc_sidecar_makes_contributor_meta_available_for_the_named_samples(tmp_path):
    root = tmp_path / "voc"
    write_voc_entry(root, "a", 40, 50, [("tank", 2, 2, 10, 10)])
    write_voc_entry(root, "b", 40, 50, [("tank", 2, 2, 10, 10)])
    (root / "contributors.yaml").write_text("a: acme\n")
    ds = VOCLoader().load(root)
    a, b = ds.sample("a"), ds.sample("b")
    assert (a.contributor, a.contributor_source) == ("acme", ContributorSource.SIDECAR)
    assert b.contributor is None
    assert Capability.DATASET_CONTRIBUTOR_META in ds.capabilities().caps


# --- VOC safety: S5 (the XML), S6 (the filename) ---------------------------
_ENTITY_DOC = '<!DOCTYPE annotation [<!ENTITY a "x">]>'

DTD_PAYLOADS = {
    "doctype_with_entity": f'<?xml version="1.0"?>{_ENTITY_DOC}'
                           "<annotation><filename>&a;</filename></annotation>",
    "doctype_external": '<!DOCTYPE annotation SYSTEM "http://127.0.0.1:1/x.dtd"><annotation/>',
    "bare_entity": '<annotation><!ENTITY a "x"></annotation>',
    "mixed_case": '<!DocType annotation [<!EnTiTy a "x">]><annotation/>',
    "billion_laughs": (
        '<!DOCTYPE l [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">'
        '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;">]><annotation><filename>&c;</filename>'
        "</annotation>"),
}


def _voc_with_raw_xml(tmp_path: Path, payload: str | bytes) -> Path:
    root = tmp_path / "voc"
    write_voc_entry(root, "a", 40, 50, [("tank", 2, 2, 10, 10)])
    data = payload.encode() if isinstance(payload, str) else payload
    (root / "Annotations" / "a.xml").write_bytes(data)
    return root


@pytest.mark.parametrize("payload", list(DTD_PAYLOADS.values()), ids=list(DTD_PAYLOADS))
def test_voc_an_annotation_with_a_dtd_or_entity_is_refused_by_s5(tmp_path, payload):
    root = _voc_with_raw_xml(tmp_path, payload)
    with pytest.raises(UnsafeArtifact) as ei:
        VOCLoader().load(root)
    assert ei.value.control == "S5"


def test_voc_a_dtd_pushed_past_the_first_64k_is_still_refused_by_s5(tmp_path):
    """The scan is a prefix scan, and a prefix is exactly what an attacker pads past.
    XML allows whitespace before the DOCTYPE, so 70 KB of it must not buy a DTD through."""
    payload = " " * 70_000 + _ENTITY_DOC + (
        "<annotation><filename>a.png</filename></annotation>")
    root = _voc_with_raw_xml(tmp_path, payload)
    with pytest.raises(UnsafeArtifact) as ei:
        VOCLoader().load(root)
    assert ei.value.control == "S5"


def test_voc_an_annotation_over_the_size_cap_is_refused_before_it_is_parsed(tmp_path):
    root = _voc_with_raw_xml(
        tmp_path, b"<annotation>" + b" " * XML_MAX_BYTES + b"</annotation>")
    assert (root / "Annotations" / "a.xml").stat().st_size > XML_MAX_BYTES
    with pytest.raises(UnsafeArtifact) as ei:
        VOCLoader().load(root)
    assert ei.value.control == "S5"
    assert "cap" in ei.value.reason


def test_voc_an_annotation_exactly_at_the_cap_still_loads(tmp_path):
    """The cap is a ceiling, not an off-by-one that rejects the largest legal file."""
    head = b"<annotation><filename>a.png</filename></annotation>"
    root = _voc_with_raw_xml(tmp_path, head + b" " * (XML_MAX_BYTES - len(head)))
    assert (root / "Annotations" / "a.xml").stat().st_size == XML_MAX_BYTES
    assert [s.sample_id for s in VOCLoader().load(root).samples] == ["a"]


@pytest.mark.parametrize("filename", [
    "../../etc/passwd",
    "/etc/passwd",
    "../JPEGImages_evil/a.png",       # a sibling that merely shares the directory-name prefix
], ids=["traversal", "absolute", "prefix_sibling"])
def test_voc_a_filename_escaping_the_dataset_is_refused_by_s6(tmp_path, filename):
    root = tmp_path / "voc"
    write_voc_entry(root, "a", 40, 50, [("tank", 2, 2, 10, 10)], filename=filename,
                    with_image=False)
    write_png(root / "JPEGImages_evil" / "a.png", 10, 10)      # the sibling is a real image
    with pytest.raises(UnsafeArtifact) as ei:
        VOCLoader().load(root)
    assert ei.value.control == "S6"


def test_voc_a_symlink_planted_in_the_dataset_is_refused_by_s6(tmp_path):
    outside = tmp_path / "outside.png"
    write_png(outside, 10, 10)
    root = tmp_path / "voc"
    write_voc_entry(root, "a", 40, 50, [("tank", 2, 2, 10, 10)], filename="link.png",
                    with_image=False)
    os.symlink(outside, root / "JPEGImages" / "link.png")
    with pytest.raises(UnsafeArtifact) as ei:
        VOCLoader().load(root)
    assert ei.value.control == "S6"


def test_voc_malformed_xml_is_a_value_error_not_an_unsafe_artifact(tmp_path):
    """A truncated download and an attack are different events and the report treats them
    differently, so a parse failure must not be dressed as a control firing."""
    root = _voc_with_raw_xml(tmp_path, "<annotation><object>")
    with pytest.raises(ValueError, match="not valid XML") as ei:
        VOCLoader().load(root)
    assert not isinstance(ei.value, UnsafeArtifact)


# ==========================================================================
# ImageFolderLoader
# ==========================================================================
def test_imagefolder_yields_one_boxless_label_per_image_from_the_class_folder(imagefolder_tree):
    ds = ImageFolderLoader().load(imagefolder_tree)
    assert [(c.category_id, c.name, c.source_id) for c in ds.categories] == [
        (0, "cats", "cats"), (1, "dogs", "dogs")]
    assert sorted(s.sample_id for s in ds.samples) == ["cats/c1", "cats/c2", "dogs/d1"]
    for s in ds.samples:
        (label,) = s.labels
        assert label.bbox is None
        assert ds.category_name(label.category_id) == s.sample_id.split("/")[0]
    assert Capability.DATASET_LABELS in ds.capabilities().caps


def test_imagefolder_class_folder_is_a_label_never_a_contributor(imagefolder_tree):
    ds = ImageFolderLoader().load(imagefolder_tree)
    for s in ds.samples:
        assert s.contributor is None
        assert s.contributor_source == ContributorSource.NONE
    assert Capability.DATASET_CONTRIBUTOR_META not in ds.capabilities().caps


def test_imagefolder_a_contrib_folder_is_a_contributor_as_well_as_a_class(tmp_path):
    root = tmp_path / "folders"
    write_png(root / "contrib_vendor" / "x.png", 20, 20)
    write_png(root / "cats" / "y.png", 20, 20)
    write_png(root / "contrib_" / "z.png", 20, 20)     # a bare prefix names nobody
    ds = ImageFolderLoader().load(root)
    x, y, z = ds.sample("contrib_vendor/x"), ds.sample("cats/y"), ds.sample("contrib_/z")

    assert (x.contributor, x.contributor_source) == ("vendor", ContributorSource.DIRECTORY)
    assert ds.category_name(x.labels[0].category_id) == "contrib_vendor"
    assert y.contributor is None and y.contributor_source == ContributorSource.NONE
    assert z.contributor is None
    assert Capability.DATASET_CONTRIBUTOR_META in ds.capabilities().caps


def test_imagefolder_ignores_hidden_folders_and_non_images(tmp_path):
    root = tmp_path / "folders"
    write_png(root / "cats" / "c.png", 20, 20)
    write_png(root / ".hidden" / "h.png", 20, 20)
    (root / "cats" / "notes.txt").write_text("not an image")
    ds = ImageFolderLoader().load(root)
    assert [s.sample_id for s in ds.samples] == ["cats/c"]
    assert [c.name for c in ds.categories] == ["cats"]


def test_imagefolder_supports_is_false_without_class_folders(tmp_path, imagefolder_tree):
    assert ImageFolderLoader().supports(imagefolder_tree)

    (tmp_path / "empty").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.txt").write_text("hello")
    write_png(tmp_path / "loose" / "a.png", 20, 20)        # images, but no class folder
    write_png(tmp_path / "hidden" / ".secret" / "a.png", 20, 20)
    for d in ("empty", "docs", "loose", "hidden"):
        assert not ImageFolderLoader().supports(tmp_path / d), d
    assert not ImageFolderLoader().supports(tmp_path / "does-not-exist")


# ==========================================================================
# GenericDataset
# ==========================================================================
def _flat(tmp_path: Path, names=("a.png", "b.png")) -> Path:
    root = tmp_path / "flat"
    for i, n in enumerate(names):
        write_png(root / n, 30 + i, 20, seed=i)
    return root


def test_generic_label_fn_returning_a_class_name_labels_the_whole_image(tmp_path):
    root = _flat(tmp_path)
    ds = GenericDataset(root, label_fn=lambda p: "cat" if p.stem == "a" else "dog").load()
    assert [c.name for c in ds.categories] == ["cat", "dog"]
    for s in ds.samples:
        (label,) = s.labels
        assert label.bbox is None
        assert ds.category_name(label.category_id) == ("cat" if s.sample_id == "a.png" else "dog")


def test_generic_label_fn_returning_pairs_gives_boxes_and_boxless_labels(tmp_path):
    root = _flat(tmp_path, ("a.png",))
    ds = GenericDataset(
        root, label_fn=lambda p: [("tank", (1, 2, 3, 4)), ("apc", None)]).load()
    (sample,) = ds.samples
    tank, apc = sample.labels
    assert tank.bbox == (1.0, 2.0, 3.0, 4.0)
    assert all(isinstance(v, float) for v in tank.bbox)
    assert apc.bbox is None
    assert (tank.label_id, apc.label_id) == ("a.png:0", "a.png:1")
    assert ds.category_name(tank.category_id) == "tank"


def test_generic_contributor_fn_is_recorded_as_a_format_field(tmp_path):
    root = _flat(tmp_path)
    ds = GenericDataset(
        root, contributor_fn=lambda p: "acme" if p.stem == "a" else None).load()
    a, b = ds.sample("a.png"), ds.sample("b.png")
    assert (a.contributor, a.contributor_source) == ("acme", ContributorSource.FORMAT_FIELD)
    assert (b.contributor, b.contributor_source) == (None, ContributorSource.NONE)
    assert Capability.DATASET_CONTRIBUTOR_META in ds.capabilities().caps


def test_generic_without_a_label_fn_carries_no_labels_and_says_so(tmp_path):
    ds = GenericDataset(_flat(tmp_path)).load()
    assert len(ds) == 2
    assert all(s.labels == [] for s in ds.samples)
    caps = ds.capabilities()
    assert Capability.DATASET_IMAGES in caps.caps
    assert Capability.DATASET_LABELS not in caps.caps
    assert caps.note_for(Capability.DATASET_LABELS)


def test_generic_is_never_auto_detected(tmp_path):
    root = _flat(tmp_path)
    gen = GenericDataset(root, label_fn=lambda p: "cat")
    assert gen.supports(root) is False
    assert gen.supports(tmp_path) is False
    with pytest.raises(ValueError, match="no dataset loader supports"):
        load_dataset(root)          # loose images, nothing else claims them


def test_generic_walks_subfolders_and_resolves_the_contributor_precedence(tmp_path):
    """sidecar > contrib_ directory > the caller's own declaration — the same order the
    format loaders use, so a hand-written adapter cannot outrank independent evidence."""
    root = tmp_path / "flat"
    write_png(root / "contrib_acme" / "x.png", 20, 20)
    write_png(root / "contrib_acme" / "y.png", 20, 20)
    write_png(root / "z.png", 20, 20)
    (root / "contributors.yaml").write_text("x: sidecar-co\n")
    ds = GenericDataset(root, contributor_fn=lambda p: "declared").load()

    x, y, z = (ds.sample(i) for i in ("contrib_acme/x.png", "contrib_acme/y.png", "z.png"))
    assert (x.contributor, x.contributor_source) == ("sidecar-co", ContributorSource.SIDECAR)
    assert (y.contributor, y.contributor_source) == ("acme", ContributorSource.DIRECTORY)
    assert (z.contributor, z.contributor_source) == ("declared", ContributorSource.FORMAT_FIELD)


def test_generic_load_accepts_an_explicit_path_and_a_none_from_label_fn(tmp_path):
    root = _flat(tmp_path)
    ds = GenericDataset(tmp_path / "elsewhere", label_fn=lambda p: None).load(root)
    assert len(ds) == 2 and all(s.labels == [] for s in ds.samples)


# ==========================================================================
# auto-detection: the order is the contract
# ==========================================================================
def test_detect_coco_wins_even_though_its_images_dir_looks_like_a_class_folder(coco_tree):
    assert ImageFolderLoader().supports(coco_tree), "the trap: ImageFolder also claims this"
    assert _fmt(load_dataset(coco_tree)) == {"coco"}


def test_detect_yolo_wins_over_imagefolder(yolo_tree):
    assert ImageFolderLoader().supports(yolo_tree)
    assert _fmt(load_dataset(yolo_tree)) == {"yolo"}


def test_detect_voc_wins_over_imagefolder(voc_tree):
    assert ImageFolderLoader().supports(voc_tree)
    assert _fmt(load_dataset(voc_tree)) == {"voc"}


def test_detect_falls_through_to_imagefolder_last(imagefolder_tree):
    assert _fmt(load_dataset(imagefolder_tree)) == {"imagefolder"}


def test_detect_a_manifest_beside_class_folders_resolves_to_the_format_with_markers(tmp_path):
    root = tmp_path / "mixed"
    write_png(root / "images" / "a.png", 40, 20)
    write_png(root / "cats" / "x.png", 20, 20)
    (root / "instances.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 40, "height": 20}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 8]}],
        "categories": [{"id": 1, "name": "tank"}]}))
    ds = load_dataset(root)
    assert _fmt(ds) == {"coco"}
    assert [s.sample_id for s in ds.samples] == ["1"]


@pytest.mark.parametrize("manifest", ['{"note": "not coco"}', "[]", "{not json"],
                         ids=["wrong_shape", "wrong_type", "corrupt"])
def test_detect_a_manifest_that_is_not_coco_does_not_capture_the_directory(tmp_path, manifest):
    root = tmp_path / "mixed"
    write_png(root / "cats" / "x.png", 20, 20)
    (root / "annotations.json").write_text(manifest)
    assert _fmt(load_dataset(root)) == {"imagefolder"}


def test_detect_an_unrecognised_directory_raises_value_error(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.txt").write_text("hello")
    for d in ("empty", "docs"):
        with pytest.raises(ValueError, match="no dataset loader supports"):
            load_dataset(tmp_path / d)


# ==========================================================================
# InMemoryDataset — the API Module A relies on
# ==========================================================================
def test_inmemory_dataset_len_sample_and_category_name(voc_tree):
    ds = VOCLoader().load(voc_tree)
    assert len(ds) == len(ds.samples) == 3
    first = ds.samples[0]
    assert ds.sample(first.sample_id) is first
    assert ds.category_name(0) == "apc"
    assert ds.category_name(12345) == "12345", "an unknown id falls back to its own text"
    with pytest.raises(KeyError):
        ds.sample("no-such-sample")


def test_inmemory_dataset_sample_stays_correct_after_samples_are_appended(voc_tree):
    import dataclasses

    ds = VOCLoader().load(voc_tree)
    first = ds.samples[0]
    assert ds.sample(first.sample_id) is first            # builds the index
    added = dataclasses.replace(first, sample_id="appended-later")
    ds.samples.append(added)

    assert len(ds) == 4
    assert ds.sample("appended-later") is added
    assert ds.sample(first.sample_id) is first, "the old entries must survive the rebuild"


# ==========================================================================
# black-box models: shared helpers
# ==========================================================================
def _script(tmp_path: Path, body: str, name: str = "child.py") -> list[str]:
    p = tmp_path / name
    p.write_text(body)
    return [sys.executable, str(p)]


def _assert_query_only(h) -> None:
    caps = h.capabilities()
    assert Capability.MODEL_PREDICT in caps.caps
    assert Capability.MODEL_LOGITS in caps.caps
    for cap in (Capability.MODEL_WEIGHTS, Capability.MODEL_ACTIVATIONS,
                Capability.MODEL_GRADIENTS, Capability.MODEL_ARCHITECTURE):
        assert cap not in caps.caps
        assert caps.note_for(cap), f"{cap} is absent with no recorded reason"
    assert h.weight_digest() == "unavailable:black-box"
    assert (h.num_classes, tuple(h.input_shape)) == (NUM_CLASSES, INPUT_SHAPE)


def _assert_distribution(h) -> None:
    x = np.ones(INPUT_SHAPE, dtype=np.float32)
    p = h.predict(x)
    assert p.shape == (1, NUM_CLASSES)
    assert np.isclose(p.sum(), 1.0, atol=1e-5)
    # The endpoint computes logits [mean, -mean] = [1, -1]; the ordering proves the tensor
    # actually crossed the boundary and came back, rather than a constant being returned.
    assert p[0, 0] > p[0, 1]
    assert np.allclose(h.logits(x), [[1.0, -1.0]])


# ==========================================================================
# SubprocessModel
# ==========================================================================
@pytest.fixture(scope="module")
def sub_model(child_command):
    return SubprocessModel(child_command, "child", INPUT_SHAPE, NUM_CLASSES)


def test_subprocess_model_reports_query_only_capabilities(sub_model):
    assert sub_model.fmt == "subprocess"
    _assert_query_only(sub_model)


def test_subprocess_model_predict_returns_a_distribution(sub_model):
    _assert_distribution(sub_model)


def test_subprocess_model_sends_float32_batches_over_stdin(tmp_path):
    log = tmp_path / "seen.txt"
    body = """\
import io
import sys

import numpy as np

x = np.load(io.BytesIO(sys.stdin.buffer.read()), allow_pickle=False)
with open(sys.argv[1], "a") as fh:
    fh.write(f"{x.dtype} {x.shape}\\n")
m = x.reshape(len(x), -1).mean(axis=1)
out = io.BytesIO()
np.save(out, np.stack([m, -m], axis=1).astype(np.float32), allow_pickle=False)
sys.stdout.buffer.write(out.getvalue())
"""
    h = SubprocessModel([*_script(tmp_path, body), str(log)], "child", INPUT_SHAPE, NUM_CLASSES)
    h.predict(np.zeros(INPUT_SHAPE, dtype=np.float64))                # one image, wrong dtype
    h.predict(np.zeros((4, *INPUT_SHAPE), dtype=np.float64))          # a batch
    assert log.read_text().splitlines() == [
        "float32 (1, 3, 8, 8)",       # the construction probe
        "float32 (1, 3, 8, 8)",       # a lone image gains a batch axis, dtype forced to f32
        "float32 (4, 3, 8, 8)"]


def test_subprocess_a_pickled_object_array_is_rejected_and_never_executed(tmp_path):
    """The process on the other end is the thing under audit, so its stdout is untrusted
    bytes. A `.npy` holding an object array is a pickle; loading it runs its `__reduce__`."""
    sentinel = tmp_path / "pwned"
    body = f"""\
import io
import sys

import numpy as np


class Evil:
    def __reduce__(self):
        return (open, ({str(sentinel)!r}, "w"))


sys.stdin.buffer.read()
out = io.BytesIO()
np.save(out, np.array([Evil()], dtype=object), allow_pickle=True)
sys.stdout.buffer.write(out.getvalue())
"""
    cmd = _script(tmp_path, body)

    # Non-vacuity: prove the payload is live by unpickling it once, deliberately, here. It is
    # self-authored and only creates an empty file inside tmp_path.
    raw = subprocess.run(cmd, input=b"", capture_output=True, check=True).stdout
    np.load(io.BytesIO(raw), allow_pickle=True)
    assert sentinel.exists(), "the payload must be live or this test proves nothing"
    sentinel.unlink()

    h = SubprocessModel(cmd, "evil", INPUT_SHAPE, NUM_CLASSES)
    assert not sentinel.exists(), "the adapter unpickled the child's output"
    caps = h.capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert "pickle" in (caps.note_for(Capability.MODEL_PREDICT) or "")
    with pytest.raises(ValueError):
        h.predict(np.zeros(INPUT_SHAPE, dtype=np.float32))
    assert not sentinel.exists()


def test_subprocess_a_command_that_exits_nonzero_leaves_predict_absent_without_raising(tmp_path):
    cmd = _script(tmp_path, "import sys\nsys.stderr.write('boom: model crashed\\n')\nsys.exit(3)\n")
    h = SubprocessModel(cmd, "bad", INPUT_SHAPE, NUM_CLASSES)          # must not raise
    caps = h.capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    note = caps.note_for(Capability.MODEL_PREDICT) or ""
    assert "exited 3" in note and "boom" in note


def test_subprocess_a_command_that_never_answers_is_bounded_by_the_timeout(tmp_path):
    cmd = _script(tmp_path, "import time\ntime.sleep(30)\n")
    h = SubprocessModel(cmd, "hang", INPUT_SHAPE, NUM_CLASSES, timeout_s=0.5)
    caps = h.capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert "timed out" in (caps.note_for(Capability.MODEL_PREDICT) or "")


def test_subprocess_output_over_the_cap_is_refused(child_command, monkeypatch):
    monkeypatch.setattr("cva.loaders.models.subprocess_model.MAX_OUTPUT_BYTES", 10)
    h = SubprocessModel(child_command, "big", INPUT_SHAPE, NUM_CLASSES)
    caps = h.capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert "cap" in (caps.note_for(Capability.MODEL_PREDICT) or "")


@pytest.mark.parametrize("command", [[], ()], ids=["list", "tuple"])
def test_subprocess_an_empty_command_is_a_value_error(command):
    with pytest.raises(ValueError, match="non-empty"):
        SubprocessModel(command, "x", INPUT_SHAPE, NUM_CLASSES)


def test_subprocess_a_command_is_argv_and_never_a_shell_line(tmp_path, child_command):
    sentinel = tmp_path / "shell-ran"

    # Shell metacharacters in an ARGUMENT are just bytes for the child to ignore.
    h = SubprocessModel([*child_command, f"; touch {sentinel}"], "x", INPUT_SHAPE, NUM_CLASSES)
    assert Capability.MODEL_PREDICT in h.capabilities().caps
    assert not sentinel.exists()

    # A whole command line as one list element is one program NAME, and no such program exists.
    h = SubprocessModel(["echo hello"], "x", INPUT_SHAPE, NUM_CLASSES)
    caps = h.capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert "echo hello" in (caps.note_for(Capability.MODEL_PREDICT) or "")

    # A bare string is not run through a shell either — it must never reach `touch`.
    h = SubprocessModel(f"touch {sentinel}", "x", INPUT_SHAPE, NUM_CLASSES)   # type: ignore[arg-type]
    assert Capability.MODEL_PREDICT not in h.capabilities().caps
    assert not sentinel.exists()


# ==========================================================================
# HTTPModel — §5.8, loopback only
# ==========================================================================
def _http(port: int, host: str = "127.0.0.1", **kw) -> HTTPModel:
    return HTTPModel(f"http://{host}:{port}/predict", "endpoint", INPUT_SHAPE, NUM_CLASSES, **kw)


def _counting_handler():
    """A handler that answers everything and records what it was asked — the spy that lets a
    test say "this server was never contacted" instead of inferring it from a timeout."""
    hits: list[str] = []

    class Spy(QuietHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.reply(200, b"{}")

        def do_POST(self) -> None:
            self.read_body()
            hits.append(self.path)
            self.reply(200, b"{}")

    return Spy, hits


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_http_model_works_against_loopback_by_address_and_by_name(serve, host):
    h = _http(serve(LogitsHandler), host)
    assert h.fmt == "http"
    _assert_query_only(h)
    _assert_distribution(h)


def test_http_model_posts_the_documented_json_shape(serve):
    seen: dict = {}

    class Recording(LogitsHandler):
        def read_body(self) -> bytes:
            body = super().read_body()
            seen["body"], seen["ctype"] = body, self.headers.get("Content-Type")
            return body

    _http(serve(Recording))
    assert seen["ctype"] == "application/json"
    inputs = np.asarray(json.loads(seen["body"])["inputs"])
    assert inputs.shape == (1, *INPUT_SHAPE)


@pytest.mark.parametrize("url", [
    "http://example.com/predict",
    "http://10.0.0.5:8000/",
    "http://192.0.2.1/",
    "ftp://127.0.0.1/",
    "file:///etc/passwd",
    "http:///no-host",
    "http://127.0.0.1.evil.example/",         # a name that merely STARTS like loopback
    "http://localhost.evil.example/",
    "http://127.0.0.1@evil.example/",         # userinfo trick: the host is evil.example
    "http://[::ffff:8.8.8.8]/",
])
def test_http_non_loopback_urls_are_rejected_before_any_connection(url):
    with pytest.raises(ValueError):
        require_loopback(url)
    with pytest.raises(ValueError):
        HTTPModel(url, "x", INPUT_SHAPE, NUM_CLASSES)


@pytest.mark.parametrize("url", [
    "http://[::1]:1/", "http://127.0.0.2:1/", "http://127.255.255.254/",
    "http://localhost:1/", "http://LOCALHOST:1/", "https://127.0.0.1:1/",
])
def test_http_loopback_urls_are_accepted(url):
    assert require_loopback(url) == url


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_http_a_redirect_is_refused_and_never_followed(serve, code):
    """A loopback server answering `3xx -> elsewhere` would otherwise walk the audited data
    off the machine. The spy is on a different origin: it must record nothing."""
    spy_cls, hits = _counting_handler()
    spy_port = serve(spy_cls)

    class Redirect(QuietHandler):
        def do_POST(self) -> None:
            self.read_body()
            self.send_response(code)
            self.send_header("Location", f"http://localhost:{spy_port}/steal")
            self.send_header("Content-Length", "0")
            self.end_headers()

    caps = _http(serve(Redirect)).capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert "redirect" in (caps.note_for(Capability.MODEL_PREDICT) or "").lower()
    assert hits == [], "the redirect target was contacted"


def test_http_proxy_environment_is_ignored(serve, monkeypatch):
    """An `HTTP_PROXY` in the environment would carry a loopback request to a remote proxy."""
    spy_cls, hits = _counting_handler()
    spy = serve(spy_cls)
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, f"http://127.0.0.1:{spy}")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    h = _http(serve(LogitsHandler), timeout_s=5)
    assert Capability.MODEL_PREDICT in h.capabilities().caps
    _assert_distribution(h)
    assert hits == [], "the request was routed through the proxy"


def test_http_an_unroutable_proxy_in_the_environment_changes_nothing(serve, monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://10.255.255.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    h = _http(serve(LogitsHandler), timeout_s=3)
    assert Capability.MODEL_PREDICT in h.capabilities().caps


def test_http_an_oversized_response_is_refused(serve, monkeypatch):
    monkeypatch.setattr("cva.loaders.models.http_model.MAX_RESPONSE_BYTES", 1024)

    class Big(QuietHandler):
        def do_POST(self) -> None:
            self.read_body()
            self.reply(200, b'{"outputs": [[' + b"0," * 2000 + b"0]]}")

    caps = _http(serve(Big)).capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert "exceeds" in (caps.note_for(Capability.MODEL_PREDICT) or "")

    # Control: under the same lowered cap a normal-sized answer is still accepted, so the
    # refusal above is about SIZE and not about the cap having broken everything.
    assert Capability.MODEL_PREDICT in _http(serve(LogitsHandler)).capabilities().caps


def _http_500(serve):
    class Broken(QuietHandler):
        def do_POST(self) -> None:
            self.read_body()
            self.reply(500, b"internal error", "text/plain")
    return _http(serve(Broken))


def _http_not_json(serve):
    class Html(QuietHandler):
        def do_POST(self) -> None:
            self.read_body()
            self.reply(200, b"<html>login</html>", "text/html")
    return _http(serve(Html))


def _http_no_outputs(serve):
    class Wrong(QuietHandler):
        def do_POST(self) -> None:
            self.read_body()
            self.reply(200, b'{"nope": 1}')
    return _http(serve(Wrong))


@pytest.fixture(params=["subprocess_exit", "http_500", "http_not_json", "http_no_outputs"])
def failing_model(request, tmp_path, serve):
    if request.param == "subprocess_exit":
        cmd = _script(tmp_path, "import sys\nsys.exit(2)\n")
        return SubprocessModel(cmd, "bad", INPUT_SHAPE, NUM_CLASSES)
    return {"http_500": _http_500, "http_not_json": _http_not_json,
            "http_no_outputs": _http_no_outputs}[request.param](serve)


def test_a_failing_black_box_leaves_predict_absent_with_a_reason_instead_of_raising(failing_model):
    caps = failing_model.capabilities()
    assert Capability.MODEL_PREDICT not in caps.caps
    assert caps.note_for(Capability.MODEL_PREDICT)


def test_a_failing_black_box_still_explains_every_absent_capability(failing_model):
    """The shared contract: an absent MODEL_* capability always names why."""
    from cva.core.capability import MODEL_CAPS

    caps = failing_model.capabilities()
    for cap in MODEL_CAPS - caps.caps:
        assert caps.note_for(cap), f"{cap} is absent with no recorded reason"


# ==========================================================================
# KerasLoader
# ==========================================================================
def _patch_find_spec(monkeypatch, missing=(), present=()):
    """Make tensorflow/tf2onnx look installed or not, whatever this environment has."""
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name in missing:
            return None
        if name in present:
            return object()
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def _savedmodel(tmp_path: Path) -> Path:
    sm = tmp_path / "sm"
    sm.mkdir()
    (sm / "saved_model.pb").write_bytes(b"\x00")
    return sm


def test_keras_supports_savedmodel_dirs_keras_and_h5_files(tmp_path):
    k = KerasLoader()
    assert k.supports(_savedmodel(tmp_path))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not k.supports(empty), "a directory without saved_model.pb is not a SavedModel"
    for name in ("m.keras", "m.h5", "m.hdf5", "M.KERAS"):
        (tmp_path / name).write_bytes(b"x")
        assert k.supports(tmp_path / name), name
    for name in ("m.onnx", "m.pt", "m.pb", "saved_model.pb"):
        (tmp_path / name).write_bytes(b"x")
        assert not k.supports(tmp_path / name), name


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".H5"])
@pytest.mark.parametrize("deps", ["missing", "present"])
def test_keras_legacy_hdf5_is_refused_by_s2_whatever_is_installed(
        tmp_path, monkeypatch, suffix, deps):
    """HDF5 can embed a pickled Lambda layer that runs on load. The refusal must not be a
    side effect of TensorFlow being absent: with both packages "installed" it still fires."""
    if deps == "missing":
        _patch_find_spec(monkeypatch, missing=("tensorflow", "tf2onnx"))
    else:
        _patch_find_spec(monkeypatch, present=("tensorflow", "tf2onnx"))
    p = tmp_path / f"model{suffix}"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    with pytest.raises(UnsafeArtifact) as ei:
        KerasLoader().load(p)
    assert ei.value.control == "S2"


@pytest.mark.parametrize("kind", ["savedmodel_dir", "keras_file"])
def test_keras_without_tensorflow_raises_loader_unavailable_naming_both_packages(
        tmp_path, monkeypatch, kind):
    _patch_find_spec(monkeypatch, missing=("tensorflow", "tf2onnx"))
    if kind == "savedmodel_dir":
        p = _savedmodel(tmp_path)
    else:
        p = tmp_path / "m.keras"
        p.write_bytes(b"PK")
    with pytest.raises(LoaderUnavailable) as ei:
        KerasLoader().load(p)
    assert "tensorflow and tf2onnx are not installed" in str(ei.value)
    assert isinstance(ei.value, RuntimeError)
    assert not isinstance(ei.value, UnsafeArtifact), "an environment gap is not an attack"


def test_keras_names_only_the_package_that_is_actually_missing(tmp_path, monkeypatch):
    _patch_find_spec(monkeypatch, missing=("tf2onnx",), present=("tensorflow",))
    with pytest.raises(LoaderUnavailable) as ei:
        KerasLoader().load(_savedmodel(tmp_path))
    msg = str(ei.value)
    assert "tf2onnx is not installed" in msg
    assert "tensorflow is not installed" not in msg and "tensorflow and" not in msg


# ==========================================================================
# detect_and_load
# ==========================================================================
def test_detect_routes_a_savedmodel_dir_to_keras_without_prescanning_a_directory(
        tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the S1 prescan hashes ONE FILE and must not see a directory")

    monkeypatch.setattr(detect_mod, "prescan", boom)
    _patch_find_spec(monkeypatch, missing=("tensorflow", "tf2onnx"))
    with pytest.raises(LoaderUnavailable):
        detect_and_load(_savedmodel(tmp_path))


def test_detect_a_directory_that_is_not_a_savedmodel_is_unsupported(tmp_path):
    (tmp_path / "junk").mkdir()
    with pytest.raises(ValueError, match="no loader supports"):
        detect_and_load(tmp_path / "junk")


def test_detect_a_legacy_h5_file_reaches_the_keras_refusal(tmp_path):
    p = tmp_path / "model.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    with pytest.raises(UnsafeArtifact) as ei:
        detect_and_load(p)
    assert ei.value.control == "S2"


def test_detect_still_loads_an_onnx_file_and_attaches_the_s1_report(tmp_path):
    from cva.fixtures import build_model

    path = build_model(tmp_path / "m.onnx")
    handle = detect_and_load(path)

    assert handle.fmt == "onnx"
    assert Capability.MODEL_PREDICT in handle.capabilities().caps
    assert handle.safety.safe_to_load is True
    assert handle.safety.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_detect_still_refuses_a_file_that_fails_the_s1_prescan(tmp_path, monkeypatch):
    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"\x08\x07this is not a serialised ONNX graph")

    def boom(*a, **k):
        raise AssertionError("the loader must not be reached for an unsafe file")

    monkeypatch.setattr(ONNXLoader, "load", boom)
    with pytest.raises(RuntimeError, match="refusing to load broken.onnx"):
        detect_and_load(bad)
    # Contrast: with the guard off the same file DOES reach the loader, so the refusal above
    # is the prescan's doing and not a coincidence of the stub.
    with pytest.raises(AssertionError, match="loader must not be reached"):
        detect_and_load(bad, enforce_safety=False)
