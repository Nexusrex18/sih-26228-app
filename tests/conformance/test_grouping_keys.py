"""The batch and source grouping levels (PS §2.2.1) from the directory convention (B7).

Contributor, batch and source are three independent mechanisms, all resolved in
`cva.loaders.datasets.base` and applied by all five loaders:

- `batch_*` / `batch-*` directory        -> `Sample.batch`
- `source_*` / `source-*` directory      -> `Sample.source_meta["source"]`
- `contrib_*` directory (existing tier 2) -> `Sample.contributor`
- COCO `images[].batch` sets the batch and beats a directory; COCO `images[].source` stays the
  contributor declaration and is NOT also mapped to the source level.

Every dataset is built in `tmp_path` with real PNGs. Directory names carrying control characters
are only used with the loaders whose manifest is not XML (VOC's annotation would be invalid XML).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cva.core.types import ContributorSource, Sample
from cva.loaders.datasets import (
    COCOLoader,
    GenericDataset,
    ImageFolderLoader,
    VOCLoader,
    YOLOLoader,
    resolve_grouping,
)
from cva.loaders.datasets.base import GROUPING_MAX_LEN
from cva.risk.contributor import GROUP_KEYS, assess_groups
from cva.risk.disposition import default_policy

from .conftest import write_png, write_voc_entry

FORMATS = ["coco", "yolo", "voc", "generic"]
ALL_FORMATS = [*FORMATS, "imagefolder"]
POLICY = default_policy()
NO_META = {"file_name", "format"}

# (folder relative to the image root, batch, source, contributor)
CASES: list[tuple[str, str | None, str | None, str | None]] = [
    ("", None, None, None),
    ("plain", None, None, None),
    ("batch_a", "a", None, None),
    ("batch-b", "b", None, None),
    ("BATCH_Upper", "Upper", None, None),                 # the prefix is case-insensitive ...
    ("Batch-Mixed", "Mixed", None, None),
    ("source_scan1", None, "scan1", None),
    ("source-cam2", None, "cam2", None),
    ("Source_Up", None, "Up", None),                      # ... the remainder is kept as written
    ("batch_a/source_scan1", "a", "scan1", None),
    ("source_scan1/batch_a", "a", "scan1", None),         # order of the two does not matter
    ("contrib_x", None, None, "x"),
    ("contrib_x/batch_a/source_s1", "a", "s1", "x"),      # all three mechanisms, independent
    ("batch_outer/batch_inner", "inner", None, None),     # the component nearest the file wins
    ("source_outer/mid/source_inner", None, "inner", None),
    ("batch_o/source_o/batch_i", "i", "o", None),
    ("batch_", None, None, None),                         # an empty remainder is not a batch
    ("source-", None, None, None),
    ("batch_ok/batch_", "ok", None, None),                # ... and is passed over for the next one out
    ("batch", None, None, None),                          # no separator, no prefix
    ("batches_x", None, None, None),
    ("sourcery", None, None, None),
    ("batch_a b", "a b", None, None),
]


# --------------------------------------------------------------------------------------
# builders: the same folder layout in each format
# --------------------------------------------------------------------------------------
def _rel(i: int, folder: str) -> str:
    return f"{folder}/img{i}.png" if folder else f"img{i}.png"


def build(fmt: str, parent: Path, folders: list[str], root_name: str = "ds",
          coco_extra: dict[str, dict[str, Any]] | None = None) -> Path:
    """One image per entry of `folders`, named `img<i>.png`, inside that folder."""
    root = parent / root_name
    root.mkdir(parents=True, exist_ok=True)
    if fmt == "coco":
        images = []
        for i, f in enumerate(folders):
            write_png(root / "images" / _rel(i, f), 10 + i, 12 + i, seed=i)
            images.append({"id": i, "file_name": _rel(i, f), "width": 10 + i, "height": 12 + i,
                           **(coco_extra or {}).get(f, {})})
        (root / "instances.json").write_text(json.dumps({
            "images": images, "annotations": [], "categories": [{"id": 1, "name": "t"}]}))
    elif fmt == "yolo":
        (root / "labels").mkdir(exist_ok=True)
        for i, f in enumerate(folders):
            write_png(root / "images" / _rel(i, f), 10 + i, 12 + i, seed=i)
    elif fmt == "voc":
        for i, f in enumerate(folders):
            write_voc_entry(root, f"img{i}", 10 + i, 12 + i, [], filename=_rel(i, f))
    elif fmt == "generic":
        for i, f in enumerate(folders):
            write_png(root / _rel(i, f), 10 + i, 12 + i, seed=i)
    else:
        raise AssertionError(fmt)
    return root


def load(fmt: str, root: Path) -> list[Sample]:
    loader = {"coco": COCOLoader, "yolo": YOLOLoader, "voc": VOCLoader,
              "imagefolder": ImageFolderLoader}.get(fmt)
    ds = loader().load(root) if loader else GenericDataset(root).load()
    return ds.samples


def load_by_folder(fmt: str, parent: Path, folders: list[str], **kw: Any) -> dict[str, Sample]:
    root = build(fmt, parent, folders, **kw)
    by_stem = {Path(s.path).stem: s for s in load(fmt, root)}
    assert len(by_stem) == len(folders), "a loader dropped an image"
    return {f: by_stem[f"img{i}"] for i, f in enumerate(folders)}


def build_imagefolder(parent: Path, classes: list[str], root_name: str = "ds") -> Path:
    root = parent / root_name
    for i, cls in enumerate(classes):
        write_png(root / cls / f"img{i}.png", 10 + i, 12 + i, seed=i)
    return root


@pytest.fixture(scope="module", params=FORMATS)
def loaded(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
           ) -> tuple[str, dict[str, Sample]]:
    return request.param, load_by_folder(
        request.param, tmp_path_factory.mktemp(f"grp_{request.param}"), [c[0] for c in CASES])


# --------------------------------------------------------------------------------------
# the convention, in every loader that has a directory tree
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("folder,batch,source,contributor", CASES,
                         ids=[c[0] or "<root>" for c in CASES])
def test_directory_convention(loaded: tuple[str, dict[str, Sample]], folder: str,
                              batch: str | None, source: str | None,
                              contributor: str | None) -> None:
    fmt, samples = loaded
    s = samples[folder]
    assert s.batch == batch, fmt
    assert s.source_meta.get("source") == source, fmt
    assert s.contributor == contributor, fmt
    assert s.contributor_source == (
        ContributorSource.DIRECTORY if contributor else ContributorSource.NONE), fmt


def test_a_file_with_no_source_directory_keeps_source_meta_exactly_as_before(
        loaded: tuple[str, dict[str, Sample]]) -> None:
    """The existing shape is byte-identical: `source` is added only when a directory gave one."""
    _, samples = loaded
    assert set(samples["plain"].source_meta) == NO_META
    assert set(samples["batch_a"].source_meta) == NO_META
    assert set(samples["source_scan1"].source_meta) == NO_META | {"source"}


def test_the_root_directorys_own_name_is_never_read(
        tmp_path_factory: pytest.TempPathFactory) -> None:
    for fmt in FORMATS:
        for root_name in ("batch_root", "source_root", "contrib_root"):
            samples = load_by_folder(fmt, tmp_path_factory.mktemp(f"r_{fmt}"), ["", "plain"],
                                     root_name=root_name)
            for s in samples.values():
                assert (s.batch, s.source_meta.get("source"), s.contributor) == (None, None, None), (
                    fmt, root_name)


def test_a_prefixed_directory_inside_a_prefixed_root_is_read_and_the_root_still_is_not(
        tmp_path: Path) -> None:
    for fmt in FORMATS:
        (s,) = load_by_folder(fmt, tmp_path / fmt, ["source_s"], root_name="batch_root").values()
        assert (s.batch, s.source_meta["source"]) == (None, "s"), fmt


def test_a_sidecar_contributor_does_not_touch_batch_or_source(tmp_path: Path) -> None:
    root = build("generic", tmp_path, ["batch_a/source_s"])
    (root / "contributors.yaml").write_text("img0: vendor\n")
    (s,) = load("generic", root)
    assert s.contributor == "vendor" and s.contributor_source == ContributorSource.SIDECAR
    assert (s.batch, s.source_meta["source"]) == ("a", "s")


# --------------------------------------------------------------------------------------
# ImageFolder: the class folder is a label
# --------------------------------------------------------------------------------------
def test_imagefolder_a_class_folder_alone_is_neither_batch_nor_source(tmp_path: Path) -> None:
    samples = load("imagefolder", build_imagefolder(tmp_path, ["cats", "dogs", "batches"]))
    assert len(samples) == 3
    for s in samples:
        assert s.batch is None and "source" not in s.source_meta and s.contributor is None
        assert set(s.source_meta) == NO_META


def test_imagefolder_a_class_folder_that_carries_the_prefix_is_both_a_label_and_a_group(
        tmp_path: Path) -> None:
    root = build_imagefolder(tmp_path, ["batch_a", "source-cams", "contrib_x", "cats"])
    ds = ImageFolderLoader().load(root)
    got = {Path(s.path).parent.name: s for s in ds.samples}
    assert got["batch_a"].batch == "a" and got["batch_a"].labels[0].category_id >= 0
    assert got["source-cams"].source_meta["source"] == "cams"
    assert got["contrib_x"].contributor == "x"
    assert got["cats"].batch is None and "source" not in got["cats"].source_meta
    assert {c.name for c in ds.categories} == {"batch_a", "source-cams", "contrib_x", "cats"}


def test_imagefolder_the_root_name_is_not_read(tmp_path: Path) -> None:
    (s,) = load("imagefolder", build_imagefolder(tmp_path, ["cats"], root_name="batch_root"))
    assert s.batch is None and "source" not in s.source_meta


# --------------------------------------------------------------------------------------
# COCO: images[].batch wins over a directory; images[].source is the contributor, not the source
# --------------------------------------------------------------------------------------
def test_coco_images_batch_wins_over_a_batch_directory(tmp_path: Path) -> None:
    extra = {
        "batch_a": {"batch": "m1"},              # manifest beats directory
        "batch_b": {"batch": 7},                 # a number becomes its string
        "batch_c": {"batch": True},              # a bool is not a batch: the directory answers
        "batch_d": {"batch": ""},                # empty: the directory answers
        "batch_e": {"batch": ["x"]},             # not a scalar: the directory answers
        "batch_f": {"batch": "   "},             # blank: the directory answers
        "batch_g": {"batch": "m\x07g"},          # cleaned like a directory value
        "plain": {"batch": "solo"},              # no directory at all
        "source_s": {"batch": 2.5},
    }
    got = load_by_folder("coco", tmp_path, list(extra), coco_extra=extra)
    assert {f: s.batch for f, s in got.items()} == {
        "batch_a": "m1", "batch_b": "7", "batch_c": "c", "batch_d": "d", "batch_e": "e",
        "batch_f": "f", "batch_g": "m_g", "plain": "solo", "source_s": "2.5"}


def test_coco_images_source_is_the_contributor_and_is_not_copied_to_the_source_level(
        tmp_path: Path) -> None:
    extra = {"plain": {"source": "unit-1"},
             "source_scan": {"source": "unit-2"},
             "contrib_x": {"source": "unit-3"}}
    got = load_by_folder("coco", tmp_path, list(extra), coco_extra=extra)
    assert (got["plain"].contributor, got["plain"].contributor_source) == (
        "unit-1", ContributorSource.FORMAT_FIELD)
    assert "source" not in got["plain"].source_meta
    # a directory gives the source; the manifest field stays the contributor
    assert got["source_scan"].source_meta["source"] == "scan"
    assert got["source_scan"].contributor == "unit-2"
    # tier 2 still beats tier 3, as before
    assert (got["contrib_x"].contributor, got["contrib_x"].contributor_source) == (
        "x", ContributorSource.DIRECTORY)
    assert "source" not in got["contrib_x"].source_meta


def test_coco_images_source_alone_yields_contributor_rows_and_no_source_rows(
        tmp_path: Path) -> None:
    folders = [f"p{i}" for i in range(6)]
    extra = {f: {"source": f"unit-{i % 2}"} for i, f in enumerate(folders)}
    samples = list(load_by_folder("coco", tmp_path, folders, coco_extra=extra).values())
    rows, _ = assess_groups(samples, {s.sample_id for s in samples[:2]}, POLICY, 7)
    assert {r["group_key"] for r in rows} == {"contributor"}


# --------------------------------------------------------------------------------------
# supplier-controlled text: cleaned, trimmed, capped
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["generic", "coco", "yolo"])     # not VOC: XML forbids these
def test_control_and_format_characters_are_replaced_with_underscore(
        fmt: str, tmp_path: Path) -> None:
    got = load_by_folder(fmt, tmp_path, [
        "batch_a\x1b[31mb", "source_x\ny", "batch_‮evil", "source_a​b\tc"])
    assert got["batch_a\x1b[31mb"].batch == "a_[31mb"
    assert got["source_x\ny"].source_meta["source"] == "x_y"
    assert got["batch_‮evil"].batch == "_evil"
    assert got["source_a​b\tc"].source_meta["source"] == "a_b_c"
    for s in got.values():
        for v in (s.batch, s.source_meta.get("source")):
            assert v is None or (v.isprintable() and "\x1b" not in v)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_long_name_is_capped(fmt: str, tmp_path: Path) -> None:
    got = load_by_folder(fmt, tmp_path, ["batch_" + "x" * 200, "source_" + "y" * 200])
    assert GROUPING_MAX_LEN == 128
    assert got["batch_" + "x" * 200].batch == "x" * GROUPING_MAX_LEN
    assert got["source_" + "y" * 200].source_meta["source"] == "y" * GROUPING_MAX_LEN


def test_a_declared_coco_batch_is_capped_too(tmp_path: Path) -> None:
    extra = {"plain": {"batch": "z" * 500}}
    (s,) = load_by_folder("coco", tmp_path, ["plain"], coco_extra=extra).values()
    assert s.batch == "z" * GROUPING_MAX_LEN


def test_a_remainder_that_is_blank_after_cleaning_is_ignored(tmp_path: Path) -> None:
    got = load_by_folder("generic", tmp_path, ["batch_ok/batch_   ", "source_s/source_  "])
    assert got["batch_ok/batch_   "].batch == "ok"
    assert got["source_s/source_  "].source_meta["source"] == "s"


def test_resolve_grouping_values_are_plain_strings_and_trimmed(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    batch, source = resolve_grouping(root, root / "batch_ a " / "source_\tb\t" / "f.png")
    assert (batch, source) == ("a", "_b_")       # trimmed of whitespace; a tab inside is a control
    assert type(batch) is str and type(source) is str


def test_resolve_grouping_a_control_only_remainder_becomes_underscores_not_empty(
        tmp_path: Path) -> None:
    root = tmp_path / "ds"
    assert resolve_grouping(root, root / "batch_\n\x1b" / "f.png")[0] == "__"


def test_resolve_grouping_a_path_outside_the_root_gives_nothing(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    assert resolve_grouping(root, tmp_path / "other" / "batch_a" / "f.png") == (None, None)
    # ... but a declared batch still stands: it is the manifest's own statement
    assert resolve_grouping(root, tmp_path / "other" / "f.png", "m") == ("m", None)


def test_resolve_grouping_the_file_name_is_never_a_component(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    assert resolve_grouping(root, root / "batch_a.png") == (None, None)
    assert resolve_grouping(root, root / "source_s" / "batch_a.png") == (None, "s")


def test_resolve_grouping_declared_batch_beats_a_directory_but_not_when_unusable(
        tmp_path: Path) -> None:
    root = tmp_path / "ds"
    p = root / "batch_dir" / "source_s" / "f.png"
    assert resolve_grouping(root, p, "decl") == ("decl", "s")
    assert resolve_grouping(root, p, 3) == ("3", "s")
    assert resolve_grouping(root, p, False) == ("dir", "s")
    assert resolve_grouping(root, p, None) == ("dir", "s")
    assert resolve_grouping(root, p, {"a": 1}) == ("dir", "s")


# --------------------------------------------------------------------------------------
# end to end: the risk layer now sees all three levels
# --------------------------------------------------------------------------------------
REQUIRED_ROW_KEYS = {"group_key", "group_value", "n_samples", "n_flagged", "posterior_mean",
                     "ci_low", "ci_high"}
ALLOWED_ROW_KEYS = REQUIRED_ROW_KEYS | {"contributor_source", "excludes_cohort_rate",
                                        "excludes_reference_rate", "disposition"}


def _rows_of(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {r["group_value"]: r for r in rows if r["group_key"] == key}


@pytest.mark.parametrize("fmt", ["generic", "coco", "yolo", "voc"])
def test_assess_groups_ranks_by_contributor_batch_and_source(fmt: str, tmp_path: Path) -> None:
    folders = [f"contrib_{c}/batch_{b}/source_{s}/n{n}"
               for c, b, s in (("x", "a", "s1"), ("x", "b", "s1"), ("y", "a", "s2"), ("y", "b", "s2"))
               for n in range(3)]
    samples = list(load_by_folder(fmt, tmp_path, folders).values())
    flagged = {s.sample_id for s in samples if s.contributor == "x" and s.batch == "a"}
    assert len(flagged) == 3

    rows, perm = assess_groups(samples, flagged, POLICY, 7)

    assert {r["group_key"] for r in rows} == set(GROUP_KEYS)
    for key, values in (("contributor", {"x", "y"}), ("batch", {"a", "b"}),
                        ("source", {"s1", "s2"})):
        got = _rows_of(rows, key)
        assert set(got) == values
        assert sum(r["n_samples"] for r in got.values()) == 12
    assert {v: (r["n_samples"], r["n_flagged"]) for v, r in _rows_of(rows, "contributor").items()
            } == {"x": (6, 3), "y": (6, 0)}
    assert {v: (r["n_samples"], r["n_flagged"]) for v, r in _rows_of(rows, "batch").items()
            } == {"a": (6, 3), "b": (6, 0)}
    assert {v: (r["n_samples"], r["n_flagged"]) for v, r in _rows_of(rows, "source").items()
            } == {"s1": (6, 3), "s2": (6, 0)}
    assert perm is not None and "contributor" in perm["conclusion"]     # first key with data

    # the tier rides only on contributor rows, and it is the directory tier
    for r in rows:
        assert ("contributor_source" in r) == (r["group_key"] == "contributor")
        if r["group_key"] == "contributor":
            assert r["contributor_source"] == "directory"
        # the report row shape is unchanged
        assert REQUIRED_ROW_KEYS <= set(r) <= ALLOWED_ROW_KEYS
        assert isinstance(r["group_value"], str)
        assert type(r["n_samples"]) is int and type(r["n_flagged"]) is int
        assert 0 <= r["n_flagged"] <= r["n_samples"]
        assert 0.0 <= r["ci_low"] <= r["posterior_mean"] <= r["ci_high"] <= 1.0
        assert r["disposition"] in {"accept", "review", "quarantine"}


def test_assess_groups_with_only_batch_directories_gives_only_batch_rows(tmp_path: Path) -> None:
    folders = [f"batch_{b}/n{n}" for b in ("a", "b") for n in range(4)]
    samples = list(load_by_folder("generic", tmp_path, folders).values())
    rows, perm = assess_groups(samples, {s.sample_id for s in samples if s.batch == "a"},
                               POLICY, 7)
    assert {r["group_key"] for r in rows} == {"batch"}
    assert perm is not None and "batch" in perm["conclusion"]
