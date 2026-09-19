"""Gate B2a's round trip: a COCO dataset survives a trip through YOLO and back.

The gate names two things that must be preserved, and they fail differently.

**Boxes** fail by arithmetic. The conversion is a centre/corner change AND a
normalise/denormalise pair, so a transposed width and height, or a centre computed before
the scale instead of after, produces boxes that look plausible and are wrong. The fixture
uses three images at DIFFERENT sizes precisely so a width/height swap cannot survive.

**Categories** fail by mapping, and that is the one that would actually ship. COCO ids are
arbitrary and need not be contiguous; YOLO indices are dense and zero-based. A loader that
passes the source id straight through round-trips its own output perfectly and corrupts
everyone else's data — so the assertion here is on the SOURCE id, not the dense index.
"""
from __future__ import annotations

import pytest

from cva.loaders.datasets import COCOLoader, YOLOLoader, to_yolo

# YOLO text carries 6 decimals, so the trip is lossy by construction. The tolerance is in
# PIXELS and is far tighter than anything that could hide a real conversion bug: at these
# image sizes a transposed axis moves a box by tens of pixels, not thousandths.
BBOX_TOL_PX = 1e-3


@pytest.fixture
def round_tripped(coco_tree, tmp_path):
    original = COCOLoader().load(coco_tree)
    exported = to_yolo(original, tmp_path / "as_yolo")
    return original, YOLOLoader().load(exported)


def test_coco_to_yolo_and_back_preserves_every_sample(round_tripped):
    original, back = round_tripped
    assert len(back.samples) == len(original.samples)
    # Matched by file identity, not by sample_id: COCO's id is `images[].id` and YOLO's is
    # the filename stem, so requiring the ids to match would test the export's naming
    # convention rather than the data.
    assert ({s.content_sha256 for s in back.samples}
            == {s.content_sha256 for s in original.samples})


def test_coco_to_yolo_and_back_preserves_bbox_geometry(round_tripped):
    original, back = round_tripped
    by_digest = {s.content_sha256: s for s in back.samples}
    compared = 0
    for src in original.samples:
        dst = by_digest[src.content_sha256]
        assert (dst.width, dst.height) == (src.width, src.height)
        assert len(dst.labels) == len(src.labels)
        for a, b in zip(sorted(src.labels, key=lambda z: z.bbox),
                        sorted(dst.labels, key=lambda z: z.bbox), strict=True):
            for u, v in zip(a.bbox, b.bbox, strict=True):
                assert abs(u - v) < BBOX_TOL_PX, f"{a.bbox} != {b.bbox}"
            compared += 1
    assert compared == sum(len(s.labels) for s in original.samples)


def test_coco_to_yolo_and_back_preserves_category_identity(round_tripped):
    """COCO 17 → YOLO index → back to 17. This is what `Category.source_id` is for: without
    it the export is lossy and "class 3" silently changes meaning between the two files."""
    original, back = round_tripped
    assert [c.source_id for c in back.categories] == [c.source_id for c in original.categories]
    assert [c.name for c in back.categories] == [c.name for c in original.categories]

    src_by_digest = {s.content_sha256: s for s in original.samples}
    orig_src_id = {c.category_id: c.source_id for c in original.categories}
    back_src_id = {c.category_id: c.source_id for c in back.categories}
    for dst in back.samples:
        src = src_by_digest[dst.content_sha256]
        assert (sorted(back_src_id[lb.category_id] for lb in dst.labels)
                == sorted(orig_src_id[lb.category_id] for lb in src.labels))


def test_the_dense_index_is_not_the_source_id_in_this_fixture():
    """Guards the test above from becoming vacuous.

    If the fixture's COCO ids ever became 0,1,2 the round trip would pass whether or not a
    mapping existed at all, and the category assertions would quietly stop testing
    anything while still reporting success.
    """
    from .conftest import COCO_CATEGORIES

    ids = sorted(c["id"] for c in COCO_CATEGORIES)
    assert ids != list(range(len(ids))), "fixture ids must be non-contiguous to mean anything"
