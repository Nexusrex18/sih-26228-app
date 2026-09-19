from __future__ import annotations

import pytest

from attacklab.annotation_geometry_attack import tamper_geometry
from attacklab.contributor_metadata import assign_contributors
from attacklab.synth_dataset import make_clean_dataset
from cva.detectors.data.annotation_geometry import AnnotationGeometry
from conftest import resolve


@pytest.fixture(scope="module")
def det_ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("geo")
    ds = make_clean_dataset(d / "c", seed=61, n=400, n_classes=4, objects=2)
    ds, _ = assign_contributors(ds, d / "c", seed=61)
    return ds


def test_registry_row():
    assert {str(c) for c in AnnotationGeometry.requires} == {"DATASET_LABELS", "DATASET_CONTRIBUTOR_META"}


def test_clean_near_zero(det_ds):
    assert resolve(AnnotationGeometry, det_ds).runnable
    fs = AnnotationGeometry().detect(det_ds, None, None)
    assert len(fs) <= 0.01 * len(det_ds), [f.reason for f in fs]


def test_shrunk_boxes_found(det_ds):
    bad, m = tamper_geometry(det_ds, seed=62, target_contributor="B", shrink=0.5)
    fs = AnnotationGeometry().detect(bad, None, None)
    flagged, truth = {f.target_ref for f in fs}, set(m["changed"])
    assert flagged and flagged <= truth                              # nothing clean flagged
    assert len(flagged) >= 0.5 * len(truth)
    f = fs[0]
    assert f.attack_class == "annotation_geometry_tamper" and f.nature.value == "indeterminate"
    assert "box area" in f.reason and "cohort-MADs below" in f.reason and "'B'" in f.reason
    assert f.target_type == "sample" and f.score_raw >= f.threshold


def test_shifted_boxes_found_on_position_features(det_ds):
    bad, m = tamper_geometry(det_ds, seed=63, target_contributor="C", shrink=1.0, shift=0.8)
    fs = AnnotationGeometry().detect(bad, None, None)
    assert fs and {f.target_ref for f in fs} <= set(m["changed"])
    assert any("horizontal box position" in f.reason for f in fs)
