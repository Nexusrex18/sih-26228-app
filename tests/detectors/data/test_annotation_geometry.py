from __future__ import annotations

import pytest

from attacklab.annotation_geometry_attack import tamper_geometry
from attacklab.contributor_metadata import assign_contributors
from attacklab.synth_dataset import make_clean_dataset
from cva.detectors.data.annotation_geometry import AnnotationGeometry
from tests.detectors.data.helpers import detect, resolve


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
    fs = detect(AnnotationGeometry, det_ds, None, None)
    assert len(fs) <= 0.01 * len(det_ds.samples), [f.reason for f in fs]


def test_shrunk_boxes_found(det_ds):
    bad, m = tamper_geometry(det_ds, seed=62, target_contributor="B", shrink=0.5)
    fs = detect(AnnotationGeometry, bad, None, None)
    flagged, truth = {f.target_ref for f in fs}, set(m["changed"])
    assert flagged and flagged <= truth                              # nothing clean flagged
    assert len(flagged) >= 0.5 * len(truth)
    f = fs[0]
    assert f.attack_class == "annotation_geometry_tamper" and f.nature.value == "indeterminate"
    assert "box area" in f.reason and "cohort-MADs below" in f.reason and "'B'" in f.reason
    assert f.target_type == "sample" and f.score_raw >= f.threshold


def test_shifted_boxes_found_on_position_features(det_ds):
    bad, m = tamper_geometry(det_ds, seed=63, target_contributor="C", shrink=1.0, shift=0.8)
    fs = detect(AnnotationGeometry, bad, None, None)
    assert fs and {f.target_ref for f in fs} <= set(m["changed"])
    assert any("horizontal box position" in f.reason for f in fs)


@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_no_false_positives_on_clean_data_at_six_categories(seed, tmp_path):
    ds = make_clean_dataset(tmp_path / "c", seed=seed, n=600, n_classes=6, objects=2)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=seed)
    assert detect(AnnotationGeometry, ds) == []


def test_multiplicity_policy_is_stamped_on_the_finding(det_ds):
    bad, _ = tamper_geometry(det_ds, seed=62, target_contributor="B", shrink=0.5)
    f = detect(AnnotationGeometry, bad)[0]
    assert any("Bonferroni" in l and "NOT at COCO scale" in l for l in f.limitations)
    assert "after correcting for" in f.reason


def test_small_dataset_is_reported_not_silently_empty(tmp_path):
    """A10: no (contributor, category) group reaching min_n used to return [] — indistinguishable
    from 'checked, nothing wrong'."""
    ds = make_clean_dataset(tmp_path / "c", seed=5, n=40, n_classes=4, objects=2)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=5)
    fs = detect(AnnotationGeometry, ds)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
    assert "at least 15 boxes" in fs[0].reason and "peer group" in fs[0].reason


def test_no_contributors_is_reported_not_silently_empty(det_ds):
    import dataclasses
    anon = type(det_ds)([dataclasses.replace(s, contributor=None, contributor_source=None)
                         for s in det_ds.samples], det_ds.categories)
    fs = detect(AnnotationGeometry, anon)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE"
