"""Module A against the REAL core types and loaders — not against a test double.

The original stand-in ``Dataset`` offered ``sample(id)``, ``category_name(id)`` and ``len()``; the
real ``InMemoryDataset`` has none of them, and five of six detectors crashed on a real dataset the
moment they built a finding. These tests keep that from coming back: a static guard on the detector
source, a check that the fixtures really are the real type, and a round trip through the real
``COCOLoader``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from attacklab.coco_export import export_coco
from attacklab.contributor_metadata import assign_contributors
from attacklab.label_flip_attack import flip_labels
from attacklab.near_dup_ood_attack import inject_near_duplicates
from attacklab.script_batch_attack import script_generate_batch
from attacklab.synth_dataset import make_clean_dataset
from cva.core.capability import Capability
from cva.core.types import ContributorSource, Dataset, Sample
from cva.detectors.data._stub_types import stub_embeddings
from cva.detectors.data.duplicate_label_conflict import DuplicateLabelConflict
from cva.detectors.data.label_flip import LabelConsistency
from cva.detectors.data.metadata_anomaly import MetadataAnomaly
from cva.detectors.data.near_duplicate import NearDuplicate
from cva.detectors.data.systematic_mislabel import SystematicMislabel
from cva.loaders.datasets.base import InMemoryDataset, _load_sidecar
from cva.loaders.datasets.coco import COCOLoader
from tests.detectors.data.helpers import detect

ROOT = Path(__file__).resolve().parents[3]
# The whole frozen ``Dataset`` Protocol (cva/core/types.py). Nothing else may be assumed.
CONTRACT = {"samples", "categories", "annotations", "capabilities"}


def test_static_guard_detectors_use_only_the_dataset_contract():
    """Any attribute access on a dataset-named variable that is not in the Protocol, and any
    ``len(dataset)``, is a bug waiting for a real dataset."""
    bad = []
    for py in (ROOT / "cva" / "detectors" / "data").glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in {"dataset", "ds"} and node.attr not in CONTRACT):
                bad.append(f"{py.name}:{node.lineno} {node.value.id}.{node.attr}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"
                    and node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in {"dataset", "ds"}):
                bad.append(f"{py.name}:{node.lineno} len({node.args[0].id})")
    assert not bad, "non-contract Dataset use (real datasets lack it): " + "; ".join(bad)


def test_fixtures_are_the_real_dataset_type_with_no_conveniences(clean):
    assert isinstance(clean, InMemoryDataset) and isinstance(clean, Dataset)
    assert isinstance(clean.samples[0], Sample)
    for attr in ("sample", "category_name", "__len__"):
        assert not hasattr(clean, attr), f"the test dataset must not offer {attr}()"
    assert not hasattr(type(clean), "__len__")


def test_the_dataset_capability_probe_is_the_real_one(clean):
    caps = clean.capabilities()
    assert Capability.DATASET_IMAGES in caps and Capability.DATASET_LABELS in caps
    assert Capability.DATASET_CONTRIBUTOR_META in caps


def test_attack_lab_sidecar_is_in_the_format_the_real_loader_reads(tmp_path):
    """Regression: the attack lab once wrote a nested contributors.yaml the loader silently ignored,
    so every contributor was lost on load."""
    ds = make_clean_dataset(tmp_path / "c", seed=3, n=40)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=3)
    side = _load_sidecar(tmp_path / "c")
    assert side, "the real loader found no contributors in the attack lab's sidecar"
    assert all(side[Path(s.path).stem] == s.contributor for s in ds.samples)


# ---------------------------------------------------------------- the real COCO path -------------
@pytest.fixture(scope="module")
def coco_clean(clean, tmp_path_factory):
    root = export_coco(clean, tmp_path_factory.mktemp("coco_clean"))
    return COCOLoader().load(root)


def test_coco_round_trip_preserves_what_detectors_read(clean, coco_clean):
    assert isinstance(coco_clean, InMemoryDataset)
    assert len(coco_clean.samples) == len(clean.samples)
    a = {s.sample_id: s for s in clean.samples}
    for s in coco_clean.samples:
        o = a[s.sample_id]
        assert s.contributor == o.contributor
        assert s.contributor_source == ContributorSource.SIDECAR
        assert [(lb.category_id, lb.bbox) for lb in s.labels] == [(lb.category_id, lb.bbox) for lb in o.labels]
        assert (s.width, s.height) == (o.width, o.height)
    assert [c.name for c in coco_clean.categories] == [c.name for c in clean.categories]


def test_detectors_run_clean_on_a_dataset_loaded_by_the_real_loader(coco_clean):
    emb = stub_embeddings(coco_clean)
    assert detect(MetadataAnomaly, coco_clean) == []
    assert detect(NearDuplicate, coco_clean, emb) == []
    assert detect(LabelConsistency, coco_clean, emb) == []
    assert detect(SystematicMislabel, coco_clean, emb) == []


def _through_coco(ds, root):
    return COCOLoader().load(export_coco(ds, root))


def test_label_flips_are_found_on_a_real_loaded_dataset(clean, tmp_path):
    flipped, m = flip_labels(clean, seed=7, flip_rate=0.1)
    loaded = _through_coco(flipped, tmp_path / "flip")
    fs = detect(LabelConsistency, loaded, stub_embeddings(loaded))       # builds findings: the crash path
    flagged, truth = {f.target_ref for f in fs}, set(m["flips"])
    assert len(flagged & truth) >= 0.8 * len(truth)
    assert all("contributors.yaml sidecar" in f.reason for f in fs)


def test_systematic_mislabelling_on_a_real_loaded_dataset(clean, tmp_path):
    bad, _ = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair",
                         target_contributor="A", class_pair=(0, 1))
    loaded = _through_coco(bad, tmp_path / "sys")
    fs = detect(SystematicMislabel, loaded, stub_embeddings(loaded))
    assert [f.target_ref for f in fs] == ["A"]
    assert "'tank'" in fs[0].reason and "'truck'" in fs[0].reason          # category names, from the real categories


def test_near_duplicates_and_conflicts_on_a_real_loaded_dataset(clean, tmp_path):
    ds, m = inject_near_duplicates(clean, tmp_path / "inj", seed=32, n_sources=4, variants_per_source=6,
                                   target_contributor="B", conflict_rate=0.3)
    loaded = _through_coco(ds, tmp_path / "nd")
    emb = stub_embeddings(loaded)
    nd = detect(NearDuplicate, loaded, emb)
    injected = set(m["injected"])
    assert len({f.target_ref for f in nd} & injected) >= 0.85 * len(injected)
    dl = detect(DuplicateLabelConflict, loaded, emb)
    conflicted = {sid for sid, v in m["injected"].items() if v["label_conflict"]}
    assert conflicted and len({f.target_ref for f in dl} & conflicted) >= 0.8 * len(conflicted)


def test_script_generated_batch_survives_the_real_loader(clean, tmp_path):
    """mtimes, EXIF and the JFIF comment must survive export -> load, or the metadata detector is
    being tested against something a real submission would never look like."""
    ds, _ = script_generate_batch(clean, tmp_path / "sb", seed=3, target_contributor="C")
    loaded = _through_coco(ds, tmp_path / "meta")
    fs = detect(MetadataAnomaly, loaded)
    assert [(f.target_type, f.target_ref) for f in fs] == [("contributor", "C")]
