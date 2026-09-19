from __future__ import annotations

from attacklab.label_flip_attack import flip_labels
from cva.detectors.data._stub_types import stub_embeddings
from cva.detectors.data.systematic_mislabel import SystematicMislabel
from conftest import resolve


def test_registry_row():
    assert {str(c) for c in SystematicMislabel.requires} == {
        "DATASET_IMAGES", "DATASET_LABELS", "DATASET_CONTRIBUTOR_META"}


def test_clean_zero_findings(clean, clean_emb):
    assert resolve(SystematicMislabel, clean).runnable
    assert SystematicMislabel().detect(clean, clean_emb, None) == []


def test_consistent_class_pair_is_found_at_contributor_level(clean):
    bad, m = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair",
                         target_contributor="A", class_pair=(0, 1))
    fs = SystematicMislabel().detect(bad, stub_embeddings(bad), None)
    assert [(f.target_type, f.target_ref) for f in fs] == [("contributor", "A")]
    f = fs[0]
    assert f.attack_class == "systematic_mislabelling" and f.nature.value == "indeterminate"
    assert f.severity.value == "high"
    assert "'tank'" in f.reason and "'truck'" in f.reason and "contributors.yaml sidecar" in f.reason
    assert f.score_raw >= f.threshold == 0.25


def test_random_flips_are_not_systematic(clean):
    bad, _ = flip_labels(clean, seed=52, flip_rate=0.1, mode="random", target_contributor="A")
    assert SystematicMislabel().detect(bad, stub_embeddings(bad), None) == []


def test_needs_a_cohort(clean):
    one = type(clean)([s for s in clean.samples if s.contributor == "A"], clean.categories)
    fs = SystematicMislabel().detect(one, stub_embeddings(one), None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "cohort" in fs[0].reason
