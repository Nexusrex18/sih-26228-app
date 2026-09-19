from __future__ import annotations

from attacklab._types import ContributorSource

import pytest

from attacklab.label_flip_attack import flip_labels
from cva.detectors.data._stub_types import stub_embeddings
from cva.detectors.data.systematic_mislabel import SystematicMislabel
from tests.detectors.data.helpers import detect, resolve


def test_registry_row():
    assert {str(c) for c in SystematicMislabel.requires} == {
        "DATASET_IMAGES", "DATASET_LABELS", "DATASET_CONTRIBUTOR_META"}


def test_clean_zero_findings(clean, clean_emb):
    assert resolve(SystematicMislabel, clean).runnable
    assert detect(SystematicMislabel, clean, clean_emb, None) == []


def test_consistent_class_pair_is_found_at_contributor_level(clean):
    bad, m = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair",
                         target_contributor="A", class_pair=(0, 1))
    fs = detect(SystematicMislabel, bad, stub_embeddings(bad), None)
    assert [(f.target_type, f.target_ref) for f in fs] == [("contributor", "A")]
    f = fs[0]
    assert f.attack_class == "systematic_mislabelling" and f.nature.value == "indeterminate"
    assert f.severity.value == "high"
    assert "'tank'" in f.reason and "'truck'" in f.reason and "contributors.yaml sidecar" in f.reason
    assert f.score_raw >= f.threshold == 0.25


def test_random_flips_are_not_systematic(clean):
    bad, _ = flip_labels(clean, seed=52, flip_rate=0.1, mode="random", target_contributor="A")
    assert detect(SystematicMislabel, bad, stub_embeddings(bad), None) == []


def test_needs_a_cohort(clean):
    one = type(clean)([s for s in clean.samples if s.contributor == "A"], clean.categories)
    fs = detect(SystematicMislabel, one, stub_embeddings(one), None)
    assert len(fs) == 1 and fs[0].availability.value == "UNAVAILABLE" and "cohort" in fs[0].reason


def test_reason_preserves_the_contributor_id_and_the_hypothesis_flag(clean):
    """Regression: str.capitalize() lower-cased the rest of the sentence, turning contributor 'A'
    into 'a' and flattening HYPOTHESIS to 'hypothesis' — on the one contributor-level finding that
    can reach high severity."""
    import dataclasses
    bad, _ = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair",
                         target_contributor="A", class_pair=(0, 1))
    bad = type(bad)([dataclasses.replace(s, contributor_source=ContributorSource.EXIF_CLUSTER) if s.contributor == "A"
                     else s for s in bad.samples], bad.categories)
    f = detect(SystematicMislabel, bad, stub_embeddings(bad))[0]
    assert f.reason.startswith("Contributor 'A' (attribution from EXIF")   # id keeps its case; EXIF too
    assert "(a HYPOTHESIS, not a fact)" in f.reason
    assert "'a'" not in f.reason and "hypothesis" not in f.reason         # old bug: 'a' / lower-cased flag


@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_no_false_positives_on_clean_data_at_six_classes(seed, tmp_path):
    """Multiplicity: at K=6 an uncorrected 1e-3 over contributors x K x (K-1) cells produced a
    false positive on clean data (reviewer's seed 303). The corrected policy must not."""
    from attacklab.contributor_metadata import assign_contributors
    from attacklab.synth_dataset import make_clean_dataset
    ds = make_clean_dataset(tmp_path / "c", seed=seed, n=600, n_classes=6)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=seed)
    assert detect(SystematicMislabel, ds, stub_embeddings(ds)) == []


def test_still_detects_at_six_classes(tmp_path):
    from attacklab.contributor_metadata import assign_contributors
    from attacklab.synth_dataset import make_clean_dataset
    ds = make_clean_dataset(tmp_path / "c", seed=77, n=600, n_classes=6)
    ds, _ = assign_contributors(ds, tmp_path / "c", seed=77)
    bad, _ = flip_labels(ds, seed=78, flip_rate=0.9, mode="class_pair", target_contributor="A", class_pair=(0, 1))
    fs = detect(SystematicMislabel, bad, stub_embeddings(bad))
    assert [f.target_ref for f in fs] == ["A"]


def test_KNOWN_LIMITATION_two_contributors_sharing_a_mislabelling_mask_each_other(clean):
    """PINNED, not celebrated. Each contributor must beat the WORST peer, so when A and B apply the
    SAME wrong mapping each is the other's worst peer and NEITHER is flagged (a lone poisoned
    contributor IS found — test_consistent_class_pair_is_found_at_contributor_level). A collusion /
    shared-SOP case therefore evades this detector. Asserted exactly so that fixing it (e.g. a
    median-of-peers gate) makes this test fail and forces the limitation text to be revisited."""
    bad, _ = flip_labels(clean, seed=61, flip_rate=0.9, mode="class_pair", target_contributor="A", class_pair=(0, 1))
    bad, _ = flip_labels(bad, seed=62, flip_rate=0.9, mode="class_pair", target_contributor="B", class_pair=(0, 1))
    refs = {f.target_ref for f in detect(SystematicMislabel, bad, stub_embeddings(bad))}
    assert refs == set(), f"behaviour changed: now flags {sorted(refs)} — update the limitation text"
    assert not refs & {"C", "D"}                     # and the clean contributors are never accused


def test_the_masking_limitation_is_stamped_on_the_finding(clean):
    bad, _ = flip_labels(clean, seed=51, flip_rate=0.9, mode="class_pair", target_contributor="A", class_pair=(0, 1))
    f = detect(SystematicMislabel, bad, stub_embeddings(bad))[0]
    assert any("SAME wrong mapping" in l and "neither is flagged" in l for l in f.limitations)
