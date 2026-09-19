"""Isotonic calibration (cva/risk/calibration.py): PAV, fitting, application, and the strict,
untrusted-input file loader. A calibration file rewrites every calibrated confidence, and
confidence drives D3/D5, so `load_calibration` is tested as a security boundary."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cva.core.capability import Availability
from cva.core.types import Finding, Severity
from cva.risk import calibration as calmod
from cva.risk.calibration import (
    CalibrationSet,
    Calibrator,
    apply_calibration,
    fit_calibrators,
    load_calibration,
    pav,
    save_calibration,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def separable(det: str, n_each: int = 20) -> list[tuple[str, float, bool]]:
    """Negatives score 0.00..0.19, positives 0.80..0.99: a perfectly separable detector."""
    neg = [(det, i / 100.0, False) for i in range(n_each)]
    pos = [(det, 0.8 + i / 100.0, True) for i in range(n_each)]
    return neg + pos


def finding(det: str, score_raw: float, confidence: float = 0.3,
            availability: Availability = Availability.OK) -> Finding:
    return Finding(detector_id=det, detector_version="0.0.1", target_type="model",
                   target_ref="t", severity=Severity.HIGH, confidence=confidence,
                   reason="r", attack_class="model_substitution", score_raw=score_raw,
                   availability=availability)


def valid_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "isotonic",
        "calibrators": {"det.a": {"edges": [0.3, 0.6, 1.0], "values": [0.0, 0.5, 1.0]}},
        "brier": 0.1,
        "reliability_bins": [{"p_mean": 0.5, "empirical": 0.4, "n": 10}],
        "excluded_detectors": ["prov.chain"],
    }


def write_json(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# pav
# --------------------------------------------------------------------------------------
def test_pav_values_are_monotone_non_decreasing_and_edges_ascending() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < scores).astype(float)         # informative but noisy
    edges, values = pav(scores, labels)
    assert len(edges) == len(values) >= 1
    assert all(b >= a for a, b in zip(edges, edges[1:], strict=False))
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_pav_pools_an_inverted_signal_into_a_single_block() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([1, 1, 0, 0])                           # wholly anti-correlated
    edges, values = pav(scores, labels)
    assert values == [0.5]
    assert edges == [0.4]


def test_pav_is_independent_of_input_order() -> None:
    scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    labels = np.array([1, 0, 0, 0, 1])
    perm = np.array([3, 0, 4, 1, 2])
    assert pav(scores, labels) == pav(scores[perm], labels[perm])


def test_pav_separable_data_gives_a_zero_then_one_step() -> None:
    edges, values = pav(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert values == [0.0, 1.0]
    assert edges == [0.2, 0.9]


def test_pav_block_value_is_the_mean_of_its_labels() -> None:
    edges, values = pav(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))
    assert values == [0.0, 0.5, 1.0]                          # the (1, 0) violation pools to 0.5
    assert edges == [0.1, 0.3, 0.4]


# --------------------------------------------------------------------------------------
# Calibrator
# --------------------------------------------------------------------------------------
def test_calibrator_maps_a_score_to_the_value_of_the_block_it_falls_in() -> None:
    c = Calibrator((0.3, 0.6, 1.0), (0.0, 0.5, 1.0))
    assert c(0.0) == 0.0
    assert c(0.2) == 0.0
    assert c(0.3) == 0.0                                       # an edge is an UPPER bound
    assert c(0.31) == 0.5
    assert c(0.6) == 0.5
    assert c(0.61) == 1.0
    assert c(1.0) == 1.0


def test_calibrator_clamps_out_of_range_scores_to_the_end_blocks() -> None:
    c = Calibrator((0.3, 0.6, 1.0), (0.1, 0.5, 0.9))
    assert c(-5.0) == 0.1
    assert c(50.0) == 0.9


def test_calibrator_is_monotone_over_a_grid() -> None:
    c = Calibrator((0.2, 0.5, 0.8), (0.05, 0.4, 0.95))
    grid = [i / 50 for i in range(-5, 60)]
    out = [c(s) for s in grid]
    assert out == sorted(out)


def test_calibrator_single_block_is_constant() -> None:
    c = Calibrator((0.5,), (0.25,))
    assert {c(-1.0), c(0.5), c(9.0)} == {0.25}


# --------------------------------------------------------------------------------------
# fit_calibrators
# --------------------------------------------------------------------------------------
def test_fit_excludes_prov_detectors_they_never_get_a_calibrator() -> None:
    records = separable("det.a") + separable("prov.chain") + separable("prov.verify")
    cal = fit_calibrators(records)
    assert "prov.chain" in cal.excluded and "prov.verify" in cal.excluded
    assert not any(k.startswith("prov.") for k in cal.calibrators)
    assert "det.a" in cal.calibrators
    assert "det.a" not in cal.excluded


def test_fit_with_only_prov_records_yields_an_empty_set() -> None:
    cal = fit_calibrators(separable("prov.chain"))
    assert cal.calibrators == {}
    assert cal.excluded == ["prov.chain"]
    assert cal.brier is None
    assert cal.bins == []


def test_fit_exclude_prefixes_is_a_parameter() -> None:
    cal = fit_calibrators(separable("det.a") + separable("x.b"), exclude_prefixes=("x.",))
    assert "x.b" in cal.excluded
    assert set(cal.calibrators) == {"det.a"}


def test_fit_skips_a_detector_below_min_points_and_keeps_the_rest() -> None:
    few = separable("det.few", n_each=14)                     # 28 points < default 30
    enough = separable("det.many", n_each=15)                 # 30 points
    cal = fit_calibrators(few + enough)
    assert "det.few" not in cal.calibrators
    assert "det.many" in cal.calibrators


def test_fit_min_points_is_a_parameter() -> None:
    records = separable("det.a", n_each=5)                    # 10 points
    assert fit_calibrators(records).calibrators == {}
    assert "det.a" in fit_calibrators(records, min_points=10).calibrators


def test_fit_with_nothing_fitted_has_no_brier_and_no_bins() -> None:
    cal = fit_calibrators(separable("det.a", n_each=3))
    assert cal.calibrators == {}
    assert cal.brier is None and cal.bins == []


def test_fit_accepts_any_iterable_of_records() -> None:
    def gen() -> Iterator[tuple[str, float, bool]]:
        yield from separable("det.a")
    assert "det.a" in fit_calibrators(gen()).calibrators


def test_fit_separable_set_has_zero_brier_and_two_pure_bins() -> None:
    cal = fit_calibrators(separable("det.a"))
    assert cal.brier == 0.0
    assert [b["n"] for b in cal.bins] == [20, 20]
    assert [b["p_mean"] for b in cal.bins] == [0.0, 1.0]
    assert [b["empirical"] for b in cal.bins] == [0.0, 1.0]
    c = cal.calibrators["det.a"]
    assert c(0.05) == 0.0 and c(0.95) == 1.0


def test_fit_brier_on_an_uninformative_set_is_sane_and_positive() -> None:
    # labels alternate irrespective of score: nothing to learn, so PAV pools to the base rate
    records = [("det.a", i / 60.0, i % 2 == 0) for i in range(60)]
    cal = fit_calibrators(records)
    assert cal.brier is not None
    assert 0.0 < cal.brier <= 0.25 + 1e-6
    assert sum(b["n"] for b in cal.bins) == 60
    assert all(0.0 <= b["p_mean"] <= 1.0 and 0.0 <= b["empirical"] <= 1.0 for b in cal.bins)


def test_fit_calibrated_values_are_within_unit_interval_and_monotone() -> None:
    rng = np.random.default_rng(1)
    records = [("det.a", float(s), bool(rng.random() < s)) for s in rng.random(300)]
    c = fit_calibrators(records).calibrators["det.a"]
    assert all(0.0 <= v <= 1.0 for v in c.values)
    assert list(c.values) == sorted(c.values)
    assert list(c.edges) == sorted(c.edges)


def test_fit_is_deterministic() -> None:
    records = separable("det.a") + separable("det.b")
    a, b = fit_calibrators(records), fit_calibrators(records)
    assert a.calibrators == b.calibrators and a.brier == b.brier and a.bins == b.bins


def test_calibration_set_summary_lists_excluded_sorted() -> None:
    cal = fit_calibrators(separable("det.a") + separable("prov.z") + separable("prov.a"))
    s = cal.summary()
    assert s["method"] == "isotonic"
    assert s["excluded_detectors"] == ["prov.a", "prov.z"]
    assert s["brier"] == cal.brier
    assert s["reliability_bins"] == cal.bins


# --------------------------------------------------------------------------------------
# apply_calibration
# --------------------------------------------------------------------------------------
def test_apply_changes_a_fitted_detectors_confidence() -> None:
    cal = fit_calibrators(separable("det.a"))
    hi, lo = finding("det.a", 0.95, confidence=0.3), finding("det.a", 0.05, confidence=0.7)
    apply_calibration([hi, lo], cal)
    assert hi.confidence == 1.0
    assert lo.confidence == 0.0


def test_apply_leaves_prov_findings_untouched_even_if_a_calibrator_is_present() -> None:
    cal = CalibrationSet(calibrators={"prov.chain": Calibrator((1.0,), (0.0,))})
    f = finding("prov.chain", 0.9, confidence=1.0)
    apply_calibration([f], cal)
    assert f.confidence == 1.0


def test_apply_leaves_unfitted_detectors_untouched() -> None:
    cal = fit_calibrators(separable("det.a"))
    f = finding("det.other", 0.95, confidence=0.42)
    apply_calibration([f], cal)
    assert f.confidence == 0.42


@pytest.mark.parametrize("state", [Availability.UNAVAILABLE, Availability.ERROR])
def test_apply_leaves_findings_that_did_not_run_untouched(state: Availability) -> None:
    cal = fit_calibrators(separable("det.a"))
    f = finding("det.a", 0.95, confidence=0.42, availability=state)
    apply_calibration([f], cal)
    assert f.confidence == 0.42


def test_apply_calibrates_a_degraded_finding() -> None:
    cal = fit_calibrators(separable("det.a"))
    f = finding("det.a", 0.95, confidence=0.42, availability=Availability.DEGRADED)
    apply_calibration([f], cal)
    assert f.confidence == 1.0


def test_apply_with_no_calibration_is_a_no_op() -> None:
    f = finding("det.a", 0.95, confidence=0.42)
    apply_calibration([f], None)
    assert f.confidence == 0.42


def test_apply_respects_a_custom_exclude_prefix() -> None:
    cal = fit_calibrators(separable("x.b"), exclude_prefixes=())
    f = finding("x.b", 0.95, confidence=0.4)
    apply_calibration([f], cal, exclude_prefixes=("x.",))
    assert f.confidence == 0.4


def test_apply_keeps_confidence_in_the_unit_interval() -> None:
    cal = CalibrationSet(calibrators={"det.a": Calibrator((1.0,), (1.0,))})
    f = finding("det.a", 0.5, confidence=0.1)
    apply_calibration([f], cal)
    assert 0.0 <= f.confidence <= 1.0


# --------------------------------------------------------------------------------------
# save_calibration / load_calibration
# --------------------------------------------------------------------------------------
def fitted() -> CalibrationSet:
    return fit_calibrators(separable("det.a") + separable("det.b") + separable("prov.chain"))


def test_round_trip_reproduces_identical_calibrator_outputs(tmp_path: Path) -> None:
    cal = fitted()
    p = save_calibration(cal, tmp_path / "cal.json")
    back = load_calibration(p)
    assert set(back.calibrators) == set(cal.calibrators)
    for det, c in cal.calibrators.items():
        assert back.calibrators[det] == c
        for s in [i / 40 for i in range(-2, 45)]:
            assert back.calibrators[det](s) == c(s)
    assert back.brier == cal.brier
    assert back.bins == cal.bins
    assert back.excluded == sorted(cal.excluded)


def test_save_returns_the_path_and_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "cal.json"
    assert save_calibration(fitted(), target) == target
    assert target.is_file()


def test_second_save_is_byte_identical(tmp_path: Path) -> None:
    cal = fitted()
    p1 = save_calibration(cal, tmp_path / "one.json")
    p2 = save_calibration(cal, tmp_path / "two.json")
    p3 = save_calibration(load_calibration(p1), tmp_path / "three.json")
    assert p1.read_bytes() == p2.read_bytes() == p3.read_bytes()


def test_saved_file_has_sorted_keys_at_every_level(tmp_path: Path) -> None:
    p = save_calibration(fitted(), tmp_path / "cal.json")
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n")
    data = json.loads(text)
    assert list(data) == sorted(data)
    assert list(data["calibrators"]) == sorted(data["calibrators"])
    for c in data["calibrators"].values():
        assert list(c) == sorted(c)
    for b in data["reliability_bins"]:
        assert list(b) == sorted(b)
    assert data["schema_version"] == 1 and data["method"] == "isotonic"
    assert not any(k.startswith("prov.") for k in data["calibrators"])


def test_empty_set_round_trips(tmp_path: Path) -> None:
    p = save_calibration(CalibrationSet(), tmp_path / "empty.json")
    back = load_calibration(p)
    assert back.calibrators == {} and back.brier is None and back.bins == []


def test_save_refuses_a_set_carrying_a_prov_calibrator_and_writes_nothing(
        tmp_path: Path) -> None:
    bad = CalibrationSet(calibrators={"prov.chain": Calibrator((1.0,), (0.5,))})
    target = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="prov"):
        save_calibration(bad, target)
    assert not target.exists()


def test_load_accepts_the_valid_payload_fixture(tmp_path: Path) -> None:
    """Guards the rejection tests below: each one mutates exactly one thing in this payload."""
    cal = load_calibration(write_json(tmp_path / "ok.json", valid_payload()))
    assert cal.calibrators["det.a"](0.4) == 0.5
    assert cal.brier == 0.1
    assert cal.excluded == ["prov.chain"]


def test_load_accepts_a_minimal_payload_with_only_required_keys(tmp_path: Path) -> None:
    p = write_json(tmp_path / "min.json",
                   {"schema_version": 1, "method": "isotonic", "calibrators": {}})
    cal = load_calibration(p)
    assert cal.calibrators == {} and cal.brier is None


def mutate(**changes: Any) -> dict[str, Any]:
    payload = valid_payload()
    payload.update(changes)
    return payload


def cal_with(edges: Any, values: Any, det: str = "det.a") -> dict[str, Any]:
    return mutate(calibrators={det: {"edges": edges, "values": values}})


REJECTED: list[tuple[str, dict[str, Any]]] = [
    ("unknown top-level key", mutate(surprise=1)),
    ("prov calibrator", cal_with([0.5, 1.0], [0.1, 0.9], det="prov.chain")),
    ("prov calibrator nested prefix", cal_with([1.0], [0.5], det="prov.")),
    ("descending edges", cal_with([0.9, 0.5, 0.1], [0.1, 0.5, 0.9])),
    ("value above one", cal_with([0.5, 1.0], [0.1, 1.5])),
    ("value below zero", cal_with([0.5, 1.0], [-0.1, 0.5])),
    ("non-monotone values", cal_with([0.3, 0.6, 1.0], [0.2, 0.8, 0.5])),
    ("edges and values differ in length", cal_with([0.5, 1.0], [0.5])),
    ("empty calibrator", cal_with([], [])),
    ("boolean value", cal_with([0.5], [True])),
    ("string edge", cal_with(["0.5"], [0.5])),
    ("null value", cal_with([0.5], [None])),
    ("edges not an array", cal_with(0.5, [0.5])),
    ("calibrator with an extra key",
     mutate(calibrators={"det.a": {"edges": [1.0], "values": [0.5], "note": "x"}})),
    ("calibrator missing values", mutate(calibrators={"det.a": {"edges": [1.0]}})),
    ("calibrator not an object", mutate(calibrators={"det.a": [1, 2]})),
    ("calibrators not an object", mutate(calibrators=[])),
    ("empty detector id", cal_with([1.0], [0.5], det="")),
    ("unsupported schema version", mutate(schema_version=2)),
    ("boolean schema version", mutate(schema_version=True)),
    ("unsupported method", mutate(method="platt")),
    ("brier above one", mutate(brier=1.5)),
    ("brier not a number", mutate(brier="0.1")),
    ("bin with an unexpected key",
     mutate(reliability_bins=[{"p_mean": 0.5, "empirical": 0.5, "n": 1, "x": 1}])),
    ("bin with a negative count",
     mutate(reliability_bins=[{"p_mean": 0.5, "empirical": 0.5, "n": -1}])),
    ("bin with a fractional count",
     mutate(reliability_bins=[{"p_mean": 0.5, "empirical": 0.5, "n": 1.5}])),
    ("bin probability out of range",
     mutate(reliability_bins=[{"p_mean": 1.5, "empirical": 0.5, "n": 1}])),
    ("bins not an array", mutate(reliability_bins={})),
    ("excluded not an array", mutate(excluded_detectors="prov.chain")),
    ("excluded with an empty name", mutate(excluded_detectors=[""])),
    ("excluded with a non-string", mutate(excluded_detectors=[3])),
]


@pytest.mark.parametrize("label,payload", REJECTED, ids=[r[0] for r in REJECTED])
def test_load_rejects_malformed_payload_as_value_error(
        tmp_path: Path, label: str, payload: dict[str, Any]) -> None:
    p = write_json(tmp_path / "bad.json", payload)
    with pytest.raises(ValueError):
        load_calibration(p)


@pytest.mark.parametrize("key", ["schema_version", "method", "calibrators"])
def test_load_rejects_a_missing_required_key(tmp_path: Path, key: str) -> None:
    payload = valid_payload()
    del payload[key]
    with pytest.raises(ValueError, match="missing"):
        load_calibration(write_json(tmp_path / "bad.json", payload))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_non_finite_numbers(tmp_path: Path, token: str) -> None:
    p = tmp_path / "nan.json"
    p.write_text('{"schema_version": 1, "method": "isotonic", "calibrators": '
                 '{"det.a": {"edges": [0.5], "values": [' + token + ']}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_calibration(p)


def test_load_rejects_nan_in_brier(tmp_path: Path) -> None:
    p = tmp_path / "nan.json"
    p.write_text('{"schema_version": 1, "method": "isotonic", "calibrators": {}, '
                 '"brier": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibration(p)


def test_load_rejects_duplicate_top_level_keys(tmp_path: Path) -> None:
    p = tmp_path / "dup.json"
    p.write_text('{"schema_version": 1, "method": "isotonic", "calibrators": {}, '
                 '"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_calibration(p)


def test_load_rejects_duplicate_detector_ids(tmp_path: Path) -> None:
    """A repeated detector id would silently let the LAST calibrator win under plain json.loads."""
    p = tmp_path / "dup.json"
    p.write_text('{"schema_version": 1, "method": "isotonic", "calibrators": {'
                 '"det.a": {"edges": [1.0], "values": [0.0]}, '
                 '"det.a": {"edges": [1.0], "values": [1.0]}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_calibration(p)


def test_load_rejects_a_file_over_the_size_cap(tmp_path: Path) -> None:
    p = tmp_path / "huge.json"
    body = json.dumps(valid_payload())
    p.write_text(body + " " * (calmod.MAX_FILE_BYTES + 1 - len(body)), encoding="utf-8")
    assert p.stat().st_size > calmod.MAX_FILE_BYTES
    with pytest.raises(ValueError, match="larger than"):
        load_calibration(p)


def test_load_accepts_a_file_at_the_size_cap(tmp_path: Path) -> None:
    p = tmp_path / "edge.json"
    body = json.dumps(valid_payload())
    p.write_text(body + " " * (calmod.MAX_FILE_BYTES - len(body)), encoding="utf-8")
    assert p.stat().st_size == calmod.MAX_FILE_BYTES
    assert "det.a" in load_calibration(p).calibrators


@pytest.mark.parametrize("text", ["not json at all", "", "{", "{'a': 1}", "[1, 2, 3]",
                                  "null", "42", '"a string"'])
def test_load_rejects_non_json_and_non_object_json(tmp_path: Path, text: str) -> None:
    p = tmp_path / "junk.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibration(p)


def test_load_rejects_undecodable_bytes(tmp_path: Path) -> None:
    p = tmp_path / "binary.json"
    p.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ValueError, match="cannot read"):
        load_calibration(p)


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_calibration(tmp_path / "does_not_exist.json")


def test_load_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_calibration(tmp_path)


def test_load_error_names_the_file(tmp_path: Path) -> None:
    p = write_json(tmp_path / "named.json", mutate(surprise=1))
    with pytest.raises(ValueError, match="named.json"):
        load_calibration(p)
