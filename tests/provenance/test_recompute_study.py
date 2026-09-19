"""The C6 gate on a REAL ONNX model: no false `output_mismatch` across ORT configurations (plan §12)."""
from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from attacklab.recompute_study import jitter_monte_carlo, stability  # noqa: E402


def test_no_false_mismatch_across_ort_thread_counts_and_optimisation_levels(tmp_path):
    results = stability(tmp_path, n_frames=40)
    assert len(results) == 30
    total = sum(r.frames for r in results)
    assert total == 1200
    assert sum(r.mismatch for r in results) == 0, [r for r in results if r.mismatch]
    assert sum(r.other_findings for r in results) == 0
    # every record is accounted for as verified or as a boundary flip — nothing silently dropped
    assert all(r.exact + r.decision + r.flip == r.frames for r in results)
    print("\nstability:", {"records": total, "exact": sum(r.exact for r in results),
                           "decision": sum(r.decision for r in results), "flip": sum(r.flip for r in results)})


def test_synthetic_jitter_never_becomes_a_mismatch_and_small_jitter_rarely_flips_at_all(tmp_path):
    res = jitter_monte_carlo(tmp_path, n_frames=150)
    print("\njitter monte-carlo:", res)
    for mag, c in res.items():
        assert c.get("class:output_mismatch", 0) == 0, (mag, c)
    assert res["1e-07"]["coarse_changed"] <= 2                # at 1e-7 (ORT-scale noise) a flip is a rarity
