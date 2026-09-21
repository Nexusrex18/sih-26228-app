"""Fixture generator (plan §C0): schema-shaped `report.json` covering every state.

Hand-written `Finding`s, not a real scan: the dashboard codes against `report.schema.json`,
not against the producer, and the states that matter most (`UNAVAILABLE` everywhere, a
`prov.*` quarantine, a hostile contributor name) are exactly the ones a happy-path scan never
emits. Every generated report is validated against the frozen schema before it is written.

`python -m cva.web.fixtures <out_dir>` writes a `<out_dir>` with the `<scan_id>/report.json`
plus shared `evidence/` layout `cva scan` produces.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
_ZERO64 = "0" * 64
_COMMIT = "a" * 40

#: The dataset supplier is an adversary by premise (plan §3.5). These strings arrive through
#: the one channel we cannot refuse to read, and S1 asserts they render inert.
HOSTILE_STRINGS = {
    "contributor": '<img src=x onerror=alert(1)>',
    "file": '"><script>alert(2)</script>',
    "category": "{{7*7}}",
    "camera": "javascript:alert(3)",
    "batch": "batch_';DROP TABLE findings;--",
}


def finding_id(detector_id: str, detector_version: str, target_type: str, target_ref: str,
               attack_class: str) -> str:
    """Backend's derivation (schema `$defs.finding.finding_id`): scan-invariant."""
    raw = "".join((detector_id, detector_version, target_type, target_ref, attack_class))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _f(detector_id: str, target_type: str, target_ref: str, *, severity: str,
       confidence: float, disposition: str, rule: str, reason: str, attack_class: str,
       nature: str = "adversarial", availability: str = "OK", score: float = 0.9,
       threshold: float = 0.5, evidence: list[dict[str, Any]] | None = None,
       exclusion_reason: str | None = None, limitations: list[str] | None = None,
       access: list[str] | None = None, scan_id: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "finding_id": finding_id(detector_id, "1.0.0", target_type, target_ref, attack_class),
        "scan_id": scan_id,
        "detector_id": detector_id,
        "detector_version": "1.0.0",
        "target_type": target_type,
        "target_ref": target_ref,
        "severity": severity,
        "confidence": confidence,
        "score_raw": score,
        "threshold": threshold,
        "reason": reason,
        "attack_class": attack_class,
        "evidence": evidence or [],
        "access_assumptions": access or ["MODEL_PREDICT present"],
        "limitations": limitations or [],
        "disposition": disposition,
        "disposition_rule": rule,
        "produced_by": detector_id,
        "nature": nature,
        "availability": availability,
    }
    if exclusion_reason is not None:
        out["exclusion_reason"] = exclusion_reason
    return out


def _ev(kind: str, caption: str, path: str | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"kind": kind, "caption": caption}
    e["path"] = path
    return e


def _evidence_payloads() -> dict[str, bytes]:
    """Content-addressed evidence files, keyed by their own sha256 (the bare hash)."""
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 13
           + b"\x00\x00\x00\x00IEND\xaeB`\x82")
    svg = ('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="80" '
           'height="40"><script>alert("svg")</script>'
           f'<text x="2" y="20">{HOSTILE_STRINGS["category"]}</text></svg>').encode()
    table = json.dumps({"rows": [["contributor", HOSTILE_STRINGS["contributor"]],
                                ["flagged", 3]]}, indent=2).encode()
    primary = json.dumps({"primary_check": "phash_hamming", "cascade_suppressed": 4},
                         indent=2).encode()
    return {hashlib.sha256(b).hexdigest(): b for b in (png, svg, table, primary)}


def build_reports() -> dict[str, dict[str, Any]]:
    """Every fixture scan, keyed by `scan_id`."""
    payloads = _evidence_payloads()
    hashes = list(payloads)
    png_h, svg_h, table_h, primary_h = hashes[0], hashes[1], hashes[2], hashes[3]
    return {
        "s-2026-09-19-0001": _rich_scan(png_h, svg_h, table_h, primary_h),
        "s-2026-09-19-0002": _all_unavailable_scan(),
        "s-2026-09-19-0003": _clean_scan(),
        "s-2026-09-19-0004": _blackbox_scan(),
    }


def _rich_scan(png_h: str, svg_h: str, table_h: str, primary_h: str) -> dict[str, Any]:
    sid = "s-2026-09-19-0001"
    findings = [
        _f("prov.ledger_verify", "record", "seq:412", severity="critical", confidence=1.0,
           disposition="quarantine", rule="D1",
           reason="Record seq 412 does not verify: the stored signature does not match the "
                  "canonical bytes of the record under key 3f9c…a21b.",
           attack_class="record_edit", nature="adversarial", scan_id=sid,
           evidence=[_ev("json", "record seq 412, as stored", table_h)],
           limitations=["Verification is only as strong as the trust root supplied."]),
        _f("prov.ledger_verify", "record", "seq:418", severity="low", confidence=1.0,
           disposition="review", rule="D5",
           reason="A degraded_marker declares 37 unsealed inferences between "
                  "2026-09-18T04:11:02Z and 2026-09-18T04:19:55Z.",
           attack_class="degraded_gap", nature="indeterminate", scan_id=sid),
        _f("prov.ledger_verify", "record", "seq:501", severity="info", confidence=1.0,
           disposition="accept", rule="D7",
           reason="created_at_utc at seq 501 precedes seq 500 by 4.2 s; the host clock is "
                  "untrusted and ordering is by ledger seq.",
           attack_class="clock_regression", nature="indeterminate", scan_id=sid),
        _f("data.near_dup", "sample", HOSTILE_STRINGS["file"], severity="medium",
           confidence=0.71, disposition="review", rule="D5",
           reason=f"Perceptual hash within Hamming distance 2 of 11 other samples "
                  f"attributed to contributor {HOSTILE_STRINGS['contributor']}.",
           attack_class="near_duplicate", nature="quality", scan_id=sid,
           evidence=[_ev("contact_sheet", "12 near-identical crops", png_h),
                     _ev("json", "primary_check", primary_h)],
           limitations=["No embedding index was built: perceptual hashing only."]),
        _f("data.label_flip", "sample", "img_04471.jpg", severity="high", confidence=0.93,
           disposition="quarantine", rule="D3",
           reason=f"Label '{HOSTILE_STRINGS['category']}' disagrees with 19 of its 20 "
                  "nearest neighbours in embedding space.",
           attack_class="label_flip", nature="adversarial", scan_id=sid,
           evidence=[_ev("image_crop", "sample and its neighbours", png_h),
                     _ev("plot", "neighbourhood label distribution", svg_h)]),
        _f("data.metadata_anomaly", "contributor", HOSTILE_STRINGS["contributor"],
           severity="high", confidence=0.88, disposition="quarantine", rule="D4",
           reason="Posterior mean 0.41 (95% CI 0.27-0.56) against a leave-one-out cohort "
                  "rate of 0.06; the interval excludes the cohort rate.",
           attack_class="contributor_outlier", nature="adversarial", scan_id=sid,
           evidence=[_ev("table", "per-contributor flag counts", table_h)]),
        _f("model.fingerprint", "model", "resnet18_field.onnx", severity="low",
           confidence=0.34, disposition="review", rule="D2",
           reason="Output divergence 0.004 over 64 deterministic probes, inside the "
                  "re-export/quantisation band.",
           attack_class="benign_conversion", nature="indeterminate", scan_id=sid,
           evidence=[_ev("plot", "per-probe divergence", svg_h)]),
        _f("model.neural_cleanse", "model", "resnet18_field.onnx", severity="medium",
           confidence=0.62, disposition="review", rule="D5",
           reason="Class 3's reconstructed trigger has an L1 norm 3.1 MAD below the median "
                  "across classes (anomaly index 3.1).",
           attack_class="backdoor_trigger", nature="adversarial", scan_id=sid,
           evidence=[_ev("heatmap", "reconstructed mask, class 3", png_h)]),
        _f("model.weight_stats", "model", "resnet18_field.onnx", severity="medium",
           confidence=0.55, disposition="review", rule="D6",
           reason="Boundary flip on 2 of 64 probes between the supplied and reference "
                  "weight sets.",
           attack_class="boundary_flip", nature="indeterminate", scan_id=sid),
        _f("model.strip", "model", "resnet18_field.onnx", severity="info", confidence=0.0,
           disposition="accept", rule="D7", availability="UNAVAILABLE",
           exclusion_reason="capability",
           reason="STRIP was not performed: it needs SUSPECT_INPUTS, and no suspect set was "
                  "supplied. This is an absence of evidence, not evidence of absence.",
           attack_class="backdoor_trigger", nature="indeterminate", scan_id=sid,
           access=["SUSPECT_INPUTS ABSENT"]),
        _f("model.universal_margin", "model", "resnet18_field.onnx", severity="info",
           confidence=0.0, disposition="accept", rule="D7", availability="UNAVAILABLE",
           exclusion_reason="budget",
           reason="Excluded by the 'standard' budget tier; estimated cost 41 min at the "
                  "measured CPU throughput.",
           attack_class="universal_perturbation", nature="indeterminate", scan_id=sid),
        _f("data.annotation_geometry", "batch", HOSTILE_STRINGS["batch"], severity="low",
           confidence=0.44, disposition="review", rule="D5",
           reason="Box aspect ratios in this batch depart from the dataset distribution "
                  "(KS D=0.31, p=0.004).",
           attack_class="annotation_geometry", nature="quality", scan_id=sid,
           evidence=[_ev("plot", "aspect-ratio ECDF", svg_h)]),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scan_id": sid,
        "created_at_utc": "2026-09-19T09:14:22+00:00",
        "produced_by": {"code_commit": _COMMIT, "profile_hash": "1" * 64,
                        "profile_name": "baseline", "budget_tier": "standard",
                        "tool_version": "0.1.0"},
        "target": {"model_id": "resnet18_field.onnx", "model_format": "onnx",
                   "model_sha256": "2" * 64, "arch_hash": "3" * 64,
                   "preprocess_hash": "4" * 64, "preprocess_ref": "sha256:" + "4" * 64,
                   "model_opset": 17, "dataset_path": "/data/field_batch_7",
                   "dataset_format": "coco", "n_samples": 4820, "n_categories": 12},
        "verdict": "QUARANTINE",
        "access_assumptions": {
            "capabilities_present": ["DATASET_IMAGES", "DATASET_LABELS", "MODEL_PREDICT",
                                     "MODEL_LOGITS", "MODEL_WEIGHTS", "MODEL_ARCHITECTURE",
                                     "INFERENCE_LEDGER"],
            "capabilities_absent": [
                {"capability": "MODEL_GRADIENTS",
                 "reason": "onnxruntime-training not bundled (declared narrowing, ADR-001)"},
                {"capability": "SUSPECT_INPUTS",
                 "reason": "no suspect input set was supplied with --corpus"},
                {"capability": "REFERENCE_CLEAN_SET",
                 "reason": "no --reference-dataset was given"}],
            "consequence": "Gradient-based trigger reconstruction ran on the NES estimator "
                           "rather than true gradients, at a lower sensitivity."},
        "plan": [
            {"check_id": "data.near_dup", "state": "DEGRADED",
             "reason": "no embedding index; perceptual hashing only", "missing": [],
             "mode": "phash_only", "exclusion_reason": None, "estimated_cost": None,
             "attack_classes": ["near_duplicate"], "elapsed_s": 41.2},
            {"check_id": "data.label_flip", "state": "OK", "reason": "all inputs present",
             "attack_classes": ["label_flip"], "elapsed_s": 190.4},
            {"check_id": "model.strip", "state": "UNAVAILABLE",
             "reason": "requires SUSPECT_INPUTS", "missing": ["SUSPECT_INPUTS"],
             "exclusion_reason": "capability", "attack_classes": ["backdoor_trigger"],
             "elapsed_s": None},
            {"check_id": "model.universal_margin", "state": "UNAVAILABLE",
             "reason": "excluded by budget tier 'standard'",
             "exclusion_reason": "budget", "estimated_cost": "41 min",
             "attack_classes": ["universal_perturbation"], "elapsed_s": None},
            {"check_id": "prov.ledger_verify", "state": "OK",
             "reason": "an inference ledger and a trust root were supplied",
             "attack_classes": ["record_edit", "degraded_gap", "clock_regression"],
             "elapsed_s": 3.1},
        ],
        "findings": findings,
        "contributor_risk": [
            {"group_key": "contributor", "group_value": HOSTILE_STRINGS["contributor"],
             "contributor_source": "sidecar", "n_samples": 310, "n_flagged": 127,
             "posterior_mean": 0.41, "ci_low": 0.27, "ci_high": 0.56,
             "excludes_cohort_rate": True, "cohort_rate_used": 0.061,
             "disposition": "quarantine"},
            {"group_key": "contributor", "group_value": "survey_team_north",
             "contributor_source": "directory", "n_samples": 1980, "n_flagged": 104,
             "posterior_mean": 0.053, "ci_low": 0.043, "ci_high": 0.063,
             "excludes_cohort_rate": False, "cohort_rate_used": 0.072,
             "disposition": "accept"},
            {"group_key": "contributor", "group_value": "unattributed_camera_cluster_2",
             "contributor_source": "exif_cluster", "n_samples": 95, "n_flagged": 21,
             "posterior_mean": 0.22, "ci_low": 0.15, "ci_high": 0.31,
             "excludes_cohort_rate": True, "cohort_rate_used": 0.064,
             "disposition": "review"},
            {"group_key": "batch", "group_value": HOSTILE_STRINGS["batch"],
             "n_samples": 640, "n_flagged": 58, "posterior_mean": 0.091,
             "ci_low": 0.071, "ci_high": 0.115, "excludes_cohort_rate": False,
             "cohort_rate_used": 0.068, "disposition": "review"},
            {"group_key": "source", "group_value": "source_alpha", "n_samples": 2400,
             "n_flagged": 160, "posterior_mean": 0.067, "ci_low": 0.057, "ci_high": 0.077,
             "excludes_cohort_rate": False, "cohort_rate_used": 0.070,
             "disposition": "accept"},
        ],
        "contributor_baseline": None,
        "contributor_baseline_unavailable":
            "no reference dataset was supplied with --reference-dataset",
        "permutation_test": {"statistic": 0.214, "p_value": 0.001, "n_permutations": 5000,
                             "conclusion": "Flags are not randomly distributed across "
                                           "contributors (p = 0.001)."},
        "provenance_summary": {"records_verified": 512, "anchors_checked": 1,
                               "declared_degraded_intervals": 1,
                               "records_after_last_anchor": 88, "custody_type": "file",
                               "durability_window": "0 records (per_record fsync)",
                               "ledger_state": "sealed"},
        "drift_summary": None,
        "calibration": {"method": "isotonic", "brier": 0.084, "scored_on": "out_of_fold",
                        "calibrated_detectors": ["data.near_dup", "data.label_flip",
                                                 "model.neural_cleanse"],
                        "reliability_bins": [{"p_mean": 0.1, "empirical": 0.08, "n": 240},
                                             {"p_mean": 0.3, "empirical": 0.34, "n": 180},
                                             {"p_mean": 0.5, "empirical": 0.47, "n": 120},
                                             {"p_mean": 0.7, "empirical": 0.74, "n": 90},
                                             {"p_mean": 0.9, "empirical": 0.91, "n": 60}],
                        "excluded_detectors": ["prov.ledger_verify"]},
        "coverage": {
            "counts_only_kind": "attack",
            "assessed": {"label_flip": ["data.label_flip"],
                         "near_duplicate": ["data.near_dup"],
                         "contributor_outlier": ["data.metadata_anomaly"],
                         "annotation_geometry": ["data.annotation_geometry"],
                         "record_edit": ["prov.ledger_verify"],
                         "backdoor_trigger": ["model.neural_cleanse"],
                         "model_substitution": ["model.fingerprint"]},
            "not_assessed": {"universal_perturbation": ["model.universal_margin"],
                             "trigger_ood": ["model.strip"]},
            "never_covered": ["latent_space_poisoning", "clean_label_feature_collision",
                              "gradient_matching_poison"],
            "operational_reports": ["degraded_gap", "clock_regression"],
            "standing_limitations": [
                "We detect poisoning that leaves a regularity in pixel, frequency, "
                "label-neighbourhood or activation space; poisoning engineered to carry its "
                "regularity only in a space we do not measure is invisible to us.",
                "Detection rates are measured against our own attack families and do not "
                "generalise to attacks outside them."]},
        "reproduction": {
            "command": "cva scan --model resnet18_field.onnx --dataset /data/field_batch_7 "
                       "--profile baseline --out artifacts/reports",
            "volatile_paths": ["$.scan_id", "$.created_at_utc", "$.findings[*].scan_id"],
            "seeds": {"global": 7, "numpy": 7, "torch": 7},
            "env": {"python": "3.12.3", "torch": "2.6.0", "numpy": "1.26.4"},
            "version_pins_hash": "5" * 64,
            "determinism_notes": ["ONNX Runtime session options pinned to 1 intra-op thread."]},
    }


def _all_unavailable_scan() -> dict[str, Any]:
    """Gate N1: a scan where everything was UNAVAILABLE must not look green."""
    sid = "s-2026-09-19-0002"
    checks = [("model.weight_digest", "MODEL_WEIGHTS", "model_substitution"),
              ("model.weight_stats", "MODEL_WEIGHTS", "weight_anomaly"),
              ("model.graph_structure", "MODEL_ARCHITECTURE", "graph_tamper"),
              ("model.neural_cleanse", "MODEL_GRADIENTS", "backdoor_trigger"),
              ("data.label_flip", "DATASET_LABELS", "label_flip")]
    return {
        "schema_version": SCHEMA_VERSION, "scan_id": sid,
        "created_at_utc": "2026-09-19T11:02:00+00:00",
        "produced_by": {"code_commit": _COMMIT, "profile_hash": "6" * 64,
                        "profile_name": "blackbox", "budget_tier": "triage"},
        "target": {"model_id": "vendor_api_v3", "model_format": "http", "model_opset": None,
                   "dataset_path": None, "dataset_format": None, "n_samples": None,
                   "n_categories": None},
        "verdict": "REVIEW",
        "access_assumptions": {
            "capabilities_present": ["MODEL_PREDICT"],
            "capabilities_absent": [
                {"capability": c, "reason": f"a query-only model exposes no {c.lower()}"}
                for c in ("MODEL_WEIGHTS", "MODEL_ARCHITECTURE", "MODEL_GRADIENTS",
                          "MODEL_ACTIVATIONS")]
            + [{"capability": "DATASET_IMAGES", "reason": "no dataset was supplied"},
               {"capability": "DATASET_LABELS", "reason": "no dataset was supplied"}],
            "consequence": "Nothing in this scan inspected the model's internals or any "
                           "data. A short report here is a small statement, not a clean one."},
        "plan": [{"check_id": cid, "state": "UNAVAILABLE", "reason": f"requires {cap}",
                  "missing": [cap], "exclusion_reason": "capability",
                  "attack_classes": [ac], "elapsed_s": None}
                 for cid, cap, ac in checks],
        "findings": [
            _f(cid, "model", "vendor_api_v3", severity="info", confidence=0.0,
               disposition="accept", rule="D7", availability="UNAVAILABLE",
               exclusion_reason="capability",
               reason=f"{cid} was not performed: it requires {cap}, which this "
                      "query-only model does not expose.",
               attack_class=ac, nature="indeterminate", scan_id=sid,
               access=[f"{cap} ABSENT"])
            for cid, cap, ac in checks],
        "contributor_risk": [],
        "contributor_baseline": None,
        "permutation_test": None,
        "provenance_summary": {"records_verified": 0, "anchors_checked": 0,
                               "declared_degraded_intervals": 0,
                               "records_after_last_anchor": 0, "custody_type": "none",
                               "durability_window": "n/a", "ledger_state": "not_sealed",
                               "not_assessed_reason":
                                   "an inference ledger was supplied and INFERENCE_LEDGER "
                                   "resolved, but no prov.* check is registered, so nothing "
                                   "in it was verified"},
        "drift_summary": None,
        "calibration": None,
        "coverage": {"counts_only_kind": "attack", "assessed": {},
                     "not_assessed": {cap_ac[2]: [cap_ac[0]] for cap_ac in checks},
                     "never_covered": ["latent_space_poisoning", "universal_perturbation",
                                       "near_duplicate", "contributor_outlier"],
                     "operational_reports": [],
                     "standing_limitations": [
                         "No check in this scan read the model's weights or architecture."]},
        "reproduction": {"command": "cva scan --model-url http://127.0.0.1:9000/predict "
                                    "--input-shape 3,224,224 --num-classes 12 "
                                    "--profile blackbox",
                         "volatile_paths": ["$.scan_id", "$.created_at_utc",
                                            "$.findings[*].scan_id"],
                         "seeds": {"global": 7}, "env": {"python": "3.12.3"},
                         "version_pins_hash": None, "determinism_notes": []},
    }


def _clean_scan() -> dict[str, Any]:
    """Demo step 1: green, and the provenance section is present anyway."""
    sid = "s-2026-09-19-0003"
    return {
        "schema_version": SCHEMA_VERSION, "scan_id": sid,
        "created_at_utc": "2026-09-19T08:00:00+00:00",
        "produced_by": {"code_commit": _COMMIT, "profile_hash": "7" * 64,
                        "profile_name": "baseline", "budget_tier": "standard"},
        "target": {"model_id": "resnet18_clean.onnx", "model_format": "onnx",
                   "model_sha256": "8" * 64, "arch_hash": "9" * 64, "model_opset": 17,
                   "dataset_path": "/data/clean_batch", "dataset_format": "coco",
                   "n_samples": 1200, "n_categories": 12},
        "verdict": "ACCEPT",
        "access_assumptions": {
            "capabilities_present": ["DATASET_IMAGES", "DATASET_LABELS", "MODEL_PREDICT",
                                     "MODEL_WEIGHTS", "MODEL_ARCHITECTURE",
                                     "INFERENCE_LEDGER"],
            "capabilities_absent": [{"capability": "MODEL_GRADIENTS",
                                     "reason": "onnxruntime-training not bundled"}],
            "consequence": "Trigger reconstruction ran on the NES estimator."},
        "plan": [{"check_id": "data.label_flip", "state": "OK", "reason": "inputs present",
                  "attack_classes": ["label_flip"], "elapsed_s": 52.0},
                 {"check_id": "prov.ledger_verify", "state": "OK",
                  "reason": "ledger and trust root supplied",
                  "attack_classes": ["record_edit"], "elapsed_s": 2.0}],
        "findings": [],
        "contributor_risk": [
            {"group_key": "contributor", "group_value": "survey_team_north",
             "contributor_source": "directory", "n_samples": 1200, "n_flagged": 3,
             "posterior_mean": 0.003, "ci_low": 0.001, "ci_high": 0.008,
             "excludes_cohort_rate": False, "cohort_rate_used": 0.004,
             "disposition": "accept"}],
        "contributor_baseline": {"reference_n": 900, "reference_flagged": 4,
                                 "reference_rate": 0.0044,
                                 "reference_rate_ci_high": 0.0102, "cohort_rate": 0.0025,
                                 "cohort_exceeds_reference": False},
        "permutation_test": {"statistic": 0.02, "p_value": 0.41, "n_permutations": 5000,
                             "conclusion": "No evidence that flags cluster by contributor."},
        "provenance_summary": {"records_verified": 240, "anchors_checked": 2,
                               "declared_degraded_intervals": 0,
                               "records_after_last_anchor": 12, "custody_type": "file",
                               "durability_window": "0 records (per_record fsync)",
                               "ledger_state": "sealed"},
        "drift_summary": None,
        "calibration": {"method": "isotonic", "brier": 0.061, "scored_on": "out_of_fold",
                        "calibrated_detectors": ["data.label_flip"],
                        "reliability_bins": [{"p_mean": 0.1, "empirical": 0.09, "n": 300}],
                        "excluded_detectors": ["prov.ledger_verify"]},
        "coverage": {"counts_only_kind": "attack",
                     "assessed": {"label_flip": ["data.label_flip"],
                                  "record_edit": ["prov.ledger_verify"]},
                     "not_assessed": {}, "never_covered": ["latent_space_poisoning"],
                     "operational_reports": [], "standing_limitations": []},
        "reproduction": {"command": "cva scan --model resnet18_clean.onnx "
                                    "--dataset /data/clean_batch --profile baseline",
                         "volatile_paths": ["$.scan_id", "$.created_at_utc",
                                            "$.findings[*].scan_id"],
                         "seeds": {"global": 7}, "env": {"python": "3.12.3"},
                         "version_pins_hash": "5" * 64, "determinism_notes": []},
    }


def _blackbox_scan() -> dict[str, Any]:
    """Gate N6: the same asset under `blackbox` — a visibly smaller coverage statement."""
    r = _clean_scan()
    sid = "s-2026-09-19-0004"
    r["scan_id"] = sid
    r["created_at_utc"] = "2026-09-19T08:40:00+00:00"
    r["produced_by"] = {"code_commit": _COMMIT, "profile_hash": "b" * 64,
                        "profile_name": "blackbox", "budget_tier": "standard"}
    r["target"] = {"model_id": "resnet18_clean.onnx", "model_format": "onnx",
                   "model_opset": 17, "dataset_path": "/data/clean_batch",
                   "dataset_format": "coco", "n_samples": 1200, "n_categories": 12}
    r["access_assumptions"] = {
        "capabilities_present": ["DATASET_IMAGES", "DATASET_LABELS", "MODEL_PREDICT"],
        "capabilities_absent": [
            {"capability": "MODEL_WEIGHTS",
             "reason": "disabled by the 'blackbox' policy, not absent from the artefact"},
            {"capability": "MODEL_ARCHITECTURE",
             "reason": "disabled by the 'blackbox' policy"},
            {"capability": "MODEL_GRADIENTS", "reason": "onnxruntime-training not bundled"}],
        "consequence": "Weight, graph and trigger-reconstruction checks were not asked to "
                       "run. The coverage statement below is correspondingly smaller."}
    r["coverage"] = {"counts_only_kind": "attack",
                     "assessed": {"label_flip": ["data.label_flip"]},
                     "not_assessed": {"model_substitution": ["model.weight_digest"],
                                      "graph_tamper": ["model.graph_structure"],
                                      "weight_anomaly": ["model.weight_stats"],
                                      "backdoor_trigger": ["model.neural_cleanse"]},
                     "never_covered": ["latent_space_poisoning", "record_edit"],
                     "operational_reports": [],
                     "standing_limitations": [
                         "Run under the 'blackbox' policy: white-box checks were disabled "
                         "up front. This is 'was not asked to', not 'could not'."]}
    r["provenance_summary"] = None
    r["plan"] = [{"check_id": "data.label_flip", "state": "OK", "reason": "inputs present",
                  "attack_classes": ["label_flip"], "elapsed_s": 52.0},
                 {"check_id": "model.weight_digest", "state": "UNAVAILABLE",
                  "reason": "disabled by the 'blackbox' policy",
                  "exclusion_reason": "capability",
                  "attack_classes": ["model_substitution"], "elapsed_s": None}]
    r["reproduction"]["command"] = ("cva scan --model resnet18_clean.onnx "
                                    "--dataset /data/clean_batch --profile blackbox")
    return r


# --- writing --------------------------------------------------------------------------------

def schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "report.schema.json"


def validate(report: dict[str, Any]) -> list[str]:
    """Validate against the frozen schema. An invalid fixture is worse than no fixture."""
    from jsonschema import Draft202012Validator
    schema = json.loads(schema_path().read_text())
    return [f"{'.'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(report)]


def write(out_dir: Path) -> Path:
    """Write the `<out_dir>/<scan_id>/report.json` + shared `<out_dir>/evidence/` layout."""
    out_dir = Path(out_dir)
    ev_dir = out_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    for digest, payload in _evidence_payloads().items():
        (ev_dir / digest).write_bytes(payload)
    for scan_id, report in build_reports().items():
        errors = validate(report)
        if errors:
            raise AssertionError(f"fixture {scan_id} fails report.schema.json:\n  "
                                 + "\n  ".join(errors))
        d = out_dir / scan_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
        body = hashlib.sha256((d / "report.json").read_bytes()).hexdigest()
        (d / "seal.json").write_text(json.dumps(
            {"scan_id": scan_id, "report_file": "report.json", "report_sha256": body,
             "sealed": False, "ledger_seq": None,
             "ledger_error": None}, indent=2) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else Path("artifacts/fixtures/reports")
    write(out)
    print(f"wrote {len(build_reports())} fixture reports to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
