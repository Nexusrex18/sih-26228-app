"""Isotonic calibration per detector — the system's own confidence is evidence.

Detector score distributions are not sigmoid-shaped, so a parametric (Platt) fit would put a
smooth curve through something that is not smooth. Pool-adjacent-violators gives the monotone
step function that best matches the labelled outcomes and assumes nothing about its shape.

Fitting needs LABELLED outcomes: (detector, score, was-it-an-attack) triples from the attack
lab's benchmark. A scan of a real submission has none, so it applies calibrators it was
handed and reports `calibration: null` when it was handed none. It never invents a curve.

`prov.*` detectors are excluded, always. A hash mismatch is arithmetic, not a belief: fitting
a curve to a deterministic check produces a calibrated probability on a certainty, and feeds
the fit a degenerate class that corrupts it for every other detector.

A fitted set travels as a JSON file (`save_calibration` / `load_calibration`). SECURITY: a
calibration file changes EVERY confidence in the report, and confidence decides dispositions
(D3/D5), so a file supplied by a third party can quietly turn a quarantine into an accept.
It is therefore treated as untrusted input: `load_calibration` validates the structure,
ranges and ordering strictly and raises ValueError on anything unexpected. Validation cannot
tell a genuine fit from a forged-but-well-formed one; only take a calibration file from a
source you would trust to set the report's thresholds, and read the `calibration` block the
report prints.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cva.core.capability import Availability
from cva.core.types import Finding

EXCLUDE_PREFIXES = ("prov.",)
MIN_POINTS = 30
SCHEMA_VERSION = 1
MAX_FILE_BYTES = 8 * 1024 * 1024        # a real file is a few KB; refuse to slurp anything else


def pav(scores: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    """Pool adjacent violators. Returns (upper bin edges, calibrated values), both ascending."""
    order = np.argsort(scores, kind="stable")
    xs, ys = scores[order].astype(float), labels[order].astype(float)
    blocks: list[list[float]] = []               # [sum_y, weight, max_x]
    for x, y in zip(xs, ys, strict=True):
        blocks.append([y, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] = top[2]
    return [b[2] for b in blocks], [b[0] / b[1] for b in blocks]


@dataclass(frozen=True)
class Calibrator:
    edges: tuple[float, ...]
    values: tuple[float, ...]

    def __call__(self, score: float) -> float:
        i = min(bisect_right(self.edges, score - 1e-12), len(self.values) - 1)
        return self.values[max(i, 0)]


@dataclass
class CalibrationSet:
    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    brier: float | None = None
    bins: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """What the report prints. `scored_on` and `calibrated_detectors` are not decoration:
        a Brier score says nothing without its measurement basis, and a reader cannot tell
        which of the report's confidences went through a curve without the list."""
        return {"method": "isotonic", "brier": self.brier, "reliability_bins": self.bins,
                "scored_on": "out_of_fold" if self.brier is not None else "none",
                "calibrated_detectors": sorted(self.calibrators),
                "excluded_detectors": sorted(self.excluded)}


def _reliability(p: list[float], y: list[float]) -> tuple[float | None, list[dict[str, Any]]]:
    """Brier score and reliability bins from (predicted, outcome) pairs. Nothing is fitted
    here — the caller decides which pairs are honest to score."""
    if not p:
        return None, []
    p_arr, y_arr = np.array(p), np.array(y)
    bins: list[dict[str, Any]] = []
    for lo in np.linspace(0, 0.9, 10):
        m = (p_arr >= lo) & (p_arr < lo + 0.1 + (1e-9 if lo > 0.85 else 0))
        if m.any():
            bins.append({"p_mean": round(float(p_arr[m].mean()), 6),
                         "empirical": round(float(y_arr[m].mean()), 6),
                         "n": int(m.sum())})
    return round(float(np.mean((p_arr - y_arr) ** 2)), 6), bins


def _fit_one(pts: list[tuple[float, bool]]) -> Calibrator:
    s = np.array([p for p, _ in pts])
    y = np.array([q for _, q in pts], dtype=float)
    edges, values = pav(s, y)
    return Calibrator(tuple(edges), tuple(values))


def fit_calibrators(records: Iterable[tuple[str, float, bool]],
                    min_points: int = MIN_POINTS,
                    exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES) -> CalibrationSet:
    """`records` = (detector_id, raw score, was_attack). Detectors with fewer than
    `min_points` labelled outcomes get no calibrator, so their confidence is left as the
    detector reported it instead of being fitted to noise.

    Records carry no grouping key here, so there is nothing to hold out and the returned set
    has `brier = None` and no reliability bins — deliberately. An isotonic fit scored on the
    points it was fitted to reports a calibration quality it will not reproduce on anything
    else, and that number sits inside the coverage statement as the report's own evidence
    about how far its confidences can be trusted. Callers that want the diagram use
    `fit_calibrators_grouped`, which fits the shipped curve on everything and scores it on a
    held-out family.
    """
    return fit_calibrators_grouped(
        ((det, score, hit, "") for det, score, hit in records),
        min_points=min_points, exclude_prefixes=exclude_prefixes)


def fit_calibrators_grouped(records: Iterable[tuple[str, float, bool, str]],
                            min_points: int = MIN_POINTS,
                            exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES
                            ) -> CalibrationSet:
    """Fit per detector, and score the fit OUT OF FOLD — `backend_plan.md:1532`, gate B7:
    *"reliability diagram + Brier on the held-out family"*.

    `records` = (detector_id, raw score, was_attack, group). `group` is the attack family,
    the only grouping that makes a fold mean anything: families share base images, harness
    and pipeline, so a random split puts siblings of the held-out model in the fitting fold
    and the held-out set is not held out in any useful sense (`bench/protocol.py`).

    Two different things come out, deliberately:

      * the SHIPPED calibrator is fitted on every point, because a curve fitted on less data
        is a worse curve and the report's confidences should use the best one available;
      * `brier` and `reliability_bins` are computed only from predictions made by a curve
        that never saw the point being predicted. An isotonic fit scored in sample reports a
        quality it cannot reproduce, and this number sits inside the coverage statement where
        the plan says the system's own confidence is evidence.

    A group with too few points to leave out (one group, or a fold whose remainder falls
    under `min_points`) contributes a calibrator and no score, never an in-sample score
    dressed up as a held-out one.
    """
    by: dict[str, list[tuple[float, bool, str]]] = {}
    excluded: set[str] = set()
    for det, score, hit, group in records:
        if det.startswith(exclude_prefixes):
            excluded.add(det)
            continue
        by.setdefault(det, []).append((score, hit, group))
    out = CalibrationSet(excluded=sorted(excluded))
    oof_p: list[float] = []
    oof_y: list[float] = []
    for det, pts in sorted(by.items()):
        if len(pts) < min_points:
            continue
        out.calibrators[det] = _fit_one([(s, h) for s, h, _ in pts])
        groups = sorted({g for _, _, g in pts})
        if len(groups) < 2:
            continue                       # nothing to hold out; no score, rather than a lie
        for held in groups:
            fit = [(s, h) for s, h, g in pts if g != held]
            test = [(s, h) for s, h, g in pts if g == held]
            if len(fit) < min_points or not test:
                continue
            # A single-label fold on either side is not a fold. PAV on one class returns the
            # constant curve, so a fit fold with no negatives predicts 1.0 for everything and
            # the score measures the split rather than the detector. Skipping is the honest
            # answer; the alternative is a Brier inflated by construction.
            if len({h for _, h in fit}) < 2 or len({h for _, h in test}) < 2:
                continue
            cal = _fit_one(fit)
            oof_p += [cal(float(s)) for s, _ in test]
            oof_y += [float(h) for _, h in test]
    out.brier, out.bins = _reliability(oof_p, oof_y)
    return out


def apply_calibration(findings: list[Finding], cal: CalibrationSet | None,
                      exclude_prefixes: tuple[str, ...] = EXCLUDE_PREFIXES) -> None:
    if cal is None:
        return
    for f in findings:
        if (f.availability in (Availability.OK, Availability.DEGRADED)
                and not f.detector_id.startswith(exclude_prefixes)
                and f.detector_id in cal.calibrators):
            f.confidence = min(1.0, max(0.0, cal.calibrators[f.detector_id](f.score_raw)))


# --------------------------------------------------------------------------------------
# File I/O. The payload is validated by ONE function on both paths, so a file this module
# writes is guaranteed to load, and a hand-edited or third-party one is held to the same rules.
# --------------------------------------------------------------------------------------
_TOP_KEYS = frozenset({"schema_version", "method", "calibrators", "brier",
                       "reliability_bins", "excluded_detectors"})
_REQUIRED_KEYS = frozenset({"schema_version", "method", "calibrators"})
_BIN_KEYS = frozenset({"p_mean", "empirical", "n"})


def _number(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{what} must be a finite number, got {v!r}")
    return float(v)


def _unit(v: Any, what: str) -> float:
    x = _number(v, what)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{what} must be within [0, 1], got {x!r}")
    return x


def _calibrator_from(det: Any, raw: Any) -> Calibrator:
    if not isinstance(det, str) or not det:
        raise ValueError(f"calibrators: detector ids must be non-empty strings, got {det!r}")
    where = f"calibrators[{det!r}]"
    if det.startswith(EXCLUDE_PREFIXES):
        raise ValueError(f"{where}: {'/'.join(EXCLUDE_PREFIXES)}* detectors are deterministic "
                         "checks and may never carry a calibrator")
    if not isinstance(raw, dict) or set(raw) != {"edges", "values"}:
        raise ValueError(f"{where} must be an object with exactly the keys 'edges' and 'values'")
    if not isinstance(raw["edges"], list) or not isinstance(raw["values"], list):
        raise ValueError(f"{where}: 'edges' and 'values' must be arrays")
    if len(raw["edges"]) != len(raw["values"]):
        raise ValueError(f"{where}: {len(raw['edges'])} edges but {len(raw['values'])} values")
    if not raw["edges"]:
        raise ValueError(f"{where}: 'edges' and 'values' must not be empty")
    edges = [_number(v, f"{where}.edges[{i}]") for i, v in enumerate(raw["edges"])]
    values = [_unit(v, f"{where}.values[{i}]") for i, v in enumerate(raw["values"])]
    if any(b < a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"{where}: edges must be in ascending order")
    # Pool-adjacent-violators output is monotone; a file that is not would INVERT confidence.
    if any(b < a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"{where}: values must be non-decreasing (an isotonic calibrator)")
    return Calibrator(tuple(edges), tuple(values))


def _set_from_payload(data: Any) -> CalibrationSet:
    """Validate a decoded calibration payload and build the set. Raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("calibration file must contain a JSON object")
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level key(s) in calibration file: {unknown}")
    missing = sorted(_REQUIRED_KEYS - set(data))
    if missing:
        raise ValueError(f"calibration file is missing required key(s): {missing}")
    version = data["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration schema_version {version!r} "
                         f"(this build reads {SCHEMA_VERSION})")
    if data["method"] != "isotonic":
        raise ValueError(f"unsupported calibration method {data['method']!r} (only 'isotonic')")
    if not isinstance(data["calibrators"], dict):
        raise ValueError("'calibrators' must be an object mapping detector id to a calibrator")
    cals = {det: _calibrator_from(det, raw) for det, raw in data["calibrators"].items()}

    brier = data.get("brier")
    if brier is not None:
        brier = _unit(brier, "brier")
    raw_bins = data.get("reliability_bins", [])
    if not isinstance(raw_bins, list):
        raise ValueError("'reliability_bins' must be an array")
    bins: list[dict[str, Any]] = []
    for i, b in enumerate(raw_bins):
        if not isinstance(b, dict) or set(b) != _BIN_KEYS:
            raise ValueError(f"reliability_bins[{i}] must be an object with exactly the keys "
                             f"{sorted(_BIN_KEYS)}")
        n = b["n"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"reliability_bins[{i}].n must be a non-negative integer, got {n!r}")
        bins.append({"p_mean": _unit(b["p_mean"], f"reliability_bins[{i}].p_mean"),
                     "empirical": _unit(b["empirical"], f"reliability_bins[{i}].empirical"),
                     "n": n})
    excluded = data.get("excluded_detectors", [])
    if not isinstance(excluded, list) or not all(isinstance(e, str) and e for e in excluded):
        raise ValueError("'excluded_detectors' must be an array of non-empty strings")
    return CalibrationSet(calibrators=cals, brier=brier, bins=bins, excluded=sorted(excluded))


def _payload(cal: CalibrationSet) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "isotonic",
        "calibrators": {det: {"edges": [float(e) for e in c.edges],
                              "values": [float(v) for v in c.values]}
                        for det, c in cal.calibrators.items()},
        "brier": None if cal.brier is None else float(cal.brier),
        "reliability_bins": [{"p_mean": float(b["p_mean"]), "empirical": float(b["empirical"]),
                              "n": int(b["n"])} for b in cal.bins],
        "excluded_detectors": sorted(cal.excluded),
    }


def save_calibration(cal: CalibrationSet, path: Path) -> Path:
    """Write `cal` as deterministic JSON (sorted keys, fixed indent) and return `path`.

    The payload is validated first, so this never writes a file `load_calibration` would
    refuse — a set carrying a `prov.*` calibrator raises ValueError instead of being saved."""
    payload = _payload(cal)
    _set_from_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed in a calibration file")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r} in calibration file")
        out[k] = v
    return out


def load_calibration(path: Path) -> CalibrationSet:
    """Read and validate a calibration file. Raises ValueError on ANY problem — unreadable,
    oversized, not JSON, or failing the checks below — never a partial or repaired result.

    UNTRUSTED INPUT: applying the returned set rewrites every calibrated detector's confidence,
    which drives D3/D5 dispositions. Checked: JSON object; exactly the known top-level keys
    (no unknown ones); `schema_version` supported; `method == "isotonic"`; each calibrator has
    equal-length non-empty `edges` (finite, ascending) and `values` (in [0, 1], non-decreasing);
    no `prov.*` detector carries a calibrator; brier, reliability bins and excluded list are
    well-typed and in range; no duplicate keys, NaN or Infinity."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"calibration file {path} is larger than {MAX_FILE_BYTES} bytes")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"cannot read calibration file {path}: {e}") from e
    try:
        data = json.loads(text, parse_constant=_reject_constant,
                          object_pairs_hook=_reject_duplicate_keys)
    except ValueError as e:                       # JSONDecodeError is a ValueError
        raise ValueError(f"calibration file {path} is not valid: {e}") from e
    try:
        return _set_from_payload(data)
    except ValueError as e:
        raise ValueError(f"calibration file {path} is not valid: {e}") from e
