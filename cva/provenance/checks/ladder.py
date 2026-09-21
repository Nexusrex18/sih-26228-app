"""The recompute claim ladder (Module C plan §7.9, decision D6): what a re-derived output may honestly claim.

Re-deriving an inference (`prov.recompute`) is the strongest check in the system because it does not infer, it
re-derives. But "the output hashes equal" is a much bigger claim when the two runs share a bit-exact
environment than when they do not, and one exact hash on a 6-dp confidence would raise a FALSE ALARM on a
clean record whenever a GPU-sealed / CPU-recomputed pair differs by float jitter. So the ladder:

  R0  fine hash (`output.jcs_sha256`) equal, AND the recompute runtime string equals the sealing one
        -> `verified_exact`     certain, bit-for-bit
  R1  coarse hash (`output.decision_sha256`: labels + whole-pixel boxes, no confidences) equal
        -> `verified_decision`  certain about WHAT was detected WHERE — not about how confident
  R2  R1 failed: is every difference explained by float jitter at a decision boundary?
        every differing detection within `eps` of the confidence threshold / a rounding boundary / the NMS
        IoU threshold  -> `boundary_flip`  (medium, indeterminate — REVIEW, never quarantine)
        otherwise                          -> `output_mismatch` (critical)
  R3  the quantised numbers, compared: max confidence / box difference. DIAGNOSTIC ONLY — attached as evidence,
      never carrying a claim of certainty. (A tolerance band is a place to hide a nudged confidence.)

A bitwise claim where only a decision-level claim is true would be dishonest, so the result always says which
rung it reached and why it did not reach the one above.

Pure Python, no numpy: every rung is unit-testable with hand-made dictionaries.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..seal.canonical import canonical_bytes
from ..seal.outputs import build_output_objects

DEFAULT_EPS_CONF = 1e-4        # a confidence within this of the threshold may flip presence on float jitter
DEFAULT_EPS_PX = 1e-3          # a box coordinate within this of a rounding boundary may flip a whole pixel
DEFAULT_EPS_IOU = 1e-4         # an IoU within this of the NMS threshold may flip a suppression
Q64 = 1.0 / 64.0               # the sealed fine boxes carry 1/64 px of quantisation error


@dataclass
class Recomputed:
    """Everything the ladder needs about a re-derived output: the objects and their hashes."""

    raw_floats: Mapping[str, Any]                       # what the pipeline produced before filtering
    filtered_floats: Mapping[str, Any]                  # what it decided (floats, with the filter config)
    raw_obj: Mapping[str, Any]                          # quantised
    fine_obj: Mapping[str, Any]
    coarse_obj: Mapping[str, Any]
    raw_sha256: str
    jcs_sha256: str
    decision_sha256: str


def recompute_objects(raw: Mapping[str, Any], filtered: Mapping[str, Any] | None, *,
                      on_nonfinite: str = "seal_marker") -> Recomputed:
    """Quantise and hash a re-derived output EXACTLY as `Sealer.commit` does — same functions, same order."""
    raw_o, fine_o, coarse_o = build_output_objects(raw, filtered, on_nonfinite=on_nonfinite)
    h = {k: hashlib.sha256(canonical_bytes(o, max_bytes=None)).hexdigest()
         for k, o in (("raw", raw_o), ("fine", fine_o), ("coarse", coarse_o))}
    return Recomputed(raw, filtered if filtered is not None else raw, raw_o, fine_o, coarse_o,
                      h["raw"], h["fine"], h["coarse"])


@dataclass
class LadderResult:
    verdict: str                     # verified_exact | verified_decision | boundary_flip | output_mismatch
    rung: str                        # R0 | R1 | R2
    reason: str
    r0_applicable: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)     # R3 — evidence, never a claim
    explanations: list[str] = field(default_factory=list)         # R2 — why each difference is / is not jitter


# --- geometry -----------------------------------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    import math
    return int(math.floor(x + 0.5))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _iou_slack(a: Sequence[float], b: Sequence[float]) -> float:
    """How far an IoU computed from a SEALED box can be from the truth. A sealed box carries up to Q64/2 of
    rounding error per coordinate; moving one edge of a box with shortest side `s` by δ moves its IoU by at most
    about δ/s per edge. Without this allowance a genuine NMS-boundary flip on a 10 px box (an IoU error of
    ~8e-4 from quantisation alone) would be reported as a mismatch — a false alarm."""
    side = max(1.0, min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]))
    return 4 * (Q64 / 2) / side


def _sealed_dets(fine: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"cls": d["cls"], "conf": d["conf_e6"] / 1e6, "box": [q / 64.0 for q in d["box_q64"]]}
            for d in fine.get("detections", [])]


def _re_dets(filtered: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"cls": d["cls"], "conf": float(d["conf"]), "box": [float(x) for x in d["box"]]}
            for d in filtered.get("detections", [])]


# --- R3: diagnostics ----------------------------------------------------------------------------------------

def diagnostics(sealed_fine: Mapping[str, Any], re_fine: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the QUANTISED numbers. Never a claim, always evidence."""
    out: dict[str, Any] = {"task": sealed_fine.get("task")}
    if sealed_fine.get("task") == "classify":
        s, r = sealed_fine.get("top", []), re_fine.get("top", [])
        out["max_abs_conf_e6"] = max((abs(a["conf_e6"] - b["conf_e6"]) for a, b in zip(s, r, strict=False)), default=0)
        out["sealed_classes"], out["recomputed_classes"] = [d["cls"] for d in s], [d["cls"] for d in r]
        return out
    s, r = sealed_fine.get("detections", []), re_fine.get("detections", [])
    out["sealed_detections"], out["recomputed_detections"] = len(s), len(r)
    pairs = list(zip(s, r, strict=False))
    out["max_abs_conf_e6"] = max((abs(a["conf_e6"] - b["conf_e6"]) for a, b in pairs), default=0)
    out["max_abs_box_q64"] = max((abs(x - y) for a, b in pairs for x, y in zip(a["box_q64"], b["box_q64"], strict=False)),
                                 default=0)
    return out


# --- R2: is every difference explained by jitter at a decision boundary? ------------------------------------

def _filter_thresholds(sealed_fine: Mapping[str, Any], re_filtered: Mapping[str, Any]) -> tuple[float | None, float | None]:
    sf = sealed_fine.get("filter") or {}
    rf = re_filtered.get("filter") or {}
    thr = sf["conf_thr_e6"] / 1e6 if "conf_thr_e6" in sf else (float(rf["conf_thr"]) if "conf_thr" in rf else None)
    nms = sf["nms_iou_e6"] / 1e6 if "nms_iou_e6" in sf else (float(rf["nms_iou"]) if "nms_iou" in rf else None)
    return thr, nms


def _classify_boundary(sealed_fine: Mapping[str, Any], re_filtered: Mapping[str, Any], eps: float) -> tuple[bool, list[str]]:
    s = [(d["cls"], d["conf_e6"] / 1e6) for d in sealed_fine.get("top", [])]
    r = [(d["cls"], float(d["conf"])) for d in re_filtered.get("top", [])]
    why: list[str] = []
    if len(s) != len(r):
        return False, [f"the number of ranked classes differs ({len(s)} sealed, {len(r)} recomputed)"]
    ok = True
    for i, ((ca, pa), (cb, pb)) in enumerate(zip(s, r, strict=True)):
        if ca == cb:
            continue
        if abs(pa - pb) <= eps:
            why.append(f"rank {i + 1}: class {ca} vs {cb} is a near-tie ({pa:.6f} vs {pb:.6f}, within {eps:g})")
        else:
            ok = False
            why.append(f"rank {i + 1}: class {ca} ({pa:.6f}) became class {cb} ({pb:.6f}) — not a tie")
    return ok, why


def _detect_boundary(sealed_fine: Mapping[str, Any], re_filtered: Mapping[str, Any], eps_conf: float, eps_px: float,
                     eps_iou: float, re_raw: Mapping[str, Any] | None = None) -> tuple[bool, list[str]]:
    thr, nms = _filter_thresholds(sealed_fine, re_filtered)
    S, R = _sealed_dets(sealed_fine), _re_dets(re_filtered)
    RAW = _re_dets(re_raw) if re_raw is not None else []
    why: list[str] = []
    ok = True
    used: set[int] = set()
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for s in S:                                                     # greedy: same class, nearest box within 1 px
        best, best_d = None, 1.0 + 2 * Q64
        for j, r in enumerate(R):
            if j in used or r["cls"] != s["cls"]:
                continue
            d = max(abs(a - b) for a, b in zip(s["box"], r["box"], strict=True))
            if d <= best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            matched.append((s, R[best]))
    matched_s = {id(s) for s, _ in matched}
    matched_r = {id(R[j]) for j in used}

    for s, r in matched:
        for k, (x, y) in enumerate(zip(s["box"], r["box"], strict=True)):
            if _round_half_up(x) == _round_half_up(y):
                continue
            if abs(x - y) <= eps_px + Q64:
                why.append(f"class {s['cls']} box coordinate {k}: {x:.4f} vs {y:.4f} straddles a whole-pixel "
                           f"rounding boundary (Δ {abs(x - y):.5f} px ≤ {eps_px + Q64:.5f})")
            else:
                ok = False
                why.append(f"class {s['cls']} box coordinate {k} moved {abs(x - y):.4f} px "
                           f"({_round_half_up(x)} -> {_round_half_up(y)}): a real change, not a rounding flip")

    def candidate(d: dict[str, Any]) -> dict[str, Any] | None:
        """The re-derived PRE-FILTER detection the same class and (within 1 px) box as `d`, or None."""
        best, best_d = None, 1.0 + 2 * Q64
        for c in RAW:
            if c["cls"] != d["cls"]:
                continue
            dist = max(abs(a - b) for a, b in zip(d["box"], c["box"], strict=True))
            if dist <= best_d:
                best, best_d = c, dist
        return best

    def presence(d: dict[str, Any], side: str, other_kept: list[dict[str, Any]]) -> None:
        nonlocal ok
        if side == "sealed":
            # A detection ONLY the sealer claims. Its own confidence is whatever the sealer chose, so it can never be
            # what excuses it: the recomputation must independently have produced that candidate before the filter,
            # and it is THAT confidence that is compared with the threshold (plan §7.9; the fabricated near-threshold
            # detection is otherwise a certain tamper downgraded to "review").
            cand = candidate(d)
            if cand is None:
                ok = False
                why.append(f"class {d['cls']} detection only in the sealed output (confidence {d['conf']:.6f}) has no "
                           "counterpart, at any confidence, in the re-derived pre-filter output: the pipeline never "
                           "produced it")
                return
            d = {**d, "conf": cand["conf"]}
        if thr is not None and abs(d["conf"] - thr) <= eps_conf:
            why.append(f"class {d['cls']} detection only in the {side} output has confidence {d['conf']:.6f}, "
                       f"within {eps_conf:g} of the threshold {thr:.6f}")
            return
        if nms is not None:
            higher = [o for o in other_kept if o["cls"] == d["cls"] and o["conf"] >= d["conf"]]
            scored = [(_iou(d["box"], o["box"]), _iou_slack(d["box"], o["box"])) for o in higher]
            near = [(v, sl) for v, sl in scored if abs(v - nms) <= eps_iou + sl]
            if near:
                v, sl = min(near, key=lambda t: abs(t[0] - nms))
                why.append(f"class {d['cls']} detection only in the {side} output overlaps a kept box at IoU "
                           f"{v:.6f}, within {eps_iou:g} (+{sl:.4f} for the sealed boxes' 1/64 px quantisation) of "
                           f"the NMS threshold {nms:.6f}")
                return
        ok = False
        near_thr = f"threshold {thr:.6f}" if thr is not None else "an unknown threshold"
        why.append(f"class {d['cls']} detection only in the {side} output (confidence {d['conf']:.6f}) is not "
                   f"explained by float jitter: it is not within {eps_conf:g} of {near_thr}")

    for s in S:
        if id(s) not in matched_s:
            presence(s, "sealed", R)
    for r in R:
        if id(r) not in matched_r:
            presence(r, "recomputed", S)
    return ok, why


def analyse_boundary(sealed_fine: Mapping[str, Any], re_filtered: Mapping[str, Any], *, eps_conf: float = DEFAULT_EPS_CONF,
                     eps_px: float = DEFAULT_EPS_PX, eps_iou: float = DEFAULT_EPS_IOU,
                     re_raw: Mapping[str, Any] | None = None) -> tuple[bool, list[str]]:
    """(explained_by_jitter, one explanation per difference). `re_raw` is the re-derived PRE-FILTER output: without
    it a detection that only the sealed output holds can never be explained as jitter."""
    if sealed_fine.get("task") != re_filtered.get("task"):
        return False, [f"the task differs ({sealed_fine.get('task')!r} sealed, {re_filtered.get('task')!r} recomputed)"]
    if sealed_fine.get("task") == "classify":
        return _classify_boundary(sealed_fine, re_filtered, eps_conf)
    if sealed_fine.get("task") == "detect":
        return _detect_boundary(sealed_fine, re_filtered, eps_conf, eps_px, eps_iou, re_raw)
    return False, ["the sealed output is not a classification or detection result"]


# --- the ladder ------------------------------------------------------------------------------------------------

def climb(sealed: Mapping[str, Any], sealed_fine: Mapping[str, Any] | None, re: Recomputed, *, sealed_runtime: str,
          recompute_runtime: str, eps_conf: float = DEFAULT_EPS_CONF, eps_px: float = DEFAULT_EPS_PX,
          eps_iou: float = DEFAULT_EPS_IOU) -> LadderResult:
    """Compare a re-derived output with a sealed one. `sealed` is the record's `output` section (hashes);
    `sealed_fine` the stored fine payload (needed only to explain a mismatch — `None` if it is missing)."""
    r0 = sealed_runtime == recompute_runtime
    diag = diagnostics(sealed_fine, re.fine_obj) if sealed_fine is not None else {}
    fine_ok = re.jcs_sha256 == sealed["jcs_sha256"]
    coarse_ok = re.decision_sha256 == sealed["decision_sha256"]
    # The coarse hash is computed from the ORIGINAL floats, so it is not implied by the fine one: two outputs can
    # quantise to the same 1/64 px yet round to different whole pixels. Calling that "bit-identical" would be
    # false, so R0 needs BOTH hashes; a fine match with a coarse difference is a rounding-boundary question (R2).
    if fine_ok and coarse_ok:
        if r0:
            return LadderResult("verified_exact", "R0", "the recomputed output is bit-identical to the sealed one "
                                f"(same runtime: {recompute_runtime})", True)
        return LadderResult("verified_decision", "R1",
                            "the recomputed output is bit-identical to the sealed one, but the runtimes differ "
                            f"(sealed on {sealed_runtime!r}, recomputed on {recompute_runtime!r}), so only the "
                            "decision-level claim is made", False, diag)
    if coarse_ok:
        why = ("R0 attempted (same runtime) and the confidences differ in the last quanta; the decisions match — "
               "diagnostic only" if r0 else
               f"R0 not attempted: sealed on {sealed_runtime!r}, recomputed on {recompute_runtime!r}")
        return LadderResult("verified_decision", "R1", f"what was detected where matches exactly; {why}", r0, diag)
    if sealed_fine is None:
        return LadderResult("output_mismatch", "R2",
                            "the decision-level hash does not match and the sealed payload is unavailable, so the "
                            "difference cannot be explained as float jitter", r0, diag,
                            ["the sealed decision payload is missing"])
    explained, why_list = analyse_boundary(sealed_fine, re.filtered_floats, re_raw=re.raw_floats, eps_conf=eps_conf, eps_px=eps_px,
                                           eps_iou=eps_iou)
    if explained:
        return LadderResult("boundary_flip", "R2",
                            "the decisions differ, but every difference sits within float jitter of a decision "
                            "boundary — indeterminate, for review, not proof of tampering", r0, diag, why_list)
    return LadderResult("output_mismatch", "R2",
                        "the recomputed decisions differ from the sealed ones by more than float jitter can explain",
                        r0, diag, why_list)
