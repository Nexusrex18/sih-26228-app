"""The disposition table (backend_plan.md §7.9), evaluated from the PROFILE.

The rules are data (`profile.disposition.rules`), first match wins, and whichever fired is
written to `Finding.disposition_rule`. Three orderings are load-bearing and each has a
failure that flatters the tool:

  * D1 is scoped to `prov.*` at `severity >= high`. `prov.*` findings bypass calibration and
    arrive at confidence 1.0, so an unfloored D1 quarantines boundary_flip, degraded_gap and
    clock_regression — the exact false alarm boundary_flip was approved to remove. And an
    unscoped D1 would swallow every `model.weight_digest` mismatch before D2 ran.
  * D2 keys on `attack_class == "benign_conversion"` and sits ABOVE D3, or a benign
    conversion matches D3 first. The tolerance band is detector-internal; the engine sees
    Findings, not fingerprints.
  * D5 floors at `severity >= low`, so an `info` finding never reaches `review` on
    confidence alone.

The engine owns `disposition` and `disposition_rule` for every finding that ran and never
reads a detector-set `Finding.disposition`. A method-level ceiling a detector's author
intended (C1..C3) is therefore a `cap` row in the profile, not a field the engine honours.
"""
from __future__ import annotations

from typing import Any

from cva.core.capability import Availability
from cva.core.taxonomy import TAXONOMY
from cva.core.types import Disposition, Finding, Severity

_SEV = {s.value: s.rank for s in Severity}
_DISP = {Disposition.ACCEPT: 0, Disposition.REVIEW: 1, Disposition.QUARANTINE: 2}

#: D6 is written as `module:D` because the schema's `attack_class` is a single string and
#: Module D owns three classes. Placed BEFORE D3 so a drift finding is capped, not quarantined.
DEFAULT_RULES: list[dict[str, Any]] = [
    {"id": "D1", "detector_prefix": "prov.", "min_severity": "high",
     "disposition": "quarantine",
     "description": "deterministic provenance failure"},
    {"id": "D2", "attack_class": "benign_conversion", "disposition": "review",
     "description": "benign conversion: never the same rule as a genuine substitution"},
    {"id": "D6", "attack_class": "module:D", "min_confidence": 0.6, "min_severity": "low",
     "disposition": "review", "cap": "review",
     "description": "drift-vs-manipulation: capped at review, never quarantine alone"},
    # C1..C3 are method ceilings: facts about a detector, held here as profile data because the
    # engine never reads a detector-set `Finding.disposition`. Each sits ABOVE D3 so a method
    # that "cannot justify quarantine" is capped at review whatever its confidence. All three
    # floor at `low` so an `info` finding (nominal, or a check that could not run) falls through
    # to D5/D7 and is not pulled into review, and none sets a `min_confidence`: these methods'
    # own confidence tops out near 0.6, and D5's floor would silently accept a flagged outlier.
    {"id": "C1", "detector_prefix": "model.intrinsic_probes", "min_severity": "low",
     "disposition": "review", "cap": "review",
     "description": "intrinsic probes are individually weak signals whose value is the ranking "
                    "they hand Neural Cleanse; a high score alone cannot justify quarantine"},
    {"id": "C2", "attack_class": "activation_anomaly", "min_severity": "low",
     "disposition": "review", "cap": "review",
     "description": "the activation-statistics half of weight_stats is reference-free and "
                    "detects gross structural damage, not a backdoor: capped at review"},
    {"id": "C3", "attack_class": "model_anomalous", "min_severity": "low",
     "disposition": "review", "cap": "review",
     "description": "'something is wrong and we cannot name it' is deliberately not an attack "
                    "attribution: capped at review, never quarantine"},
    {"id": "D3", "min_confidence": 0.9, "min_severity": "high", "disposition": "quarantine"},
    {"id": "D4", "min_posterior": 0.25, "requires_ci_excludes_cohort": True,
     "disposition": "quarantine",
     "description": "contributor posterior; evaluated on contributor rows, not findings"},
    {"id": "D5", "min_confidence": 0.6, "min_severity": "low", "disposition": "review"},
    {"id": "D7", "disposition": "accept", "description": "otherwise"},
]


def default_policy() -> dict[str, Any]:
    return {"rules": [dict(r) for r in DEFAULT_RULES]}


def _matches(rule: dict[str, Any], f: Finding) -> bool:
    if rule.get("min_posterior") is not None:
        return False                                  # a contributor rule (D4)
    prefix = rule.get("detector_prefix")
    if prefix and not f.detector_id.startswith(prefix):
        return False
    want = rule.get("attack_class")
    if want:
        if want.startswith("module:"):
            cls = TAXONOMY.get(f.attack_class)
            if cls is None or cls.module != want.split(":", 1)[1]:
                return False
        elif f.attack_class != want:
            return False
    if rule.get("min_confidence") is not None and f.confidence < rule["min_confidence"]:
        return False
    return not (rule.get("min_severity") is not None
                and f.severity.rank < _SEV[rule["min_severity"]])


def decide(policy: dict[str, Any], f: Finding) -> tuple[Disposition, str]:
    for rule in policy["rules"]:
        if not _matches(rule, f):
            continue
        disp = Disposition(rule["disposition"])
        cap = rule.get("cap")
        if cap and _DISP[disp] > _DISP[Disposition(cap)]:
            disp = Disposition(cap)
        return disp, rule["id"]
    return Disposition.ACCEPT, "D7"


def apply_dispositions(findings: list[Finding], policy: dict[str, Any]) -> None:
    """Route every finding that actually ran. A finding for a check that could not run keeps
    the disposition the orchestrator gave it: the gap, not a judgement about the data."""
    for f in findings:
        if f.availability in (Availability.OK, Availability.DEGRADED):
            f.disposition, f.disposition_rule = decide(policy, f)


def defer_digest_to_fingerprint(findings: list[Finding]) -> None:
    """§7.9: the fingerprint owns the substitution verdict whenever it ran.

    `model.weight_digest` and `model.fingerprint` both emit `model_substitution`. On a benign
    conversion that produced `benign_conversion` (D2 -> review) AND a bare digest mismatch at
    critical (D3 -> quarantine): D2 fixed one row and did nothing about the other, so the
    routine conversion still read SUBSTITUTED. The digest is still computed, still reported
    and still auditable; it stops racing the detector that can tell the two causes apart.
    """
    ran = (Availability.OK, Availability.DEGRADED)
    if not any(f.detector_id == "model.fingerprint" and f.availability in ran
               for f in findings):
        return
    for f in findings:
        if (f.detector_id == "model.weight_digest" and f.attack_class == "model_substitution"
                and f.availability in ran and f.severity != Severity.INFO):
            f.severity = Severity.INFO
            f.reason = ("Digest comparison recorded; the substitution verdict is deferred to "
                        "model.fingerprint, which ran. " + f.reason)
            f.limitations.append("Severity lowered to info because model.fingerprint ran "
                                 "and owns the substitution verdict (plan §7.9).")


def contributor_disposition(policy: dict[str, Any], posterior: float,
                            excludes_cohort: bool) -> str:
    """D4 on a contributor row. A row that does not meet it is `review` when its interval
    still excludes the cohort rate, else `accept` — never quarantine on the point estimate."""
    for rule in policy["rules"]:
        floor = rule.get("min_posterior")
        if floor is None:
            continue
        if posterior >= floor and (excludes_cohort or not rule.get("requires_ci_excludes_cohort")):
            return str(rule["disposition"])
    return "review" if excludes_cohort else "accept"
