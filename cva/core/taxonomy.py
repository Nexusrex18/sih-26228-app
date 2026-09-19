"""The single `attack_class` list, owned by Backend.

Module C open item **O3**, adopted in backend_plan.md §9.5. Each module plan defines its
own `attack_class` strings and the coverage generator groups on them — so a typo, a
synonym or a drifted string makes a detector's coverage row **silently vanish**. That is
the worst failure shape this system has: a shorter coverage statement looks like a
cleaner result.

One list, here, and **a detector registering an `attack_class` not in it is a startup
error, not a warning** — the same rule as unknown profile keys, for the same reason.

**Why `kind` and not a severity filter.** An earlier draft proposed filtering coverage on
`severity == "warning"`. Two defects in one line: `warning` is not in the frozen severity
enum (`info | low | medium | high | critical`), *and severity is the wrong axis anyway* —
once `degraded_gap` maps to `low`, filtering `low` out of coverage would also drop genuine
low-severity attack detections. Filter on a property of the **class**, not of the
instance. The coverage generator counts `kind == "attack"` only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Module = Literal["A", "B", "C", "D", "core"]
Kind = Literal["attack", "operational"]


@dataclass(frozen=True)
class AttackClass:
    name: str
    module: Module
    kind: Kind
    desc: str


def _c(name: str, module: Module, kind: Kind, desc: str) -> AttackClass:
    return AttackClass(name, module, kind, desc)


# --- Module A — training-data integrity (PS §2.2.1) ------------------------
_A = [
    _c("near_duplicate_flooding", "A", "attack",
       "Near-duplicate images inflating a class or a contributor's apparent volume"),
    _c("label_flipping", "A", "attack", "Labels changed to a wrong class"),
    _c("systematic_mislabelling", "A", "attack",
       "A consistent, directed labelling error rather than scattered noise"),
    _c("trigger_injection", "A", "attack", "A trigger pattern planted in training images"),
    _c("out_of_distribution", "A", "attack", "Imagery that does not belong to the task at all"),
    _c("negative_space_poisoning", "A", "attack",
       "Poisoning by what is NOT annotated — objects present and deliberately unlabelled"),
    _c("annotation_geometry_tamper", "A", "attack",
       "Boxes subtly shifted, shrunk or grown to degrade localisation"),
    _c("duplicate_label_conflict", "A", "attack",
       "The same object annotated twice with conflicting classes"),
    _c("script_generated_batch", "A", "attack",
       "A contribution whose metadata shows machine generation, not collection"),
]

# --- Module B — model integrity (PS §2.2.2) --------------------------------
# Confirmed by ML-2, Plan/Module-B-Model-Integrity-Plan.md §5.
_B = [
    _c("model_substitution", "B", "attack", "Supplied model is not the declared model"),
    _c("benign_conversion", "B", "operational",
       "Digest mismatch, fingerprint inside tolerance — re-export, opset change or "
       "quantisation. ADR-007: this must NOT read as 'SUBSTITUTED'"),
    _c("weight_anomaly", "B", "attack",
       "Per-layer weight distribution diverges from the reference battery"),
    _c("activation_anomaly", "B", "attack",
       "Activation distribution anomaly on clean probes"),
    _c("backdoor_trigger", "B", "attack", "Trigger-conditioned backdoor"),
    _c("architectural_backdoor", "B", "attack", "Malice in the graph, not in the weights"),
    # `model_anomalous` is `attack`, decided in backend_plan.md §9.5 against ML-2's flag
    # that it was a judgment call. PS §2.2.2 names "anomalous" as one of three required
    # model verdicts, so it must appear as CLAIMED COVERAGE even though it is a generic
    # divergence report rather than a named attacker goal. Classing it `operational`
    # would drop a literally-enumerated PS requirement out of the coverage statement.
    _c("model_anomalous", "B", "attack",
       "Corrupted, truncated or badly trained — PS §2.2.2's third verdict, not "
       "classifiable as any of the above"),
    _c("data_model_discrepancy", "B", "attack",
       "Model behaviour the supplied training data cannot explain"),
]

# --- Module C — inference provenance (PS §2.2.3) ---------------------------
# C names THE OBSERVATION (`record_edit`), where A and D name THE ATTACK
# (`label_flipping`). That difference is deliberate and correct: a provenance check does
# not infer an attacker's goal, it reports which arithmetic failed, and forcing C's
# strings into intent-shaped names would overclaim exactly where the module is strongest.
_C_TAMPER = [
    "record_edit", "record_replace", "record_replay", "record_delete", "record_reorder",
    "input_swap", "model_swap", "output_payload_mismatch", "output_mismatch",
    "boundary_flip", "tail_truncation", "ledger_fork", "key_unauthorised", "chain_broken",
    "nonce_reuse", "genesis_mismatch", "checkpoint_mismatch", "ledger_incomplete",
    "non_canonical_encoding", "derived_column_mismatch",
]
_C = [_c(n, "C", "attack", f"Provenance verification failure: {n.replace('_', ' ')}")
      for n in _C_TAMPER] + [
    _c("degraded_gap", "C", "operational",
       "A DECLARED interval in which the ledger was unwritable — a true report of a known "
       "condition, not an attack. Maps to severity `low`"),
    _c("clock_regression", "C", "operational",
       "NTP stepped the host clock backwards. Must NEVER affect disposition"),
]

# --- Module D — distribution shift (PS §2.2.4) -----------------------------
_D = [
    _c("distribution_shift", "D", "attack", "Deviation from the declared reference distribution"),
    _c("semantic_shift", "D", "attack", "The content mix moved, not just the pixel statistics"),
    _c("suspicious_manipulation", "D", "attack",
       "Shift whose shape is better explained by manipulation than by drift — capped at "
       "`review`, never `quarantine` alone"),
]

# --- core — the two the orchestrator itself emits --------------------------
# Neither is in any module plan's list, because neither is a module's to emit. Both are
# Backend amendments, declared here rather than invented three ways in three packages.
_CORE = [
    _c("unsafe_artifact", "core", "attack",
       "The artefact was refused before loading — deserialisation payload, custom ONNX "
       "operator, decompression bomb. An attack on the ASSURANCE TOOL, and the one attack "
       "class we detect without running a single detector"),
    _c("tool.error", "core", "operational",
       "A check raised. A defect in the assurance tool, never a property of the model, and "
       "never folded into DEGRADED"),
    _c("not_assessed", "core", "operational",
       "A registered check did not run — capability or budget. Carries the UNAVAILABLE "
       "finding when a check declares more than one attack class and none of them is the "
       "honest single answer"),
]

TAXONOMY: dict[str, AttackClass] = {a.name: a for a in (_A + _B + _C + _D + _CORE)}


class UnknownAttackClass(KeyError):
    """Raised at registration/startup, never at render time.

    A warning here would be the worst of both worlds: the run completes, the report looks
    clean, and one detector's coverage row is simply absent.
    """


def require(name: str) -> AttackClass:
    """Look up an attack class, or fail loudly. The registry calls this at startup."""
    try:
        return TAXONOMY[name]
    except KeyError:
        near = sorted(k for k in TAXONOMY if k[:4] == name[:4] or name in k)
        raise UnknownAttackClass(
            f"attack_class {name!r} is not in core/taxonomy.py. Every class the coverage "
            f"generator groups on must be declared there — an undeclared string makes the "
            f"detector's coverage row silently vanish."
            + (f" Did you mean: {', '.join(near)}?" if near else "")
        ) from None


def of_kind(kind: Kind) -> set[str]:
    return {k for k, v in TAXONOMY.items() if v.kind == kind}


def of_module(module: Module) -> set[str]:
    return {k for k, v in TAXONOMY.items() if v.module == module}


#: What the coverage generator counts. `operational` entries are true reports about the
#: system's own state, not attack classes we claim to detect; counting them would inflate
#: the statement with things no attacker does.
CLAIMABLE: set[str] = of_kind("attack")
