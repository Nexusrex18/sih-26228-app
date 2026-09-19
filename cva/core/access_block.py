"""§7.4's access-assumptions block, generated from the probed CapabilitySet."""
from __future__ import annotations

from typing import Any

from cva.core.capability import Availability, Capability, CapabilitySet

WIDTH = 27

# Printed in this order regardless of what is present: a capability that is absent has to
# occupy a line, or the block silently shortens and a missing row reads as a clean scan.
ROWS: list[tuple[str, Capability]] = [
    ("Weights", Capability.MODEL_WEIGHTS),
    ("Activations", Capability.MODEL_ACTIVATIONS),
    ("Gradients", Capability.MODEL_GRADIENTS),
    ("Architecture", Capability.MODEL_ARCHITECTURE),
    ("Logits", Capability.MODEL_LOGITS),
    ("Dataset images", Capability.DATASET_IMAGES),
    ("Dataset labels", Capability.DATASET_LABELS),
    ("Contributor metadata", Capability.DATASET_CONTRIBUTOR_META),
    ("Reference clean set", Capability.REFERENCE_CLEAN_SET),
    ("Suspect inputs", Capability.SUSPECT_INPUTS),
    ("Reference battery", Capability.REFERENCE_MODEL_BATTERY),
    ("Registered fingerprint", Capability.REFERENCE_MANIFEST),
    ("Inference ledger", Capability.INFERENCE_LEDGER),
    ("Signing key", Capability.SIGNING_KEY),
]


def _line(label: str, present: bool, note: str | None, detail: str | None) -> str:
    dots = "." * max(1, WIDTH - len(label))
    if present:
        tail = f"AVAILABLE ({detail})" if detail else "AVAILABLE"
    else:
        tail = f"NOT AVAILABLE — {note}" if note else "NOT AVAILABLE"
    return f"  {label} {dots} {tail}"


def render_access_block(
    caps: CapabilitySet,
    model_fmt: str | None = None,
    model_detail: str | None = None,
    details: dict[Capability, str] | None = None,
    plan: list[Any] | None = None,
) -> str:
    details = details or {}
    out = ["Access assumptions for this scan"]
    if model_fmt:
        # NOT run through _line(): the model format is a FACT about the artefact, not a
        # capability that can be AVAILABLE or not. §7.4 prints it bare, and wrapping it in
        # "AVAILABLE (...)" would invite reading a format as a permission.
        label = "Model format"
        dots = "." * max(1, WIDTH - len(label))
        out.append(f"  {label} {dots} {model_detail or model_fmt}")
    for label, cap in ROWS:
        out.append(_line(label, cap in caps.caps, caps.note_for(cap), details.get(cap)))
    if plan is not None:
        out.append("  Consequence: " + consequence(plan))
    return "\n".join(out)


def consequence(plan: list[Any]) -> str:
    """The one line that makes the block actionable — what the gaps cost this scan."""
    cap_u = sum(1 for r in plan if r.resolution.state == Availability.UNAVAILABLE
                and r.resolution.exclusion_reason != "budget")
    bud_u = sum(1 for r in plan if r.resolution.state == Availability.UNAVAILABLE
                and r.resolution.exclusion_reason == "budget")
    deg = sum(1 for r in plan if r.resolution.state == Availability.DEGRADED)
    err = sum(1 for r in plan if r.resolution.state == Availability.ERROR)
    parts = [f"{cap_u} check(s) UNAVAILABLE (capability)",
             f"{deg} DEGRADED", f"{bud_u} UNAVAILABLE (budget)"]
    if err:
        parts.append(f"{err} ERROR")
    if not plan:
        return "no checks registered — this scan assessed nothing."
    return ", ".join(parts) + "."
