"""model.data_consistency — a Detector, not a ModelCheck. The strongest reference-free idea
in Module B.

**Why a Detector.** It needs the CONTRIBUTED training data, the dataset under audit.
`ModelCheck.check()` offers `reference_probes`, but that is the REFERENCE_CLEAN_SET — a
different artefact, and there is no way to swap the contributed dataset in. `Detector.detect
(dataset, embeddings, model)` takes exactly this pair, and carries `model` for precisely
this kind of check. Same reason activation clustering is a Detector despite being a
model-side method.

**Both directions matter, and the second is the valuable one.**

Corroboration: Module A finds trigger artefacts in a contributor's samples; this finds the
model responds to that pattern. Neither alone is conclusive; together they are a finding
with no external reference involved.

The contrapositive: take a trigger candidate recovered from the model and search the
supplied training data for it. If the model responds strongly to a pattern ABSENT from the
data we were given, either the model was trained on data we were never shown, or it was
modified after training. That is reference-free, defensible and high severity — and it is
the actual supply-chain question the PS asks: not "is this model bad" but "is this model
the thing you were told it was."
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.finding import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register_detector


def trigger_response(model, x: np.ndarray, mask: np.ndarray, pattern: np.ndarray,
                     target: int) -> float:
    """Fraction of inputs driven to `target` once the candidate trigger is stamped on."""
    stamped = np.clip((1 - mask) * x + mask * pattern, 0, 1).astype(np.float32)
    return float((np.asarray(model.predict(stamped)).argmax(1) == target).mean())


def data_match_rate(dataset_x: np.ndarray, mask: np.ndarray, pattern: np.ndarray,
                    tol: float = 0.12) -> float:
    """Fraction of supplied training images already carrying the candidate pattern,
    measured only where the mask is active."""
    m = mask[0] if mask.ndim == 4 else mask
    active = m > 0.5
    if active.sum() < 4:
        return 0.0
    tgt = (pattern * m)[..., active] if pattern.ndim == 3 else pattern[..., active]
    diffs = np.abs(dataset_x[..., active] - tgt[None, ...]).mean(axis=(1, 2))
    return float((diffs < tol).mean())


@register_detector
class DataConsistencyCheck:
    id = "model.data_consistency"
    version = "1.0.0"
    requires = {Capability.DATASET_IMAGES, Capability.MODEL_PREDICT}
    optional: set = set()
    attack_classes = {"data_model_discrepancy"}

    def detect(self, dataset, embeddings, model, ctx: CheckContext) -> list[Finding]:
        cand = ctx.profile.get("trigger_candidate")
        if cand is None:
            f = unavailable_finding(
                self.id, self.version, getattr(model, "model_id", "?"),
                "no trigger candidate was recovered this scan (model.neural_cleanse "
                "produced none, or ran UNAVAILABLE), so there is nothing to search the "
                "supplied training data for.",
                (), "data_model_discrepancy", Availability.DEGRADED)
            f.scan_id = ctx.scan_id
            return [f]

        mask = np.asarray(cand["mask"], dtype=np.float32)
        pattern = np.asarray(cand["pattern"], dtype=np.float32)
        target = int(cand["target"])
        x = np.asarray(dataset, dtype=np.float32) if not hasattr(dataset, "x") \
            else np.asarray(dataset.x, dtype=np.float32)

        resp = trigger_response(model, x[: min(len(x), 256)], mask, pattern, target)
        present = data_match_rate(x, mask, pattern)
        thr_resp = float(ctx.opt("dc_response_threshold", 0.6))
        thr_present = float(ctx.opt("dc_presence_threshold", 0.005))

        discrepancy = resp >= thr_resp and present < thr_present
        corroborated = resp >= thr_resp and present >= thr_present

        ev = [Evidence("table", "model response vs data presence", data={
            "trigger_drives_to_target": round(resp, 4),
            "fraction_of_training_data_carrying_it": round(present, 6),
            "response_threshold": thr_resp, "presence_threshold": thr_present,
            "target_class": target})]

        if discrepancy:
            reason = (f"The model is driven to class {target} by a pattern for "
                      f"{resp*100:.0f}% of inputs, but that pattern appears in only "
                      f"{present*100:.3f}% of the supplied training data. **The model "
                      "exhibits a behaviour the supplied training data cannot explain** — "
                      "either it was trained on data we were not shown, or it was modified "
                      "after training.")
            sev, disp, rule = Severity.HIGH, Disposition.QUARANTINE, "data_consistency.discrepancy"
            nat, conf = Nature.ADVERSARIAL, min(0.9, resp)
        elif corroborated:
            reason = (f"The model responds to a pattern ({resp*100:.0f}% driven to class "
                      f"{target}) that IS present in {present*100:.2f}% of the supplied "
                      "training data. Model-side and data-side evidence corroborate: this "
                      "looks like a backdoor trained in from the data we were given.")
            sev, disp, rule = Severity.HIGH, Disposition.QUARANTINE, "data_consistency.corroborated"
            nat, conf = Nature.ADVERSARIAL, min(0.85, resp)
        else:
            reason = (f"The recovered candidate does not reliably drive the model to class "
                      f"{target} ({resp*100:.0f}% < {thr_resp*100:.0f}%), so no data/model "
                      "discrepancy can be asserted.")
            sev, disp, rule = Severity.INFO, Disposition.ACCEPT, "data_consistency.no_signal"
            nat, conf = Nature.INDETERMINATE, 0.0

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=getattr(model, "model_id", "?"),
            severity=sev, confidence=float(conf), score_raw=resp, threshold=thr_resp,
            reason=reason, attack_class="data_model_discrepancy", evidence=ev,
            access_assumptions=["contributed training data supplied", "query access to the model",
                                "a trigger candidate was recovered earlier this scan"],
            limitations=[
                "Identifies a DISCREPANCY, not the backdoor itself.",
                "A trigger genuinely present in the supplied data produces corroboration, "
                "not discrepancy — the data explains the behaviour, even though the data is "
                "poisoned. Read with Module A's contributor findings.",
                "Presence matching is pixel-space and tolerant; a semantically equivalent "
                "but pixel-different trigger will read as absent."],
            disposition=disp, disposition_rule=rule, nature=nat,
        )]
