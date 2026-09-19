"""model.anomalous — PS 2.2.2's THIRD verdict.

The clause names "anomalous, substituted or backdoor-like". Substitution and backdoor have
their own checks; this is the honest catch-all for a model that is neither, but is
corrupted, truncated, badly trained or simply strange against the probe set.

Split into its own file 2026-09-19 to match the ratified layout in
Plan/Module-B-Model-Integrity-Plan.md §11 — it is a distinct verdict, not a weight statistic.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.finding import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register


@register
class AnomalousBehaviourCheck:
    id = "model.anomalous"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET}
    optional: set = set()
    attack_classes = {"model_anomalous"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x, y = ctx.probes_x, ctx.probes_y
        p = np.asarray(model.predict(x.astype(np.float32)))
        pred = p.argmax(1)
        acc = float((pred == y).mean()) if y is not None else float("nan")
        conf = float(p.max(1).mean())
        K = model.num_classes
        used = len(np.unique(pred))
        dist = np.bincount(pred, minlength=K) / len(pred)
        collapse = float(dist.max())

        problems = []
        if not np.isnan(acc) and acc < float(ctx.opt("min_task_acc", 0.4)):
            problems.append(f"accuracy {acc:.3f} is near chance ({1/K:.2f})")
        if used < max(2, K // 2):
            problems.append(f"only {used} of {K} classes are ever predicted")
        if collapse > float(ctx.opt("collapse_threshold", 0.6)):
            problems.append(f"{collapse*100:.0f}% of predictions fall in one class")
        if conf > 0.999:
            problems.append("predictions are saturated at ~1.0 confidence")

        flagged = bool(problems)
        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.MEDIUM if flagged else Severity.INFO,
            confidence=0.8 if flagged else 0.0,
            score_raw=collapse, threshold=float(ctx.opt("collapse_threshold", 0.6)),
            reason=("Model behaves anomalously without matching a known attack signature: "
                    + "; ".join(problems) + "."
                    if flagged else
                    f"Behaviour is nominal on the probe set (accuracy {acc:.3f}, "
                    f"{used}/{K} classes used, mean confidence {conf:.3f})."),
            attack_class="model_anomalous",
            evidence=[Evidence("table", "behavioural summary", data={
                "probe_accuracy": round(acc, 4), "mean_confidence": round(conf, 4),
                "classes_predicted": f"{used}/{K}",
                "largest_class_share": round(collapse, 4),
                "prediction_distribution": [round(float(v), 4) for v in dist]})],
            access_assumptions=["query access only"],
            limitations=["This is the 'something is wrong and we cannot name it' verdict. "
                         "It is deliberately not an attack attribution."],
            disposition=Disposition.REVIEW if flagged else Disposition.ACCEPT,
            disposition_rule="anomalous.heuristics" if flagged else "anomalous.nominal",
            nature=Nature.QUALITY if flagged else Nature.INDETERMINATE,
        )]
