"""Weight and activation statistics, plus the generic 'anomalous' verdict.

PS 2.2.2 names four approaches: fingerprinting, trigger reconstruction, "parameter OR
ACTIVATION statistics", and comparison against a reference battery. Activation statistics
as a MODEL check was missing from the plan — the activation work lived in Module A
answering a data question. It needs MODEL_ACTIVATIONS, so it runs on ONNX where Neural
Cleanse cannot.

2.2.2 also names THREE verdicts — "anomalous, substituted or backdoor-like". model.anomalous
is the third: a model that is neither substituted nor backdoored but is corrupted,
truncated, badly trained or simply strange against the battery.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.finding import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register


def layer_moments(weights: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out = {}
    for name, w in weights.items():
        a = np.asarray(w, dtype=np.float64).ravel()
        if a.size < 2:
            continue
        out[name] = {
            "mean": float(a.mean()), "std": float(a.std()),
            "kurtosis": float(((a - a.mean()) ** 4).mean() / max(a.std() ** 4, 1e-12) - 3),
            "max_abs": float(np.abs(a).max()),
            "frac_gt_3sigma": float((np.abs(a - a.mean()) > 3 * a.std()).mean()),
        }
    return out


@register
class WeightStatisticsCheck:
    id = "model.weight_statistics"
    version = "1.0.0"
    requires = {Capability.MODEL_WEIGHTS}
    optional = {Capability.REFERENCE_MODEL_BATTERY}
    attack_classes = {"model.weight_modification"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        w = model.get_weights()
        mine = layer_moments(w)
        battery = (ctx.battery.models if ctx.battery else None) or []

        if not battery:
            f = unavailable_finding(
                self.id, self.version, model.model_id,
                "no reference model battery was supplied, so per-layer moments have "
                "nothing to be anomalous against. Reporting the model's own statistics only.",
                (Capability.REFERENCE_MODEL_BATTERY,),
                "model.weight_modification", Availability.DEGRADED)
            f.scan_id = ctx.scan_id
            f.evidence.append(Evidence("table", "per-layer weight moments", data={
                k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in mine.items()}))
            return [f]

        ref = [layer_moments(b.get_weights() or {}) for b in battery]
        rows, worst_z, worst_layer = {}, 0.0, ""
        for layer, stats in mine.items():
            vals = [r[layer]["std"] for r in ref if layer in r]
            if len(vals) < 2:
                continue
            mu, sd = float(np.mean(vals)), float(np.std(vals)) or 1e-9
            z = abs(stats["std"] - mu) / sd
            rows[layer] = {"std": round(stats["std"], 5), "battery_mean": round(mu, 5),
                           "z": round(z, 2)}
            if z > worst_z:
                worst_z, worst_layer = z, layer

        thr = float(ctx.opt("weight_stat_z", 4.0))
        flagged = worst_z > thr
        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.HIGH if flagged else Severity.INFO,
            confidence=float(np.clip(worst_z / (thr * 2), 0, 0.9)),
            score_raw=worst_z, threshold=thr,
            reason=(f"Layer '{worst_layer}' weight spread is {worst_z:.1f}σ from the "
                    f"reference battery mean, above the {thr}σ threshold. Consistent with "
                    "weight-level modification after training."
                    if flagged else
                    f"All layer moments lie within {thr}σ of the reference battery "
                    f"(worst: '{worst_layer}' at {worst_z:.1f}σ)."),
            attack_class="model.weight_modification",
            evidence=[Evidence("table", "per-layer spread vs battery", data=rows)],
            access_assumptions=["weights readable", f"battery of {len(battery)} reference models"],
            limitations=["Catches crude weight edits. A backdoor trained in normally leaves "
                         "moments indistinguishable from a clean model — most real backdoors "
                         "are invisible here.",
                         "Reference battery must not be siblings of the model under test, or "
                         "this comparison is unrealistically easy."],
            disposition=Disposition.QUARANTINE if flagged else Disposition.ACCEPT,
            disposition_rule="weight_stats.outlier" if flagged else "weight_stats.nominal",
            nature=Nature.INDETERMINATE,
        )]


@register
class ActivationStatisticsCheck:
    id = "model.activation_statistics"
    version = "1.0.0"
    requires = {Capability.MODEL_ACTIVATIONS, Capability.REFERENCE_CLEAN_SET}
    optional: set = set()
    attack_classes = {"model.anomalous_behaviour", "model.weight_modification"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        x = ctx.probes_x[: int(ctx.opt("activation_probes", 128))]
        acts = model.activations(x.astype(np.float32))
        if not acts:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "activation extraction returned nothing",
                (Capability.MODEL_ACTIVATIONS,), "model.anomalous_behaviour")]

        rows, dead_total, sat_total, n_layers = {}, 0.0, 0.0, 0
        for name, a in acts.items():
            a = np.asarray(a, dtype=np.float64)
            if a.ndim < 2 or a.size == 0:
                continue
            flat = a.reshape(a.shape[0], -1)
            per_unit = flat.mean(0)
            dead = float((np.abs(per_unit) < 1e-6).mean())
            sat = float((flat > np.percentile(flat, 99.9)).mean()) if flat.size else 0.0
            rng = float(flat.max() - flat.min())
            rows[name] = {"dead_frac": round(dead, 4), "sat_frac": round(sat, 5),
                          "dyn_range": round(rng, 3)}
            dead_total += dead; sat_total += sat; n_layers += 1

        dead_mean = dead_total / max(n_layers, 1)
        thr = float(ctx.opt("dead_unit_threshold", 0.35))
        flagged = dead_mean > thr
        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.MEDIUM if flagged else Severity.INFO,
            confidence=float(np.clip(dead_mean / max(thr, 1e-6), 0, 1) * 0.6),
            score_raw=dead_mean, threshold=thr,
            reason=(f"{dead_mean*100:.1f}% of units are inert across {n_layers} layers on "
                    f"clean probes, above the {thr*100:.0f}% threshold — consistent with "
                    "pruning damage, truncation, or a model that did not train."
                    if flagged else
                    f"Activation distributions are nominal ({dead_mean*100:.1f}% inert units "
                    f"across {n_layers} layers)."),
            attack_class="model.anomalous_behaviour",
            evidence=[Evidence("table", "per-layer activation statistics", data=rows)],
            access_assumptions=["activations extractable — on ONNX this required graph surgery",
                                "clean probe set supplied"],
            limitations=["Reference-free and therefore coarse. Detects gross structural "
                         "damage, not a trained-in backdoor."],
            disposition=Disposition.REVIEW if flagged else Disposition.ACCEPT,
            disposition_rule="activation_stats.dead_units" if flagged else "activation_stats.nominal",
            nature=Nature.QUALITY if flagged else Nature.INDETERMINATE,
        )]


@register
class AnomalousBehaviourCheck:
    id = "model.anomalous"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT, Capability.REFERENCE_CLEAN_SET}
    optional: set = set()
    attack_classes = {"model.anomalous_behaviour"}

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
            attack_class="model.anomalous_behaviour",
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
