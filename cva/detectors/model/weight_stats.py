"""model.weight_stats — PS 2.2.2's "parameter OR ACTIVATION statistics", one check.

The PS names the two together and the plan's definition of done requires the activation
path to be "thorough enough to close PS-audit G3 outright (no separate detector)" — so this
is one check emitting two findings, not two checks. The activation half needs only
MODEL_ACTIVATIONS, so it runs on ONNX (after graph surgery) where Neural Cleanse cannot.

The parameter half is battery-relative and therefore gated on battery size: a z-score
against two reference models is meaningless — with n=2 the sample standard deviation is a
terrible estimator and small differences explode (measured: 170 sigma on a clean layer).
That is the same failure as ranking contributors by raw flag rate, and it fails the same
way: confidently, toward false alarms.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.types import Disposition, Evidence, Finding, Nature, Severity, unavailable_finding
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
    id = "model.weight_stats"
    version = "1.0.0"
    requires = {Capability.MODEL_WEIGHTS}
    optional = {Capability.REFERENCE_MODEL_BATTERY, Capability.MODEL_ACTIVATIONS}
    attack_classes = {"weight_anomaly", "activation_anomaly"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        return self._weight_findings(model, ctx) + self._activation_findings(model, ctx)

    # ---- parameter half ----------------------------------------------------
    def _weight_findings(self, model, ctx: CheckContext) -> list[Finding]:
        w = model.get_weights()
        mine = layer_moments(w)
        battery = (ctx.battery.models if ctx.battery else None) or []

        if not battery:
            f = unavailable_finding(
                self.id, self.version, model.model_id,
                "no reference model battery was supplied, so per-layer moments have "
                "nothing to be anomalous against. Reporting the model's own statistics only.",
                (Capability.REFERENCE_MODEL_BATTERY,),
                "weight_anomaly", Availability.DEGRADED)
            f.scan_id = ctx.scan_id
            f.evidence.append(Evidence("table", "per-layer weight moments", data={
                k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in mine.items()}))
            return [f]

        # A z-score against a two-model battery is meaningless: with n=2 the sample
        # standard deviation is a terrible estimator, so a small difference explodes
        # (measured: 170 sigma on conv2.bias against a clean model). This is the same
        # error as ranking contributors by raw flag rate, and it fails the same way —
        # confidently, and in the direction that produces false alarms.
        MIN_BATTERY = int(ctx.opt("weight_stat_min_battery", 5))
        if len(battery) < MIN_BATTERY:
            f = unavailable_finding(
                self.id, self.version, model.model_id,
                f"reference battery has {len(battery)} model(s); at least {MIN_BATTERY} "
                "are needed before a per-layer z-score carries any information. Reporting "
                "the model's own statistics only.",
                (Capability.REFERENCE_MODEL_BATTERY,),
                "weight_anomaly", Availability.DEGRADED)
            f.scan_id = ctx.scan_id
            f.evidence.append(Evidence("table", "per-layer weight moments", data={
                k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in mine.items()}))
            return [f]

        ref = [layer_moments(b.get_weights() or {}) for b in battery]
        rows, worst_z, worst_layer = {}, 0.0, ""
        for layer, stats in mine.items():
            vals = [r[layer]["std"] for r in ref if layer in r]
            if len(vals) < MIN_BATTERY:
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
            attack_class="weight_anomaly",
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

    # ---- activation half: PS 2.2.2's "or activation statistics" -------------
    def _activation_findings(self, model, ctx: CheckContext) -> list[Finding]:
        if Capability.MODEL_ACTIVATIONS not in model.capabilities():
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "activations are not extractable from this model, so the activation half "
                "of PS 2.2.2's 'parameter or activation statistics' was not assessed.",
                (Capability.MODEL_ACTIVATIONS,), "activation_anomaly",
                Availability.DEGRADED)]
        if ctx.probes_x is None or not len(ctx.probes_x):
            # Activations are extractable but there is nothing to extract them FROM: without
            # probe inputs this half cannot run, and saying so beats indexing None.
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "no probe inputs were supplied, so there was nothing to run the model on and "
                "the activation half of PS 2.2.2's 'parameter or activation statistics' was "
                "not assessed.",
                (Capability.REFERENCE_CLEAN_SET,), "activation_anomaly",
                Availability.DEGRADED)]
        x = ctx.probes_x[: int(ctx.opt("activation_probes", 128))]
        acts = model.activations(x.astype(np.float32))
        if not acts:
            return [unavailable_finding(
                self.id, self.version, model.model_id,
                "activation extraction returned nothing",
                (Capability.MODEL_ACTIVATIONS,), "activation_anomaly",
                Availability.DEGRADED)]

        rows, dead_total, n_layers = {}, 0.0, 0
        for name, a in acts.items():
            a = np.asarray(a, dtype=np.float64)
            if a.ndim < 2 or a.size == 0:
                continue
            flat = a.reshape(a.shape[0], -1)
            per_unit = flat.mean(0)
            dead = float((np.abs(per_unit) < 1e-6).mean())
            sat = float((flat > np.percentile(flat, 99.9)).mean()) if flat.size else 0.0
            rows[name] = {"dead_frac": round(dead, 4), "sat_frac": round(sat, 5),
                          "dyn_range": round(float(flat.max() - flat.min()), 3)}
            dead_total += dead; n_layers += 1

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
                    f"Activation distributions are nominal ({dead_mean*100:.1f}% inert "
                    f"units across {n_layers} layers)."),
            attack_class="activation_anomaly",
            evidence=[Evidence("table", "per-layer activation statistics", data=rows)],
            access_assumptions=["activations extractable — on ONNX this required graph surgery",
                                "clean probe set supplied"],
            limitations=["Reference-free and therefore coarse. Detects gross structural "
                         "damage, not a trained-in backdoor."],
            disposition=Disposition.REVIEW if flagged else Disposition.ACCEPT,
            disposition_rule=("weight_stats.activation.dead_units" if flagged
                              else "weight_stats.activation.nominal"),
            nature=Nature.QUALITY if flagged else Nature.INDETERMINATE,
        )]

