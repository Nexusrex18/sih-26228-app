"""model.behavioural_fingerprint — substitution detection with a tolerance band.

The digest answers "did the bytes change?". This answers "did the BEHAVIOUR change, and by
how much?" — which is what separates a hostile substitution from a benign re-export or
quantisation. A conversion moves the fingerprint slightly, a retrain moves it a lot, a
different model moves it completely.

Deterministic probe battery: fixed seed, fixed inputs, so the vector is reproducible.
"""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, Capability
from cva.core.finding import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register


def probe_battery(shape: tuple[int, ...], n: int = 64, seed: int = 20260919) -> np.ndarray:
    """Deterministic, model-independent probes spanning several input regimes."""
    rng = np.random.default_rng(seed)
    parts = [
        rng.uniform(0, 1, size=(n // 4, *shape)),                       # uniform noise
        np.clip(rng.normal(0.5, 0.15, size=(n // 4, *shape)), 0, 1),    # gaussian
        np.tile(np.linspace(0, 1, shape[-1], dtype=np.float32),
                (n // 4, shape[0], shape[1], 1)),                       # ramps
        np.repeat(np.linspace(0, 1, n // 4, dtype=np.float32)
                  .reshape(-1, 1, 1, 1), shape[0], 1)
        * np.ones((1, *shape), dtype=np.float32),                       # flat fields
    ]
    return np.concatenate(parts).astype(np.float32)


def fingerprint(model, n: int = 64, seed: int = 20260919) -> np.ndarray:
    x = probe_battery(tuple(model.input_shape), n, seed)
    return np.asarray(model.predict(x), dtype=np.float64).ravel()


def divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over the probe response. Bounded [0, 2] for probabilities."""
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).mean())


@register
class FingerprintCheck:
    id = "model.behavioural_fingerprint"
    version = "1.0.0"
    requires = {Capability.MODEL_PREDICT}
    optional = {Capability.REFERENCE_MANIFEST}
    attack_classes = {"model.substitution", "model.weight_modification",
                      "model.quantisation_divergence"}

    # Tolerance band: below BENIGN a difference is explained by re-export/quantisation;
    # above HOSTILE the model is behaviourally different. Between them: review.
    BENIGN = 0.01
    HOSTILE = 0.05

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        n = ctx.opt("fingerprint_probes", 64)
        fp = fingerprint(model, n)
        manifest = ctx.battery.manifest if ctx.battery else None
        benign = ctx.opt("fingerprint_benign", self.BENIGN)
        hostile = ctx.opt("fingerprint_hostile", self.HOSTILE)

        if manifest is None or not manifest.fingerprint:
            f = unavailable_finding(
                self.id, self.version, model.model_id,
                "no registered fingerprint was supplied, so behavioural substitution "
                "cannot be assessed. The fingerprint was computed and emitted for "
                "registration.",
                (Capability.REFERENCE_MANIFEST,),
                "model.substitution", Availability.DEGRADED,
            )
            f.scan_id = ctx.scan_id
            f.evidence.append(Evidence("json", "computed fingerprint (first 8 of %d)" % len(fp),
                                       data=[round(v, 6) for v in fp[:8]]))
            return [f]

        d = divergence(fp, np.asarray(manifest.fingerprint, dtype=np.float64))
        ev = [Evidence("table", "fingerprint divergence", data={
            "divergence": round(d, 6), "benign_band": benign, "hostile_band": hostile,
            "probes": n})]

        if d <= benign:
            sev, conf, disp, rule = Severity.INFO, 1 - d, Disposition.ACCEPT, "fingerprint.match"
            reason = (f"Behavioural fingerprint matches the registered one "
                      f"(divergence {d:.5f} ≤ {benign}). Model behaves identically on the "
                      "probe battery.")
            nature = Nature.INDETERMINATE
        elif d <= hostile:
            sev, conf, disp, rule = (Severity.LOW, 0.5, Disposition.REVIEW,
                                     "fingerprint.within_tolerance")
            reason = (f"Behaviour differs slightly from the registered fingerprint "
                      f"(divergence {d:.5f}, inside the {hostile} tolerance band). This is "
                      "the signature of a benign re-export, opset change or quantisation "
                      "rather than a substituted model.")
            nature = Nature.QUALITY
        else:
            sev, conf, disp, rule = (Severity.HIGH, min(0.99, d / hostile * 0.6 + 0.35),
                                     Disposition.QUARANTINE, "fingerprint.divergent")
            reason = (f"Behaviour diverges materially from the registered fingerprint "
                      f"(divergence {d:.5f} > {hostile}). Consistent with a substituted or "
                      "retrained model, not with a format conversion.")
            nature = Nature.ADVERSARIAL

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=sev, confidence=float(conf), score_raw=d, threshold=hostile,
            reason=reason, attack_class="model.substitution", evidence=ev,
            access_assumptions=["query access only — no weights required"],
            limitations=["Detects only substitution that changes behaviour on this probe "
                         "battery. A substituted model with near-identical decision "
                         "boundaries would pass."],
            disposition=disp, disposition_rule=rule, nature=nature,
        )]
