"""model.weight_digest — deterministic substitution detection.

A hash catches ANY byte change, which is also its false-positive mode: a model quantised
or format-converted with no malice produces a different digest. The finding therefore
carries the benign-explanation caveat, and the fingerprint runs alongside it to separate
the two. Certainty about the bytes is not certainty about intent.
"""
from __future__ import annotations

from cva.core.capability import Availability, Capability
from cva.core.finding import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register


@register
class WeightDigestCheck:
    id = "model.weight_digest"
    version = "1.0.0"
    requires: set = set()                                   # file access only
    optional = {Capability.REFERENCE_MANIFEST, Capability.MODEL_WEIGHTS}
    attack_classes = {"model_substitution", "weight_anomaly"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        digest = model.weight_digest()
        manifest = ctx.battery.manifest if ctx.battery else None

        if manifest is None or not manifest.weights_sha256:
            f = unavailable_finding(
                self.id, self.version, model.model_id,
                "no reference manifest was supplied, so there is no declared digest to "
                "compare against. Substitution cannot be decided deterministically.",
                (Capability.REFERENCE_MANIFEST,),
                "model_substitution", Availability.DEGRADED,
            )
            f.scan_id = ctx.scan_id
            f.evidence.append(Evidence("hash", "computed weight digest", data=digest))
            f.limitations.append(
                "A reference manifest has been emitted for this model — register it at "
                "acceptance and every future scan detects substitution deterministically."
            )
            return [f]

        if digest == manifest.weights_sha256:
            return [Finding(
                detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
                target_type="model", target_ref=model.model_id,
                severity=Severity.INFO, confidence=1.0, score_raw=0.0, threshold=0.0,
                reason=f"Weight digest matches the declared manifest exactly ({digest[:16]}…).",
                attack_class="model_substitution",
                evidence=[Evidence("hash", "digest", data=digest)],
                access_assumptions=["weights readable", "manifest supplied"],
                limitations=["Proves byte identity only; says nothing about whether the "
                             "declared model was itself trustworthy."],
                disposition=Disposition.ACCEPT, disposition_rule="digest.match",
                nature=Nature.INDETERMINATE,
            )]

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=Severity.CRITICAL, confidence=1.0, score_raw=1.0, threshold=0.0,
            reason=(f"Weight digest does NOT match the declared manifest. "
                    f"Declared {manifest.weights_sha256[:16]}…, found {digest[:16]}…. "
                    "The supplied model is not the model that was registered."),
            attack_class="model_substitution",
            evidence=[Evidence("table", "digest comparison", data={
                "declared": manifest.weights_sha256, "computed": digest})],
            access_assumptions=["weights readable", "manifest supplied"],
            limitations=[
                "A hash cannot distinguish malice from a benign re-export, quantisation or "
                "opset conversion. Read this together with model.behavioural_fingerprint: "
                "if the fingerprint is inside tolerance, a benign conversion is the likely "
                "explanation."],
            disposition=Disposition.QUARANTINE, disposition_rule="digest.mismatch",
            nature=Nature.INDETERMINATE,
        )]
