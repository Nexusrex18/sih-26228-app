"""`prov.recompute` — re-derive sealed inferences and compare (Module C plan §7.9, gate C6).

It is the strongest check in the system because it does not infer, it re-derives: load the referenced input,
load the stored preprocessing spec, run the referenced model, apply the stored postprocessing spec, quantise,
hash, compare. What it may CLAIM depends on how the two runs relate — see `ladder.py`.

  Scope (§7.9). Recompute costs one inference per record, so the default is: every record that a chain check
  already flagged, plus a seeded random sample of the rest (the seed and the chosen seqs are recorded in the
  summary, so the sample is itself reproducible). `scope="all"` re-derives everything.

  Only records whose SIGNATURE verified are re-derived: re-deriving a record that was edited after signing would
  prove nothing about the output that was sealed.

  It also performs, when it has the artefacts, the two checks that are cheaper as a by-product:
    * input swap — sha256(the referenced image) != `input.sha256`. The HASH COMPARISON carries that certainty,
      not the recompute. For `source_kind = "model_input_tensor"` records the sealed hash covers the
      post-preprocessing tensor, so the check needs a raw frame to re-preprocess; without one it is
      UNAVAILABLE for that record, with a reason, never a pass.
    * model swap — the loaded model's weight digest != the digest the record names.

  Never a false pass. A record that cannot be re-derived (no input, a missing payload, a model that is not the
  one named) is counted, with a reason, in the summary — which is DEGRADED whenever anything was skipped.
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cva.core.capability import Availability, Capability, CapabilitySet, Resolution
from cva.core.types import Disposition, Evidence, Finding, Nature, Severity

from ..seal.keys import TrustRoot
from ..seal.verify import JsonlSource, SqliteSource, _analyse, open_source, verify_ledger
from .ladder import (
    DEFAULT_EPS_CONF,
    DEFAULT_EPS_IOU,
    DEFAULT_EPS_PX,
    LadderResult,
    climb,
    recompute_objects,
)
from .pipeline import Pipeline

DETECTOR_ID = "prov.recompute"
VERSION = "1"

LIMITATIONS = (
    "R1 (decision level: which objects, where) is the CERTAIN claim; bit-exact R0 is claimed only when the "
    "recompute runtime string equals the sealing one, and does not vouch for the hardware it ran on.",
    "A boundary_flip is a review item, not proof: a difference within float jitter of a decision boundary cannot "
    "be told apart from a deliberate nudge of that size.",
    "Recompute re-derives only a sample plus the already-flagged records unless scope='all'.",
    "Confidence values that differ in their last digits while the decisions match are reported as a diagnostic, "
    "never as a finding.",
)

_SEVERITY = {"output_mismatch": ("critical", "indeterminate"), "boundary_flip": ("medium", "indeterminate"),
             "input_swap": ("critical", "indeterminate"),
             # A scanner-side digest mismatch is two possible facts: the ledger names a model the auditor did not load
             # (a routine model update on the scanner) or the record was altered — and the second is `prov.ledger_verify`'s
             # to decide, from the ledger's own registrations. Here it only means "cannot re-derive with this model".
             "model_swap": ("medium", "indeterminate")}


@dataclass
class _Tally:
    counts: Counter[str] = field(default_factory=Counter)
    unavailable: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[int]] = field(default_factory=dict)

    def skip(self, why: str, seq: int) -> None:
        self.unavailable[why] += 1
        self.examples.setdefault(why, [])
        if len(self.examples[why]) < 5:
            self.examples[why].append(seq)


class Recompute:
    """Registrable plug-in shape (`core.registry.Registrable`). Like `LedgerVerify` it is deliberately not forced
    into `core.registry.REGISTRY` — wiring it into the scan (a model handle, a pipeline, an input resolver) is
    Backend's step."""

    id = DETECTOR_ID
    version = VERSION
    requires = frozenset({Capability.INFERENCE_LEDGER, Capability.MODEL_PREDICT})
    optional: frozenset[Capability] = frozenset()      # degrades at RUNTIME (payload_missing), not by capability
    attack_classes = frozenset({"output_mismatch", "boundary_flip", "input_swap", "model_swap", "recompute_verified"})

    def resolve(self, caps: CapabilitySet) -> Resolution:
        return caps.resolve(set(self.requires), set(self.optional))

    # ---------------------------------------------------------------------------------------------------

    def recompute(self, source: Any, trust_root: TrustRoot | str, *, model: Any, pipeline: Pipeline,
                  input_resolver: Callable[[Mapping[str, Any]], bytes | None],
                  scope: str = "flagged+sample", sample: int = 50, seed: int | None = None,
                  payloads: Mapping[str, bytes] | None = None, weights_digest: str | None = None,
                  eps_conf: float = DEFAULT_EPS_CONF, eps_px: float = DEFAULT_EPS_PX, eps_iou: float = DEFAULT_EPS_IOU,
                  on_nonfinite: str = "seal_marker", scan_id: str = "", produced_by: str = "") -> list[Finding]:
        if scope not in ("flagged+sample", "all"):
            raise ValueError("scope must be 'flagged+sample' or 'all'")
        report = verify_ledger(_fresh(source, payloads), trust_root=trust_root)
        src = _fresh(source, payloads)
        try:
            records = self._eligible(src, report.verified_seqs)
            flagged = {f.seq for f in report.findings if f.severity != "info" and f.seq is not None}
            priority = sorted(s for s in records if s in flagged)
            rest = sorted(s for s in records if s not in flagged)
            if seed is None:
                seed = secrets.randbits(63)          # unpredictable by default: a fixed default seed tells an adversary which
            chosen = list(rest) if scope == "all" else random.Random(seed).sample(rest, min(sample, len(rest)))
            todo = sorted(set(priority) | set(chosen))
            digest = weights_digest if weights_digest is not None else _safe_digest(model)

            tally, out = _Tally(), []
            swapped: dict[str, list[int]] = {}           # one model_swap finding per sealed digest, not per record
            for seq in todo:
                rec = records[seq]
                res = self._one(rec, src, pipeline, model, digest, input_resolver, tally, eps_conf, eps_px, eps_iou,
                                on_nonfinite)
                if res is not None and res[0] == "model_swap":
                    swapped.setdefault(res[1]["sealed"], []).append(seq)
                    if len(swapped[res[1]["sealed"]]) == 1:
                        out.append(self._finding(res, rec, scan_id, produced_by, pipeline, swapped[res[1]["sealed"]]))
                elif res is not None:
                    out.append(self._finding(res, rec, scan_id, produced_by, pipeline))
            out.append(self._summary(len(records), todo, priority, scope, sample, seed, tally, pipeline, digest,
                                     scan_id, produced_by))
            return out
        finally:
            src.close()

    # -- per record --------------------------------------------------------------------------------------

    @staticmethod
    def _eligible(src: SqliteSource | JsonlSource, verified: set[int]) -> dict[int, dict[str, Any]]:
        recs: dict[int, dict[str, Any]] = {}
        for row in src.rows():
            rec = _analyse(row.data)[0]
            if rec is not None and rec["type"] == "inference" and rec["seq"] in verified:
                recs[rec["seq"]] = rec
        return recs

    def _one(self, rec: dict[str, Any], src: SqliteSource | JsonlSource, pipe: Pipeline, model: Any, digest: str,
             resolver: Callable[[Mapping[str, Any]], bytes | None], tally: _Tally, eps_conf: float, eps_px: float,
             eps_iou: float, on_nonfinite: str) -> tuple[str, dict[str, Any]] | None:
        seq = rec["seq"]
        # model swap: the model we hold must be the model the record names
        if digest != "unavailable" and rec["model"]["weights_sha256"] != digest:
            tally.counts["model_swap"] += 1
            return "model_swap", {"sealed": rec["model"]["weights_sha256"], "loaded": digest}
        if digest == "unavailable":
            tally.skip("model weight digest unavailable — cannot confirm this is the model the record names", seq)
            return None
        data = resolver(rec)
        if data is None:
            tally.skip("the input image was not supplied", seq)
            return None
        try:
            pre = self._payload_json(src, rec["config"]["preprocess_ref"], "preprocess spec")
            post = self._payload_json(src, "sha256:" + rec["config"]["postprocess_hash"], "postprocess spec")
        except _Missing as e:
            tally.skip(str(e), seq)
            return None
        prepared = pipe.prepare(data, rec["input"]["source_kind"], pre)
        if rec["input"]["source_kind"] == "encoded_file":
            actual = hashlib.sha256(data).hexdigest()
        else:                                        # the sealed hash covers the post-preprocessing tensor
            try:
                actual = hashlib.sha256(pipe.tensor_bytes(prepared)).hexdigest()
            except NotImplementedError:
                tally.skip("the pipeline cannot serialise its model-input tensor, so this record's input hash "
                           "cannot be checked", seq)
                return None
        if actual != rec["input"]["sha256"]:
            tally.counts["input_swap"] += 1
            return "input_swap", {"sealed": rec["input"]["sha256"], "found": actual}
        raw, filtered = pipe.decode(pipe.infer(prepared), post)
        re = recompute_objects(raw, filtered, on_nonfinite=on_nonfinite)
        try:
            sealed_fine = self._payload_json(src, "sha256:" + rec["output"]["jcs_sha256"], "sealed decision payload")
        except _Missing:
            sealed_fine = None                        # only needed to EXPLAIN a mismatch; exact/decision still work
        result = climb(rec["output"], sealed_fine, re, sealed_runtime=rec["config"]["runtime"],
                       recompute_runtime=pipe.runtime, eps_conf=eps_conf, eps_px=eps_px, eps_iou=eps_iou)
        tally.counts[result.verdict] += 1
        if result.verdict in ("verified_exact", "verified_decision"):
            return None
        return result.verdict, {"ladder": result, "recomputed": re}

    @staticmethod
    def _payload_json(src: SqliteSource | JsonlSource, ref: str, what: str) -> dict[str, Any]:
        addr = ref.removeprefix("sha256:")
        exists, data = src.payload(addr)
        if not exists or data is None:
            raise _Missing(f"payload missing: {what}")
        if hashlib.sha256(data).hexdigest() != addr:
            raise _Missing(f"payload corrupt: {what} does not hash to its address")
        return json.loads(data)                        # type: ignore[no-any-return]

    # -- findings -----------------------------------------------------------------------------------------

    def _finding(self, res: tuple[str, dict[str, Any]], rec: dict[str, Any], scan_id: str, produced_by: str,
                 pipe: Pipeline, seqs: list[int] | None = None) -> Finding:
        cls, info = res
        sev, nat = _SEVERITY[cls]
        seq = rec["seq"]
        access = ["INFERENCE_LEDGER with its payload table; MODEL_PREDICT via the supplied pipeline",
                  f"recomputed on {pipe.runtime!r}; sealed on {rec['config']['runtime']!r}"]
        limits = list(LIMITATIONS)
        if cls == "model_swap":
            reason = (f"Inference seq {seq} names model '{rec['model']['id']}' with weights "
                      f"{info['sealed'][:12]}…, but the model loaded for recompute has weights {info['loaded'][:12]}…: "
                      "so this record was NOT re-derived. Either the auditor loaded a different model (for example after a legitimate "
                      "update) or the record names the wrong one; which of the two is the ledger check's to decide from the "
                      "ledger's own model registrations — this check only reports that it could not re-derive.")
            ev = [Evidence("hash", "weights digest", data={**info, "affected_seqs": seqs or [seq]})]
        elif cls == "input_swap":
            reason = (f"The input supplied for inference seq {seq} does not match the hash sealed with the record "
                      f"(sealed {info['sealed'][:12]}…, found {info['found'][:12]}…): the image was swapped after "
                      "sealing. The hash comparison, not the recompute, carries this certainty.")
            ev = [Evidence("hash", "input sha256", data=info)]
        else:
            lad: LadderResult = info["ladder"]
            head = (f"Recomputing inference seq {seq} does not reproduce the sealed decisions: "
                    if cls == "output_mismatch" else
                    f"Recomputing inference seq {seq} flips a decision at a boundary: ")
            reason = head + lad.reason + ". " + " ".join(f"({i + 1}) {w}." for i, w in enumerate(lad.explanations[:6]))
            reason += (" Certain that the sealed output is not what this pipeline produces from this input; not "
                       "certain why." if cls == "output_mismatch" else "")
            ev = [Evidence("table", "the differences (R2) and the numeric diagnostics (R3, never a claim)",
                           data={"rung": lad.rung, "r0_applicable": lad.r0_applicable, "explanations": lad.explanations,
                                 "diagnostics": lad.diagnostics}),
                  Evidence("hash", "decision hash", data={"sealed": rec["output"]["decision_sha256"],
                                                          "recomputed": info["recomputed"].decision_sha256})]
        disposition = Disposition.QUARANTINE if sev in ("critical", "high") else Disposition.REVIEW
        return Finding(detector_id=DETECTOR_ID, detector_version=VERSION, target_type="record",
                       target_ref=f"seq:{seq}", severity=Severity(sev), confidence=1.0, reason=reason, attack_class=cls,
                       scan_id=scan_id, produced_by=produced_by, score_raw=1.0, threshold=1.0, evidence=ev,
                       access_assumptions=access, limitations=limits, disposition=disposition,
                       disposition_rule="prov.provisional — the risk engine owns the final disposition",
                       nature=Nature(nat), availability=Availability.OK)

    def _summary(self, eligible: int, todo: list[int], priority: list[int], scope: str, sample: int, seed: int,
                 tally: _Tally, pipe: Pipeline, digest: str, scan_id: str, produced_by: str) -> Finding:
        c = tally.counts
        skipped = sum(tally.unavailable.values())
        done = c["verified_exact"] + c["verified_decision"] + c["boundary_flip"] + c["output_mismatch"]
        stats = {"eligible_inference_records": eligible, "selected": len(todo), "flagged_priority": len(priority),
                 "scope": scope, "sample_size": sample, "seed": seed, "sampled_seqs": [s for s in todo if s not in set(priority)][:200],
                 "verified_exact": c["verified_exact"], "verified_decision": c["verified_decision"],
                 "boundary_flip": c["boundary_flip"], "output_mismatch": c["output_mismatch"],
                 "input_swap": c["input_swap"], "model_swap": c["model_swap"], "could_not_be_rederived": skipped,
                 "why_not": dict(tally.unavailable), "examples": tally.examples,
                 "recompute_runtime": pipe.runtime, "loaded_weights_digest": digest}
        frac = (len(todo) / eligible) if eligible else 0.0
        stats["sampled_fraction"] = round(frac, 6)
        reason = (f"Recompute: {len(todo)} of {eligible} sealed inference(s) selected ({frac:.1%}; {len(priority)} already "
                  f"flagged, the rest a random sample, seed {seed}"
                  + ("" if scope == "all" or len(todo) >= eligible else
                     " — a SAMPLE, so the unselected records are unassessed, not cleared") + "); {c['verified_exact']} re-derived bit-exact, "
                  f"{c['verified_decision']} re-derived at decision level, {c['boundary_flip']} boundary flip(s), "
                  f"{c['output_mismatch']} mismatch(es), {skipped} could not be re-derived"
                  + (f" ({'; '.join(f'{n}× {w}' for w, n in tally.unavailable.items())})" if skipped else "") + ".")
        degraded = skipped > 0 or done == 0
        limits = list(LIMITATIONS)
        if skipped:
            limits.append(f"{skipped} record(s) were NOT re-derived: they are unassessed, not passed.")
        if done == 0 and not skipped:
            limits.append("No inference record was selected: nothing was re-derived.")
        return Finding(detector_id=DETECTOR_ID, detector_version=VERSION, target_type="record", target_ref="ledger",
                       severity=Severity.INFO, confidence=1.0, reason=reason, attack_class="recompute_verified",
                       scan_id=scan_id, produced_by=produced_by, score_raw=1.0, threshold=1.0,
                       evidence=[Evidence("table", "what was re-derived", data=stats)],
                       access_assumptions=[f"recomputed on {pipe.runtime!r}"], limitations=limits,
                       disposition=Disposition.ACCEPT, disposition_rule="prov.summary — information only",
                       nature=Nature.INDETERMINATE,
                       availability=Availability.DEGRADED if degraded else Availability.OK)


class _Missing(Exception):
    pass


def _safe_digest(model: Any) -> str:
    try:
        return str(model.weight_digest())
    except Exception:                                # noqa: BLE001 - a model that cannot say is "unavailable", not a swap
        return "unavailable"


def _fresh(source: Any, payloads: Mapping[str, bytes] | None) -> SqliteSource | JsonlSource:
    """A NEW source each time (`verify_ledger` closes the one it is given), so `source` must be a path or the
    bytes of an export — not a prepared source object."""
    if isinstance(source, (SqliteSource, JsonlSource)):
        raise TypeError("pass a path or the bytes of an export: verify_ledger closes the source it is given")
    return open_source(source, payloads)


__all__ = ["DETECTOR_ID", "LIMITATIONS", "Recompute"]
