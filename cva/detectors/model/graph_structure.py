"""model.graph_structure — architectural backdoors, and unsafe artifacts.

Every other Module B check reads weights or behaviour. Nothing read STRUCTURE. A backdoor
can live in the graph itself — a conditional, a custom operator, a dead branch that
activates on a pattern — with entirely ordinary weights.

Cheap, deterministic, and it runs on ONNX where gradients do not. It is also where the
custom-operator risk surfaces: an ONNX custom op can load a shared library, which is the
same class of problem as torch.load executing pickle.
"""
from __future__ import annotations

from cva.core.capability import Availability, Capability
from cva.core.types import (Disposition, Evidence, Finding, Nature, Severity,
                              unavailable_finding)
from cva.detectors.base import CheckContext, register

CONTROL_FLOW_OPS = {"If", "Loop", "Scan", "Where"}


@register
class GraphStructureCheck:
    id = "model.graph_structure"
    version = "1.0.0"
    requires = {Capability.MODEL_ARCHITECTURE}
    optional = {Capability.REFERENCE_MANIFEST}
    attack_classes = {"architectural_backdoor", "unsafe_artifact"}

    def check(self, model, ctx: CheckContext) -> list[Finding]:
        if not hasattr(model, "op_inventory"):
            f = unavailable_finding(
                self.id, self.version, model.model_id,
                f"graph inventory is only implemented for ONNX; this model is "
                f"'{model.fmt}'. Structural assessment was not performed.",
                (Capability.MODEL_ARCHITECTURE,), "architectural_backdoor",
                Availability.DEGRADED)
            f.scan_id = ctx.scan_id
            return [f]

        inv = model.op_inventory()
        nonstd = model.nonstandard_ops()
        graph = model.get_graph()

        produced = {o for n in graph.node for o in n.output}
        consumed = {i for n in graph.node for i in n.input}
        graph_outputs = {o.name for o in graph.output}
        dead = sorted(produced - consumed - graph_outputs)
        control = {op: c for op, c in inv.items() if op in CONTROL_FLOW_OPS}

        problems, severity, ac = [], Severity.INFO, "architectural_backdoor"
        if nonstd:
            problems.append(
                f"non-standard operators present: {', '.join(nonstd)} — a custom ONNX "
                "operator can load a shared library at session-creation time")
            severity, ac = Severity.CRITICAL, "unsafe_artifact"
        if control:
            problems.append(
                f"data-dependent control flow: {control} — a conditional branch can gate "
                "behaviour on an input pattern without any weight carrying the signature")
            severity = max(severity, Severity.HIGH, key=lambda s: s.rank)
        if dead:
            problems.append(
                f"{len(dead)} tensor(s) produced but never consumed or output "
                f"(e.g. {dead[:3]}) — a dormant subgraph")
            severity = max(severity, Severity.MEDIUM, key=lambda s: s.rank)

        manifest = ctx.battery.manifest if ctx.battery else None
        arch_note = None
        if manifest and manifest.architecture_hash:
            import hashlib
            h = hashlib.sha256(
                ",".join(f"{k}:{v}" for k, v in sorted(inv.items())).encode()).hexdigest()
            if h != manifest.architecture_hash:
                problems.append("operator inventory does not match the declared "
                                "architecture hash")
                severity = max(severity, Severity.HIGH, key=lambda s: s.rank)
            arch_note = h

        flagged = bool(problems)
        ev = [Evidence("table", "operator inventory", data=inv)]
        if dead:
            ev.append(Evidence("json", "dead tensors", data=dead[:20]))
        if arch_note:
            ev.append(Evidence("hash", "computed architecture hash", data=arch_note))

        return [Finding(
            detector_id=self.id, detector_version=self.version, scan_id=ctx.scan_id,
            target_type="model", target_ref=model.model_id,
            severity=severity, confidence=0.95 if flagged else 0.0,
            score_raw=float(len(problems)), threshold=1.0,
            reason=("Graph structure findings: " + "; ".join(problems) + "."
                    if flagged else
                    f"Graph structure is nominal: {sum(inv.values())} nodes across "
                    f"{len(inv)} operator types, all standard-domain, no data-dependent "
                    "control flow, no dead subgraphs."),
            attack_class=ac, evidence=ev,
            access_assumptions=["graph readable — no weights, gradients or execution needed"],
            limitations=["Reads structure only. A backdoor expressed purely in weight "
                         "values is invisible to this check by construction."],
            disposition=Disposition.QUARANTINE if severity.rank >= Severity.HIGH.rank
            else (Disposition.REVIEW if flagged else Disposition.ACCEPT),
            disposition_rule="graph.structural_finding" if flagged else "graph.nominal",
            nature=Nature.ADVERSARIAL if flagged else Nature.INDETERMINATE,
        )]
