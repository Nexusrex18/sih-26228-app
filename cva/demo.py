"""Four scenarios that demonstrate the design decisions, each producing a report.

1. ONNX backdoored, NO reference   — the expected real case. Gradients absent, so Neural
                                     Cleanse runs DEGRADED via NES; STRIP and the intrinsic
                                     probes carry the finding. Coverage shrinks honestly.
2. PyTorch backdoored, NO reference — same model family with gradients. Neural Cleanse runs
                                     at full confidence and RECONSTRUCTS the trigger image.
3. Benign re-export WITH reference  — digest mismatches (different bytes) but the
                                     fingerprint is inside tolerance. Must NOT read hostile.
4. Substituted model WITH reference — digest mismatches AND fingerprint diverges. Hostile.

3 and 4 together are the point: a hash alone cannot tell them apart.
"""
from __future__ import annotations

import json
from pathlib import Path

from cva.loaders.models import load_model
from attacklab.arch import ARCH_REGISTRY
from cva.cli import build_battery, emit_reference, load_probes, manifest_from
from cva.core.model import ModelBattery
import cva.detectors.model.registry  # noqa: F401 — registration happens at the entrypoint, never in core
from cva.core.orchestrator import RunContext, scan
from cva.report.render_html import render


def run(corpus: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((corpus / "manifest.json").read_text())
    by_id = {m["id"]: m for m in man["models"]}
    x, y = load_probes(corpus, 400)
    results, notes = [], []

    def do(label, path, mid, manifest=None, battery=True, profile="deep"):
        model = load_model(corpus / path, ARCH_REGISTRY, mid)
        bat = build_battery(corpus, mid) if battery else ModelBattery()
        if manifest:
            bat.manifest = manifest
        res = scan(model, RunContext(probes_x=x, probes_y=y, battery=bat,
                                     out_dir=out, seed=7), profile)
        results.append(res)
        fired = [f.detector_id.replace("model.", "") for f in res.findings
                 if f.disposition.value in ("quarantine",) and f.severity.rank >= 3]
        notes.append((label, mid, res.verdict, fired))
        print(f"  {label:38} {res.verdict:10} fired: {', '.join(fired) or '—'}")
        return model, res

    print("\n[1] Backdoored ONNX, no reference — the expected real case")
    do("backdoored ONNX / no reference", by_id["bd_patch_08"]["onnx"], "bd_patch_08.onnx")

    print("\n[2] Same backdoor as PyTorch — gradients available")
    do("backdoored PyTorch / no reference", by_id["bd_patch_08"]["pt"], "bd_patch_08.pt")

    print("\n[3] Benign re-export WITH a registered reference")
    base = load_model(corpus / by_id["clean_a"]["pt"], ARCH_REGISTRY, "clean_a")
    ref_path = emit_reference(base, out / "clean_a.reference.json")
    ref = manifest_from(ref_path)
    if "clean_a_reexport" in by_id:
        do("benign re-export / reference registered",
           by_id["clean_a_reexport"]["onnx"], "clean_a_reexport.onnx", manifest=ref)

    print("\n[4] Different model presented under clean_a's reference — substitution")
    do("substituted model / reference registered",
       by_id["clean_b"]["pt"], "clean_b_as_clean_a", manifest=ref)

    render(results, out / "demo.report.html", "CV Assurance — Module B — demo scenarios")
    print(f"\n  report: {out}/demo.report.html")
