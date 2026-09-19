"""Benchmark: run every detector over the whole model corpus and score it against ground
truth. One artifact, three jobs — the pitch numbers, the calibration set, and the
regression suite.

Reports per-detector detection rate and false-alarm rate SEPARATELY. A detector tuned only
against attacks has no measured false-alarm rate at all, and the false-alarm column is the
one that decides whether an analyst keeps reading the reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cva.detectors.model.registry  # noqa: F401 — registration happens at the entrypoint, never in core
from attacklab.arch import ARCH_REGISTRY
from cva.core.capability import Availability
from cva.core.model import ModelBattery
from cva.core.orchestrator import RunContext, scan
from cva.core.types import Disposition
from cva.loaders.models import load_model
from cva.report.render_html import render


def _fired(f) -> bool:
    return (f.availability in (Availability.OK, Availability.DEGRADED)
            and f.disposition in (Disposition.QUARANTINE, Disposition.REVIEW)
            and f.severity.rank >= 2)          # medium or above


def run_bench(corpus: Path, out: Path, profile: str = "deep",
              formats=("pt", "onnx")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((corpus / "manifest.json").read_text())
    x = np.load(corpus / "probe_x.npy")[:400]
    y = np.load(corpus / "probe_y.npy")[:400]

    refs = [m for m in man["models"] if not m["backdoored"] and "pt" in m
            and not m.get("modified") and not m.get("benign_variant")]
    rows, results = [], []

    for entry in man["models"]:
        for fmt in formats:
            if fmt not in entry:
                continue
            mid = f"{entry['id']}[{fmt}]"
            try:
                model = load_model(corpus / entry[fmt], ARCH_REGISTRY, mid)
            except Exception as exc:
                rows.append({"model": mid, "error": str(exc)[:120]})
                continue
            battery = ModelBattery(models=[
                load_model(corpus / r["pt"], ARCH_REGISTRY, r["id"])
                for r in refs if r["id"] != entry["id"]][:2])
            ctx = RunContext(probes_x=x, probes_y=y, battery=battery,
                             out_dir=out, seed=7)
            res = scan(model, ctx, profile)
            results.append(res)
            row = {"model": mid, "truth": "backdoored" if entry["backdoored"] else "clean",
                   "trigger": entry.get("trigger"), "asr": entry.get("asr"),
                   "verdict": res.verdict}
            for f in res.findings:
                row[f.detector_id] = (
                    "FIRE" if _fired(f) else
                    ("-" if f.availability in (Availability.OK, Availability.DEGRADED)
                     else f.availability.value[:4]))
            rows.append(row)
            print(f"  {mid:28} truth={row['truth']:11} verdict={res.verdict}")

    # --- per-detector scoring ---------------------------------------------
    detectors = sorted({k for r in rows for k in r
                        if k.startswith("model.")})
    scoreboard = []
    for d in detectors:
        bd = [r for r in rows if r.get("truth") == "backdoored" and d in r]
        cl = [r for r in rows if r.get("truth") == "clean" and d in r]
        tp = sum(1 for r in bd if r[d] == "FIRE")
        fp = sum(1 for r in cl if r[d] == "FIRE")
        ran = sum(1 for r in rows if r.get(d) in ("FIRE", "-"))
        scoreboard.append({
            "detector": d,
            "detection_rate": round(tp / len(bd), 3) if bd else None,
            "false_alarm_rate": round(fp / len(cl), 3) if cl else None,
            "ran_on": f"{ran}/{len(rows)}",
        })

    summary = {"rows": rows, "scoreboard": scoreboard}
    (out / "bench.json").write_text(json.dumps(summary, indent=2))
    _render_matrix(rows, scoreboard, detectors, out / "bench.html")
    render(results, out / "all_models.report.html", "CV Assurance — Module B — full corpus")

    print("\n  detector                          detect   false-alarm   ran")
    for s in scoreboard:
        dr = "  n/a" if s["detection_rate"] is None else f"{s['detection_rate']:5.2f}"
        fa = "  n/a" if s["false_alarm_rate"] is None else f"{s['false_alarm_rate']:5.2f}"
        print(f"  {s['detector']:32} {dr}      {fa}    {s['ran_on']}")
    return summary


def _render_matrix(rows, scoreboard, detectors, path: Path) -> None:
    import html as H

    from cva.report.render_html import CSS
    p = [f"<!doctype html><meta charset=utf-8><title>Module B benchmark</title>"
         f"<style>{CSS}.fire{{background:#fdeceb;color:#b3261e;font-weight:700}}"
         f".na{{color:#9aa0aa}}td,th{{white-space:nowrap}}</style>"
         "<div class=wrap><h1>Module B — benchmark</h1>"
         "<div class=sub>Ground truth from the attack lab manifest. "
         "FIRE = detector raised a medium-or-above finding with a review/quarantine "
         "disposition.</div>"]

    p.append("<h2>Per-detector scoreboard</h2><div class=scroll><table>"
             "<tr><th>detector</th><th>detection rate</th><th>false-alarm rate</th>"
             "<th>ran on</th></tr>")
    for s in scoreboard:
        dr = "n/a" if s["detection_rate"] is None else f"{s['detection_rate']:.2f}"
        fa = "n/a" if s["false_alarm_rate"] is None else f"{s['false_alarm_rate']:.2f}"
        p.append(f"<tr><td><code>{s['detector']}</code></td><td>{dr}</td>"
                 f"<td>{fa}</td><td class=lim>{s['ran_on']}</td></tr>")
    p.append("</table></div>")

    p.append("<h2>Detection matrix</h2><div class=scroll><table><tr><th>model</th>"
             "<th>truth</th><th>trigger</th><th>ASR</th><th>verdict</th>"
             + "".join(f"<th>{d.replace('model.','')}</th>" for d in detectors) + "</tr>")
    for r in rows:
        if "error" in r:
            p.append(f"<tr><td><code>{H.escape(r['model'])}</code></td>"
                     f"<td colspan=99 class=lim>load error: {H.escape(r['error'])}</td></tr>")
            continue
        asr = "" if r.get("asr") is None else f"{r['asr']:.2f}"
        cells = ""
        for d in detectors:
            v = r.get(d, "")
            cls = "fire" if v == "FIRE" else ("na" if v not in ("-", "") else "")
            cells += f"<td class={cls}>{v}</td>"
        p.append(f"<tr><td><code>{H.escape(r['model'])}</code></td>"
                 f"<td>{r['truth']}</td><td class=lim>{r.get('trigger') or '—'}</td>"
                 f"<td class=lim>{asr}</td>"
                 f"<td><span class='pill p-{r['verdict']}'>{r['verdict']}</span></td>"
                 f"{cells}</tr>")
    p.append("</table></div></div>")
    path.write_text("".join(p), encoding="utf-8")
