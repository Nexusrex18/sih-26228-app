"""Single-file, self-contained HTML assurance report. No CDN — the air gap forbids it,
and a report that loses its styling offline looks broken, so the analyst concludes the
tool is broken too.

Read order is deliberate: verdict, then ACCESS ASSUMPTIONS, then what was NOT checked,
then findings. An analyst has to know the scope of the assessment before its conclusions.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from cva.core.capability import Availability, Capability

CSS = """
:root{--bg:#fbfbfd;--fg:#1a1d23;--mut:#666c78;--line:#e2e5ea;--card:#fff;
--crit:#b3261e;--high:#c8641b;--med:#a37c10;--low:#4a6fa5;--info:#6b7280;
--ok:#1f7a45;--warn:#a37c10;--bad:#b3261e;--code:#f4f5f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 12px;
padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:0 0 6px}
.sub{color:var(--mut);font-size:13px;margin-bottom:24px}
.verdict{padding:16px 20px;border-radius:10px;font-size:20px;font-weight:600;
margin:18px 0;border:1px solid}
.v-QUARANTINE{background:#fdeceb;border-color:#f3b6b1;color:var(--bad)}
.v-REVIEW{background:#fdf6e3;border-color:#ecd9a0;color:var(--warn)}
.v-ACCEPT{background:#e9f7ef;border-color:#a9dcc0;color:var(--ok)}
table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);
vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin:12px 0}
.card.crit{border-left:4px solid var(--crit)}.card.high{border-left:4px solid var(--high)}
.card.med{border-left:4px solid var(--med)}.card.low{border-left:4px solid var(--low)}
.card.info{border-left:4px solid var(--info)}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;
font-weight:600;letter-spacing:.03em;text-transform:uppercase}
.p-critical{background:#fdeceb;color:var(--crit)}.p-high{background:#fdf0e6;color:var(--high)}
.p-medium{background:#fdf8e8;color:var(--med)}.p-low{background:#eef3fa;color:var(--low)}
.p-info{background:#f1f2f4;color:var(--info)}
.p-OK{background:#e9f7ef;color:var(--ok)}.p-DEGRADED{background:#fdf8e8;color:var(--warn)}
.p-UNAVAILABLE{background:#f1f2f4;color:var(--mut)}.p-ERROR{background:#fdeceb;color:var(--bad)}
.meta{color:var(--mut);font-size:12px;margin:6px 0}
.reason{margin:8px 0 10px}
details{margin-top:8px}summary{cursor:pointer;font-size:12px;color:var(--mut)}
pre{background:var(--code);padding:10px 12px;border-radius:7px;overflow-x:auto;
font-size:12px;margin:8px 0}
img{max-width:100%;border:1px solid var(--line);border-radius:7px;margin:8px 0;
background:#fff}
ul{margin:6px 0;padding-left:20px}li{margin:3px 0}
.lim{color:var(--mut);font-size:12.5px}
.scroll{overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:20px}.stat span{color:var(--mut);font-size:12px}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--mut:#9aa0aa;
--line:#2a2e35;--card:#1b1e23;--code:#20242a}
.v-QUARANTINE{background:#2a1614}.v-REVIEW{background:#2a2414}.v-ACCEPT{background:#14241b}}
"""


def _b64(p: Path) -> str | None:
    try:
        return base64.b64encode(p.read_bytes()).decode()
    except Exception:
        return None


def _ev_html(ev, base: Path) -> str:
    out = []
    for e in ev:
        cap = html.escape(e.caption)
        if e.path:
            b = _b64(base / e.path)
            if b:
                out.append(f'<div class="meta">{cap}</div>'
                           f'<img alt="{cap}" src="data:image/png;base64,{b}">')
                continue
        if e.data is not None:
            out.append(f'<details><summary>{cap}</summary>'
                       f'<pre>{html.escape(json.dumps(e.data, indent=2)[:4000])}</pre></details>')
    return "".join(out)


def render(results, out_path: Path, title: str = "CV Assurance — Module B") -> Path:
    base = out_path.parent
    if not isinstance(results, list):
        results = [results]

    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style><div class=wrap><h1>{html.escape(title)}</h1>"]

    for r in results:
        parts.append(f'<div class="sub">model <code>{html.escape(r.model_id)}</code> · '
                     f'format <code>{html.escape(r.model_fmt)}</code> · '
                     f'scan <code>{r.scan_id}</code></div>')
        parts.append(f'<div class="verdict v-{r.verdict}">VERDICT: {r.verdict}</div>')

        # --- access assumptions, before any conclusion ---------------------
        parts.append("<h2>Access assumptions for this scan</h2><div class=scroll><table>"
                     "<tr><th>capability</th><th>state</th><th>why</th></tr>")
        for cap in Capability:
            if not cap.name.startswith(("MODEL_", "REFERENCE_")):
                continue
            have = cap in r.capabilities
            note = r.capabilities.note_for(cap) or ""
            parts.append(
                f"<tr><td><code>{cap.value}</code></td>"
                f"<td><span class='pill p-{'OK' if have else 'UNAVAILABLE'}'>"
                f"{'available' if have else 'absent'}</span></td>"
                f"<td class=lim>{html.escape(note)}</td></tr>")
        parts.append("</table></div>")

        # --- the plan -----------------------------------------------------
        counts = {}
        for row in r.plan:
            counts[row.resolution.state.value] = counts.get(row.resolution.state.value, 0) + 1
        parts.append("<h2>Check plan — resolved before anything ran</h2><div class=grid>")
        for k in ("OK", "DEGRADED", "UNAVAILABLE"):
            parts.append(f"<div class=stat><b>{counts.get(k,0)}</b><span>{k}</span></div>")
        parts.append("</div><div class=scroll><table>"
                     "<tr><th>check</th><th>state</th><th>reason</th><th>time</th></tr>")
        for row in sorted(r.plan, key=lambda x: x.check_id):
            parts.append(
                f"<tr><td><code>{row.check_id}</code></td>"
                f"<td><span class='pill p-{row.resolution.state.value}'>"
                f"{row.resolution.state.value}</span></td>"
                f"<td class=lim>{html.escape(row.resolution.reason)}</td>"
                f"<td class=lim>{r.timings.get(row.check_id,'—')}s</td></tr>")
        parts.append("</table></div>")

        # --- findings, worst first ----------------------------------------
        ran = [f for f in r.findings
               if f.availability in (Availability.OK, Availability.DEGRADED)]
        notrun = [f for f in r.findings if f not in ran]
        parts.append(f"<h2>Findings ({len(ran)} assessed)</h2>")
        for f in sorted(ran, key=lambda x: (-x.severity.rank, -x.confidence)):
            parts.append(
                f'<div class="card {f.severity.value}">'
                f'<h3><code>{f.detector_id}</code> '
                f'<span class="pill p-{f.severity.value}">{f.severity.value}</span> '
                f'<span class="pill p-{f.availability.value}">{f.availability.value}</span></h3>'
                f'<div class=meta>confidence {f.confidence:.2f} · score {f.score_raw:.4f} '
                f'· threshold {f.threshold} · disposition <b>{f.disposition.value}</b> '
                f'(<code>{f.disposition_rule}</code>) · nature {f.nature.value}</div>'
                f'<div class=reason>{html.escape(f.reason)}</div>'
                + _ev_html(f.evidence, base)
                + ("<div class=lim><b>Access:</b> " +
                   html.escape("; ".join(f.access_assumptions)) + "</div>"
                   if f.access_assumptions else "")
                + ("<div class=lim><b>Limitations:</b><ul>" +
                   "".join(f"<li>{html.escape(l)}</li>" for l in f.limitations) + "</ul></div>"
                   if f.limitations else "")
                + "</div>")

        # --- generated coverage -------------------------------------------
        assessed, missed = {}, {}
        for row in r.plan:
            tgt = assessed if row.resolution.runnable else missed
            for ac in row.attack_classes:
                tgt.setdefault(ac, []).append(row.check_id)
        parts.append("<h2>Coverage — generated, not written</h2>"
                     "<div class=scroll><table><tr><th>attack class</th><th>assessed by</th>"
                     "<th>not assessed</th></tr>")
        for ac in sorted(set(assessed) | set(missed)):
            parts.append(
                f"<tr><td><code>{ac}</code></td>"
                f"<td>{', '.join(f'<code>{c}</code>' for c in assessed.get(ac, [])) or '—'}</td>"
                f"<td class=lim>{', '.join(missed.get(ac, [])) or '—'}</td></tr>")
        parts.append("</table></div>")
        if notrun:
            parts.append("<h3>Not assessed in this scan</h3><ul class=lim>")
            for f in notrun:
                parts.append(f"<li><code>{f.detector_id}</code> — "
                             f"{html.escape(f.reason)}</li>")
            parts.append("</ul>")

    parts.append("</div>")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path
