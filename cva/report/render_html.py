"""Single-file, self-contained HTML assurance report. No CDN — the air gap forbids it,
and a report that loses its styling offline looks broken, so the analyst concludes the
tool is broken too.

Read order is deliberate: verdict, then ACCESS ASSUMPTIONS, then contributor risk, then
findings grouped by what to DO about them, then provenance, shift, coverage and how to
reproduce. An analyst has to know the scope of the assessment before its conclusions.

Three sections always print, and say why when they are empty (contributor risk, provenance,
shift). A report with no provenance section is indistinguishable from a scan where
provenance was never checked; absence of evidence must not render as evidence of absence.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from cva.core.access_block import consequence
from cva.core.capability import Availability, Capability
from cva.report.report_json import (
    coverage_of,
    provenance_summary_of,
    reproduction_of,
    standing_limitations,
)

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
margin:18px 0 6px;border:1px solid}
.v-QUARANTINE{background:#fdeceb;border-color:#f3b6b1;color:var(--bad)}
.v-REVIEW{background:#fdf6e3;border-color:#ecd9a0;color:var(--warn)}
.v-ACCEPT{background:#e9f7ef;border-color:#a9dcc0;color:var(--ok)}
.scope{margin:0 0 18px;font-size:14px}
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
.p-quarantine{background:#fdeceb;color:var(--bad)}.p-review{background:#fdf8e8;color:var(--warn)}
.p-accept{background:#e9f7ef;color:var(--ok)}
.meta{color:var(--mut);font-size:12px;margin:6px 0}
.reason{margin:8px 0 10px}
.note{background:var(--card);border:1px dashed var(--line);border-radius:8px;
padding:10px 14px;margin:10px 0;color:var(--mut);font-size:13px}
details{margin-top:8px}summary{cursor:pointer;font-size:12px;color:var(--mut)}
pre{background:var(--code);padding:10px 12px;border-radius:7px;overflow-x:auto;
font-size:12px;margin:8px 0}
img{max-width:100%;border:1px solid var(--line);border-radius:7px;margin:8px 0;
background:#fff}
ul{margin:6px 0;padding-left:20px}li{margin:3px 0}
.lim{color:var(--mut);font-size:12.5px}
.scroll{overflow-x:auto}
.rd{max-width:360px;width:100%;height:auto;display:block;margin:8px 0}
.rd .frame{fill:var(--card);stroke:var(--line)}
.rd .diag{stroke:var(--mut);stroke-dasharray:4 3;fill:none}
.rd .pt{fill:var(--low);fill-opacity:.65;stroke:var(--low)}
.rd text{fill:var(--mut);font-size:11px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:20px}.stat span{color:var(--mut);font-size:12px}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--mut:#9aa0aa;
--line:#2a2e35;--card:#1b1e23;--code:#20242a}
.v-QUARANTINE{background:#2a1614}.v-REVIEW{background:#2a2414}.v-ACCEPT{background:#14241b}}
"""

# What the renderer inlines when the profile carries no `evidence` block. The tiers in
# core/profile.py override these.
DEFAULT_CAPS = {"max_images_per_finding": 4, "max_findings_rendered": 200,
                "max_report_bytes": 15_000_000}
_SEV_CLS = {"critical": "crit", "high": "high", "medium": "med", "low": "low", "info": "info"}
_GROUPS = (("quarantine", "Quarantine"), ("review", "Review"), ("accept", "Accept"))
_RAN = (Availability.OK, Availability.DEGRADED)


class _Doc:
    """Accumulates the page and tracks its size so image inlining can honour the cap."""

    def __init__(self, budget: int) -> None:
        self.parts: list[str] = []
        self.size = 0
        self.budget = budget

    def add(self, s: str) -> None:
        self.parts.append(s)
        self.size += len(s)


def _e(x: Any) -> str:
    return html.escape(str(x))


def _pill(cls: str, text: str) -> str:
    return f"<span class='pill p-{_e(cls)}'>{_e(text)}</span>"


def _mime(blob: bytes) -> str | None:
    """Sniffed from magic bytes: a bare content hash has no suffix to go on."""
    head = blob.lstrip()[:512]
    if blob.startswith(b"\x89PNG"):
        return "image/png"
    if blob.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head):
        return "image/svg+xml"
    return None


def _blob(root: Path, ref: str) -> bytes | None:
    """`root` is the evidence directory itself; `ref` names a file directly inside it.

    The schema says the ref is a bare hash, but Module A's EvidenceStore returns
    `evidence/<sha256>.<ext>` (a documented Module A contract deviation), so exactly one
    leading `evidence/` prefix is stripped. Anything still shaped like a path is refused,
    not resolved."""
    if ref.startswith("evidence/"):
        ref = ref[len("evidence/"):]
    if not ref or "/" in ref or "\\" in ref or ref.startswith("."):
        return None
    try:
        return (root / ref).read_bytes()
    except OSError:
        return None


def _text_of(blob: bytes | None, data: Any) -> str | None:
    if blob is not None:
        text = blob.decode("utf-8", "replace")
        try:
            return json.dumps(json.loads(text), indent=2, default=str)
        except ValueError:
            return text
    if data is not None:
        return json.dumps(data, indent=2, default=str)
    return None


def _evidence(f, root: Path, doc: _Doc, max_images: int, stats: dict[str, int]) -> str:
    out: list[str] = []
    shown = used = 0
    for ev in f.evidence:
        cap = _e(ev.caption)
        blob = _blob(root, ev.path) if ev.path else None
        mime = _mime(blob) if blob is not None else None
        if mime and blob is not None:
            b64 = base64.b64encode(blob).decode()
            if shown >= max_images or doc.size + used + len(b64) > doc.budget:
                stats["images_omitted"] += 1
                continue
            shown += 1
            piece = (f'<div class="meta">{cap}</div>'
                     f'<img alt="{cap}" src="data:{mime};base64,{b64}">')
            used += len(piece)
            out.append(piece)
            continue
        text = _text_of(blob, ev.data)
        if text is not None:
            out.append(f"<details><summary>{cap}</summary>"
                       f"<pre>{_e(text[:4000])}</pre></details>")
        elif ev.path:
            # A referenced artefact that cannot be read must be visible, not silently absent.
            out.append(f"<div class=meta>{cap} — evidence file not found: {_e(ev.path)}</div>")
    return "".join(out)


def _card(f, root: Path, doc: _Doc, max_images: int, stats: dict[str, int]) -> str:
    return (
        f'<div class="card {_SEV_CLS.get(f.severity.value, "info")}">'
        f'<h3><code>{_e(f.detector_id)}</code> '
        f'{_pill(f.severity.value, f.severity.value)} '
        f'{_pill(f.availability.value, f.availability.value)}</h3>'
        f'<div class=meta>confidence {f.confidence:.2f} · score {f.score_raw:.4f} '
        f'· threshold {f.threshold} · disposition <b>{_e(f.disposition.value)}</b> '
        f'(<code>{_e(f.disposition_rule)}</code>) · nature {_e(f.nature.value)}</div>'
        f'<div class=reason>{_e(f.reason)}</div>'
        + _evidence(f, root, doc, max_images, stats)
        + ("<div class=lim><b>Access:</b> " + _e("; ".join(f.access_assumptions)) + "</div>"
           if f.access_assumptions else "")
        + ("<div class=lim><b>Limitations:</b><ul>"
           + "".join(f"<li>{_e(lim)}</li>" for lim in f.limitations) + "</ul></div>"
           if f.limitations else "")
        + "</div>")


def _verdict_section(doc: _Doc, r) -> None:
    errored = {f.detector_id for f in r.findings if f.availability == Availability.ERROR}
    ran = sum(1 for row in r.plan
              if row.resolution.runnable and row.check_id not in errored)
    total = len(r.plan)
    doc.add(f'<div class="verdict v-{_e(r.verdict)}">VERDICT: {_e(r.verdict)}</div>')
    if ran == 0:
        scope = ("<b>No check ran.</b> This verdict reflects nothing having been assessed, "
                 "not a clean result.")
    else:
        scope = (f"<b>{ran} of {total} planned checks ran; this verdict covers only "
                 f"those.</b> The rest are listed below as not assessed.")
    doc.add(f'<div class="scope">{scope}</div>')


def _access_section(doc: _Doc, r) -> None:
    doc.add("<h2>Access assumptions for this scan</h2><div class=scroll><table>"
            "<tr><th>capability</th><th>state</th><th>why</th></tr>")
    for cap in Capability:
        have = cap in r.capabilities
        note = "" if have else (r.capabilities.note_for(cap) or "not available for this model")
        doc.add(f"<tr><td><code>{cap.value}</code></td>"
                f"<td>{_pill('OK' if have else 'UNAVAILABLE', 'available' if have else 'absent')}"
                f"</td><td class=lim>{_e(note)}</td></tr>")
    doc.add(f"</table></div><div class=lim>{_e(consequence(r.plan))}</div>")


def _contributor_section(doc: _Doc, r) -> None:
    doc.add("<h2>Contributor risk</h2>")
    rows = getattr(r, "contributor_risk", None) or []
    if not rows:
        doc.add("<div class=note>Not computed in this build. Source-level aggregation "
                "(beta-binomial posterior per contributor and batch) is not part of this "
                "report yet, so the absence of a contributor flag here means nothing.</div>")
        return
    doc.add("<div class=scroll><table><tr><th>group</th><th>n</th><th>flagged</th>"
            "<th>posterior</th><th>95% CI</th></tr>")
    for c in rows:
        doc.add(f"<tr><td><code>{_e(c['group_key'])}:{_e(c['group_value'])}</code></td>"
                f"<td>{c['n_samples']}</td><td>{c['n_flagged']}</td>"
                f"<td>{c['posterior_mean']:.3f}</td>"
                f"<td>{c['ci_low']:.3f}–{c['ci_high']:.3f}</td></tr>")
    doc.add("</table></div>")
    perm = getattr(r, "permutation_test", None)
    if perm:
        doc.add(f"<div class=note>Permutation test ({perm['n_permutations']} permutations, "
                f"p = {perm['p_value']:.4f}): {_e(perm['conclusion'])}</div>")
    base = getattr(r, "contributor_baseline", None)
    why = getattr(r, "contributor_baseline_unavailable", None)
    if base:
        line = (f"Reference dataset: {base['reference_flagged']} of {base['reference_n']} images "
                f"flagged ({base['reference_rate']:.1%}); this cohort's baseline rate is "
                f"{base['cohort_rate']:.1%}.")
        if base["cohort_exceeds_reference"]:
            line += (" Cohort prior unreliable here: the baseline every group is compared "
                     "with is itself higher than clean data produces, so a contaminated "
                     "cohort can make its contributors look ordinary.")
    elif why:
        line = (f"A reference dataset was supplied but not used ({why}), so a contaminated "
                "cohort would not be detected.")
    else:
        line = ("No reference dataset supplied, so a contaminated cohort would not be "
                "detected.")
    doc.add(f"<div class=note>{_e(line)}</div>")


def _findings_section(doc: _Doc, r, root: Path, caps: dict[str, int]) -> None:
    ran = [f for f in r.findings if f.availability in _RAN]
    notrun = [f for f in r.findings if f.availability not in _RAN]
    order = {name: i for i, (name, _) in enumerate(_GROUPS)}
    ran.sort(key=lambda f: (order.get(f.disposition.value, len(order)),
                            -f.severity.rank, -f.confidence))
    limit = caps["max_findings_rendered"]
    stats = {"images_omitted": 0}

    body = _Doc(doc.budget)
    body.size = doc.size            # image inlining sees the page built so far
    current = None
    for f in ran[:limit]:
        if f.disposition.value != current:
            current = f.disposition.value
            label = dict(_GROUPS).get(current, current)
            body.add(f"<h3 style='margin-top:20px'>{_pill(current, label)}</h3>")
        body.add(_card(f, root, body, caps["max_images_per_finding"], stats))

    doc.add(f"<h2>Findings ({len(ran)} assessed)</h2>")
    if len(ran) > limit:
        doc.add(f"<div class=note>{limit} of {len(ran)} findings rendered "
                f"(highest disposition and severity first); the full set is in "
                f"report.json.</div>")
    if stats["images_omitted"]:
        doc.add(f"<div class=note>{stats['images_omitted']} images omitted to stay within "
                f"the tier's size and per-finding caps; full set in report.json and the "
                f"evidence directory.</div>")
    if not ran:
        doc.add("<div class=note>No check produced an assessed finding.</div>")
    for part in body.parts:
        doc.add(part)

    doc.add("<h3 style='margin-top:26px'>Not assessed / check failed</h3>")
    if not notrun:
        doc.add("<div class=lim>Every planned check ran.</div>")
        return
    doc.add("<ul class=lim>")
    for f in notrun:
        pill = _pill(f.availability.value, f.availability.value)
        doc.add(f"<li><code>{_e(f.detector_id)}</code> {pill} — {_e(f.reason)}</li>")
    doc.add("</ul>")


def _provenance_section(doc: _Doc, r) -> None:
    doc.add("<h2>Provenance</h2>")
    if Capability.INFERENCE_LEDGER not in r.capabilities:
        doc.add("<div class=note>No inference ledger was supplied, so provenance was not "
                "checked. A report with no provenance findings is NOT evidence that "
                "provenance is intact.</div>")
    else:
        doc.add("<div class=note>An inference ledger was supplied, but the provenance "
                "summary is not computed in this build.</div>")
    # The same object report.json carries; a field absent there is unknown, not zero.
    summ = provenance_summary_of(r) or {}
    labels = (("records_verified", "records verified"), ("anchors_checked", "anchors checked"),
              ("declared_degraded_intervals", "declared degraded intervals"),
              ("records_after_last_anchor", "records after last anchor"),
              ("custody_type", "custody type"), ("durability_window", "durability window"))
    stats = [f"<div class=stat><b>{_e(summ[k])}</b><span>{lab}</span></div>"
             for k, lab in labels if k in summ]
    if stats:
        doc.add("<div class=grid>" + "".join(stats) + "</div>")
    # ledger_state is omitted from the JSON when a signing key exists (the scan record is
    # appended after report.json is hashed); the sequence number is known by render time.
    seq = getattr(r, "ledger_seq", None)
    err = getattr(r, "ledger_error", None)
    if seq:
        state = f"sealed, seq {_e(seq)}"
    elif err:
        # The ledger RAISED. Distinct from "no key was supplied", and previously
        # indistinguishable from it: `append_scan_record` swallowed the exception, so the
        # report said the record was not sealed and could not say why.
        state = f"NOT SEALED — the audit ledger raised: {_e(err)}"
    elif summ.get("ledger_state") == "not_sealed":
        state = "not sealed — no audit ledger with a signing key"
    else:
        state = "not sealed — no scan record was appended to the audit ledger"
    doc.add(f"<div class=lim>Scan record: {state}.</div>")


def _shift_section(doc: _Doc) -> None:
    doc.add("<h2>Distribution shift</h2>"
            "<div class=note>Not computed in this build. No drift summary is present in "
            "this report; that is not a finding of no shift.</div>")


def _plan_section(doc: _Doc, r) -> None:
    """All four states at equal prominence: a shorter report must not look like a cleaner one."""
    counts: dict[str, int] = {}
    for row in r.plan:
        counts[row.resolution.state.value] = counts.get(row.resolution.state.value, 0) + 1
    doc.add("<h2>Check plan — resolved before anything ran</h2><div class=grid>")
    for state in ("OK", "DEGRADED", "UNAVAILABLE", "ERROR"):
        doc.add(f"<div class=stat><b>{counts.get(state, 0)}</b><span>{state}</span></div>")
    doc.add("</div><div class=scroll><table>"
            "<tr><th>check</th><th>state</th><th>reason</th><th>time</th></tr>")
    for row in sorted(r.plan, key=lambda x: x.check_id):
        res = row.resolution
        why = f" [{_e(res.exclusion_reason)}]" if res.exclusion_reason else ""
        doc.add(f"<tr><td><code>{_e(row.check_id)}</code></td>"
                f"<td>{_pill(res.state.value, res.state.value)}</td>"
                f"<td class=lim>{_e(res.reason)}{why}</td>"
                f"<td class=lim>{_e(r.timings.get(row.check_id, '—'))}s</td></tr>")
    doc.add("</table></div>")


def _brier(cal: dict[str, Any]) -> str:
    b = cal.get("brier")
    return f"{b:.4f}" if isinstance(b, (int, float)) else "not computed"


def _reliability_svg(bins: list[dict[str, Any]]) -> str:
    """Inline SVG: predicted probability (x) against observed frequency (y), the y=x
    diagonal, marker area scaled by bin count. No xmlns (HTML needs none, and the report
    carries no URL at all) and colours come from the CSS variables, so dark mode works."""
    pts: list[tuple[float, float, int]] = []
    for b in bins:
        try:
            pts.append((min(max(float(b["p_mean"]), 0.0), 1.0),
                        min(max(float(b["empirical"]), 0.0), 1.0), max(int(b["n"]), 0)))
        except (KeyError, TypeError, ValueError):
            continue
    if not pts:
        return "<div class=lim>No reliability bins were recorded, so no diagram is drawn.</div>"
    size, pad = 320, 40
    span = size - 2 * pad

    def px(v: float) -> str:
        return f"{pad + v * span:.1f}"

    def py(v: float) -> str:
        return f"{size - pad - v * span:.1f}"

    nmax = max(n for _, _, n in pts) or 1
    parts = [f'<svg class=rd viewBox="0 0 {size} {size}" role="img" '
             f'aria-label="Reliability diagram: predicted probability against observed '
             f'frequency">',
             f'<rect class=frame x="{pad}" y="{pad}" width="{span}" height="{span}"/>']
    for t in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{px(t)}" y="{size - pad + 16}" text-anchor="middle">{t:g}</text>')
        parts.append(f'<text x="{pad - 6}" y="{float(py(t)) + 4:.1f}" '
                     f'text-anchor="end">{t:g}</text>')
    parts.append(f'<line class=diag x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}"/>')
    for p, e, n in pts:
        rad = 3.0 + 9.0 * (n / nmax) ** 0.5
        parts.append(f'<circle class=pt cx="{px(p)}" cy="{py(e)}" r="{rad:.1f}">'
                     f'<title>predicted {p:.3f}, observed {e:.3f}, n = {n}</title></circle>')
    parts.append(f'<text x="{size / 2:.0f}" y="{size - 6}" text-anchor="middle">'
                 f'predicted probability (bin mean)</text>')
    parts.append(f'<text x="12" y="{size / 2:.0f}" text-anchor="middle" '
                 f'transform="rotate(-90 12 {size / 2:.0f})">observed frequency</text>')
    parts.append("</svg>")
    return ("<div class=meta>Reliability diagram: points on the dashed diagonal are "
            "perfectly calibrated; marker size scales with bin count n.</div>"
            + "".join(parts))


def _coverage_section(doc: _Doc, r) -> None:
    cov = coverage_of(r)
    doc.add("<h2>Coverage — generated, not written</h2>"
            "<div class=scroll><table><tr><th>attack class</th><th>assessed by</th>"
            "<th>not assessed</th></tr>")
    for ac in sorted(set(cov["assessed"]) | set(cov["not_assessed"])):
        got = ", ".join(f"<code>{_e(c)}</code>" for c in cov["assessed"].get(ac, [])) or "—"
        missed = ", ".join(_e(c) for c in cov["not_assessed"].get(ac, [])) or "—"
        doc.add(f"<tr><td><code>{_e(ac)}</code></td><td>{got}</td>"
                f"<td class=lim>{missed}</td></tr>")
    doc.add("</table></div>")
    if cov["never_covered"]:
        doc.add("<div class=lim><b>No check covers at all:</b> "
                + ", ".join(f"<code>{_e(c)}</code>" for c in cov["never_covered"]) + "</div>")
    if cov["operational_reports"]:
        doc.add("<div class=lim><b>Operational reports (not counted as coverage):</b> "
                + ", ".join(f"<code>{_e(c)}</code>" for c in cov["operational_reports"])
                + "</div>")
    cal = getattr(r, "calibration", None)
    doc.add("<div class=note>" + (
        f"Calibration: {_e(cal.get('method', 'unknown'))}, Brier {_e(_brier(cal))}; "
        f"excluded detectors {_e(', '.join(cal.get('excluded_detectors') or []) or 'none')}."
        if cal else
        "Calibration: not applied. No labelled benchmark was supplied, so every confidence "
        "shown is the detector's own and has not been checked against outcomes.") + "</div>")
    if cal:
        doc.add(_reliability_svg(cal.get("reliability_bins") or []))
    doc.add("<h3 style='margin-top:20px'>Standing limitations</h3><ul class=lim>"
            + "".join(f"<li>{_e(s)}</li>"
                      for s in standing_limitations(getattr(r, "target", None) or {}))
            + "</ul>")


def _reproduction_section(doc: _Doc, r, command: str | None) -> None:
    rep = reproduction_of(r, command)
    prof = getattr(r, "profile", None) or {}
    doc.add("<h2>Reproduction</h2>"
            f"<pre>{_e(rep['command'])}</pre>"
            f"<div class=meta>profile <code>{_e(r.profile_name)}</code> "
            f"(tier <code>{_e(prof.get('budget_tier', '—'))}</code>) · "
            f"profile hash <code>{_e(r.profile_hash[:16])}</code> · "
            f"commit <code>{_e(r.code_commit)}</code> · seeds "
            + ", ".join(f"{_e(k)}=<code>{_e(v)}</code>" for k, v in rep["seeds"].items())
            + "</div>"
            "<div class=lim><b>Fields expected to differ between runs:</b> "
            + ", ".join(f"<code>{_e(p)}</code>" for p in rep["volatile_paths"]) + "</div>"
            "<div class=lim><b>Environment:</b> "
            + ", ".join(f"{_e(k)} {_e(v)}" for k, v in rep["env"].items()) + "</div>"
            "<ul class=lim>" + "".join(f"<li>{_e(n)}</li>" for n in rep["determinism_notes"])
            + "</ul>")


def _caps_of(r) -> dict[str, int]:
    return {**DEFAULT_CAPS, **((getattr(r, "profile", None) or {}).get("evidence") or {})}


def render(results, out_path: Path, title: str = "CV Assurance — Module B",
           evidence_root: Path | None = None, command: str | None = None) -> Path:
    """Evidence is resolved against `evidence_root` — the shared `<out>/evidence/` store —
    which defaults to a sibling of the report. `cmd_scan` passes it explicitly because the
    report sits one level down, in `<out>/<scan_id>/`."""
    root = evidence_root if evidence_root is not None else out_path.parent / "evidence"
    if not isinstance(results, list):
        results = [results]
    # One page holds every result, so the tightest tier among them sets the byte budget.
    budget = min((_caps_of(r)["max_report_bytes"] for r in results),
                 default=DEFAULT_CAPS["max_report_bytes"])

    doc = _Doc(budget)
    doc.add(f"<!doctype html><meta charset=utf-8><title>{_e(title)}</title>"
            f"<style>{CSS}</style><div class=wrap><h1>{_e(title)}</h1>")

    for r in results:
        r_caps = _caps_of(r)
        doc.add(f'<div class="sub">model <code>{_e(r.model_id)}</code> · '
                f'format <code>{_e(r.model_fmt)}</code> · scan <code>{_e(r.scan_id)}</code></div>')
        _verdict_section(doc, r)
        _access_section(doc, r)
        _contributor_section(doc, r)
        _findings_section(doc, r, root, r_caps)
        _provenance_section(doc, r)
        _shift_section(doc)
        _plan_section(doc, r)
        _coverage_section(doc, r)
        _reproduction_section(doc, r, command)

    doc.add("</div>")
    out_path.write_text("".join(doc.parts), encoding="utf-8")
    return out_path
