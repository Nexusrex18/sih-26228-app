"""The read-only views (plan §7.2), in the single-file report's own read order.

verdict banner -> access assumptions -> contributor risk -> findings -> provenance -> shift
-> coverage -> reproduction. Both artefacts tell the same story in the same sequence, and
the contributor table precedes individual findings because it is what an acceptance analyst
actually acts on.
"""
from __future__ import annotations

from typing import Any

from flask import Blueprint, redirect, render_template, request, url_for

from .. import services
from ..auth import require_login
from ..coverage.view import coverage_groups, compare_coverage
from ..reports.index import severity_sorted
from ..reports.loader import load, validate_scan_id
from ..workflow.events import allowed_reason_codes, is_prov, new_request_id

bp = Blueprint("main", __name__)


@bp.get("/")
@require_login
def scans() -> Any:
    idx = services.index()
    idx.refresh()
    rows = idx.scans()
    records = services.ledger_records(types=("scan_record",))
    sealed_by_scan = {}
    if records is not None:
        from ..reports import seal_check
        for row in rows:
            if row.readable:
                sealed_by_scan[row.scan_id] = seal_check.check(
                    row.report_sha256, row.scan_id, records, ledger_available=True)
    return render_template("scans.html", rows=rows, sealed=sealed_by_scan,
                           ledger_readable=records is not None)


@bp.get("/scans/<scan_id>")
@require_login
def scan(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    data = view.report.data
    effective = _effective_counts(view)
    return render_template(
        "scan.html", view=view, d=data, effective=effective,
        access=data.get("access_assumptions") or {},
        plan_rows=data.get("plan") or [],
        contributors=(data.get("contributor_risk") or [])[:8],
        contributor_total=len(data.get("contributor_risk") or []),
        provenance=data.get("provenance_summary"),
        permutation=data.get("permutation_test"),
        baseline=data.get("contributor_baseline"),
        baseline_unavailable=data.get("contributor_baseline_unavailable"),
        coverage=coverage_groups(data.get("coverage") or {}))


@bp.get("/scans/<scan_id>/findings")
@require_login
def findings(scan_id: str) -> Any:
    scan_id = validate_scan_id(scan_id)
    view = services.load_scan(scan_id)
    cfg = services.config()
    idx = services.index()

    filters = {k: (request.args.get(k) or None)
               for k in ("disposition", "nature", "availability", "module")}
    page = max(1, _int(request.args.get("page"), 1))
    per_page = cfg.findings_per_page
    total = idx.count(scan_id, **filters)
    ids = idx.finding_ids(scan_id, limit=per_page, offset=(page - 1) * per_page, **filters)
    rows = severity_sorted([view.report.findings_by_id[i] for i in ids
                            if i in view.report.findings_by_id])

    return render_template(
        "findings.html", view=view, rows=rows, filters=filters, page=page,
        per_page=per_page, total=total,
        pages=max(1, (total + per_page - 1) // per_page),
        modules=idx.modules(scan_id),
        unfiltered_total=idx.count(scan_id),
        plan_rows=view.report.data.get("plan") or [],
        effective=_effective_counts(view))


@bp.get("/scans/<scan_id>/findings/<finding_id>")
@require_login
def finding(scan_id: str, finding_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    f = view.finding(finding_id)
    if f is None:
        return render_template(
            "error.html", code=404, title="No such finding in this report",
            detail=f"{finding_id} is not one of the {len(view.report.findings)} findings "
                   f"in scan {view.scan_id}."), 404
    state = view.state_of(finding_id)
    prov = is_prov(f.get("detector_id"))
    return render_template("finding.html", view=view, f=f, state=state, is_prov=prov,
                           reason_codes=allowed_reason_codes(prov),
                           # One id per form render. A double-click or a retry after a
                           # timeout submits the SAME id, and ledgerd returns the original
                           # seq rather than recording the decision twice (§7.3).
                           request_id=new_request_id(),
                           health=services.ledger_health())


@bp.get("/scans/<scan_id>/contributors")
@require_login
def contributors(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    rows = view.report.data.get("contributor_risk") or []
    group = request.args.get("group") or "contributor"
    present = sorted({str(r.get("group_key")) for r in rows})
    shown = [r for r in rows if str(r.get("group_key")) == group]
    return render_template(
        "contributors.html", view=view, rows=shown, group=group, groups=present,
        baseline=view.report.data.get("contributor_baseline"),
        baseline_unavailable=view.report.data.get("contributor_baseline_unavailable"),
        permutation=view.report.data.get("permutation_test"))


@bp.get("/scans/<scan_id>/provenance")
@require_login
def provenance(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    prov_findings = [f for f in view.report.findings
                     if is_prov(f.get("detector_id"))]
    return render_template("provenance.html", view=view,
                           summary=view.report.data.get("provenance_summary"),
                           rows=severity_sorted(prov_findings),
                           health=services.ledger_health())


@bp.get("/scans/<scan_id>/coverage")
@require_login
def coverage(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    other_id = request.args.get("compare")
    other = None
    if other_id:
        other = load(services.config().reports_dir, validate_scan_id(other_id))
    return render_template(
        "coverage.html", view=view,
        groups=coverage_groups(view.report.data.get("coverage") or {}),
        calibration=view.report.data.get("calibration"),
        other=other,
        other_groups=coverage_groups(other.data.get("coverage") or {}) if other else None,
        diff=compare_coverage(view.report.data.get("coverage") or {},
                              other.data.get("coverage") or {}) if other else None,
        candidates=[r for r in services.index().scans()
                    if r.readable and r.scan_id != view.scan_id])


@bp.get("/scans/<scan_id>/reproduction")
@require_login
def reproduction(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    return render_template("reproduction.html", view=view,
                           repro=view.report.data.get("reproduction") or {},
                           produced_by=view.report.data.get("produced_by") or {},
                           target=view.report.data.get("target") or {})


@bp.get("/scans/<scan_id>/drift")
@require_login
def drift(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    return render_template("drift.html", view=view,
                           summary=view.report.data.get("drift_summary"))


@bp.get("/health")
def health() -> Any:
    """Unauthenticated liveness only. It says nothing about any scan."""
    return {"status": "ok"}, 200


def _int(raw: str | None, default: int) -> int:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def _effective_counts(view: services.ScanView) -> dict[str, int]:
    """Counts AFTER the fold — kept next to the tool's own, never instead of them (D-E5)."""
    out = {"accept": 0, "review": 0, "quarantine": 0, "pending": 0}
    for st in view.state.findings.values():
        out[st.effective] = out.get(st.effective, 0) + 1
        if st.pending:
            out["pending"] += 1
    return out
