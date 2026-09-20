"""The remediation front-end (plan §7.5, PS G5).

The wording rule is enforced in the templates, not just intended: `tests/web/test_wording.py`
renders these pages and asserts the forbidden words never appear.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from .. import services
from ..auth import current_account, require_login, require_role
from ..remediation.jobs import JobRunner, entry_point_available
from ..reports.loader import load, validate_scan_id

bp = Blueprint("remediation", __name__)


def _runner() -> JobRunner:
    runner = current_app.extensions.get("cva_remediation")
    if runner is None:
        runner = JobRunner(services.config().reports_dir.parent / "remediated")
        current_app.extensions["cva_remediation"] = runner
    return runner


@bp.get("/scans/<scan_id>/remediation")
@require_login
def index(scan_id: str) -> Any:
    scan_id = validate_scan_id(scan_id)
    view = services.load_scan(scan_id)
    available, why = entry_point_available()
    quarantined = [t for t in view.state.targets.values() if t.status == "quarantined"]
    return render_template("remediation.html", view=view, available=available, why=why,
                           quarantined=quarantined, jobs=_runner().list_jobs(scan_id))


@bp.post("/scans/<scan_id>/remediation/start")
@require_role("approver")
def start(scan_id: str) -> Any:
    """Only an approver, and only from an approved quarantine (plan §7.5 step 1)."""
    scan_id = validate_scan_id(scan_id)
    view = services.load_scan(scan_id)
    account = current_account()
    assert account is not None

    target_type = (request.form.get("target_type") or "").strip()
    target_ref = request.form.get("target_ref") or ""
    from ..workflow.events import encode_text
    target = view.state.targets.get((target_type, encode_text(target_ref)))
    if target is None or target.status != "quarantined":
        flash(("failed",
               "Remediation starts from an approved quarantine. That asset is not under "
               "one, so there is nothing recorded to justify removing it."), "remediation")
        return redirect(url_for("remediation.index", scan_id=scan_id))

    dataset = (view.report.data.get("target") or {}).get("dataset_path")
    if not dataset:
        flash(("failed",
               "This scan records no dataset_path, so there is no artefact to re-emit."),
              "remediation")
        return redirect(url_for("remediation.index", scan_id=scan_id))

    job = _runner().start(scan_id=scan_id, target_type=target_type, target_ref=target_ref,
                          requested_by=account.actor_id, dataset=Path(str(dataset)))
    return redirect(url_for("remediation.job", scan_id=scan_id, job_id=job.job_id))


@bp.get("/scans/<scan_id>/remediation/jobs/<job_id>")
@require_login
def job(scan_id: str, job_id: str) -> Any:
    scan_id = validate_scan_id(scan_id)
    view = services.load_scan(scan_id)
    j = _runner().get(job_id)
    if j is None or j.scan_id != scan_id:
        return render_template("error.html", code=404, title="No such remediation job",
                               detail="Jobs do not survive a restart of the dashboard."), 404
    rescan = None
    rescan_id = request.args.get("rescan")
    if rescan_id:
        rescan = load(services.config().reports_dir, validate_scan_id(rescan_id))
    return render_template(
        "remediation_job.html", view=view, job=j, rescan=rescan,
        candidates=[r for r in services.index().scans()
                    if r.readable and r.scan_id != scan_id])
