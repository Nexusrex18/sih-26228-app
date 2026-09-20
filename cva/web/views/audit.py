"""The audit-trail views and the bridge to `cva-seal verify` (plan §7.4)."""
from __future__ import annotations

from typing import Any

from flask import Blueprint, flash, redirect, render_template, request, url_for

from .. import services
from ..audit import timeline
from ..audit.verify_bridge import export
from ..auth import require_login, require_role
from ..reports.loader import SCAN_ID_RE

bp = Blueprint("audit", __name__, url_prefix="/audit")


@bp.get("/")
@require_login
def index() -> Any:
    scan_id = request.args.get("scan_id") or None
    if scan_id and not SCAN_ID_RE.fullmatch(scan_id):
        scan_id = None
    records = services.ledger_records(
        scan_id=scan_id, types=("analyst_event", "scan_record"))
    health = services.ledger_health()
    rows = timeline.rows(records or [])
    return render_template("timeline.html", rows=rows, health=health,
                           readable=records is not None, scan_id=scan_id,
                           decisions=timeline.decision_count(rows),
                           scans=[r for r in services.index().scans() if r.readable])


@bp.post("/verify")
@require_login
def reverify() -> Any:
    """Run the verifier now. Read-only: it signs nothing and writes nothing."""
    services.verify_cache().invalidate()
    health = services.ledger_health()
    state = health.verify_state
    if state is None or state.state == "UNAVAILABLE":
        flash(("unavailable", health.banner_text), "verify")
    elif state.ok:
        flash(("ok", f"Verified: {state.summary}"), "verify")
    else:
        flash(("failed", f"Verification FAILED: {state.summary}"), "verify")
    return redirect(request.referrer or url_for("audit.index"))


@bp.get("/verification")
@require_login
def verification() -> Any:
    """The full verifier result, including what it does NOT claim."""
    health = services.ledger_health()
    return render_template("verification.html", health=health, state=health.verify_state)


@bp.post("/export")
@require_role("approver", "admin")
def export_ledger() -> Any:
    """Write the strict JSONL export — verifiable elsewhere with the trust root alone."""
    cfg = services.config()
    if cfg.ledger_path is None:
        flash(("unavailable", "No ledger_path is configured, so there is nothing to export."),
              "verify")
        return redirect(url_for("audit.verification"))
    out = cfg.ledger_path.with_suffix(".export.jsonl")
    ok, detail = export(cfg.ledger_path, out, timeout_s=cfg.subprocess_timeout_s)
    flash(("ok" if ok else "failed",
           (f"Exported to {out}. Verify it on another machine with the trust root alone; "
            "docs/VERIFICATION-PROCEDURE.md has the commands.") if ok
           else f"Export failed: {detail}"), "verify")
    return redirect(url_for("audit.verification"))
