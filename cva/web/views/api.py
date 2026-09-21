"""JSON API for the Next.js dashboard.

The same data the server-rendered templates use, over `/api/*`. Both consumers go through
`services.load_scan`, so the fold, the seal check and the ledger-trust state are computed in
exactly one place and the two front ends cannot drift.

Security notes, because an API is a second authenticated surface:

  * Session cookie auth, same as the templates. No bearer tokens, no second credential path.
  * Every state-changing route still requires the CSRF token, now via the `X-CSRF-Token`
    header (the `before_request` hook accepts either that or the form field).
  * `SameSite=Strict` on the session cookie means a cross-site request never carries it.
  * Responses are JSON, so the XSS surface moves to the client. React escapes by default and
    nothing in the frontend uses `dangerouslySetInnerHTML` — asserted by a test.
"""
from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request, session

from .. import services
from ..auth import current_account, require_login
from ..coverage.view import compare_coverage, coverage_groups
from ..reports.loader import load, validate_scan_id
from ..security import new_csrf_token
from ..workflow import events as E
from ..workflow.fold import ROLE_ACTIONS
from ..workflow.states import FindingState, TargetState

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.get("/session")
def whoami() -> Any:
    """Who am I, what may I do, and the CSRF token to send back."""
    account = current_account()
    if session.get("csrf") is None:
        session["csrf"] = new_csrf_token()
    if account is None:
        return jsonify({"authenticated": False, "csrf_token": session["csrf"]})
    return jsonify({
        "authenticated": True,
        "actor_id": account.actor_id,
        "role": account.role,
        "can": sorted(ROLE_ACTIONS.get(account.role, frozenset())),
        "csrf_token": session["csrf"],
        "policy": {
            "four_eyes": services.config().four_eyes,
            "min_justification_chars": services.config().min_justification_chars,
            "min_justification_words": services.config().min_justification_words,
            "min_justification_chars_other": services.config().min_justification_chars_other,
        },
    })


@bp.get("/health")
@require_login
def ledger_health() -> Any:
    h = services.ledger_health()
    v = h.verify_state
    return jsonify({
        "reachable": h.reachable,
        "writable": h.writable,
        "verified": h.verified,
        "kind": h.banner_kind,
        "text": h.banner_text,
        "detail": h.detail,
        "verify": None if v is None else {
            "state": v.state, "summary": v.summary, "detail": v.detail,
            "checked_at": v.checked_at_label,
            "records_checked": v.records_checked,
            "anchors_verified": v.anchors_verified,
            "anchors_in_chain": v.anchors_in_chain,
            "unwitnessed_records": v.unwitnessed_records,
            "declared_gaps": v.declared_gaps,
            "durability": v.durability, "loss_window": v.loss_window,
            "command": v.command_line,
            "findings": list(v.findings),
            "limitations": list(v.limitations),
        },
    })


@bp.get("/scans")
@require_login
def scans() -> Any:
    idx = services.index()
    idx.refresh()
    rows = idx.scans()
    records = services.ledger_records(types=("scan_record",))
    sealed: dict[str, Any] = {}
    if records is not None:
        from ..reports import seal_check
        for r in rows:
            if r.readable:
                s = seal_check.check(r.report_sha256, r.scan_id, records,
                                     ledger_available=True)
                sealed[r.scan_id] = _seal(s)
    return jsonify({
        "ledger_readable": records is not None,
        "scans": [{
            "scan_id": r.scan_id, "created_at_utc": r.created_at_utc,
            "verdict": r.verdict, "profile_name": r.profile_name,
            "budget_tier": r.budget_tier, "readable": r.readable,
            "unreadable_reason": r.unreadable_reason,
            "n_findings": r.n_findings, "counts": {
                "accept": r.n_accept, "review": r.n_review,
                "quarantine": r.n_quarantine},
            "seal": sealed.get(r.scan_id),
        } for r in rows],
    })


@bp.get("/scans/<scan_id>")
@require_login
def scan(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    d = view.report.data
    groups = coverage_groups(d.get("coverage") or {})
    effective = {"accept": 0, "review": 0, "quarantine": 0, "pending": 0}
    for st in view.state.findings.values():
        effective[st.effective] = effective.get(st.effective, 0) + 1
        if st.pending:
            effective["pending"] += 1
    return jsonify({
        "scan_id": view.scan_id,
        "created_at_utc": d.get("created_at_utc"),
        "verdict": d.get("verdict"),
        "produced_by": d.get("produced_by"),
        "target": d.get("target"),
        "access_assumptions": d.get("access_assumptions"),
        "plan": d.get("plan"),
        "contributor_risk": d.get("contributor_risk"),
        "contributor_baseline": d.get("contributor_baseline"),
        "contributor_baseline_unavailable": d.get("contributor_baseline_unavailable"),
        "permutation_test": d.get("permutation_test"),
        "provenance_summary": d.get("provenance_summary"),
        "drift_summary": d.get("drift_summary"),
        "calibration": d.get("calibration"),
        "reproduction": d.get("reproduction"),
        "report_sha256": view.report.sha256,
        "seal": _seal(view.seal),
        "events_available": view.events_available,
        "effective": effective,
        "tool_counts": view.report.counts_by_disposition(),
        "coverage": _coverage(groups),
        "n_findings": len(view.report.findings),
    })


@bp.get("/scans/<scan_id>/findings")
@require_login
def findings(scan_id: str) -> Any:
    scan_id = validate_scan_id(scan_id)
    view = services.load_scan(scan_id)
    idx = services.index()
    cfg = services.config()

    filters = {k: (request.args.get(k) or None)
               for k in ("disposition", "nature", "availability", "module")}
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    per_page = min(500, max(1, int(request.args.get("per_page", cfg.findings_per_page))))
    total = idx.count(scan_id, **filters)
    ids = idx.finding_ids(scan_id, limit=per_page, offset=(page - 1) * per_page, **filters)
    rows = [view.report.findings_by_id[i] for i in ids if i in view.report.findings_by_id]
    return jsonify({
        "scan_id": scan_id,
        "total": total,
        "unfiltered_total": idx.count(scan_id),
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "modules": idx.modules(scan_id),
        "filters": filters,
        "plan": view.report.data.get("plan") or [],
        "findings": [_finding(f, view.state_of(str(f["finding_id"]))) for f in rows],
    })


@bp.get("/scans/<scan_id>/findings/<finding_id>")
@require_login
def finding(scan_id: str, finding_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    f = view.finding(finding_id)
    if f is None:
        return jsonify({"error": "unknown_finding",
                        "detail": f"{finding_id} is not in scan {view.scan_id}"}), 404
    prov = E.is_prov(f.get("detector_id"))
    state = view.state_of(finding_id)
    return jsonify({
        "scan_id": view.scan_id,
        "finding": _finding(f, state),
        "is_prov": prov,
        "reason_codes": list(E.allowed_reason_codes(prov)),
        "request_id": E.new_request_id(),
        "events": [_event(e) for e in (state.events if state else [])],
        "events_available": view.events_available,
    })


@bp.get("/scans/<scan_id>/targets")
@require_login
def targets(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    return jsonify({"targets": [_target(t) for t in view.state.targets.values()]})


@bp.get("/scans/<scan_id>/coverage")
@require_login
def coverage(scan_id: str) -> Any:
    view = services.load_scan(validate_scan_id(scan_id))
    other_id = request.args.get("compare")
    payload: dict[str, Any] = {
        "scan_id": view.scan_id,
        "profile_name": view.report.profile_name,
        "budget_tier": view.report.budget_tier,
        "coverage": _coverage(coverage_groups(view.report.data.get("coverage") or {})),
        "calibration": view.report.data.get("calibration"),
    }
    if other_id:
        other = load(services.config().reports_dir, validate_scan_id(other_id))
        diff = compare_coverage(view.report.data.get("coverage") or {},
                                other.data.get("coverage") or {})
        payload["other"] = {
            "scan_id": other.scan_id, "profile_name": other.profile_name,
            "budget_tier": other.budget_tier,
            "coverage": _coverage(coverage_groups(other.data.get("coverage") or {})),
        }
        payload["diff"] = {
            "only_left": list(diff.only_left), "only_right": list(diff.only_right),
            "both": list(diff.both), "shrank": diff.shrank,
            "lost_reasons": {k: list(v) for k, v in diff.lost_reasons.items()},
        }
    return jsonify(payload)


@bp.get("/audit")
@require_login
def audit() -> Any:
    from ..audit import timeline
    scan_id = request.args.get("scan_id") or None
    if scan_id:
        scan_id = validate_scan_id(scan_id)
    records = services.ledger_records(scan_id=scan_id,
                                      types=("analyst_event", "scan_record"))
    rows = timeline.rows(records or [])
    return jsonify({
        "readable": records is not None,
        "decisions": timeline.decision_count(rows),
        "rows": [{
            "seq": r.seq, "kind": r.kind, "created_at_utc": r.created_at_utc,
            "actor": r.actor, "role": r.role, "action": r.action, "target": r.target,
            "detail": r.detail, "justification": r.justification,
            "reason_code": r.reason_code, "refs_seq": r.refs_seq,
            "scan_id": r.scan_id, "audit_flagged": r.audit_flagged,
            "key_id": r.key_id,
        } for r in rows],
    })


# --- the write path -------------------------------------------------------------------

@bp.post("/scans/<scan_id>/findings/<finding_id>/act")
@require_login
def act(scan_id: str, finding_id: str) -> Any:
    """The write path (plan §5.3), six steps, and nothing local changes before step 6:
    authn/role/CSRF (the before-request hook) → validate and build the request, with no
    local mutation → ask cva-ledgerd → ledgerd checks peercred, schema, role/state and
    optimistic concurrency → SealedLedger.append, failing closed → refold on ack only.
    No optimistic UI and no local queue (D-E4)."""
    from ..workflow.fold import FoldRejection, check_admissible
    from ..workflow.ledgerd_client import LedgerRefused, LedgerUnreachable
    from ..workflow.states import LedgerEvent

    scan_id = validate_scan_id(scan_id)
    account = current_account()
    assert account is not None
    view = services.load_scan(scan_id)
    f = view.finding(finding_id)
    if f is None:
        return _refusal("unknown_finding", f"{finding_id} is not in scan {scan_id}"), 404

    health = services.ledger_health()
    if not health.writable:
        return _refusal("ledger_unavailable", health.banner_text), 503

    body = request.get_json(silent=True) or {}
    state = view.state_of(finding_id)
    cfg = services.config()
    try:
        req = E.build(
            actor_id=account.actor_id, role=account.role,
            action=str(body.get("action") or ""), scan_id=scan_id,
            target_type=str(f.get("target_type", "sample")),
            target_ref=str(f.get("target_ref", "")), finding_id=finding_id,
            justification=str(body.get("justification") or ""),
            new_disposition=body.get("new_disposition") or None,
            reason_code=body.get("reason_code") or None,
            assignee=(body.get("assignee") or None),
            refs_seq=body.get("refs_seq"),
            expected_prev_seq=state.last_seq if state else 0,
            request_id=str(body.get("request_id") or "") or None,
            is_prov_finding=E.is_prov(f.get("detector_id")),
            min_chars=cfg.min_justification_chars,
            min_words=cfg.min_justification_words,
            min_chars_other=cfg.min_justification_chars_other)
    except E.EventError as e:
        return _refusal(e.code, e.detail), 400

    try:
        check_admissible(view.state,
                         LedgerEvent.from_record({"seq": 0, "analyst": req.body}),
                         four_eyes=cfg.four_eyes)
    except FoldRejection as e:
        return _refusal(e.code, e.detail), 409

    try:
        written = services.client().append_analyst_event(req.as_wire())
    except LedgerUnreachable as e:
        return _refusal("ledger_unavailable", str(e)), 503
    except LedgerRefused as e:
        return _refusal(e.code, e.detail), 409

    services.verify_cache().invalidate()
    refreshed = services.load_scan(scan_id)
    return jsonify({
        "ok": True, "seq": written.seq, "deduped": written.deduped,
        "action": req.action,
        "message": f"{req.action} recorded at ledger seq {written.seq}"
                   + (" (already recorded; it was not recorded twice)"
                      if written.deduped else ""),
        "finding": _finding(refreshed.finding(finding_id) or f,
                            refreshed.state_of(finding_id)),
        "events": [_event(e) for e in (refreshed.state_of(finding_id).events
                                       if refreshed.state_of(finding_id) else [])],
    })


@bp.post("/scans/<scan_id>/targets/act")
@require_login
def target_act(scan_id: str) -> Any:
    from ..workflow.fold import FoldRejection, check_admissible
    from ..workflow.ledgerd_client import LedgerRefused, LedgerUnreachable
    from ..workflow.states import LedgerEvent

    scan_id = validate_scan_id(scan_id)
    account = current_account()
    assert account is not None
    view = services.load_scan(scan_id)
    health = services.ledger_health()
    if not health.writable:
        return _refusal("ledger_unavailable", health.banner_text), 503

    body = request.get_json(silent=True) or {}
    target_type = str(body.get("target_type") or "")
    target_ref = str(body.get("target_ref") or "")
    target = view.state.targets.get((target_type, E.encode_text(target_ref)))
    cfg = services.config()
    try:
        req = E.build(
            actor_id=account.actor_id, role=account.role,
            action=str(body.get("action") or ""), scan_id=scan_id,
            target_type=target_type, target_ref=target_ref,
            finding_id=E.target_event_finding_id(target_type, target_ref,
                                                 body.get("finding_id")),
            justification=str(body.get("justification") or ""),
            reason_code=body.get("reason_code") or None,
            refs_seq=body.get("refs_seq"),
            expected_prev_seq=target.last_seq if target else 0,
            request_id=str(body.get("request_id") or "") or None,
            min_chars=cfg.min_justification_chars,
            min_words=cfg.min_justification_words,
            min_chars_other=cfg.min_justification_chars_other)
    except E.EventError as e:
        return _refusal(e.code, e.detail), 400

    try:
        check_admissible(view.state,
                         LedgerEvent.from_record({"seq": 0, "analyst": req.body}),
                         four_eyes=cfg.four_eyes)
    except FoldRejection as e:
        return _refusal(e.code, e.detail), 409

    try:
        written = services.client().append_analyst_event(req.as_wire())
    except LedgerUnreachable as e:
        return _refusal("ledger_unavailable", str(e)), 503
    except LedgerRefused as e:
        return _refusal(e.code, e.detail), 409

    services.verify_cache().invalidate()
    refreshed = services.load_scan(scan_id)
    return jsonify({"ok": True, "seq": written.seq, "deduped": written.deduped,
                    "message": f"{req.action} recorded at ledger seq {written.seq}",
                    "targets": [_target(t) for t in refreshed.state.targets.values()]})


@bp.post("/audit/verify")
@require_login
def reverify() -> Any:
    services.verify_cache().invalidate()
    return ledger_health()


@bp.post("/audit/export")
@require_login
def export_ledger() -> Any:
    """Write the strict JSONL export, verifiable elsewhere with the trust root alone.

    The role check is here rather than on `require_role` because that decorator renders the
    HTML error page; an API caller gets the same refusal shape as every other refusal, with
    `nothing_changed` said out loud. Exporting reads the ledger and signs nothing.
    """
    from ..audit.verify_bridge import export

    account = current_account()
    assert account is not None
    if account.role not in ("approver", "admin"):
        return _refusal(
            "role_not_permitted",
            f"Exporting the ledger needs the approver or admin role; your account holds "
            f"'{account.role}'.",
        ), 403

    cfg = services.config()
    if cfg.ledger_path is None:
        return _refusal("ledger_unavailable",
                        "No ledger_path is configured, so there is nothing to export."), 503
    out = cfg.ledger_path.with_suffix(".export.jsonl")
    ok, detail = export(cfg.ledger_path, out, timeout_s=cfg.subprocess_timeout_s)
    if not ok:
        return _refusal("export_failed", detail), 502
    return jsonify({"ok": True, "path": str(out), "detail": detail})


# --- account management (D-E9) ---------------------------------------------------------
# `admin` manages accounts and holds no workflow rights: one person controlling both
# identity and decisions is exactly what the separation exists to prevent. Every route
# checks the role itself so a refusal is JSON, not the HTML error page.

def _admin_only() -> Any | None:
    account = current_account()
    if account is None or account.role != "admin":
        return _refusal("role_not_permitted",
                        "Managing accounts needs the admin role; your account holds "
                        f"'{account.role if account else 'none'}'."), 403
    return None


def _accounts_store():
    from flask import current_app
    return current_app.extensions["cva_accounts"]


def _account_row(a: Any) -> dict[str, Any]:
    return {"actor_id": a.actor_id, "role": a.role, "disabled": a.disabled}


def _accounts_payload(message: str | None = None) -> Any:
    from ..accounts import MIN_PASSWORD_CHARS, ROLES
    body: dict[str, Any] = {
        "accounts": [_account_row(a) for a in _accounts_store().list_accounts()],
        "roles": list(ROLES),
        "min_password_chars": MIN_PASSWORD_CHARS,
    }
    if message:
        body["ok"] = True
        body["message"] = message
    return jsonify(body)


@bp.get("/admin/accounts")
@require_login
def admin_accounts() -> Any:
    refused = _admin_only()
    return refused if refused is not None else _accounts_payload()


@bp.post("/admin/accounts")
@require_login
def admin_create_account() -> Any:
    from ..accounts import AccountError

    refused = _admin_only()
    if refused is not None:
        return refused
    body = request.get_json(silent=True) or {}
    try:
        acct = _accounts_store().create(str(body.get("actor_id") or "").strip(),
                                        str(body.get("password") or ""),
                                        str(body.get("role") or "viewer").strip())
    except AccountError as e:
        return _refusal("account_refused", str(e)), 400
    return _accounts_payload(f"Created {acct.actor_id} as {acct.role}.")


@bp.post("/admin/accounts/<actor_id>")
@require_login
def admin_update_account(actor_id: str) -> Any:
    """One route, one change per request: `role`, `disabled` or `password`.

    A role change, a disable and a password reset each end the user's live sessions
    (AccountStore bumps the session epoch), so the change binds on their next request.
    """
    from ..accounts import AccountError

    refused = _admin_only()
    if refused is not None:
        return refused
    store = _accounts_store()
    if store.get(actor_id) is None:
        return _refusal("no_such_account", f"No account {actor_id!r}."), 404
    body = request.get_json(silent=True) or {}
    try:
        if "role" in body:
            acct = store.set_role(actor_id, str(body["role"]))
            msg = (f"{acct.actor_id} is now {acct.role}. Their sessions were ended, so the "
                   "change takes effect on their next request.")
        elif "disabled" in body:
            disabled = bool(body["disabled"])
            store.set_disabled(actor_id, disabled)
            msg = (f"{actor_id} is {'disabled' if disabled else 'enabled'}. Their past "
                   "decisions stay in the ledger, which is the point of an append-only "
                   "record.")
        elif "password" in body:
            store.set_password(actor_id, str(body["password"]))
            msg = f"Password set for {actor_id}; their sessions were ended."
        else:
            return _refusal("account_refused",
                            "Send one of: role, disabled, password."), 400
    except AccountError as e:
        return _refusal("account_refused", str(e)), 400
    return _accounts_payload(msg)


# --- shaping --------------------------------------------------------------------------

#: Why a refusal happened, in the analyst's terms. The daemon's `detail` is always shown
#: as well; this is the heading above it.
REFUSAL_TITLES = {
    "ledger_unavailable": "Nothing was recorded — the ledger is not writable",
    "stale_state": "Someone else acted on this first",
    "not_permitted": "Your role does not allow this",
    "four_eyes": "This needs a second person",
    "no_such_pending": "There is no such change to approve",
    "unknown_finding": "That finding is not in this report",
    "justification_required": "A justification is required",
    "invalid_reason_code": "That reason does not apply to this finding",
    "not_quarantined": "That asset is not under quarantine",
    "read_only": "The workflow is read-only",
    "role_not_permitted": "Your role does not allow this",
    "export_failed": "The export did not complete",
    "account_refused": "The account change was refused",
    "no_such_account": "There is no such account",
}


def _refusal(code: str, detail: str) -> Any:
    return jsonify({"ok": False, "error": code,
                    "title": REFUSAL_TITLES.get(code, "That action was refused"),
                    "detail": detail,
                    # Said explicitly, on every refusal, because it is the property the
                    # whole write path exists to guarantee.
                    "nothing_changed": True})


def _seal(s: Any) -> dict[str, Any]:
    return {"state": s.state, "label": s.label, "detail": s.detail,
            "is_alarm": s.is_alarm, "sealed_digest": s.sealed_digest,
            "on_disk_digest": s.on_disk_digest, "ledger_seq": s.ledger_seq}


def _coverage(g: Any) -> dict[str, Any]:
    return {
        "assessed": [{"attack_class": r.attack_class, "checks": list(r.checks)}
                     for r in g.assessed],
        "not_assessed": [{"attack_class": r.attack_class, "checks": list(r.checks)}
                         for r in g.not_assessed],
        "never_covered": list(g.never_covered),
        "operational_reports": list(g.operational_reports),
        "standing_limitations": list(g.standing_limitations),
        "total_attack_classes": g.total_attack_classes,
        "assessed_fraction": g.assessed_fraction,
    }


def _finding(f: dict[str, Any], state: FindingState | None) -> dict[str, Any]:
    out = dict(f)
    out["state"] = None if state is None else {
        "original": state.original,
        "original_rule": state.original_rule,
        "effective": state.effective,
        "cited_seq": state.cited_seq,
        "changed_by_human": state.changed_by_human,
        "assignee": state.assignee,
        "acknowledged_by": list(state.acknowledged_by),
        "last_seq": state.last_seq,
        "pending": None if state.pending is None else {
            "seq": state.pending.seq,
            "requested_by": state.pending.requested_by,
            "action": state.pending.action,
            "new_disposition": state.pending.new_disposition,
            "reason_code": state.pending.reason_code,
            "justification": state.pending.justification_text,
        },
        "superseded": [{"seq": p.seq, "requested_by": p.requested_by,
                        "new_disposition": p.new_disposition} for p in state.superseded],
    }
    return out


def _target(t: TargetState) -> dict[str, Any]:
    return {
        "target_type": t.target_type,
        "target_ref": t.target_ref_text,
        "status": t.status,
        "cited_seq": t.cited_seq,
        "last_seq": t.last_seq,
        "pending_release": None if t.pending_release is None else {
            "seq": t.pending_release.seq,
            "requested_by": t.pending_release.requested_by,
            "justification": t.pending_release.justification_text,
        },
    }


def _event(e: Any) -> dict[str, Any]:
    return {"seq": e.seq, "actor_id": e.actor_id, "role": e.role, "action": e.action,
            "new_disposition": e.new_disposition, "reason_code": e.reason_code,
            "assignee": e.assignee, "refs_seq": e.refs_seq,
            "justification": e.justification_text,
            "target_ref": e.target_ref_text, "target_type": e.target_type,
            "created_at_utc": e.created_at_utc}
