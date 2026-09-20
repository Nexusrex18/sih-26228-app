"""The write path (plan §5.3). Six steps, and nothing local changes before step 6.

  1. authn -> role check -> CSRF (the `before_request` hook did the last one)
  2. validate lengths and charset, build the request  -- NO local mutation
  3. ask cva-ledgerd over the socket
  4. ledgerd: peercred -> schema -> role/state -> optimistic concurrency
  5. ledgerd: SealedLedger.append                      [fails closed on any error]
  6. on ack only: refold and re-render

There is no optimistic UI and no local queue. If steps 3 to 5 fail the browser is told the
reason and the state is exactly what it was (D-E4).
"""
from __future__ import annotations

from typing import Any

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from .. import services
from ..auth import current_account, require_login
from ..reports.loader import validate_scan_id
from ..workflow import events as E
from ..workflow.fold import FoldRejection, check_admissible
from ..workflow.ledgerd_client import LedgerRefused, LedgerUnreachable
from ..workflow.states import LedgerEvent

bp = Blueprint("workflow", __name__)

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
}


@bp.post("/scans/<scan_id>/findings/<finding_id>/act")
@require_login
def act(scan_id: str, finding_id: str) -> Any:
    scan_id = validate_scan_id(scan_id)
    account = current_account()
    assert account is not None
    view = services.load_scan(scan_id)
    f = view.finding(finding_id)
    if f is None:
        return _refused(view, None, "unknown_finding",
                        f"{finding_id} is not in scan {scan_id}.")

    health = services.ledger_health()
    if not health.writable:
        return _refused(view, f, "ledger_unavailable", health.banner_text)

    action = (request.form.get("action") or "").strip()
    if action not in E.ANALYST_ACTIONS:
        return _refused(view, f, "invalid_request", f"{action!r} is not an action.")

    state = view.state_of(finding_id)
    cfg = services.config()
    try:
        req = E.build(
            actor_id=account.actor_id, role=account.role, action=action, scan_id=scan_id,
            target_type=str(f.get("target_type", "sample")),
            target_ref=str(f.get("target_ref", "")), finding_id=finding_id,
            justification=request.form.get("justification") or "",
            new_disposition=request.form.get("new_disposition") or None,
            reason_code=request.form.get("reason_code") or None,
            assignee=(request.form.get("assignee") or "").strip() or None,
            refs_seq=_int(request.form.get("refs_seq")),
            expected_prev_seq=state.last_seq if state else 0,
            request_id=_request_id(),
            is_prov_finding=E.is_prov(f.get("detector_id")),
            min_chars=cfg.min_justification_chars,
            min_words=cfg.min_justification_words,
            min_chars_other=cfg.min_justification_chars_other)
    except E.EventError as e:
        return _refused(view, f, e.code, e.detail)

    # The same gate ledgerd applies, applied here first, so the analyst sees the specific
    # reason instead of a bare refusal from a daemon they cannot read the logs of. The
    # daemon repeats it independently; this is a usability layer, never a substitute.
    try:
        check_admissible(view.state, LedgerEvent.from_record({"seq": 0, "analyst": req.body}),
                         four_eyes=cfg.four_eyes)
    except FoldRejection as e:
        return _refused(view, f, e.code, e.detail)

    return _submit(view, f, req)


@bp.post("/scans/<scan_id>/targets/act")
@require_login
def target_act(scan_id: str) -> Any:
    """Quarantine or release a contributor / batch / model / dataset as a whole."""
    scan_id = validate_scan_id(scan_id)
    account = current_account()
    assert account is not None
    view = services.load_scan(scan_id)

    health = services.ledger_health()
    if not health.writable:
        return _refused(view, None, "ledger_unavailable", health.banner_text)

    action = (request.form.get("action") or "").strip()
    if action not in (*E.TARGET_LEVEL_ACTIONS, "approve"):
        return _refused(view, None, "invalid_request", f"{action!r} is not a target action.")
    target_type = (request.form.get("target_type") or "").strip()
    target_ref = request.form.get("target_ref") or ""
    cfg = services.config()

    key = (target_type, E.encode_text(target_ref))
    target = view.state.targets.get(key)
    try:
        req = E.build(
            actor_id=account.actor_id, role=account.role, action=action, scan_id=scan_id,
            target_type=target_type, target_ref=target_ref,
            finding_id=E.target_event_finding_id(
                target_type, target_ref, request.form.get("finding_id")),
            justification=request.form.get("justification") or "",
            reason_code=request.form.get("reason_code") or None,
            refs_seq=_int(request.form.get("refs_seq")),
            expected_prev_seq=target.last_seq if target else 0,
            request_id=_request_id(),
            min_chars=cfg.min_justification_chars,
            min_words=cfg.min_justification_words,
            min_chars_other=cfg.min_justification_chars_other)
    except E.EventError as e:
        return _refused(view, None, e.code, e.detail)

    try:
        check_admissible(view.state, LedgerEvent.from_record({"seq": 0, "analyst": req.body}),
                         four_eyes=cfg.four_eyes)
    except FoldRejection as e:
        return _refused(view, None, e.code, e.detail)

    return _submit(view, None, req, back=url_for("main.contributors", scan_id=scan_id))


def _submit(view: services.ScanView, f: dict[str, Any] | None,
            req: E.AnalystEventRequest, back: str | None = None) -> Any:
    try:
        written = services.client().append_analyst_event(req.as_wire())
    except LedgerUnreachable as e:
        return _refused(view, f, "ledger_unavailable", str(e))
    except LedgerRefused as e:
        return _refused(view, f, e.code, e.detail)

    # The verification cache is keyed on ledger size, so it invalidates itself; this makes
    # the banner refresh on the very next render rather than at the next TTL boundary.
    services.verify_cache().invalidate()
    # Only NOW does anything change: the fold is recomputed from the ledger on the next
    # request, because the ledger is the only place the decision lives.
    session.pop("cva_request_id", None)
    flash(("recorded" if not written.deduped else "deduped",
           f"{req.action} recorded at ledger seq {written.seq}"
           + (" (this request had already been recorded; it was not recorded twice)"
              if written.deduped else "")), "workflow")
    return redirect(back or (url_for("main.finding", scan_id=view.scan_id,
                                     finding_id=req.finding_id)))


def _refused(view: services.ScanView, f: dict[str, Any] | None, code: str,
             detail: str) -> Any:
    """Fail closed, and say which of the two kinds of failure this was.

    Nothing local changed. The page re-renders the state the analyst already had, so a
    refusal never leaves the screen showing something the ledger does not contain.
    """
    status = 409 if code in ("stale_state", "no_such_pending") else 403
    if code == "ledger_unavailable":
        status = 503
    return render_template(
        "refused.html", view=view, f=f, code=code, detail=detail,
        title=REFUSAL_TITLES.get(code, "That action was refused"),
        state=view.state_of(str(f.get("finding_id"))) if f else None), status


def _int(raw: str | None) -> int | None:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def _request_id() -> str:
    """One id per form render, carried through a retry (plan §7.3, idempotency).

    A double-click or a retry after a timeout submits the SAME id, and ledgerd returns the
    original seq instead of recording twice.
    """
    submitted = (request.form.get("request_id") or "").strip()
    if E.REQUEST_ID_RE.fullmatch(submitted):
        return submitted
    return E.new_request_id()
