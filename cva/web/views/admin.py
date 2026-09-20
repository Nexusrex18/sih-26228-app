"""Account management (plan D-E9).

`admin` manages accounts and holds no workflow rights. That is why this is a separate
blueprint behind a separate role: one person controlling both identity and decisions is
exactly what the separation exists to prevent.
"""
from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..accounts import ROLES, AccountError, AccountStore
from ..auth import require_role

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _store() -> AccountStore:
    return current_app.extensions["cva_accounts"]


@bp.get("/accounts")
@require_role("admin")
def accounts() -> Any:
    return render_template("accounts.html", accounts=_store().list_accounts(), roles=ROLES)


@bp.post("/accounts")
@require_role("admin")
def create() -> Any:
    try:
        account = _store().create((request.form.get("actor_id") or "").strip(),
                                  request.form.get("password") or "",
                                  (request.form.get("role") or "viewer").strip())
    except AccountError as e:
        flash(("failed", str(e)), "accounts")
        return redirect(url_for("admin.accounts"))
    flash(("ok", f"Created {account.actor_id} as {account.role}."), "accounts")
    return redirect(url_for("admin.accounts"))


@bp.post("/accounts/<actor_id>/role")
@require_role("admin")
def set_role(actor_id: str) -> Any:
    try:
        account = _store().set_role(actor_id, (request.form.get("role") or "").strip())
    except AccountError as e:
        flash(("failed", str(e)), "accounts")
        return redirect(url_for("admin.accounts"))
    flash(("ok", f"{account.actor_id} is now {account.role}. Their existing sessions were "
                 "ended, so the change takes effect on their next request."), "accounts")
    return redirect(url_for("admin.accounts"))


@bp.post("/accounts/<actor_id>/disabled")
@require_role("admin")
def set_disabled(actor_id: str) -> Any:
    disabled = (request.form.get("disabled") or "") == "1"
    _store().set_disabled(actor_id, disabled)
    flash(("ok", f"{actor_id} is {'disabled' if disabled else 'enabled'}. Their past "
                 "decisions stay in the ledger, which is the point of an append-only "
                 "record."), "accounts")
    return redirect(url_for("admin.accounts"))


@bp.post("/accounts/<actor_id>/password")
@require_role("admin")
def set_password(actor_id: str) -> Any:
    try:
        _store().set_password(actor_id, request.form.get("password") or "")
    except AccountError as e:
        flash(("failed", str(e)), "accounts")
        return redirect(url_for("admin.accounts"))
    flash(("ok", f"Password set for {actor_id}; their sessions were ended."), "accounts")
    return redirect(url_for("admin.accounts"))
