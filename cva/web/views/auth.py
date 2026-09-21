"""Login and logout."""
from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..accounts import AccountError, AccountStore
from ..auth import authenticate, current_account, login, logout
from ..spa_paths import safe_next as _safe_next

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
def login_view() -> Any:
    nxt = _safe_next(request.values.get("next"))
    if current_account() is not None:
        return redirect(nxt)
    if request.method == "GET":
        return render_template("login.html", next=nxt, error=None, actor="")

    store: AccountStore = current_app.extensions["cva_accounts"]
    actor = (request.form.get("actor_id") or "").strip()
    password = request.form.get("password") or ""
    try:
        account = authenticate(store, actor, password, request.remote_addr or "")
    except AccountError as e:
        remaining = store.locked_for(actor, request.remote_addr or "")
        detail = (f"Too many failed attempts. Try again in {int(remaining)} seconds."
                  if remaining > 0 else str(e))
        # The account name survives a failure (autoescaped); the password never does.
        return render_template("login.html", next=nxt, error=detail, actor=actor), 401
    login(account)
    return redirect(nxt)


@bp.post("/logout")
def logout_view() -> Any:
    logout()
    return redirect(url_for("auth.login_view"))
