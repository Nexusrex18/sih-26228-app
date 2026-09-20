"""Login, session lifetime and role gates (plan §7.3, S6, S8).

Login and lockout events go to the application log, **not** the ledger, in v1. That is a
declared limitation (§13), written down here so the next reader does not assume otherwise.
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from flask import current_app, g, redirect, render_template, request, session, url_for

from .accounts import Account, AccountError, AccountStore
from .workflow.fold import ROLE_ACTIONS

log = logging.getLogger("cva.web.auth")

SESSION_ACTOR = "actor_id"
SESSION_ROLE = "role"
SESSION_EPOCH = "epoch"
SESSION_SEEN = "seen_at"
SESSION_CSRF = "csrf"


def login(account: Account) -> None:
    session.clear()
    session[SESSION_ACTOR] = account.actor_id
    session[SESSION_ROLE] = account.role
    session[SESSION_EPOCH] = account.session_epoch
    session[SESSION_SEEN] = time.time()
    session.permanent = True
    log.info("login actor=%s role=%s ip=%s", account.actor_id, account.role,
             request.remote_addr)


def logout() -> None:
    actor = session.get(SESSION_ACTOR)
    if actor:
        store: AccountStore = current_app.extensions["cva_accounts"]
        store.revoke_sessions(actor)
        log.info("logout actor=%s", actor)
    session.clear()


def current_account() -> Account | None:
    """The live account, re-read every request.

    Re-reading is the point: a cookie is a claim about who you were when it was issued. If
    the account has since been demoted, disabled or logged out elsewhere, `session_epoch`
    has moved and the session is dead — which is what S6's "server-side revocation on
    logout or role change" means in practice.
    """
    cached = g.get("cva_account")
    if cached is not None:
        return cached
    actor = session.get(SESSION_ACTOR)
    if not actor:
        return None
    store: AccountStore = current_app.extensions["cva_accounts"]
    account = store.get(str(actor))
    if account is None or account.disabled:
        session.clear()
        return None
    if account.session_epoch != session.get(SESSION_EPOCH):
        session.clear()
        return None
    idle = current_app.config["CVA"].session_idle_timeout_s
    seen = float(session.get(SESSION_SEEN, 0.0))
    if time.time() - seen > idle:
        session.clear()
        return None
    session[SESSION_SEEN] = time.time()
    if account.role != session.get(SESSION_ROLE):
        session[SESSION_ROLE] = account.role
    g.cva_account = account
    return account


def authenticate(store: AccountStore, actor_id: str, password: str, ip: str) -> Account:
    cfg = current_app.config["CVA"]
    try:
        return store.authenticate(actor_id, password, ip,
                                  threshold=cfg.lockout_threshold,
                                  base_s=cfg.lockout_base_s, max_s=cfg.lockout_max_s)
    except AccountError:
        # S8: every failure and every lockout is logged. The log is the only record of a
        # brute-force attempt, since login events are not in the ledger in v1.
        remaining = store.locked_for(actor_id, ip)
        log.warning("login failed actor=%s ip=%s locked_for=%.0fs", actor_id, ip, remaining)
        raise


def require_login(view: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(view)
    def wrapper(*a: Any, **kw: Any) -> Any:
        if current_account() is None:
            return redirect(url_for("auth.login_view", next=request.path))
        return view(*a, **kw)
    return wrapper


def require_role(*roles: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(view: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(view)
        def wrapper(*a: Any, **kw: Any) -> Any:
            account = current_account()
            if account is None:
                return redirect(url_for("auth.login_view", next=request.path))
            if account.role not in roles:
                return render_template("error.html", code=403,
                                       title="Not permitted for your role",
                                       detail=f"This action needs one of {list(roles)}; "
                                              f"your account holds '{account.role}'."), 403
            return view(*a, **kw)
        return wrapper
    return decorator


def may(action: str, account: Account | None = None) -> bool:
    """Whether this account may take a workflow action. The UI hides what it cannot do —
    and the daemon refuses it anyway, which is the check that counts."""
    account = account or current_account()
    if account is None:
        return False
    return action in ROLE_ACTIONS.get(account.role, frozenset())


__all__ = ["SESSION_CSRF", "authenticate", "current_account", "login", "logout", "may",
           "require_login", "require_role"]
