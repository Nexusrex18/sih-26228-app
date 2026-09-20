"""The Flask application factory (plan §6, D-E1).

Server-rendered, no npm, no CDN, no web fonts. Everything the browser loads comes from
`static/`, which is what makes the empty-network-namespace test (S2) pass and what makes the
air-gap claim something an auditor can check rather than take on trust.
"""
from __future__ import annotations

import logging
from typing import Any

from flask import Flask, g, render_template, request, session

from . import security
from .accounts import AccountStore
from .config import WebConfig, load
from .reports.index import ReportIndex
from .services import VerifyCache
from .workflow.ledgerd_client import LedgerdClient

log = logging.getLogger("cva.web")


def create_app(config: WebConfig | None = None, **overrides: Any) -> Flask:
    cfg = config or load()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["CVA"] = cfg

    accounts = AccountStore(cfg.accounts_db)
    app.secret_key = accounts.secret_key()
    app.config.update(security.session_cookie_config(
        tls=bool(cfg.tls_certfile), idle_timeout_s=cfg.session_idle_timeout_s))
    # S12: a body larger than this is rejected by Werkzeug before a view ever sees it.
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_request_bytes
    app.config["JSON_SORT_KEYS"] = False
    app.config.update(overrides)

    index = ReportIndex(cfg.index_db, cfg.reports_dir)
    index.refresh()

    app.extensions["cva_accounts"] = accounts
    app.extensions["cva_index"] = index
    app.extensions["cva_ledgerd"] = LedgerdClient(cfg.ledgerd_socket)
    app.extensions["cva_verify"] = VerifyCache()
    app.extensions["cva_rate"] = security.RateLimiter(cfg.rate_limit_per_minute)

    _register_hooks(app)
    _register_blueprints(app)
    _register_errors(app)
    _register_filters(app)
    return app


def _register_blueprints(app: Flask) -> None:
    from .views import admin, api, audit, evidence, main, remediation, spa, workflow
    from .views import auth as auth_views

    app.register_blueprint(auth_views.bp)
    app.register_blueprint(main.bp)
    app.register_blueprint(evidence.bp)
    app.register_blueprint(workflow.bp)
    app.register_blueprint(audit.bp)
    app.register_blueprint(remediation.bp)
    app.register_blueprint(admin.bp)
    # The Next.js dashboard and the JSON API it reads. Both go through the same
    # `services.load_scan` as the Jinja views, so the two front ends cannot drift.
    app.register_blueprint(api.bp)
    app.register_blueprint(spa.bp)


def _register_hooks(app: Flask) -> None:
    from .auth import SESSION_CSRF, current_account

    @app.before_request
    def _csrf_and_rate() -> Any:
        g.cva_account = None
        if SESSION_CSRF not in session:
            session[SESSION_CSRF] = security.new_csrf_token()

        identity = str(session.get("actor_id") or request.remote_addr or "anonymous")
        limiter = app.extensions["cva_rate"]
        bucket = "write" if request.method in ("POST", "PUT", "DELETE") else "read"
        if not limiter.check(identity, bucket):
            return render_template("error.html", code=429, title="Too many requests",
                                   detail="Slow down and try again in a minute."), 429

        if request.method in ("POST", "PUT", "DELETE"):
            # S7: token AND origin. Either alone is a known bypass.
            if not security.csrf_ok(session.get(SESSION_CSRF),
                                    request.form.get("csrf_token")
                                    or request.headers.get("X-CSRF-Token")):
                log.warning("CSRF token rejected path=%s ip=%s", request.path,
                            request.remote_addr)
                return render_template("error.html", code=403, title="Request refused",
                                       detail="This form did not carry a valid CSRF token. "
                                              "Reload the page and try again."), 403
            if not security.origin_ok(request.headers.get("Origin"),
                                      request.headers.get("Referer"),
                                      request.headers.get("Host")):
                log.warning("CSRF origin rejected path=%s origin=%s", request.path,
                            request.headers.get("Origin"))
                return render_template("error.html", code=403, title="Request refused",
                                       detail="This request did not come from this "
                                              "dashboard."), 403
        return None

    @app.after_request
    def _headers(response):  # type: ignore[no-untyped-def]
        security.apply_headers(response.headers,
                               authenticated=current_account() is not None,
                               spa=request.path.startswith("/app"))
        return response

    @app.context_processor
    def _globals() -> dict[str, Any]:
        from . import services
        from .auth import current_account as ca
        from .auth import may
        account = ca()
        return {
            "account": account,
            "may": may,
            "csrf_token": session.get(SESSION_CSRF, ""),
            "cfg": app.config["CVA"],
            "health": services.ledger_health() if account else None,
        }


def _register_errors(app: Flask) -> None:
    from .reports.loader import ReportUnreadable

    @app.errorhandler(ReportUnreadable)
    def _unreadable(e: ReportUnreadable):  # type: ignore[no-untyped-def]
        return render_template("error.html", code=422,
                               title=f"Scan {e.scan_id} cannot be read",
                               detail=e.reason), 422

    @app.errorhandler(404)
    def _404(_e: Any):  # type: ignore[no-untyped-def]
        return render_template("error.html", code=404, title="No such page",
                               detail="That address does not exist here."), 404

    @app.errorhandler(413)
    def _413(_e: Any):  # type: ignore[no-untyped-def]
        return render_template("error.html", code=413, title="Request too large",
                               detail="Requests are capped at "
                                      f"{app.config['MAX_CONTENT_LENGTH']} bytes. "
                                      "A justification is text, not an attachment."), 413

    @app.errorhandler(500)
    def _500(e: Any):  # type: ignore[no-untyped-def]
        log.exception("unhandled error")
        return render_template("error.html", code=500, title="Something failed here",
                               detail="The error is in this dashboard's log. Nothing was "
                                      "recorded in the ledger."), 500


def _register_filters(app: Flask) -> None:
    from .workflow.events import decode_text

    @app.template_filter("wire")
    def _wire(value: str) -> str:
        """Decode a percent-encoded ledger string for DISPLAY.

        Safe because Jinja autoescapes the result: a contributor named
        `<img src=x onerror=...>` round-trips through the ledger and renders as text (S1).
        """
        return decode_text(str(value or ""))

    @app.template_filter("short")
    def _short(value: str, n: int = 12) -> str:
        value = str(value or "")
        return value if len(value) <= n else value[:n] + "…"

    @app.template_filter("pct")
    def _pct(value: float, digits: int = 1) -> str:
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "—"


def run(cfg: WebConfig | None = None) -> None:
    """Serve with waitress. Never Flask's development server — Python's own docs say the
    stdlib equivalent 'is not recommended for production'."""
    from waitress import serve

    cfg = cfg or load()
    app = create_app(cfg)
    log.info("cva-web on http://%s:%d (reports=%s, ledgerd=%s)",
             cfg.bind_host, cfg.bind_port, cfg.reports_dir, cfg.ledgerd_socket)
    serve(app, host=cfg.bind_host, port=cfg.bind_port, threads=8,
          ident="cva-web", clear_untrusted_proxy_headers=True)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="cva-web", description="The analyst dashboard (Module E)")
    ap.add_argument("--config", default=None, help="YAML or JSON; unknown keys are an error")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run(load(a.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
