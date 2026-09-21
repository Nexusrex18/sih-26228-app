"""The S2/S6/S7 header and CSRF suite (plan §7.7, Appendix B)."""
from __future__ import annotations

import pytest

from cva.web.security import CSP, SPA_CSP, origin_ok

from tests.web.conftest import sign_in


def test_every_security_header_is_present(signed_in):
    h = signed_in.get("/api/session").headers
    assert h["Content-Security-Policy"] == CSP
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["Referrer-Policy"] == "same-origin"
    assert h["X-Frame-Options"] == "DENY"
    assert "no-store" in h["Cache-Control"]


def test_the_csp_admits_no_external_origin(signed_in):
    """The property the air-gap claim rests on: nothing may be fetched from anywhere else."""
    csp = signed_in.get("/api/session").headers["Content-Security-Policy"]
    for directive in csp.split(";"):
        for token in directive.split()[1:]:
            assert token in ("'self'", "'none'", "data:", "blob:"), \
                f"{token!r} in {directive.strip()!r} reaches outside this origin"
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp


def test_the_spa_csp_also_admits_no_external_origin():
    """`/app/` relaxes inline script and style, and NOTHING else. An injection there still
    cannot exfiltrate, because there is nowhere it is allowed to send anything."""
    for directive in SPA_CSP.split(";"):
        for token in directive.split()[1:]:
            assert token in ("'self'", "'none'", "'unsafe-inline'", "data:", "blob:"), \
                f"{token!r} in {directive.strip()!r} reaches outside this origin"


def test_the_strict_csp_never_allows_inline_script(client):
    """The sign-in page is the one server-rendered page left, and it gets the strict CSP."""
    csp = client.get("/login").headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in csp
    assert "'unsafe-eval'" not in csp


# --- S7: Origin / Referer ----------------------------------------------------------------

def test_referrer_policy_leaves_referer_present_for_our_own_requests():
    """A regression test for a real defect.

    `Referrer-Policy: no-referrer` stripped `Referer`, and Chrome omits `Origin` on a
    same-origin form POST — so both signals were absent and the CSRF origin check refused
    every legitimate login with "this request did not come from this dashboard". A CSRF
    defence that rejects the only case it is supposed to admit is worse than none, because
    it looks like it is working.
    """
    from cva.web.security import SECURITY_HEADERS
    assert SECURITY_HEADERS["Referrer-Policy"] == "same-origin"


def test_origin_ok_accepts_a_same_origin_referer_with_no_origin_header():
    assert origin_ok(None, "http://localhost:8713/login?next=/app/", "localhost:8713")


def test_origin_ok_accepts_a_same_origin_origin_with_no_referer():
    assert origin_ok("http://localhost:8713", None, "localhost:8713")


def test_origin_ok_refuses_a_cross_origin_request():
    assert not origin_ok("http://evil.example", None, "localhost:8713")
    assert not origin_ok(None, "http://evil.example/page", "localhost:8713")


def test_origin_ok_refuses_when_both_signals_are_absent():
    """Fail closed. A same-origin POST from a real browser carries at least one of them."""
    assert not origin_ok(None, None, "localhost:8713")


def test_origin_ok_is_not_fooled_by_a_prefix():
    assert not origin_ok("http://localhost:8713.evil.example", None, "localhost:8713")


def test_a_login_post_succeeds_end_to_end(client):
    """The case the defect broke: sign in through the real form path."""
    resp = sign_in(client)
    assert resp.status_code in (302, 303)
    assert client.get("/api/session").get_json()["authenticated"] is True


def test_no_page_overrides_the_referrer_policy_back_to_no_referrer(client):
    """Regression, found in Firefox: `base.html` carried `<meta name="referrer"
    content="no-referrer">`. A meta tag OVERRIDES the header, and under `no-referrer`
    Firefox sends `Origin: null` on a same-origin form POST — so S7's origin check refused
    every sign-in. The header was fixed; the meta tag was not, and the test above could not
    see it because it sets `Origin` by hand. This checks what the BROWSER is told."""
    import re

    for url in ("/login", "/login?next=/app/"):
        html = client.get(url).get_data(as_text=True)
        metas = re.findall(r'<meta\s+name="referrer"\s+content="([^"]+)"', html, re.I)
        assert all(m == "same-origin" for m in metas), f"{url}: meta referrer {metas}"


# --- S7: the token ------------------------------------------------------------------------
# The write surface is /api, whose client sends the token in `X-CSRF-Token`. The same
# before-request check covers it and the sign-in form alike.

def _act_url(app, scan_id) -> str:
    from cva.web.reports.loader import load
    fid = load(app.config["CVA"].reports_dir, scan_id).findings[0]["finding_id"]
    return f"/api/scans/{scan_id}/findings/{fid}/act"


def _token(client) -> str:
    return client.get("/api/session").get_json()["csrf_token"]


def test_a_post_without_a_csrf_token_is_refused(signed_in, scan_id, app):
    resp = signed_in.post(_act_url(app, scan_id), json={"action": "acknowledge"},
                          headers={"Origin": "http://localhost", "Host": "localhost"})
    assert resp.status_code == 403
    assert b"CSRF" in resp.data


def test_a_post_with_a_wrong_csrf_token_is_refused(signed_in, scan_id, app):
    resp = signed_in.post(_act_url(app, scan_id), json={"action": "acknowledge"},
                          headers={"Origin": "http://localhost", "Host": "localhost",
                                   "X-CSRF-Token": "not-the-token"})
    assert resp.status_code == 403


def test_a_post_from_another_origin_is_refused(signed_in, scan_id, app):
    resp = signed_in.post(_act_url(app, scan_id), json={"action": "acknowledge"},
                          headers={"Origin": "http://evil.example", "Host": "localhost",
                                   "X-CSRF-Token": _token(signed_in)})
    assert resp.status_code == 403
    assert b"did not come from this dashboard" in resp.data


# --- S6: session cookie -------------------------------------------------------------------

def test_the_session_cookie_carries_the_right_flags(client):
    resp = sign_in(client)
    cookie = resp.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Path=/" in cookie


def test_a_role_change_ends_the_users_live_sessions(app, client):
    """S6: a demoted approver must not keep approving on a cookie issued a minute ago."""
    sign_in(client, "b.rao")
    assert client.get("/api/scans").status_code == 200
    app.extensions["cva_accounts"].set_role("b.rao", "viewer")
    resp = client.get("/api/scans")
    assert resp.status_code in (302, 303)
    assert "/login" in resp.headers["Location"]


def test_a_disabled_account_is_logged_out_at_once(app, client):
    sign_in(client, "v.iyer")
    app.extensions["cva_accounts"].set_disabled("v.iyer", True)
    resp = client.get("/api/scans")
    assert resp.status_code in (302, 303)


# --- S12 ------------------------------------------------------------------------------------

def test_an_oversize_body_is_refused(signed_in, scan_id, app):
    resp = signed_in.post(_act_url(app, scan_id),
                          json={"action": "override", "justification": "x" * 200_000},
                          headers={"Origin": "http://localhost", "Host": "localhost",
                                   "X-CSRF-Token": _token(signed_in)})
    assert resp.status_code == 413


# --- S5 --------------------------------------------------------------------------------------

def test_scrypt_runs_at_owasps_parameters_without_raising():
    """The `maxmem` failure mode, asserted rather than assumed.

    N = 2^17, r = 8 needs 128 * N * r = 128 MiB, and OpenSSL defaults to a 32 MiB cap — so
    the call raises `ValueError: memory limit exceeded` unless `maxmem` is raised. A silent
    fallback to weaker parameters would be invisible.
    """
    import os

    from cva.web.accounts import SCRYPT_MAXMEM, SCRYPT_N, SCRYPT_R, derive
    assert SCRYPT_N == 2 ** 17
    assert SCRYPT_R == 8
    assert SCRYPT_MAXMEM >= 128 * SCRYPT_N * SCRYPT_R
    salt = os.urandom(16)
    first = derive("a-test-password", salt)
    assert len(first) == 64
    assert first == derive("a-test-password", salt)
    assert first != derive("a-different-password", salt)


def test_an_actor_id_the_ledger_cannot_hold_is_refused_at_account_creation(app):
    """Enforced at creation, not at the first analyst event — which would refuse an
    override at the worst possible moment with no obvious cause."""
    from cva.web.accounts import AccountError
    store = app.extensions["cva_accounts"]
    for bad in ("a sharma", "a@sharma", "sharma!", "x" * 65, ""):
        with pytest.raises(AccountError):
            store.create(bad, "correct-horse-battery", "analyst")


# --- S8 ---------------------------------------------------------------------------------------

def test_repeated_failures_lock_the_account(app, client):
    from cva.web.security import new_csrf_token
    store = app.extensions["cva_accounts"]
    for _ in range(6):
        with client.session_transaction() as s:
            s["csrf"] = s.get("csrf") or new_csrf_token()
        with client.session_transaction() as s:
            token = s["csrf"]
        client.post("/login", data={"actor_id": "a.sharma", "password": "wrong",
                                    "csrf_token": token},
                    headers={"Origin": "http://localhost", "Host": "localhost"})
    assert store.locked_for("a.sharma", "") > 0
