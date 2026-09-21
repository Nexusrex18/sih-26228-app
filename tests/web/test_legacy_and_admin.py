"""The dashboard after the server-rendered one was removed.

Three things are pinned here:

  * every old server-rendered path redirects to the SPA page that now shows the same
    thing, and the old form-POST write routes are gone — /api is the only write surface;
  * the report guarantees the Jinja pages carried (gate N1) still hold in the data the SPA
    renders: a refused schema, an unreadable scan listed with its reason, pagination,
    filters, the budget exclusion's cost, provenance present or explicitly absent;
  * account management, ported from the Jinja admin page to /api/admin/accounts, is
    admin-only, refuses in the standard shape, and still ends a user's sessions on a role
    change.

What each page LOOKS like is in tests/e2e/test_spa_content.py, in a real browser.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.web.conftest import sign_in

RICH = "s-2026-09-19-0001"
ALL_UNAVAILABLE = "s-2026-09-19-0002"
CLEAN = "s-2026-09-19-0003"
BLACKBOX = "s-2026-09-19-0004"
HDRS = {"Origin": "http://localhost", "Host": "localhost"}


def _csrf(client) -> str:
    return client.get("/api/session").get_json()["csrf_token"]


def _post_json(client, url, body, *, csrf=True):
    headers = dict(HDRS)
    if csrf:
        headers["X-CSRF-Token"] = _csrf(client)
    return client.post(url, json=body, headers=headers)


# --- the old paths ------------------------------------------------------------------------

@pytest.mark.parametrize("old,new", [
    ("/", "/app/"),
    (f"/scans/{RICH}", f"/app/scan/?id={RICH}"),
    (f"/scans/{RICH}/findings", f"/app/findings/?id={RICH}"),
    (f"/scans/{RICH}/findings/0123456789abcdef", f"/app/findings/?id={RICH}"),
    (f"/scans/{RICH}/contributors", f"/app/contributors/?id={RICH}"),
    (f"/scans/{RICH}/provenance", f"/app/provenance/?id={RICH}"),
    (f"/scans/{RICH}/coverage", f"/app/coverage/?id={RICH}"),
    (f"/scans/{RICH}/reproduction", f"/app/scan/?id={RICH}"),
    (f"/scans/{RICH}/drift", f"/app/scan/?id={RICH}"),
    (f"/scans/{RICH}/remediation", f"/app/scan/?id={RICH}"),
    ("/audit/", "/app/audit/"),
    ("/audit/verification", "/app/verification/"),
    ("/admin/accounts", "/app/accounts/"),
    ("/health", "/app/verification/"),
])
def test_every_old_path_redirects_to_its_spa_page(client, old, new):
    resp = client.get(old)
    assert resp.status_code == 302, (old, resp.status_code)
    assert resp.headers["Location"].endswith(new), (old, resp.headers["Location"])


def test_the_dashboard_and_the_api_still_need_a_login(client):
    for url in ("/app/", "/api/scans", f"/api/scans/{RICH}", "/evidence/" + "0" * 64):
        resp = client.get(url)
        assert resp.status_code in (302, 303), url
        assert "/login" in resp.headers["Location"], url


def test_the_old_form_write_routes_are_gone(signed_in, app):
    """/api is the only write surface. A POST to a removed form route writes nothing."""
    from cva.web.reports.loader import load
    fid = load(app.config["CVA"].reports_dir, RICH).findings[0]["finding_id"]
    before = signed_in.get("/api/audit").get_json()["rows"]
    for url in (f"/scans/{RICH}/findings/{fid}/act", f"/scans/{RICH}/targets/act",
                "/audit/verify", "/audit/export", "/admin/accounts"):
        resp = signed_in.post(url, data={"csrf_token": _csrf(signed_in), "action": "acknowledge",
                                         "justification": "x " * 20}, headers=HDRS)
        assert resp.status_code in (404, 405), (url, resp.status_code)
    assert signed_in.get("/api/audit").get_json()["rows"] == before


# --- the report guarantees, in the data the SPA renders ------------------------------------

def test_an_unknown_schema_version_is_refused_not_served(app, signed_in):
    root = Path(app.config["CVA"].reports_dir)
    bad = root / "s-2027-01-01-0001"
    bad.mkdir()
    payload = json.loads((root / CLEAN / "report.json").read_text())
    payload["schema_version"] = "2.0.0"
    payload["scan_id"] = "s-2027-01-01-0001"
    (bad / "report.json").write_text(json.dumps(payload))
    resp = signed_in.get("/api/scans/s-2027-01-01-0001")
    assert resp.status_code == 422
    assert "2.0.0" in resp.get_data(as_text=True)


def test_an_unreadable_scan_is_listed_with_its_reason(app, signed_in):
    root = Path(app.config["CVA"].reports_dir)
    broken = root / "s-2027-02-02-0002"
    broken.mkdir()
    (broken / "report.json").write_text("{ not json")
    rows = {s["scan_id"]: s for s in signed_in.get("/api/scans").get_json()["scans"]}
    assert rows["s-2027-02-02-0002"]["readable"] is False
    assert rows["s-2027-02-02-0002"]["unreadable_reason"]


def test_pagination_holds_and_an_out_of_range_page_is_empty(signed_in):
    first = signed_in.get(f"/api/scans/{RICH}/findings?page=1&per_page=5").get_json()
    assert len(first["findings"]) <= 5 and first["pages"] >= 1
    far = signed_in.get(f"/api/scans/{RICH}/findings?page=9999").get_json()
    assert far["findings"] == []


def test_filters_narrow_and_nature_separates_quality_from_adversarial(signed_in):
    all_ = signed_in.get(f"/api/scans/{RICH}/findings").get_json()
    held = signed_in.get(f"/api/scans/{RICH}/findings?disposition=quarantine").get_json()
    assert 0 < held["total"] < all_["total"]
    quality = signed_in.get(f"/api/scans/{RICH}/findings?nature=quality").get_json()
    assert quality["findings"] and all(f["nature"] == "quality" for f in quality["findings"])


def test_capability_and_budget_exclusions_stay_distinct_and_budget_names_its_cost(signed_in):
    plan = signed_in.get(f"/api/scans/{RICH}").get_json()["plan"]
    reasons = {p.get("exclusion_reason") for p in plan}
    assert {"capability", "budget"} <= reasons
    assert any(p.get("estimated_cost") == "41 min" for p in plan)


def test_an_all_unavailable_scan_reports_no_check_that_ran(signed_in):
    plan = signed_in.get(f"/api/scans/{ALL_UNAVAILABLE}").get_json()["plan"]
    assert plan and all(p["state"] != "OK" for p in plan)


def test_provenance_is_present_when_clean_and_absent_is_said(signed_in):
    assert signed_in.get(f"/api/scans/{CLEAN}").get_json()["provenance_summary"] is not None
    assert signed_in.get(f"/api/scans/{BLACKBOX}").get_json()["provenance_summary"] is None


def test_the_blackbox_statement_is_smaller_and_names_what_it_lost(signed_in):
    rich = signed_in.get(f"/api/scans/{CLEAN}/coverage").get_json()["coverage"]
    narrow = signed_in.get(f"/api/scans/{BLACKBOX}/coverage").get_json()["coverage"]
    assert len(narrow["assessed"]) <= len(rich["assessed"])
    lost = {c for row in narrow["not_assessed"] for c in row["checks"]}
    assert "model.weight_digest" in lost


# --- account management, ported to the API --------------------------------------------------

@pytest.fixture
def admin(client):
    sign_in(client, actor="root.admin")
    return client


@pytest.mark.parametrize("actor", ["a.sharma", "b.rao", "v.iyer"])
def test_only_the_admin_may_manage_accounts(client, actor):
    sign_in(client, actor=actor)
    resp = client.get("/api/admin/accounts")
    assert resp.status_code == 403
    assert resp.get_json()["nothing_changed"] is True
    resp = _post_json(client, "/api/admin/accounts",
                      {"actor_id": "x.y", "role": "admin", "password": "password-long"})
    assert resp.status_code == 403


def test_the_admin_lists_creates_and_the_new_account_can_sign_in(admin, app):
    listing = admin.get("/api/admin/accounts").get_json()
    assert {a["actor_id"] for a in listing["accounts"]} >= {"a.sharma", "root.admin"}
    assert listing["min_password_chars"] == 8
    resp = _post_json(admin, "/api/admin/accounts",
                      {"actor_id": "Tan.00", "role": "analyst", "password": "12345678"})
    assert resp.status_code == 200, resp.get_json()
    assert any(a["actor_id"] == "Tan.00" for a in resp.get_json()["accounts"])
    other = app.test_client()
    sign_in(other, actor="Tan.00", password="12345678")
    assert other.get("/api/session").get_json()["actor_id"] == "Tan.00"


@pytest.mark.parametrize("body,why", [
    ({"actor_id": "Tan@00", "role": "analyst", "password": "12345678"}, "an @"),
    ({"actor_id": "short.pw", "role": "analyst", "password": "1234567"}, "7 chars"),
    ({"actor_id": "bad.role", "role": "root", "password": "12345678"}, "unknown role"),
])
def test_a_bad_account_is_refused_in_the_standard_shape(admin, body, why):
    resp = _post_json(admin, "/api/admin/accounts", body)
    assert resp.status_code == 400, why
    j = resp.get_json()
    assert j["ok"] is False and j["nothing_changed"] is True and j["detail"], why


def test_a_role_change_ends_that_users_live_sessions(admin, app):
    victim = app.test_client()
    sign_in(victim, actor="a.sharma")
    assert victim.get("/api/session").get_json()["authenticated"] is True
    resp = _post_json(admin, "/api/admin/accounts/a.sharma", {"role": "viewer"})
    assert resp.status_code == 200
    assert victim.get("/api/session").get_json()["authenticated"] is False


def test_disable_and_password_reset(admin, app):
    assert _post_json(admin, "/api/admin/accounts/v.iyer", {"disabled": True}).status_code == 200
    rows = {a["actor_id"]: a for a in admin.get("/api/admin/accounts").get_json()["accounts"]}
    assert rows["v.iyer"]["disabled"] is True
    assert _post_json(admin, "/api/admin/accounts/b.rao",
                      {"password": "another-password"}).status_code == 200
    other = app.test_client()
    sign_in(other, actor="b.rao", password="another-password")


def test_account_changes_need_the_csrf_header_and_a_real_account(admin):
    assert _post_json(admin, "/api/admin/accounts/a.sharma", {"role": "viewer"},
                      csrf=False).status_code == 403
    assert _post_json(admin, "/api/admin/accounts/nobody.here",
                      {"role": "viewer"}).status_code == 404
