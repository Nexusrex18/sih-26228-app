"""The JSON API the Next.js dashboard reads (plan §5.1, `cva/web/views/api.py`).

Same app, same fixtures and the same real ledgerd as the template tests. The point of these
is that the two front ends cannot drift: both go through `services.load_scan`, so a test
here that disagrees with `test_views_render.py` is a bug in one of them.
"""
from __future__ import annotations

import json

from .conftest import sign_in


def _json(resp):
    assert resp.status_code == 200, resp.data[:400]
    return json.loads(resp.data)


def _csrf_of(client) -> str:
    """The token the SPA reads from `/api/session` and echoes in `X-CSRF-Token`.

    Signing in rotates the session, so the token is read *after* login and through the same
    route the dashboard uses rather than out of the cookie.
    """
    return _json(client.get("/api/session"))["csrf_token"]


def test_session_is_anonymous_until_login_and_always_carries_a_csrf_token(client):
    body = _json(client.get("/api/session"))
    assert body["authenticated"] is False
    # The token exists before login: the sign-in form is itself a state-changing request.
    assert body["csrf_token"]

    sign_in(client)
    body = _json(client.get("/api/session"))
    assert body["authenticated"] is True
    assert body["actor_id"] == "a.sharma"
    assert body["role"] == "analyst"
    assert "override" in body["can"]
    assert body["policy"]["min_justification_words"] >= 1


def test_every_read_route_refuses_an_anonymous_caller(client, scan_id):
    for url in ("/api/health", "/api/scans", f"/api/scans/{scan_id}",
                f"/api/scans/{scan_id}/findings", f"/api/scans/{scan_id}/coverage",
                "/api/audit"):
        resp = client.get(url)
        # `require_login` redirects to the sign-in form; what matters is that no report
        # content is served.
        assert resp.status_code in (302, 401, 403), url
        assert b"finding_id" not in resp.data


def test_scan_detail_carries_the_fold_and_the_tool_counts_separately(signed_in, scan_id):
    body = _json(signed_in.get(f"/api/scans/{scan_id}"))
    assert body["scan_id"] == scan_id
    # D-E5: what the tool said stays beside what humans did. Collapsing them into one
    # number is exactly the thing this dashboard may not do.
    assert set(body["tool_counts"]) <= {"accept", "review", "quarantine"}
    assert "effective" in body
    assert body["coverage"]["total_attack_classes"] >= len(body["coverage"]["assessed"])


def test_findings_module_filter_selects_the_prov_family(signed_in, scan_id):
    """The provenance page asks the server for `module=prov` rather than fetching every
    finding and filtering in the browser — §9's 10k-finding budget is met by never
    shipping the other 9,900."""
    body = _json(signed_in.get(f"/api/scans/{scan_id}/findings?module=prov&per_page=200"))
    assert body["findings"], "the rich fixture carries a prov.* quarantine"
    assert all(f["detector_id"].startswith("prov.") for f in body["findings"])
    assert body["unfiltered_total"] >= body["total"]


def test_coverage_compare_reports_which_statement_is_smaller(signed_in, scan_id):
    scans = _json(signed_in.get("/api/scans"))["scans"]
    other = next(s["scan_id"] for s in scans if s["scan_id"] != scan_id and s["readable"])

    body = _json(signed_in.get(f"/api/scans/{scan_id}/coverage?compare={other}"))
    assert body["other"]["scan_id"] == other
    diff = body["diff"]
    assert set(diff) >= {"only_left", "only_right", "both", "shrank", "lost_reasons"}
    # `assessed_fraction` is the report's own number. The dashboard draws it and never
    # recomputes a coverage row of its own (D-E2).
    assert 0.0 <= body["coverage"]["assessed_fraction"] <= 1.0


def test_audit_distinguishes_an_unreadable_ledger_from_an_empty_one(signed_in):
    body = _json(signed_in.get("/api/audit"))
    assert body["readable"] is True
    assert isinstance(body["rows"], list)
    assert body["decisions"] >= 0


def test_audit_is_unreadable_not_empty_when_nothing_is_listening(no_ledger_config):
    from cva.web.app import create_app

    app = create_app(no_ledger_config, TESTING=True)
    app.extensions["cva_accounts"].create("a.sharma", "correct-horse-battery", "analyst")
    client = app.test_client()
    sign_in(client)
    body = _json(client.get("/api/audit"))
    # An audit trail that could not be opened is not an audit trail with nothing in it.
    assert body["readable"] is False
    assert body["rows"] == []


def test_export_refuses_an_analyst_and_says_nothing_changed(signed_in):
    resp = signed_in.post("/api/audit/export", json={},
                          headers={"Origin": "http://localhost", "Host": "localhost",
                                   "X-CSRF-Token": _csrf_of(signed_in)})
    assert resp.status_code == 403, resp.data[:400]
    body = json.loads(resp.data)
    assert body["error"] == "role_not_permitted"
    assert body["nothing_changed"] is True


def test_export_writes_a_file_an_approver_can_verify_elsewhere(client, config):
    sign_in(client, actor="b.rao")
    resp = client.post("/api/audit/export", json={},
                       headers={"Origin": "http://localhost", "Host": "localhost",
                                "X-CSRF-Token": _csrf_of(client)})
    assert resp.status_code == 200, resp.data[:400]
    body = json.loads(resp.data)
    out = config.ledger_path.with_suffix(".export.jsonl")
    assert body["path"] == str(out)
    assert out.is_file() and out.stat().st_size > 0


def test_export_without_a_csrf_token_is_refused(client):
    sign_in(client, actor="b.rao")
    resp = client.post("/api/audit/export", json={},
                       headers={"Origin": "http://localhost", "Host": "localhost"})
    assert resp.status_code == 403
