"""The complete analyst flow, end to end, against the real processes.

Sign in the way a browser does, open every page, take a decision that needs two people,
read it back in the audit trail, verify the ledger, export it, verify the export with the
CLI on its own, then run a real scan while the dashboard is up and see it arrive sealed.

Every step goes over HTTP to `cva-web`, which talks to `cva-ledgerd` over its socket, which
holds the key. Nothing is called in-process.
"""
from __future__ import annotations

import re
import subprocess
import sys
import uuid

import pytest

from .browser import Browser
from .conftest import PASSWORD, ROOT, _env

pytestmark = pytest.mark.e2e

RICH = "s-2026-09-19-0001"
SPA_PAGES = ("/app/", "/app/scan/", "/app/contributors/", "/app/findings/",
             "/app/provenance/", "/app/coverage/", "/app/audit/", "/app/verification/")


def _rid() -> str:
    return uuid.uuid4().hex


# --- 1. signing in, the way each browser actually does it -------------------------------

def test_sign_in_with_the_headers_a_browser_derives_from_the_page(stack, browser_for):
    """The regression a human found in Firefox. The login page's EFFECTIVE referrer policy —
    meta tag included — decides which Origin the browser sends. Derive it; don't set it."""
    b = Browser(stack.base)
    page = b.visit("/login?next=/app/")
    assert b.page_policy == "same-origin", (
        f"the login page puts the browser under '{b.page_policy}'. Under 'no-referrer' "
        "Firefox sends Origin: null and the CSRF origin check refuses every sign-in.")
    r = b.sign_in("a.sharma", PASSWORD)
    assert r.status in (302, 303), r.text[:300]
    assert b.get("/api/session").json()["actor_id"] == "a.sharma"


def test_a_page_under_no_referrer_would_be_refused(stack):
    """Proves the check above is load-bearing: the headers Firefox sends under `no-referrer`
    (Origin: null, no Referer) are refused, and nothing is signed in."""
    b = Browser(stack.base)
    b.visit("/login?next=/app/")
    b.page_policy = "no-referrer"
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', b.visit("/login").text)
    b.page_policy = "no-referrer"
    r = b.submit_form("/login", {"actor_id": "a.sharma", "password": PASSWORD,
                                 "csrf_token": m.group(1), "next": "/app/"})
    assert r.status == 403
    assert b.get("/api/session").json()["authenticated"] is False


def _form_token(b: Browser, path: str) -> tuple[str, str]:
    page = b.visit(path)
    token = re.search(r'name="csrf_token"\s+value="([^"]+)"', page.text).group(1)
    nxt = re.search(r'name="next"\s+value="([^"]*)"', page.text).group(1)
    return token, nxt


def test_signing_in_with_no_next_lands_on_the_dashboard(stack):
    """Regression: the login page defaulted `next` to the OLD server-rendered dashboard at
    `/`, so a person who opened /login directly signed in and landed on the wrong app."""
    b = Browser(stack.base)
    token, nxt = _form_token(b, "/login")
    assert nxt == "/app/", nxt
    r = b.submit_form("/login", {"actor_id": "a.sharma", "password": PASSWORD,
                                 "csrf_token": token, "next": nxt})
    assert r.status in (302, 303) and r.headers["location"].endswith("/app/"), r.headers


@pytest.mark.parametrize("old,new", [
    ("/", "/app/"),
    (f"/scans/{RICH}", f"/app/scan/?id={RICH}"),
    (f"/scans/{RICH}/findings", f"/app/findings/?id={RICH}"),
    (f"/scans/{RICH}/coverage", f"/app/coverage/?id={RICH}"),
    ("/audit/", "/app/audit/"),
    ("/audit/verification", "/app/verification/"),
    ("//evil.example/", "/app/"),
    ("https://evil.example/", "/app/"),
])
def test_an_old_next_is_translated_and_an_offsite_one_refused(stack, old, new):
    b = Browser(stack.base)
    token, nxt = _form_token(b, f"/login?next={old}")
    assert nxt == new, (old, nxt)


def test_signing_out_ends_the_session_and_returns_to_sign_in(stack, browser_for):
    """The SPA's Sign out button is a form POST to /logout with the session's CSRF token."""
    b = browser_for("a.sharma")
    csrf = b.open_app("/app/")
    r = b.submit_form("/logout", {"csrf_token": csrf})
    assert r.status in (302, 303) and r.headers["location"].endswith("/login"), r.headers
    assert b.get("/api/session").json()["authenticated"] is False


def test_a_signed_in_visit_to_login_goes_to_the_dashboard(stack, browser_for):
    b = browser_for("a.sharma")
    r = b.visit("/login")
    assert r.status in (302, 303) and r.headers["location"].endswith("/app/")


def test_a_wrong_password_is_refused_and_says_so(stack):
    b = Browser(stack.base)
    r = b.sign_in("v.iyer", "not-the-password-at-all")
    assert r.status == 401 and "Not signed in" in r.text
    assert b.get("/api/session").json()["authenticated"] is False


# --- 2. every page loads, with everything it needs ---------------------------------------

def test_every_dashboard_page_and_its_scripts_are_served(stack, browser_for):
    b = browser_for("a.sharma")
    for path in SPA_PAGES:
        r = b.visit(path + f"?id={RICH}")
        assert r.status == 200, (path, r.status)
        assert "text/html" in r.headers["content-type"]
        # A static export: every script the page needs must come from this server, under
        # /app/, with no network. A 404 here is a blank page in the demo.
        scripts = re.findall(r'<script[^>]+src="([^"]+)"', r.text)
        assert scripts, f"{path} references no scripts"
        for src in scripts:
            assert src.startswith("/app/_next/"), f"{path}: script from elsewhere: {src}"
            js = b.get(src, accept="*/*")
            assert js.status == 200, (path, src, js.status)
        csp = r.headers.get("content-security-policy", "")
        assert "connect-src 'self'" in csp and "http" not in csp, csp


def test_the_spa_refuses_an_anonymous_visitor(stack):
    r = Browser(stack.base).visit("/app/")
    assert r.status in (302, 303)


def test_the_old_server_rendered_paths_redirect_into_the_dashboard(stack, browser_for):
    """The server-rendered dashboard is gone; its bookmarks land on the SPA page that
    replaced each one, against the real server."""
    b = browser_for("a.sharma")
    for old, new in (("/", "/app/"), (f"/scans/{RICH}", f"/app/scan/?id={RICH}"),
                     (f"/scans/{RICH}/findings", f"/app/findings/?id={RICH}"),
                     (f"/scans/{RICH}/contributors", f"/app/contributors/?id={RICH}"),
                     (f"/scans/{RICH}/provenance", f"/app/provenance/?id={RICH}"),
                     (f"/scans/{RICH}/coverage", f"/app/coverage/?id={RICH}"),
                     ("/audit/", "/app/audit/"),
                     ("/audit/verification", "/app/verification/"),
                     ("/admin/accounts", "/app/accounts/")):
        r = b.visit(old)
        assert r.status == 302 and r.headers["location"].endswith(new), (old, r.status,
                                                                        r.headers)


# --- 3. the data every page reads ----------------------------------------------------------

def test_the_ledger_is_verified_and_the_fixtures_are_sealed(stack, browser_for):
    b = browser_for("v.iyer")
    health = b.get("/api/health").json()
    assert health["reachable"] and health["writable"], health
    assert health["kind"] == "verified", health

    scans = {s["scan_id"]: s for s in b.get("/api/scans").json()["scans"]}
    assert RICH in scans
    assert scans[RICH]["seal"]["state"] == "sealed", scans[RICH]["seal"]


def test_every_read_route_answers_for_the_rich_scan(stack, browser_for):
    b = browser_for("v.iyer")
    detail = b.get(f"/api/scans/{RICH}").json()
    assert detail["n_findings"] > 0
    assert b.get(f"/api/scans/{RICH}/findings?module=prov").json()["findings"]
    cov = b.get(f"/api/scans/{RICH}/coverage?compare=s-2026-09-19-0004").json()
    assert cov["other"]["scan_id"] == "s-2026-09-19-0004" and "diff" in cov
    # The reviewed standing file reaches the page's data (N6)... only if the report was
    # generated with it; fixtures predate it, so this checks the shape, not the content.
    assert isinstance(cov["coverage"]["standing_limitations"], list)
    assert b.get(f"/api/scans/{RICH}/targets").status == 200
    audit = b.get("/api/audit").json()
    assert audit["readable"] is True
    assert any(r["kind"] == "scan_record" for r in audit["rows"])


# --- 4. the two-person decision ------------------------------------------------------------

def _a_review_finding(b: Browser) -> dict:
    page = b.get(f"/api/scans/{RICH}/findings?disposition=review&per_page=200").json()
    for f in page["findings"]:
        if not f["detector_id"].startswith("prov.") and not (f["state"] or {}).get("pending"):
            if (f["state"] or {}).get("effective", f["disposition"]) == "review":
                return f
    pytest.fail("the rich fixture has no unpending, non-prov review finding left")


def test_lowering_needs_a_second_person_and_lands_in_the_audit_trail(stack, browser_for):
    analyst, approver = browser_for("a.sharma"), browser_for("b.rao")
    a_csrf = analyst.open_app(f"/app/findings/?id={RICH}")
    f = _a_review_finding(analyst)
    url = f"/api/scans/{RICH}/findings/{f['finding_id']}/act"

    # The analyst lowers review -> accept. It is recorded, and it is PENDING.
    r = analyst.fetch_json(url, {
        "action": "override", "new_disposition": "accept", "reason_code": "quality_issue",
        "justification": "Confirmed duplicate cluster from a scanner re-export, batch note 14.",
        "request_id": _rid()}, a_csrf)
    assert r.status == 200, r.text[:400]
    lowered = r.json()
    pending_seq = lowered["seq"]
    assert lowered["finding"]["state"]["pending"]["seq"] == pending_seq
    assert lowered["finding"]["state"]["effective"] == "review", "a lowering is not immediate"

    # The same analyst may not approve their own change.
    r = analyst.fetch_json(url, {"action": "approve", "refs_seq": pending_seq,
                                 "justification": "Approving my own change should fail.",
                                 "request_id": _rid()}, a_csrf)
    assert r.status == 409, r.text[:300]
    assert r.json()["nothing_changed"] is True

    # A different person, with the approver role, may.
    b_csrf = approver.open_app(f"/app/findings/?id={RICH}")
    r = approver.fetch_json(url, {"action": "approve", "refs_seq": pending_seq,
                                  "justification": "Reviewed the batch note and the cluster.",
                                  "request_id": _rid()}, b_csrf)
    assert r.status == 200, r.text[:400]
    state = r.json()["finding"]["state"]
    assert state["effective"] == "accept" and state["pending"] is None
    approve_seq = r.json()["seq"]
    assert approve_seq > pending_seq

    # Both events are in the audit trail, in seq order, with who did what.
    rows = analyst.get(f"/api/audit?scan_id={RICH}").json()["rows"]
    seqs = [row["seq"] for row in rows]
    assert seqs == sorted(seqs), "the trail is ordered by seq, never by the clock"
    by_seq = {row["seq"]: row for row in rows}
    assert by_seq[pending_seq]["actor"] == "a.sharma"
    assert by_seq[approve_seq]["actor"] == "b.rao"
    assert by_seq[approve_seq]["refs_seq"] == pending_seq

    # And the ledger still verifies with both events in it.
    health = analyst.fetch_json("/api/audit/verify", {}, a_csrf).json()
    assert health["kind"] == "verified", health


def test_raising_takes_effect_at_once(stack, browser_for):
    analyst = browser_for("a.sharma")
    csrf = analyst.open_app(f"/app/findings/?id={RICH}")
    f = _a_review_finding(analyst)
    r = analyst.fetch_json(f"/api/scans/{RICH}/findings/{f['finding_id']}/act", {
        "action": "override", "new_disposition": "quarantine",
        "reason_code": "insufficient_evidence",
        "justification": "Holding this until the contributor answers the batch query.",
        "request_id": _rid()}, csrf)
    assert r.status == 200, r.text[:400]
    assert r.json()["finding"]["state"]["effective"] == "quarantine"


def test_refusals_change_nothing(stack, browser_for):
    analyst = browser_for("a.sharma")
    csrf = analyst.open_app(f"/app/findings/?id={RICH}")
    f = _a_review_finding(analyst)
    url = f"/api/scans/{RICH}/findings/{f['finding_id']}/act"
    before = analyst.get("/api/audit").json()["rows"]

    # Punctuation is not a justification.
    r = analyst.fetch_json(url, {"action": "override", "new_disposition": "accept",
                                 "reason_code": "quality_issue", "justification": "." * 40,
                                 "request_id": _rid()}, csrf)
    assert r.status == 400 and r.json()["nothing_changed"] is True, r.text[:300]

    # No CSRF token.
    r = analyst.fetch_json(url, {"action": "acknowledge", "justification": "x" * 40,
                                 "request_id": _rid()}, None)
    assert r.status == 403

    # A viewer may read and may not decide.
    viewer = browser_for("v.iyer")
    v_csrf = viewer.open_app("/app/")
    r = viewer.fetch_json(url, {"action": "override", "new_disposition": "quarantine",
                                "reason_code": "quality_issue",
                                "justification": "A viewer should not be able to do this.",
                                "request_id": _rid()}, v_csrf)
    assert r.status in (400, 403, 409) and r.json().get("nothing_changed") is True

    after = analyst.get("/api/audit").json()["rows"]
    assert len(after) == len(before), "a refused request wrote to the ledger"


def test_a_retried_request_is_recorded_once(stack, browser_for):
    analyst = browser_for("a.sharma")
    csrf = analyst.open_app(f"/app/findings/?id={RICH}")
    f = _a_review_finding(analyst)
    body = {"action": "acknowledge",
            "justification": "Seen, and assigned to the batch follow-up list.",
            "request_id": _rid()}
    url = f"/api/scans/{RICH}/findings/{f['finding_id']}/act"
    first = analyst.fetch_json(url, body, csrf)
    second = analyst.fetch_json(url, body, csrf)
    assert first.status == 200, first.text[:300]
    assert second.status == 200 and second.json()["deduped"] is True
    assert second.json()["seq"] == first.json()["seq"]


# --- 5. export, and verify it with nothing but the CLI -----------------------------------

def test_an_approver_exports_and_the_export_verifies_on_its_own(stack, browser_for):
    analyst = browser_for("a.sharma")
    a_csrf = analyst.open_app("/app/verification/")
    assert analyst.fetch_json("/api/audit/export", {}, a_csrf).status == 403

    approver = browser_for("b.rao")
    csrf = approver.open_app("/app/verification/")
    r = approver.fetch_json("/api/audit/export", {}, csrf)
    assert r.status == 200, r.text[:300]
    export = r.json()["path"]

    # A third party's check: the CLI, the export and the trust root. No dashboard.
    v = subprocess.run([sys.executable, "-m", "cva.provenance.seal.cli", "verify",
                        "--records", export, "--trust", str(stack.trust_root)],
                       capture_output=True, text=True, timeout=300, env=_env())
    assert v.returncode == 0, v.stdout + v.stderr


# --- 6. a real scan, while the dashboard is up --------------------------------------------

@pytest.mark.slow
def test_a_scan_run_now_appears_sealed_without_a_restart(stack, browser_for, tmp_path):
    """Demo steps 1-2: run a scan while the dashboard is up, and it is usable at once.

    The findings page is opened FIRST, straight from the scan id, before the list has been
    loaded. Only the list refreshed the index, so this page said the new scan had no
    findings until someone happened to load the list."""
    before = {p.name for p in stack.reports.iterdir()}
    fx = tmp_path / "fixtures"
    subprocess.run([sys.executable, "-m", "cva.fixtures", "--out", str(fx),
                    "--kind", "mixed_contributors"], check=True, env=_env(), timeout=600,
                   capture_output=True)
    dataset = fx / "mixed_contributors"
    r = subprocess.run([sys.executable, "-m", "cva.cli", "scan", "--dataset", str(dataset),
                        "--profile", "baseline", "--out", str(stack.reports),
                        "--audit-ledger-socket", str(stack.socket)],
                       capture_output=True, text=True, timeout=1800, env=_env(), cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "NOT sealed" not in r.stdout + r.stderr, r.stdout[-2000:] + r.stderr[-2000:]

    import json
    import time

    created = sorted({p.name for p in stack.reports.iterdir()} - before - {"evidence"})
    assert len(created) == 1, created
    new_id = created[0]
    on_disk = json.loads((stack.reports / new_id / "report.json").read_text())
    assert on_disk["findings"], "a scan with no findings would make the check below vacuous"

    b = browser_for("v.iyer")
    time.sleep(2.1)          # past the index's refresh throttle, as any real click would be
    page = b.get(f"/api/scans/{new_id}/findings?per_page=500").json()
    assert page["unfiltered_total"] == len(on_disk["findings"]), (
        f"the findings page shows {page['unfiltered_total']} findings for a scan whose "
        f"report has {len(on_disk['findings'])}")

    deadline = time.monotonic() + 15
    new = []
    while time.monotonic() < deadline and not new:
        rows = b.get("/api/scans").json()["scans"]
        new = [s for s in rows if not s["scan_id"].startswith("s-2026-09-19-")]
        if not new:
            time.sleep(1)
    assert new, "the scan finished but the dashboard does not list it"
    scan = new[0]
    assert scan["readable"], scan
    assert scan["seal"]["state"] == "sealed", scan["seal"]

    # And its pages have data.
    detail = b.get(f"/api/scans/{scan['scan_id']}").json()
    assert detail["contributor_risk"], "the mixed-contributors scan produced no contributor rows"
    groups = {r["group_value"] for r in detail["contributor_risk"]}
    assert "guilty" in groups
