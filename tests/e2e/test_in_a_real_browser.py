"""The dashboard in a real, headless Chrome: does every page actually RENDER?

`test_full_flow.py` proves every route and API answers. It cannot see a React component
that throws on load — that serves a good HTML shell and a blank screen. This runs the
bundle in the Chrome installed on the machine and fails on:

  * any uncaught exception, console.error or error-level log entry, on any page;
  * a page that never reaches its loaded state (its data section never appears);
  * Next's "Application error" screen;
  * at 390px, anything wider than the screen (horizontal scroll hides the column that
    matters — the mobile pass's own rule);

for every page, against every fixture scan (each has a differently shaped report: rich,
all-UNAVAILABLE, clean, black-box), at desktop and phone width. Then one decision is taken
entirely through the UI — open the sheet, pick, type, submit — and checked in the ledger.

Screenshots of every page land in `.scratch/e2e-screens/` for a human to look at.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .cdp import Chrome, find_chrome
from .conftest import PASSWORD, ROOT

pytestmark = pytest.mark.e2e

SCANS = ("s-2026-09-19-0001", "s-2026-09-19-0002", "s-2026-09-19-0003",
         "s-2026-09-19-0004")
#: page -> text that appears only once the page's data has loaded.
SCAN_PAGES = {
    "scan": "Access assumptions",
    "contributors": "Source-level aggregation",
    "findings": "Keyboard:",
    "provenance": "What this does not establish",
    "coverage": "Standing limitations",
}
GLOBAL_PAGES = {
    "": "Triage queue",
    "audit": "What this trail does not record",
    "verification": "Independent verification",
}
SCREENS = ROOT / ".scratch" / "e2e-screens"


@pytest.fixture(scope="module")
def chrome_bin() -> str:
    found = find_chrome()
    if not found:
        pytest.skip("no Chrome/Chromium installed")
    return found


def _sign_in(c: Chrome, base: str, actor: str) -> None:
    """Through the real form, as a person would — the browser derives Origin itself."""
    c.goto(f"{base}/login?next=/app/")
    c.type_into("#actor_id", actor)
    c.type_into("#password", PASSWORD)
    c.click_text("Sign in", "button")
    c.wait_for("location.pathname.startsWith('/app')", what="redirect into /app/")
    c.wait_text("Triage queue")


def _check_page(c: Chrome, url: str, ready: str, shot: Path, *, phone: bool) -> list[str]:
    problems: list[str] = []
    c.clear_events()
    c.goto(url)
    try:
        c.wait_text(ready, timeout=25)
        c.wait_for("!document.body.innerText.includes('loading findings')",
                   what="findings to load", timeout=25)
    except AssertionError as e:
        problems.append(f"{url}: never reached its loaded state ({ready!r}). {e}")
    if "Application error" in c.text():
        problems.append(f"{url}: Next.js 'Application error' screen")
    problems += [f"{url}: {err}" for err in c.errors()]
    if phone:
        overflow = c.evaluate("""(() => {
            const w = window.innerWidth;
            if (document.documentElement.scrollWidth <= w + 1) return [];
            return [...document.querySelectorAll('body *')]
              .filter(e => e.getBoundingClientRect().right > w + 1 && e.offsetParent !== null)
              .slice(0, 6)
              .map(e => e.tagName.toLowerCase() + '.' + String(e.className).slice(0, 60)
                        + ' right=' + Math.round(e.getBoundingClientRect().right));
        })()""")
        if overflow:
            problems.append(f"{url}: wider than the {390}px screen: {overflow}")
    c.screenshot(shot)
    return problems


@pytest.mark.parametrize("viewport", ["desktop", "phone"])
def test_every_page_renders_without_an_error(stack, chrome_bin, viewport):
    phone = viewport == "phone"
    c = Chrome(chrome_bin)
    try:
        if phone:
            c.set_viewport(390, 844, mobile=True)
        _sign_in(c, stack.base, "a.sharma")
        problems: list[str] = []
        for page, ready in GLOBAL_PAGES.items():
            url = f"{stack.base}/app/{page + '/' if page else ''}"
            problems += _check_page(c, url, ready, SCREENS / viewport / f"{page or 'home'}.png",
                                    phone=phone)
        for scan in SCANS:
            for page, ready in SCAN_PAGES.items():
                url = f"{stack.base}/app/{page}/?id={scan}"
                problems += _check_page(c, url, ready,
                                        SCREENS / viewport / f"{page}-{scan[-4:]}.png",
                                        phone=phone)
        assert not problems, "\n".join(problems)
    finally:
        c.quit()


def test_the_trust_banner_says_verified(stack, chrome_bin):
    c = Chrome(chrome_bin)
    try:
        _sign_in(c, stack.base, "v.iyer")
        c.wait_text("Audit ledger verified")
    finally:
        c.quit()


def test_a_decision_taken_entirely_through_the_ui_reaches_the_ledger(stack, chrome_bin):
    """Filter to review, open a finding, raise it to quarantine with a reason and a typed
    justification, submit — then read it back from the API the page itself uses."""
    scan = SCANS[0]
    c = Chrome(chrome_bin)
    try:
        _sign_in(c, stack.base, "a.sharma")
        c.goto(f"{stack.base}/app/findings/?id={scan}")
        c.wait_text("Keyboard:")
        # Which finding to act on is decided from data, not from whichever card rendered
        # first: the list animates between filters, so "the first card" is a race.
        finding_id = c.evaluate(f"""fetch('/api/scans/{scan}/findings?disposition=review')
            .then(r => r.json())
            .then(j => j.findings.find(f => !(f.state && f.state.pending)
                                            && !f.detector_id.startsWith('prov.')).finding_id)""")
        c.click_text("review", "button[aria-pressed]")          # the disposition filter, as a user would
        sel = f'#f-{finding_id}'
        c.wait_for(f"document.querySelector('{sel} button[aria-expanded]')",
                   what="the chosen finding's card")
        c.evaluate(f"document.querySelector('{sel} button[aria-expanded]').click()")
        c.wait_for(f"[...document.querySelectorAll('{sel} button')]"
                   ".some(b => b.innerText.includes('Open and decide'))",
                   what="the expanded card")
        c.click_text("Open and decide", f"{sel} button")
        c.wait_for("document.querySelector('[role=dialog]')", what="the decide sheet")
        c.wait_text("New disposition")

        c.evaluate("""[...document.querySelectorAll('[role=dialog] #disp button')]
                      .find(b => b.innerText.startsWith('quarantine')).click()""")
        # The sheet offers the codes that fit THIS finding (a prov.* finding has its own
        # closed set), so take the first one it offers rather than assume one.
        c.wait_for("document.querySelector('#rc option:not([value=\"\"])')",
                   what="the reason codes")
        code = c.evaluate("document.querySelector('#rc option:not([value=\"\"])').value")
        c.select("#rc", code)
        c.type_into("#just", "Holding this until the contributor answers the batch query.")
        c.wait_for("[...document.querySelectorAll('[role=dialog] button')]"
                   ".some(b => b.innerText.includes('Record this decision') && !b.disabled)",
                   what="the submit button to enable")
        c.click_text("Record this decision", "[role=dialog] button")

        # The ledger is the source of truth, not the page: ask the API the page uses.
        c.wait_for(f"""fetch('/api/scans/{scan}/findings/{finding_id}')
                       .then(r => r.json())
                       .then(j => j.finding.state && j.finding.state.effective === 'quarantine')""",
                   what="the decision to be recorded", timeout=20)
        events = json.loads(c.evaluate(
            f"fetch('/api/scans/{scan}/findings/{finding_id}').then(r => r.text())"))["events"]
        assert events and events[-1]["actor_id"] == "a.sharma", events
        assert events[-1]["new_disposition"] == "quarantine"
        assert not c.errors(), c.errors()
    finally:
        c.quit()
