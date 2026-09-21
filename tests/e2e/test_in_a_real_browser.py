"""The dashboard in a real browser: does every page actually RENDER?

`test_full_flow.py` proves every route and API answers. It cannot see a React component
that throws on load — that serves a good HTML shell and a blank screen. This runs the
bundle in Chrome through Playwright and fails on:

  * any uncaught exception, console error, failed request or 4xx/5xx response;
  * a page that never reaches its loaded state (its data section never appears);
  * Next's "Application error" screen;
  * at 390px, anything wider than the screen (horizontal scroll hides the column that
    matters — the mobile pass's own rule);

for every page, against every fixture scan (each has a differently shaped report: rich,
all-UNAVAILABLE, clean, black-box), at desktop and phone width. Then one decision is taken
entirely through the UI — real clicks and typing — and checked in the ledger.

Playwright drives the Chrome installed on the machine (`channel="chrome"`), so nothing is
downloaded at test time. Screenshots of every page land in `.scratch/e2e-screens/`.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from .conftest import PASSWORD, ROOT

playwright_api = pytest.importorskip("playwright.sync_api",
                                     reason="pip install playwright")

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
PHONE = {"width": 390, "height": 844}


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome")
        except Exception:                          # no system Chrome: try bundled Chromium
            try:
                b = p.chromium.launch()
            except Exception as e:
                pytest.skip(f"no browser available to Playwright: {e}")
        yield b
        b.close()


class Watched:
    """A page plus every error it produced: exceptions, console errors, bad responses."""

    def __init__(self, page) -> None:
        self.page = page
        self.errors: list[str] = []
        page.on("pageerror", lambda e: self.errors.append(f"exception: {e}"))
        page.on("console", lambda m: self.errors.append(f"console.error: {m.text}")
                if m.type == "error" else None)
        # ERR_ABORTED is the browser cancelling its own request — Next prefetches linked
        # routes (`index.txt?_rsc=`) and a navigation mid-prefetch aborts them. That is not
        # a site failure. A refused connection or a 5xx still lands here or below.
        page.on("requestfailed", lambda r: self.errors.append(
            f"request failed: {r.url} ({r.failure})")
            if "ERR_ABORTED" not in (r.failure or "") else None)
        page.on("response", lambda r: self.errors.append(f"HTTP {r.status}: {r.url}")
                if r.status >= 400 else None)


def _open(browser, *, phone: bool = False) -> Watched:
    ctx = browser.new_context(
        viewport=PHONE if phone else {"width": 1280, "height": 900},
        device_scale_factor=2 if phone else 1, is_mobile=phone, has_touch=phone)
    return Watched(ctx.new_page())


def _sign_in(w: Watched, base: str, actor: str) -> None:
    """Through the real form, as a person would — the browser derives Origin itself."""
    page = w.page
    page.goto(f"{base}/login?next=/app/")
    page.fill("#actor_id", actor)
    page.fill("#password", PASSWORD)
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_url("**/app/**")
    page.get_by_text("Triage queue").wait_for()


def _check_page(w: Watched, url: str, ready: str, shot: Path, *, phone: bool) -> list[str]:
    page = w.page
    w.errors.clear()
    problems: list[str] = []
    page.goto(url)
    try:
        page.get_by_text(ready, exact=False).first.wait_for(timeout=25_000)
        page.wait_for_function(
            "!document.body.innerText.includes('loading findings')", timeout=25_000)
    except playwright_api.TimeoutError:
        problems.append(f"{url}: never reached its loaded state ({ready!r}). Page text:\n"
                        + page.inner_text("body")[:600])
    if "Application error" in page.inner_text("body"):
        problems.append(f"{url}: Next.js 'Application error' screen")
    if phone:
        overflow = page.evaluate("""() => {
            const w = window.innerWidth;
            if (document.documentElement.scrollWidth <= w + 1) return [];
            return [...document.querySelectorAll('body *')]
              .filter(e => e.getBoundingClientRect().right > w + 1 && e.offsetParent !== null)
              .slice(0, 6)
              .map(e => e.tagName.toLowerCase() + '.' + String(e.className).slice(0, 60)
                        + ' right=' + Math.round(e.getBoundingClientRect().right));
        }""")
        if overflow:
            problems.append(f"{url}: wider than the {PHONE['width']}px screen: {overflow}")
    # Let entrance animations (count-up, charts drawing in) finish, so the screenshot shows
    # the settled page a person reviews — not a frame mid-animation.
    page.wait_for_timeout(1500)
    shot.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(shot))
    problems += [f"{url}: {err}" for err in w.errors]
    return problems


@pytest.mark.parametrize("viewport", ["desktop", "phone"])
def test_every_page_renders_without_an_error(stack, browser, viewport):
    phone = viewport == "phone"
    w = _open(browser, phone=phone)
    try:
        _sign_in(w, stack.base, "a.sharma")
        problems: list[str] = []
        for page_name, ready in GLOBAL_PAGES.items():
            url = f"{stack.base}/app/{page_name + '/' if page_name else ''}"
            problems += _check_page(w, url, ready,
                                    SCREENS / viewport / f"{page_name or 'home'}.png",
                                    phone=phone)
        for scan in SCANS:
            for page_name, ready in SCAN_PAGES.items():
                url = f"{stack.base}/app/{page_name}/?id={scan}"
                problems += _check_page(w, url, ready,
                                        SCREENS / viewport / f"{page_name}-{scan[-4:]}.png",
                                        phone=phone)
        assert not problems, "\n".join(problems)
    finally:
        w.page.context.close()


def test_the_trust_banner_says_verified(stack, browser):
    w = _open(browser)
    try:
        _sign_in(w, stack.base, "v.iyer")
        w.page.get_by_text("Audit ledger verified").first.wait_for()
    finally:
        w.page.context.close()


def test_a_decision_taken_entirely_through_the_ui_reaches_the_ledger(stack, browser):
    """Filter to review, open a finding, raise it to quarantine with a reason and a typed
    justification, submit — with real clicks and keystrokes — then read it back from the
    API the page itself uses."""
    scan = SCANS[0]
    w = _open(browser)
    page = w.page
    try:
        _sign_in(w, stack.base, "a.sharma")
        page.goto(f"{stack.base}/app/findings/?id={scan}")
        page.get_by_text("Keyboard:").wait_for()

        # Which finding to act on is decided from data, not from whichever card rendered
        # first: the list animates between filters, so "the first card" is a race.
        finding_id = page.evaluate(f"""() => fetch('/api/scans/{scan}/findings?disposition=review')
            .then(r => r.json())
            .then(j => j.findings.find(f => !(f.state && f.state.pending)
                                            && !f.detector_id.startsWith('prov.')).finding_id)""")
        page.locator("button[aria-pressed]", has_text="review").first.click()

        card = page.locator(f"#f-{finding_id}")
        card.locator("button[aria-expanded]").click()
        card.get_by_role("button", name="Open and decide").click()

        dialog = page.get_by_role("dialog")
        dialog.get_by_text("New disposition").wait_for()
        dialog.locator("#disp button", has_text="quarantine").click()
        # The sheet offers the codes that fit THIS finding (a prov.* finding has its own
        # closed set), so take the first one it offers rather than assume one.
        code = dialog.locator("#rc option:not([value=''])").first.get_attribute("value")
        dialog.locator("#rc").select_option(code)
        dialog.locator("#just").fill(
            "Holding this until the contributor answers the batch query.")
        submit = dialog.get_by_role("button", name="Record this decision")
        playwright_api.expect(submit).to_be_enabled()
        submit.click()

        # The ledger is the source of truth, not the page: ask the API the page uses.
        page.wait_for_function(
            f"""() => fetch('/api/scans/{scan}/findings/{finding_id}')
                .then(r => r.json())
                .then(j => j.finding.state && j.finding.state.effective === 'quarantine')""",
            timeout=20_000)
        detail = page.evaluate(
            f"() => fetch('/api/scans/{scan}/findings/{finding_id}').then(r => r.json())")
        events = detail["events"]
        assert events and events[-1]["actor_id"] == "a.sharma", events
        assert events[-1]["new_disposition"] == "quarantine"
        assert not w.errors, w.errors
    finally:
        page.context.close()
