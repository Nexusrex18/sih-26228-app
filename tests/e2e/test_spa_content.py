"""What the dashboard SHOWS, in a real browser (gate N1 and S1, after the Jinja pages).

Two families, both of which used to be asserted against server-rendered HTML and now have
to be asserted where the content is actually rendered — by React, in a browser:

  * S1: report content is an adversary's string. The hostile fixture values must appear on
    the page as TEXT, and none of them may become an element, run a handler or be
    evaluated. A page that runs `onerror=alert(1)` opens a dialog; Playwright sees it.
  * N1: the report's guarantees are visible — a scan where nothing ran leads with that, a
    zero is shown beside the checks that ran, provenance is present when clean and its
    absence is said, an EXIF attribution is a hypothesis, budget and capability exclusions
    stay distinct and the budget one names its cost.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from cva.web.fixtures import HOSTILE_STRINGS

from .conftest import PASSWORD

playwright_api = pytest.importorskip("playwright.sync_api",
                                     reason="pip install playwright")

pytestmark = pytest.mark.e2e

RICH = "s-2026-09-19-0001"
ALL_UNAVAILABLE = "s-2026-09-19-0002"
CLEAN = "s-2026-09-19-0003"
BLACKBOX = "s-2026-09-19-0004"


@pytest.fixture(scope="module")
def page(stack) -> Iterator[object]:
    with playwright_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                pytest.skip(f"no browser available to Playwright: {e}")
        pg = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
        pg.dialogs = []                                        # type: ignore[attr-defined]
        pg.on("dialog", lambda d: (pg.dialogs.append(d.message), d.dismiss()))
        pg.goto(f"{stack.base}/login?next=/app/")
        pg.fill("#actor_id", "a.sharma")
        pg.fill("#password", PASSWORD)
        pg.get_by_role("button", name="Sign in").click()
        pg.wait_for_url("**/app/**")
        yield pg
        browser.close()


def _open(page, base: str, path: str, ready: str) -> str:
    """Navigate and wait for the page's loaded state. Returns ALL text, including text in
    collapsed sections (textContent, not innerText): a guarantee printed inside a closed
    <details> is still printed."""
    page.goto(f"{base}{path}")
    page.get_by_text(ready).first.wait_for(timeout=25_000)
    page.wait_for_timeout(600)
    return page.locator("body").text_content() or ""


# --- S1: hostile report content is inert -------------------------------------------------------

@pytest.mark.parametrize("path,ready", [
    (f"/app/contributors/?id={RICH}", "Source-level aggregation"),
    (f"/app/findings/?id={RICH}", "Keyboard:"),
    (f"/app/scan/?id={RICH}", "Access assumptions"),
])
def test_hostile_strings_render_as_text_and_never_as_markup(stack, page, path, ready):
    text = _open(page, stack.base, path, ready)
    # The injected tag, if it had become an element, would be an <img src="x">.
    assert page.locator('img[src="x"]').count() == 0, f"{path}: an <img> was injected"
    assert page.locator("script:not([src])").filter(has_text="alert(2)").count() == 0, path
    assert not page.dialogs, f"{path}: a hostile handler ran: {page.dialogs}"
    # Shown, not dropped: an analyst has to be able to see what the contributor sent.
    if "contributors" in path:
        assert HOSTILE_STRINGS["contributor"] in text, "the hostile contributor was dropped"


def test_the_findings_page_shows_hostile_values_verbatim(stack, page):
    """Expanded cards: the file name and the `{{7*7}}` category arrive as the literal
    characters — never closed as an attribute, never evaluated to 49."""
    _open(page, stack.base, f"/app/findings/?id={RICH}", "Keyboard:")
    for button in page.locator('article[id^="f-"] button[aria-expanded]').all():
        button.click()
    page.wait_for_timeout(400)
    text = page.locator("body").text_content() or ""
    assert HOSTILE_STRINGS["file"] in text
    assert HOSTILE_STRINGS["category"] in text
    assert not page.dialogs, page.dialogs


# --- N1: the report's guarantees are visible ---------------------------------------------------

def test_a_scan_where_nothing_ran_leads_with_that_and_does_not_look_green(stack, page):
    text = _open(page, stack.base, f"/app/scan/?id={ALL_UNAVAILABLE}", "Access assumptions")
    assert "No check ran for this scan" in text
    assert "A short report here is a small statement, not a clean one." in text
    assert page.locator(".hatched").count() > 0, "absence must be hatched, not greyed away"
    assert "UNAVAILABLE" in text


def test_zero_findings_is_shown_beside_the_checks_that_ran(stack, page):
    text = _open(page, stack.base, f"/app/findings/?id={CLEAN}", "Keyboard:")
    assert "This scan produced no findings" in text
    assert "data.label_flip" in text and "prov.ledger_verify" in text


def test_provenance_is_shown_when_clean_and_its_absence_is_said(stack, page):
    text = _open(page, stack.base, f"/app/scan/?id={CLEAN}", "Access assumptions")
    assert "Records verified" in text and "Unwitnessed window" in text
    text = _open(page, stack.base, f"/app/provenance/?id={BLACKBOX}",
                 "What this does not establish")
    assert "No provenance section in this report" in text
    assert "That is not a clean provenance result. It is the absence of one." in text


def test_an_exif_cluster_attribution_is_labelled_a_hypothesis(stack, page):
    text = _open(page, stack.base, f"/app/contributors/?id={RICH}", "Source-level aggregation")
    assert "hypothesis" in text
    assert "camera-serial cluster, not a declared identity" in text


def test_budget_and_capability_exclusions_stay_distinct(stack, page):
    text = _open(page, stack.base, f"/app/scan/?id={RICH}", "Access assumptions")
    assert "capability" in text and "budget" in text
    assert "41 min" in text, "a budget exclusion prints what it would have cost"


def test_the_scan_sections_follow_the_single_file_reports_order(stack, page):
    """Matched on the section HEADINGS: the tab bar above the content repeats these words,
    so the first occurrence of each string is the tab, not the section."""
    _open(page, stack.base, f"/app/scan/?id={RICH}", "Reproduction")
    headings = [h.strip() for h in page.locator("h2").all_text_contents()]
    order = ["Access assumptions", "Contributor risk", "Provenance", "Coverage",
             "Reproduction"]
    found = [h for h in headings if h in order]
    assert found == order, f"sections out of order: {headings}"


def test_remediation_is_reported_as_absent_not_offered(stack, page):
    text = _open(page, stack.base, f"/app/scan/?id={RICH}", "Reproduction")
    assert "cva remediate is absent from this build" in text
    for banned in (" cleaned", " fixed", " safe "):
        assert banned not in text.lower(), f"wording rule (§7.5): {banned!r}"
