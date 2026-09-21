"""S1, S3 and S4: a report is attacker-controlled content (plan §3.5, §7.7).

The dataset supplier is an adversary by premise, and everything derived from their data —
a contributor name, a file name, a category, an evidence file — reaches this dashboard as a
string someone else chose. `cva/web/fixtures.py:HOSTILE_STRINGS` puts that premise in the
fixtures; this file is the assertion half that was missing.

What is being tested is not "the strings appear somewhere". It is that they appear INERT:
the markup they contain is text, the template language they mimic is not evaluated, the
evidence route never takes a path from them, and an SVG they control is never inlined into
the dashboard's DOM.
"""
from __future__ import annotations

import hashlib
import re

import pytest

from cva.web.fixtures import HOSTILE_STRINGS
from cva.web.security import EVIDENCE_CSP, SVG_TYPE

from tests.web.conftest import sign_in  # noqa: F401  (fixtures come from conftest)

#: Every server-rendered page that can carry report-derived content.
PAGES = ("/scans/{scan}", "/scans/{scan}/contributors", "/scans/{scan}/findings",
         "/scans/{scan}/provenance", "/scans/{scan}/coverage",
         "/scans/{scan}/reproduction", "/audit/")


def _dangerous_markup(html: str) -> tuple[list[str], bool]:
    """`(event-handler attribute names, any inline <script>)`, from a real parse."""
    from html.parser import HTMLParser

    class Scan(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.handlers: list[str] = []
            self.inline_script = False
            self._in_script_without_src = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            names = {k.lower() for k, _ in attrs}
            self.handlers += [n for n in names if n.startswith("on")]
            if tag == "script" and "src" not in names:
                self._in_script_without_src = True

        def handle_endtag(self, tag: str) -> None:
            if tag == "script":
                self._in_script_without_src = False

        def handle_data(self, data: str) -> None:
            if self._in_script_without_src and data.strip():
                self.inline_script = True

    p = Scan()
    p.feed(html)
    return p.handlers, p.inline_script


def _pages(client, scan_id):
    for url in PAGES:
        resp = client.get(url.format(scan=scan_id))
        assert resp.status_code == 200, f"{url}: {resp.status_code}"
        yield url, resp.get_data(as_text=True)


# --- S1: autoescaping ---------------------------------------------------------------------

def test_the_hostile_contributor_name_is_rendered_as_text_not_as_an_element(signed_in,
                                                                           scan_id):
    """`<img src=x onerror=alert(1)>` must arrive escaped. If the raw tag appears anywhere,
    a contributor has injected an element into an analyst's browser."""
    raw = HOSTILE_STRINGS["contributor"]
    seen_escaped = False
    for url, html in _pages(signed_in, scan_id):
        assert raw not in html, f"{url}: the raw tag reached the document"
        if "&lt;img src=x onerror=alert(1)&gt;" in html:
            seen_escaped = True
    assert seen_escaped, "the hostile contributor must be SHOWN, escaped — not dropped"


def test_no_page_carries_an_inline_script_or_an_event_handler_attribute(signed_in, scan_id):
    """S1's structural half. The strict CSP forbids both; this fails at the template rather
    than relying on the browser to enforce it.

    Parsed, not grepped. `value="&lt;img src=x onerror=alert(1)&gt;"` contains the
    characters of an event handler and is the CORRECT rendering of a hostile contributor
    name — a regex cannot tell that from an attribute, and the parser can.
    """
    for url, html in _pages(signed_in, scan_id):
        handlers, inline_scripts = _dangerous_markup(html)
        assert not handlers, f"{url}: event-handler attributes {handlers}"
        assert not inline_scripts, f"{url}: inline script"


def test_the_hostile_file_name_cannot_close_an_attribute(signed_in, scan_id):
    raw = HOSTILE_STRINGS["file"]
    for url, html in _pages(signed_in, scan_id):
        assert raw not in html, f"{url}: `\"><script>` reached the document unescaped"
        assert "<script>alert(2)" not in html, url


def test_the_template_expression_in_a_category_is_never_evaluated(signed_in, scan_id):
    """`{{7*7}}` is server-side template injection's canary. It must render as five
    characters and never as 49."""
    found = False
    for url, html in _pages(signed_in, scan_id):
        if HOSTILE_STRINGS["category"] in html:
            found = True
        assert not re.search(r"\b49\b(?![0-9])", html) or "{{7*7}}" in html, url
    assert found, "the hostile category must be shown verbatim somewhere"


def test_a_javascript_url_never_becomes_an_href(signed_in, scan_id):
    for url, html in _pages(signed_in, scan_id):
        assert not re.search(r'href\s*=\s*["\']?\s*javascript:', html, re.I), url


def test_the_sql_shaped_batch_name_is_a_string_not_a_statement(signed_in, scan_id):
    """The index is SQLite and the batch name contains `';DROP TABLE findings;--`. If it
    were interpolated rather than bound, the findings page would fail to load at all."""
    resp = signed_in.get(f"/scans/{scan_id}/findings")
    assert resp.status_code == 200
    # And the table is still there afterwards.
    assert signed_in.get(f"/scans/{scan_id}/contributors").status_code == 200


def test_the_json_api_escapes_nothing_and_the_client_must_do_it(signed_in, scan_id):
    """The API returns data, not markup — the escaping boundary moves to React, which
    escapes by default. What matters here is that no route hands back HTML it built from
    report content, so there is exactly one boundary and not two."""
    resp = signed_in.get(f"/api/scans/{scan_id}")
    assert resp.mimetype == "application/json"
    body = resp.get_data(as_text=True)
    # JSON-encoded, so the raw characters are present; the content type is what makes them
    # inert. A `text/html` response carrying this would be the bug.
    assert "text/html" not in resp.headers.get("Content-Type", "")
    assert HOSTILE_STRINGS["contributor"] in body or "img src=x" in body


def test_the_spa_bundle_contains_no_dangerously_set_inner_html():
    """The React half of S1, asserted against the source rather than the build: there is
    exactly one escaping boundary and `dangerouslySetInnerHTML` would be a second."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "frontend" / "src"
    if not src.is_dir():
        pytest.skip("the frontend source is not present in this checkout")
    offenders = [p for p in src.rglob("*.tsx")
                 if "dangerouslySetInnerHTML" in p.read_text(encoding="utf-8")]
    assert not offenders, [str(p) for p in offenders]


# --- S3 / S4: the evidence route ----------------------------------------------------------

HEX64 = "a" * 64


@pytest.mark.parametrize("bad", [
    "../../../../etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "evidence/../../secret",
    "A" * 64,                 # uppercase: the store is lowercase hex
    "g" * 64,                 # not hex
    "a" * 63,                 # too short
    "a" * 65,                 # too long
    "",
])
def test_no_path_taken_from_a_url_reaches_the_filesystem(signed_in, bad):
    """S4. The only thing that may become a path is a 64-hex digest, so `../` has nothing
    to traverse. Anything else is refused before `evidence_path` is called."""
    resp = signed_in.get(f"/evidence/{bad}")
    assert resp.status_code in (400, 404, 308), f"{bad!r} -> {resp.status_code}"
    assert b"root:" not in resp.data


def test_a_file_whose_bytes_do_not_match_its_name_is_refused(signed_in, config):
    """S3's second rule. Evidence is content-addressed: a file that no longer hashes to its
    own name is not the evidence the finding cited, and serving it would put an unverified
    image behind a verified label."""
    store = config.reports_dir / "evidence"
    store.mkdir(parents=True, exist_ok=True)
    (store / HEX64).write_bytes(b"\x89PNG\r\n\x1a\n not the bytes this name claims")

    resp = signed_in.get(f"/evidence/{HEX64}")
    assert resp.status_code == 409
    body = resp.get_data(as_text=True)
    assert "hash mismatch" in body
    # Never a broken image and never silently omitted (§8): the page SAYS what is wrong.
    assert "not the evidence this finding cited" in body


def test_an_svg_is_served_sandboxed_and_never_inlined(signed_in, config):
    """S3's fourth rule. A matplotlib SVG carries attacker-supplied label text and SVG is a
    document that can carry script, so it leaves under a sandbox CSP — and the dashboard
    renders it through `<img>`, which is what makes the sandbox meaningful."""
    payload = (b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script>'
               b'<text>label</text></svg>')
    digest = hashlib.sha256(payload).hexdigest()
    store = config.reports_dir / "evidence"
    store.mkdir(parents=True, exist_ok=True)
    (store / digest).write_bytes(payload)

    resp = signed_in.get(f"/evidence/{digest}")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith(SVG_TYPE)
    assert resp.headers["Content-Security-Policy"] == EVIDENCE_CSP
    assert resp.headers["Content-Security-Policy"].startswith("sandbox")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"

    # The dashboard must reference it, never embed it.
    page = signed_in.get(f"/evidence/{digest}/about").get_data(as_text=True)
    assert "<svg" not in page.lower()
    assert "alert(1)" not in page


def test_a_polyglot_is_typed_by_its_magic_bytes_not_its_content(signed_in, config):
    """A file that begins with the PNG magic and continues with an HTML payload is a PNG to
    every sniffer that reads the first bytes — which, with `nosniff`, is the only reading
    that happens."""
    payload = b"\x89PNG\r\n\x1a\n<html><script>alert(1)</script></html>"
    digest = hashlib.sha256(payload).hexdigest()
    store = config.reports_dir / "evidence"
    store.mkdir(parents=True, exist_ok=True)
    (store / digest).write_bytes(payload)

    resp = signed_in.get(f"/evidence/{digest}")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("image/png")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "inline" in resp.headers["Content-Disposition"]


def test_an_unrecognised_type_is_an_attachment_not_a_rendered_document(signed_in, config):
    payload = b"MZ\x90\x00 this is not any image we recognise"
    digest = hashlib.sha256(payload).hexdigest()
    store = config.reports_dir / "evidence"
    store.mkdir(parents=True, exist_ok=True)
    (store / digest).write_bytes(payload)

    resp = signed_in.get(f"/evidence/{digest}")
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].startswith("attachment")


def test_a_missing_evidence_file_says_so_and_is_not_a_broken_image(signed_in):
    resp = signed_in.get(f"/evidence/{'b' * 64}")
    assert resp.status_code == 404
    body = resp.get_data(as_text=True)
    assert "evidence unavailable" in body
    assert "not in the evidence store" in body


def test_evidence_needs_a_session(client):
    resp = client.get(f"/evidence/{HEX64}")
    assert resp.status_code in (302, 401, 403)
