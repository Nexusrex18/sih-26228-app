"""A browser, as far as the server can tell: cookies, and the headers a browser DERIVES.

The unit and route tests set `Origin` by hand. That is exactly why a Firefox-only login
failure reached a human before it reached a test: the server's CSRF check depends on
`Origin`/`Referer`, and a browser computes those from the page's referrer policy — where a
`<meta name="referrer">` in the HTML OVERRIDES the response header. This client does the
same derivation, so a policy regression in either place shows up here as a refused request.

It follows the Fetch standard's rules for the two cases this dashboard has:

  * a form POST (navigation): `Origin` is the page's origin, or the literal `null` when
    the policy is `no-referrer` (what Firefox sent); `Referer` is sent same-origin unless
    the policy is `no-referrer`;
  * a `fetch()` from the SPA: the same, using the SPA page's policy.
"""
from __future__ import annotations

import http.cookiejar
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

#: Policies under which a same-origin request carries a Referer (Fetch §8.3).
_SENDS_SAME_ORIGIN_REFERER = {
    "same-origin", "strict-origin-when-cross-origin", "origin-when-cross-origin",
    "no-referrer-when-downgrade", "unsafe-url", "origin", "strict-origin",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a: Any, **kw: Any) -> None:  # noqa: D401
        return None


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class Browser:
    base: str
    jar: http.cookiejar.CookieJar = field(default_factory=http.cookiejar.CookieJar)
    #: The page the next request is made FROM, and the referrer policy it is under.
    page_url: str = ""
    page_policy: str = "strict-origin-when-cross-origin"   # the browser default

    def __post_init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar), _NoRedirect())

    # -- the policy a page is under ---------------------------------------------------------
    @staticmethod
    def effective_policy(headers: dict[str, str], html: str) -> str:
        """A `<meta name="referrer">` overrides the header; the last valid token wins."""
        metas = re.findall(r'<meta\s+name="referrer"\s+content="([^"]+)"', html, re.I)
        if metas:
            return metas[-1].strip().lower()
        header = headers.get("referrer-policy", "")
        tokens = [t.strip().lower() for t in header.split(",") if t.strip()]
        return tokens[-1] if tokens else "strict-origin-when-cross-origin"

    def _origin(self) -> str:
        p = urllib.parse.urlsplit(self.base)
        return f"{p.scheme}://{p.netloc}"

    def _derived_headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.page_policy == "no-referrer":
            h["Origin"] = "null"
        else:
            h["Origin"] = self._origin()
            if self.page_policy in _SENDS_SAME_ORIGIN_REFERER and self.page_url:
                h["Referer"] = (self._origin() + "/"
                                if self.page_policy in ("origin", "strict-origin")
                                else self.page_url)
        return h

    # -- requests -------------------------------------------------------------------------
    def _send(self, method: str, path: str, data: bytes | None,
              headers: dict[str, str]) -> Response:
        url = path if path.startswith("http") else self.base + path
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with self._opener.open(req, timeout=60) as r:
                return Response(r.status, {k.lower(): v for k, v in r.headers.items()},
                                r.read(), url)
        except urllib.error.HTTPError as e:
            return Response(e.code, {k.lower(): v for k, v in e.headers.items()},
                            e.read(), url)

    def visit(self, path: str) -> Response:
        """A navigation. The page becomes the one later requests are made from."""
        r = self._send("GET", path, None, {"Accept": "text/html"})
        if r.status == 200 and "text/html" in r.headers.get("content-type", ""):
            self.page_url = r.url
            self.page_policy = self.effective_policy(r.headers, r.text)
        return r

    def get(self, path: str, accept: str = "application/json") -> Response:
        return self._send("GET", path, None, {"Accept": accept})

    def submit_form(self, action: str, fields: dict[str, str]) -> Response:
        """A form POST from the current page, with the headers this page's policy yields."""
        body = urllib.parse.urlencode(fields).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded",
                   "Accept": "text/html", **self._derived_headers()}
        return self._send("POST", action, body, headers)

    def fetch_json(self, path: str, body: dict[str, Any], csrf: str | None) -> Response:
        """The SPA's `fetch()`: JSON body, CSRF token in a header, derived Origin/Referer."""
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   **self._derived_headers()}
        if csrf is not None:
            headers["X-CSRF-Token"] = csrf
        return self._send("POST", path, json.dumps(body).encode(), headers)

    # -- the two flows every test needs -----------------------------------------------------
    def sign_in(self, actor: str, password: str) -> Response:
        page = self.visit("/login?next=/app/")
        assert page.status == 200, page.status
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', page.text)
        assert m, "the login form carries no csrf_token field"
        return self.submit_form("/login", {"actor_id": actor, "password": password,
                                           "csrf_token": m.group(1), "next": "/app/"})

    def open_app(self, path: str = "/app/") -> str:
        """Load an SPA page, then read the CSRF token the way the SPA does."""
        r = self.visit(path)
        assert r.status == 200, (path, r.status)
        s = self.get("/api/session").json()
        assert s["authenticated"], s
        return s["csrf_token"]
