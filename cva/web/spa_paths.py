"""Where each old server-rendered path lives now, in the Next.js dashboard.

The dashboard used to be two front ends: Jinja pages at `/` and the SPA at `/app/`. The
Jinja dashboard is gone; its paths redirect here so a bookmark, a link in an old report or
a `next=` parameter still lands on the page it meant. One table, used by both the login
redirect and the legacy-path redirects, so the two cannot disagree.
"""
from __future__ import annotations

import re
from urllib.parse import quote

HOME = "/app/"

#: `/scans/<id>/<section>` -> the SPA page for that section.
_SECTIONS = {
    "": "scan",
    "findings": "findings",
    "contributors": "contributors",
    "provenance": "provenance",
    "coverage": "coverage",
    "reproduction": "scan",
    "drift": "scan",
    "remediation": "scan",
}
_SCAN = re.compile(r"^/scans/([^/?#]+)(?:/([a-z]+))?(?:/.*)?/?$")
_FIXED = {
    "/": HOME,
    "/health": "/app/verification/",
    "/audit": "/app/audit/",
    "/audit/": "/app/audit/",
    "/audit/verification": "/app/verification/",
    "/admin/accounts": "/app/accounts/",
}


def spa_equivalent(path: str) -> str | None:
    """The SPA URL for an old server-rendered path, or None if `path` is not one."""
    bare = path.split("?", 1)[0].split("#", 1)[0]
    if bare in _FIXED:
        return _FIXED[bare]
    m = _SCAN.match(bare)
    if m:
        page = _SECTIONS.get(m.group(2) or "", "scan")
        return f"/app/{page}/?id={quote(m.group(1), safe='-')}"
    return None


def safe_next(raw: str | None) -> str:
    """A post-login destination: same-site only (an open redirect from a login form is a
    phishing primitive), translated to the SPA, defaulting to the dashboard home."""
    if not raw or not raw.startswith("/") or raw.startswith("//") or "\\" in raw:
        return HOME
    if raw.startswith("/app"):
        return raw
    return spa_equivalent(raw) or HOME


__all__ = ["HOME", "safe_next", "spa_equivalent"]
