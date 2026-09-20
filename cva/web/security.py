"""Response headers, CSP, CSRF, session cookies and rate limiting (plan §7.7).

This is the most exposed process on the host. Everything here assumes the strings it renders
came from an adversary, because by the Problem Statement's own premise they did.
"""
from __future__ import annotations

import hmac
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

#: S2, Appendix B. `script-src 'self'` and `style-src 'self'` mean NO inline `<script>` and
#: NO inline `style=` attribute anywhere — including on an evidence thumbnail. Every rule
#: here is asserted by `tests/security/test_headers.py`.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "font-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; "
       "base-uri 'none'; form-action 'self'")

#: The Next.js dashboard at `/app/`. A STATIC EXPORT inlines its hydration bootstrap
#: (`self.__next_f.push(...)`) and its critical CSS, and a static build cannot carry a
#: per-response nonce — there is no server rendering the page to mint one.
#:
#: So `/app/` gets `'unsafe-inline'` for scripts and styles, and this is a real weakening
#: of defence in depth that is written down rather than buried:
#:
#:   * What is KEPT, and what actually carries the air-gap claim: no external origin is
#:     reachable at all. `default-src 'self'` with no host allowlist means a successful
#:     injection still cannot exfiltrate to anywhere, and the empty-network-namespace test
#:     still passes.
#:   * What is LOST: an injected inline `<script>` would execute. The mitigation is that
#:     React escapes every interpolated value by default and the frontend never calls
#:     `dangerouslySetInnerHTML` — asserted by `tests/security/test_spa_csp.py`, which
#:     greps the built bundle.
#:   * The server-rendered views keep the strict policy above, unchanged. They are the
#:     no-script fallback and the surface the XSS fixtures are asserted against.
SPA_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
           "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
           "font-src 'self' data:; connect-src 'self'; object-src 'none'; "
           "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

#: S3: an SVG is a document that can carry script, and a matplotlib SVG can carry
#: attacker-supplied label text. It is served sandboxed and rendered through `<img>`, never
#: inlined into the dashboard's DOM.
EVIDENCE_CSP = "sandbox; default-src 'none'; style-src 'unsafe-inline'"

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    # `same-origin`, NOT `no-referrer`, and the difference is load-bearing.
    #
    # S7's second half checks `Origin`/`Referer` against `Host` on every state-changing
    # request. Chrome omits `Origin` on a same-origin form POST, so `Referer` is the only
    # signal left — and `no-referrer` strips that too. The result was every login POST
    # failing the check with "this request did not come from this dashboard", which is a
    # CSRF defence rejecting the legitimate case and telling the operator nothing useful.
    #
    # `same-origin` keeps the privacy property that mattered — a referrer is never sent to
    # another origin, so no scan id or finding id leaks off the host — while leaving the
    # header present for our own requests, which is exactly what the check reads.
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), usb=(), payment=()",
}

#: S3: content type comes from MAGIC BYTES, never from an extension — the evidence route has
#: no extension to read anyway, because the path IS the content hash.
MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", True),
    (b"\xff\xd8\xff", "image/jpeg", True),
    (b"GIF87a", "image/gif", True),
    (b"GIF89a", "image/gif", True),
    (b"RIFF", "image/webp", True),
)
#: Served, but sandboxed and never inlined.
SVG_TYPE = "image/svg+xml"
#: Anything not recognised is a download, with a generic type. A polyglot that is both a
#: valid PNG and a valid HTML document is served as an attachment either way.
FALLBACK_TYPE = "application/octet-stream"


def sniff(data: bytes) -> tuple[str, bool]:
    """`(content_type, inlineable)` from the first bytes. Never from a filename."""
    head = data[:16]
    for magic, ctype, inline in MAGIC:
        if head.startswith(magic):
            if ctype == "image/webp" and data[8:12] != b"WEBP":
                continue
            return ctype, inline
    stripped = data.lstrip()[:512].lower()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<svg"):
        if b"<svg" in stripped:
            return SVG_TYPE, False
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "application/json", False
    return FALLBACK_TYPE, False


def apply_headers(headers: Any, *, authenticated: bool = True, spa: bool = False) -> None:
    for k, v in SECURITY_HEADERS.items():
        headers[k] = v
    if spa:
        headers["Content-Security-Policy"] = SPA_CSP
    if authenticated:
        # An analyst's browser must not keep a page of findings in its disk cache after they
        # log out on a shared terminal.
        headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        headers["Pragma"] = "no-cache"


# --- CSRF (S7) --------------------------------------------------------------------------

def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_ok(session_token: str | None, submitted: str | None) -> bool:
    if not session_token or not submitted:
        return False
    return hmac.compare_digest(session_token, submitted)


def origin_ok(origin: str | None, referer: str | None, host: str | None) -> bool:
    """The second half of S7: a token alone is not enough.

    A missing Origin AND a missing Referer is refused rather than waved through: the
    browsers this runs against send at least one on a same-origin POST, and "the header was
    absent" is the shape of a request that did not come from the page.
    """
    if not host:
        return False
    for header in (origin, referer):
        if not header:
            continue
        rest = header.split("://", 1)[-1]
        candidate = rest.split("/", 1)[0]
        if candidate == host:
            return True
    return False


# --- rate limiting (S12) -------------------------------------------------------------------

class RateLimiter:
    """A fixed-window counter per (identity, bucket). In-process and deliberately simple:
    this serves a handful of analysts on one host, not the internet."""

    def __init__(self, per_minute: int = 120) -> None:
        self.per_minute = per_minute
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, identity: str, bucket: str = "default",
              *, now: float | None = None, limit: int | None = None) -> bool:
        now = time.monotonic() if now is None else now
        limit = self.per_minute if limit is None else limit
        q = self._hits[(identity, bucket)]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    def reset(self, identity: str, bucket: str = "default") -> None:
        self._hits.pop((identity, bucket), None)


# --- session cookie flags (S6) --------------------------------------------------------------

def session_cookie_config(*, tls: bool, idle_timeout_s: int) -> Mapping[str, Any]:
    return {
        "SESSION_COOKIE_NAME": "cva_session",
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Strict",
        "SESSION_COOKIE_SECURE": bool(tls),
        "SESSION_COOKIE_PATH": "/",
        "PERMANENT_SESSION_LIFETIME": int(idle_timeout_s),
    }


__all__ = ["CSP", "EVIDENCE_CSP", "FALLBACK_TYPE", "MAGIC", "SECURITY_HEADERS", "SPA_CSP",
           "SVG_TYPE", "RateLimiter", "apply_headers", "csrf_ok", "new_csrf_token",
           "origin_ok", "session_cookie_config", "sniff"]
