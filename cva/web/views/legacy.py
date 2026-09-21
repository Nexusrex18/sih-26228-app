"""Where the old server-rendered dashboard used to be.

That dashboard is gone; the Next.js app at /app/ replaced it. Its GET paths answer with a
redirect to the page that now shows the same thing, so a bookmark, a link pasted into a
ticket or an old `next=` still lands somewhere true. The mapping is `spa_paths` — the same
table the login redirect uses.

302, not 301: a permanent redirect is cached by the browser for good, and these paths are
ours to reuse.
"""
from __future__ import annotations

from typing import Any

from flask import Blueprint, redirect, request

from ..spa_paths import HOME, spa_equivalent

bp = Blueprint("legacy", __name__)

_PATHS = (
    "/", "/health", "/audit", "/audit/", "/audit/verification", "/admin/accounts",
    "/scans/<scan_id>", "/scans/<scan_id>/<section>",
    "/scans/<scan_id>/<section>/<path:rest>",
)


def _to_spa(**_: Any) -> Any:
    return redirect(spa_equivalent(request.path) or HOME, code=302)


for _i, _path in enumerate(_PATHS):
    bp.add_url_rule(_path, f"legacy_{_i}", _to_spa, methods=["GET"])


__all__ = ["bp"]
