"""Serve the Next.js static export at `/app/`.

The export is a directory of plain files — no Node process at runtime — so the whole
dashboard still works with the container's network namespace empty. The build needs a
network the way `make wheelhouse` does; the running system does not.

The server-rendered Jinja templates stay at their existing routes. They are the no-script
fallback and the thing the security suite asserts against, and keeping both costs nothing
because both read the same `services.load_scan`.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, current_app, redirect, send_from_directory

from ..auth import require_login

bp = Blueprint("spa", __name__)

#: Where `npm run build` puts the export (`frontend/out`).
DEFAULT_EXPORT = Path(__file__).resolve().parents[3] / "frontend" / "out"

#: Long-lived only for content-hashed bundles under `_next/static/`. Everything else is
#: `no-store`, because an analyst's browser must not keep a page of findings in its disk
#: cache after they log out on a shared terminal.
IMMUTABLE_PREFIX = "_next/static/"


def export_dir() -> Path:
    configured = current_app.config.get("CVA_SPA_DIR")
    return Path(configured) if configured else DEFAULT_EXPORT


@bp.get("/app")
def root_redirect() -> Any:
    return redirect("/app/")


@bp.get("/app/")
@bp.get("/app/<path:subpath>")
@require_login
def serve(subpath: str = "") -> Any:
    root = export_dir()
    if not root.is_dir():
        return Response(
            "<h1>The dashboard bundle is not built</h1>"
            "<p>Run <code>npm --prefix frontend install &amp;&amp; "
            "npm --prefix frontend run build</code>, or use the server-rendered views at "
            "<a href='/'>/</a>.</p>",
            mimetype="text/html",
            status=503,
        )

    candidate = _resolve(root, subpath)
    if candidate is None:
        abort(404)

    response = send_from_directory(root, str(candidate.relative_to(root)))
    rel = str(candidate.relative_to(root))
    if rel.startswith(IMMUTABLE_PREFIX):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-store"
    guessed = mimetypes.guess_type(candidate.name)[0]
    if guessed:
        response.headers["Content-Type"] = guessed
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _resolve(root: Path, subpath: str) -> Path | None:
    """Map a URL path onto a file inside the export, refusing anything outside it.

    `send_from_directory` already refuses traversal; this resolves the trailing-slash and
    `index.html` conventions the export uses, and re-checks containment so a symlink inside
    the export cannot point out of it either.
    """
    target = (root / subpath).resolve() if subpath else root.resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None

    if target.is_dir():
        index = target / "index.html"
        return index if index.is_file() else None
    if target.is_file():
        return target
    # `trailingSlash: true` means /app/scan resolves to scan/index.html.
    alt = (root / f"{subpath}/index.html").resolve() if subpath else None
    if alt and alt.is_file():
        try:
            alt.relative_to(root.resolve())
        except ValueError:
            return None
        return alt
    return None
