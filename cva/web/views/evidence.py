"""`/evidence/<sha256>` — served defensively (plan S3, S4).

Four rules, each of which has a test:

  1. The only path segment is a 64-hex digest. Nothing else ever reaches the filesystem, so
     `../` has nothing to traverse.
  2. The hash is RECOMPUTED from the bytes and a mismatch refuses. Evidence is
     content-addressed; a file whose contents no longer hash to its name is not the evidence
     the finding cited, and serving it would put an unverified image behind a verified label.
  3. Content type comes from MAGIC BYTES, never an extension — there is no extension here.
  4. SVG is untrusted. A matplotlib SVG carries attacker-supplied label text and SVG can
     carry script, so it goes out sandboxed and is rendered through `<img>`, never inlined
     into the dashboard's DOM. Anything unrecognised is an attachment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, render_template

from .. import services
from ..auth import require_login
from ..reports.loader import EVIDENCE_HASH_RE
from ..security import EVIDENCE_CSP, SVG_TYPE, sniff

bp = Blueprint("evidence", __name__)

#: An evidence file larger than this is not rendered inline. Contact sheets are big; a
#: 200 MB "PNG" is someone probing for a memory limit.
MAX_INLINE_BYTES = 64 * 1024 * 1024


def evidence_path(digest: str) -> Path:
    """The shared store: `<out_dir>/evidence/<sha256>`, never `<scan_id>/evidence/`.

    Shared because evidence is content-addressed — an identical crop cited by four scans is
    stored once — and because a per-scan directory would force `Evidence.path` to be
    scan-relative, smuggling the volatile `scan_id` back into every finding.
    """
    return services.config().reports_dir / "evidence" / digest


@bp.get("/evidence/<digest>")
@require_login
def serve(digest: str) -> Any:
    if not EVIDENCE_HASH_RE.fullmatch(digest or ""):
        return _placeholder(digest, "not a 64-character lowercase hex digest"), 400

    path = evidence_path(digest)
    try:
        if not path.is_file():
            return _placeholder(digest, "evidence unavailable: the file is not in the "
                                        "evidence store"), 404
        if path.stat().st_size > MAX_INLINE_BYTES:
            return _placeholder(digest, f"evidence unavailable: larger than "
                                        f"{MAX_INLINE_BYTES // (1024 * 1024)} MiB"), 413
        data = path.read_bytes()
    except OSError as e:
        return _placeholder(digest, f"evidence unavailable: {e.strerror}"), 404

    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        # Never a broken image and never silently omitted (§8).
        return _placeholder(
            digest, f"evidence unavailable: hash mismatch. The stored file hashes to "
                    f"{actual[:16]}…, so it is not the evidence this finding cited."), 409

    ctype, inlineable = sniff(data)
    response = Response(data, mimetype=ctype)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "private, max-age=300"
    # Content-addressed: the bytes for a digest never change, so a strong ETag is exact.
    response.headers["ETag"] = f'"{digest}"'
    if ctype == SVG_TYPE:
        response.headers["Content-Security-Policy"] = EVIDENCE_CSP
        response.headers["Content-Disposition"] = f'inline; filename="{digest}.svg"'
    elif inlineable:
        response.headers["Content-Security-Policy"] = EVIDENCE_CSP
        response.headers["Content-Disposition"] = f'inline; filename="{digest}"'
    else:
        response.headers["Content-Security-Policy"] = EVIDENCE_CSP
        response.headers["Content-Disposition"] = f'attachment; filename="{digest}"'
    return response


@bp.get("/evidence/<digest>/about")
@require_login
def about(digest: str) -> Any:
    """What this file is, before an analyst decides to open it."""
    if not EVIDENCE_HASH_RE.fullmatch(digest or ""):
        return _placeholder(digest, "not a 64-character lowercase hex digest"), 400
    path = evidence_path(digest)
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    ctype = "unknown"
    verified = False
    if exists and size <= MAX_INLINE_BYTES:
        data = path.read_bytes()
        ctype = sniff(data)[0]
        verified = hashlib.sha256(data).hexdigest() == digest
    return render_template("evidence_about.html", digest=digest, exists=exists, size=size,
                           ctype=ctype, verified=verified)


def _placeholder(digest: str, reason: str) -> Response:
    html = render_template("evidence_missing.html", digest=digest, reason=reason)
    response = Response(html, mimetype="text/html")
    response.headers["Cache-Control"] = "no-store"
    return response
