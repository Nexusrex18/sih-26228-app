"""`scan_id` grammar and the content-addressed evidence store — backend_plan.md §5.3.

"Byte-identical report.json across two runs" is impossible as stated: `report.json` carries
`scan_id` and `created_at_utc`, which are volatile BY DESIGN — a scan that reused a scan id
would break the per-scan grouping `Finding.scan_id` exists for. The fix is three parts, and
the first of them is the actual deliverable:

  1. the reproduction block DECLARES THE VOLATILE FIELD SET explicitly. Declaring that set
     *is* the deliverable — an auditor needs to know which fields are expected to differ,
     not to be handed an assertion that cannot hold;
  2. the reproducibility test diffs everything else (V9);
  3. the `selftest` profile derives `scan_id` from `--seed` and pins the clock, so under
     `--profile selftest` the diff is empty and the assertion is unconditional (V10).

One grammar, and the seed-derived form is a legal instance of it. Stated here because
Module E validates `scan_id` as a PATH SEGMENT (Module-E S4) and refuses anything else — a
selftest report that failed the regex would be refused by the dashboard, which is the kind
of thing nobody finds until demo day.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCAN_ID_RE = re.compile(r"^s-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{4}$")

#: The volatile set, declared as JSON PATHS, not bare field names. `scan_id` appears at
#: report level AND on every Finding (`Plugin-Interfaces.md:22` puts it on the shape
#: itself), so a top-level deletion is not enough and the diff must walk the tree. V9
#: implements the walk; without it every finding differs and the test fails on a CORRECT
#: run, which is how a reproducibility test gets deleted (R9).
VOLATILE_PATHS: tuple[str, ...] = (
    "$.scan_id",
    "$.created_at_utc",
    "$.findings[*].scan_id",
)

#: The clock the `selftest` profile pins. Any fixed instant would do; this one is the date
#: the contract was frozen, so a selftest report is self-identifying at a glance.
SELFTEST_EPOCH = _dt.datetime(2026, 9, 19, 0, 0, 0, tzinfo=_dt.UTC)


def new_scan_id(now: _dt.datetime | None = None, counter: int | None = None) -> str:
    """Normal form: NNNN is a per-day counter."""
    now = now or _dt.datetime.now(_dt.UTC)
    n = counter if counter is not None else int(now.strftime("%H%M"))
    return f"s-{now:%Y-%m-%d}-{n % 10000:04d}"


def selftest_scan_id(seed: int) -> str:
    """Selftest form: NNNN = last 4 digits of sha256(seed), date pinned by the profile's
    frozen clock. Both forms match the same regex; the dashboard cannot tell them apart,
    and must not have to."""
    h = hashlib.sha256(str(int(seed)).encode()).hexdigest()
    nnnn = int(h[-8:], 16) % 10000
    return f"s-{SELFTEST_EPOCH:%Y-%m-%d}-{nnnn:04d}"


def validate_scan_id(scan_id: str) -> str:
    if not SCAN_ID_RE.match(scan_id):
        raise ValueError(
            f"scan_id {scan_id!r} does not match {SCAN_ID_RE.pattern}. It is used as a "
            "path segment and the dashboard validates it (Module-E S4).")
    return scan_id


def scan_out_dir(out_dir: Path, scan_id: str, *, allow_existing: bool = False) -> Path:
    """`<out_dir>/<scan_id>/`, and **refuse to write into an existing one**.

    A scan is an artefact; silently replacing one is data loss. And under `selftest` two
    runs derive the SAME `scan_id` by design, so overwriting is not a corner case but the
    default behaviour — which is why V10 runs `--out a/` and `--out b/` and diffs across
    them. Without that, the second run would overwrite the first and the reproducibility
    test would compare a file with itself: passing unconditionally while asserting nothing.
    """
    validate_scan_id(scan_id)
    d = Path(out_dir) / scan_id
    if d.exists() and any(d.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"{d} already holds a scan. A scan is an artefact; refusing to overwrite it. "
            f"Use --out to write elsewhere (this is exactly what V10 does, because two "
            f"selftest runs derive the same scan_id by design).")
    d.mkdir(parents=True, exist_ok=True)
    return d


class EvidenceStore:
    """`<out_dir>/evidence/<sha256>` — SHARED ACROSS SCANS, not under each scan's dir.

    Content addressing exists so an identical crop is stored once; a per-scan `evidence/`
    folder stores it once *per scan that references it*, and worse, forces `Evidence.path`
    to be scan-relative — re-coupling it to `scan_id`, the exact volatile field §5.3
    removes. `Evidence.path` is therefore the BARE HASH, and both the renderer and the
    dashboard resolve it against this directory.
    """

    def __init__(self, out_dir: Path) -> None:
        self.root = Path(out_dir) / "evidence"
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, blob: bytes, suffix: str = "") -> str:
        digest = hashlib.sha256(blob).hexdigest()
        target = self.root / (digest + suffix)
        if not target.exists():
            target.write_bytes(blob)
        return digest + suffix

    def put_json(self, payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          default=str).encode("utf-8")
        return self.put_bytes(blob, ".json")

    def path_for(self, ref: str) -> Path:
        return self.root / ref


def materialise_evidence(findings: Any, store: EvidenceStore) -> None:
    """Write every inline `Evidence.data` payload into the store and fill `path`.

    `Evidence` is frozen, so this rebuilds the list rather than mutating entries — which
    is the behaviour we want anyway: the finding's evidence list after this call is the
    one the report records, and it refers to bytes that exist on disk.
    """
    from .types import Evidence

    for f in findings:
        out = []
        for e in f.evidence:
            if e.path is None and e.data is not None:
                out.append(Evidence(e.kind, e.caption, store.put_json(e.data), e.data))
            else:
                out.append(e)
        f.evidence = out
