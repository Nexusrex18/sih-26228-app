"""Read and JSON-Schema-validate `report.json` (plan §7.1).

An unknown `schema_version` is REFUSED with an explanation, never best-effort rendered: a
report that "mostly renders" is a report that can hide a field.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: `backend_plan.md` §5.3, quoted in the schema. The `selftest` profile's seed-derived form
#: is a legal instance of the same grammar, so it is accepted rather than refused.
SCAN_ID_RE = re.compile(r"s-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{4}")
EVIDENCE_HASH_RE = re.compile(r"[0-9a-f]{64}")

SUPPORTED_SCHEMA_VERSIONS = ("1.0.0",)


class ReportUnreadable(Exception):
    """The scan is listed as unreadable WITH THE REASON; it is never partially rendered."""

    def __init__(self, scan_id: str, reason: str) -> None:
        super().__init__(f"{scan_id}: {reason}")
        self.scan_id = scan_id
        self.reason = reason


@dataclass(frozen=True)
class LoadedReport:
    scan_id: str
    path: Path
    data: dict[str, Any]
    sha256: str
    #: Backend's sidecar (`seal.json`), when present: the sealing OUTCOME, machine-readable.
    seal_sidecar: dict[str, Any] | None = None
    findings_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def created_at_utc(self) -> str:
        return str(self.data.get("created_at_utc", ""))

    @property
    def verdict(self) -> str:
        return str(self.data.get("verdict", "REVIEW"))

    @property
    def profile_name(self) -> str:
        return str(self.data.get("produced_by", {}).get("profile_name", ""))

    @property
    def budget_tier(self) -> str:
        return str(self.data.get("produced_by", {}).get("budget_tier", ""))

    @property
    def findings(self) -> list[dict[str, Any]]:
        return list(self.data.get("findings") or [])

    def originals(self) -> dict[str, tuple[str, str]]:
        """`finding_id -> (disposition, disposition_rule)` — the fold's starting point."""
        return {str(f["finding_id"]): (str(f.get("disposition", "accept")),
                                       str(f.get("disposition_rule", "")))
                for f in self.findings}

    def counts_by_disposition(self) -> dict[str, int]:
        out = {"accept": 0, "review": 0, "quarantine": 0}
        for f in self.findings:
            d = str(f.get("disposition", "accept"))
            out[d] = out.get(d, 0) + 1
        return out


@functools.cache
def _schema() -> dict[str, Any]:
    p = Path(__file__).resolve().parents[3] / "schemas" / "report.schema.json"
    return json.loads(p.read_text())


@functools.cache
def _validator() -> Any:
    from jsonschema import Draft202012Validator
    return Draft202012Validator(_schema())


def validate_scan_id(raw: str) -> str:
    """S4: the only user-supplied string that ever becomes a path segment."""
    if not isinstance(raw, str) or not SCAN_ID_RE.fullmatch(raw):
        raise ReportUnreadable(str(raw)[:64], "not a scan id (s-YYYY-MM-DD-NNNN)")
    return raw


def validate_evidence_hash(raw: str) -> str:
    """S4: 64 lowercase hex, and nothing else, ever reaches the evidence directory."""
    if not isinstance(raw, str) or not EVIDENCE_HASH_RE.fullmatch(raw):
        raise ReportUnreadable("evidence", "not a 64-character lowercase hex digest")
    return raw


def report_path(reports_dir: Path, scan_id: str) -> Path:
    return Path(reports_dir) / validate_scan_id(scan_id) / "report.json"


def load(reports_dir: Path, scan_id: str) -> LoadedReport:
    scan_id = validate_scan_id(scan_id)
    p = report_path(reports_dir, scan_id)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise ReportUnreadable(scan_id, f"cannot read {p.name}: {e.strerror}") from None
    return parse(scan_id, p, raw, _sidecar(p.parent))


def _sidecar(scan_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((scan_dir / "seal.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse(scan_id: str, path: Path, raw: bytes,
          sidecar: dict[str, Any] | None = None) -> LoadedReport:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReportUnreadable(scan_id, f"report.json is not valid JSON ({e})") from None
    if not isinstance(data, dict):
        raise ReportUnreadable(scan_id, "report.json is not a JSON object")

    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ReportUnreadable(
            scan_id,
            f"schema_version {version!r} is not one this dashboard understands "
            f"({list(SUPPORTED_SCHEMA_VERSIONS)}). It was produced by a different version of "
            "the scanner. It is NOT rendered best-effort: a report that mostly renders is a "
            "report that can hide a field, and a hidden field in an assurance report is the "
            "failure this tool exists to prevent. Upgrade the dashboard, or read the "
            "single-file report.html that shipped beside it.")

    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = ".".join(str(p) for p in first.absolute_path) or "$"
        raise ReportUnreadable(
            scan_id, f"report.json fails report.schema.json ({len(errors)} error(s); first "
                     f"at {where}: {first.message[:200]})")

    if data.get("scan_id") != scan_id:
        raise ReportUnreadable(
            scan_id, f"the report's own scan_id is {data.get('scan_id')!r} but it was found "
                     f"in the directory {scan_id!r}. Identity is not negotiable.")

    return LoadedReport(
        scan_id=scan_id, path=path, data=data,
        sha256=hashlib.sha256(raw).hexdigest(), seal_sidecar=sidecar,
        findings_by_id={str(f["finding_id"]): f for f in (data.get("findings") or [])})


def discover(reports_dir: Path) -> list[str]:
    """Every scan directory present, newest id first. Never raises on a bad directory."""
    root = Path(reports_dir)
    if not root.is_dir():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir() and SCAN_ID_RE.fullmatch(child.name) \
                and (child / "report.json").is_file():
            out.append(child.name)
    return sorted(out, reverse=True)


__all__ = ["EVIDENCE_HASH_RE", "SCAN_ID_RE", "SUPPORTED_SCHEMA_VERSIONS", "LoadedReport",
           "ReportUnreadable", "discover", "load", "parse", "report_path",
           "validate_evidence_hash", "validate_scan_id"]
