"""`index.db` — a derived, rebuildable cache for filtering (plan §7.1, D-E2).

Nothing authoritative lives here. Delete the file and it rebuilds from `reports/`. Every
value is copied from `report.json`; nothing is computed, least of all a disposition.

Corruption is handled the way the plan says: delete and rebuild, no data lost (§8).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import LoadedReport, ReportUnreadable, discover, load

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS scans (
  scan_id TEXT PRIMARY KEY, created_at_utc TEXT, verdict TEXT, profile_name TEXT,
  budget_tier TEXT, report_sha256 TEXT, n_findings INTEGER,
  n_accept INTEGER, n_review INTEGER, n_quarantine INTEGER,
  unreadable_reason TEXT, mtime REAL);
CREATE TABLE IF NOT EXISTS findings (
  scan_id TEXT, finding_id TEXT, detector_id TEXT, target_type TEXT, target_ref TEXT,
  severity TEXT, confidence REAL, disposition TEXT, disposition_rule TEXT, nature TEXT,
  attack_class TEXT, availability TEXT, exclusion_reason TEXT, reason TEXT,
  PRIMARY KEY (scan_id, finding_id));
CREATE INDEX IF NOT EXISTS findings_by_scan ON findings(scan_id, disposition, severity);
CREATE INDEX IF NOT EXISTS findings_by_nature ON findings(scan_id, nature);
"""

INDEX_VERSION = "1"

SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")
_SEV_RANK = {s: i for i, s in enumerate(SEVERITY_ORDER)}


@dataclass(frozen=True)
class ScanRow:
    scan_id: str
    created_at_utc: str
    verdict: str
    profile_name: str
    budget_tier: str
    report_sha256: str
    n_findings: int
    n_accept: int
    n_review: int
    n_quarantine: int
    unreadable_reason: str | None

    @property
    def readable(self) -> bool:
        return not self.unreadable_reason


class ReportIndex:
    #: How often a request may trigger a re-scan of `reports/`. Each refresh stats every
    #: report directory, which is cheap but not free on a store with thousands of scans.
    REFRESH_INTERVAL_S = 2.0

    def __init__(self, path: Path, reports_dir: Path) -> None:
        self.path = Path(path)
        self.reports_dir = Path(reports_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        # waitress serves on several threads that share this one connection. Writes are
        # serialised here; a refresh that races another would otherwise interleave two
        # DELETE/INSERT sequences for the same scan.
        self._lock = threading.Lock()
        self._refreshed_at = 0.0

    def refresh_if_stale(self) -> int:
        """Pick up scans that finished after the dashboard started.

        The scan LIST always refreshed, but every other reader of the index did not: open a
        new scan's findings directly (a bookmark, a link from the audit trail) before the
        list had been loaded, and the page said the scan had no findings. Found by the
        end-to-end test.
        """
        now = time.monotonic()
        if now - self._refreshed_at < self.REFRESH_INTERVAL_S:
            return 0
        return self.refresh()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            row = conn.execute("SELECT v FROM meta WHERE k='index_version'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta(k,v) VALUES('index_version',?)",
                             (INDEX_VERSION,))
                conn.commit()
            elif row["v"] != INDEX_VERSION:
                conn.close()
                return self._rebuild_from_scratch()
            return conn
        except sqlite3.DatabaseError:
            return self._rebuild_from_scratch()

    def _rebuild_from_scratch(self) -> sqlite3.Connection:
        self.path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(self.path) + suffix).unlink(missing_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('index_version',?)",
                     (INDEX_VERSION,))
        conn.commit()
        return conn

    # -- writing -------------------------------------------------------------------------
    def refresh(self) -> int:
        """Re-index every scan whose `report.json` is new or has changed on disk."""
        with self._lock:
            n = self._refresh_locked()
            self._refreshed_at = time.monotonic()
            return n

    def _refresh_locked(self) -> int:
        n = 0
        known = {r["scan_id"]: r["mtime"] for r in
                 self._conn.execute("SELECT scan_id, mtime FROM scans")}
        present = set()
        for scan_id in discover(self.reports_dir):
            present.add(scan_id)
            p = self.reports_dir / scan_id / "report.json"
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if known.get(scan_id) == mtime:
                continue
            try:
                self._upsert(load(self.reports_dir, scan_id), mtime)
            except ReportUnreadable as e:
                self._upsert_unreadable(scan_id, e.reason, mtime)
            n += 1
        gone = set(known) - present
        if gone:
            self._conn.executemany("DELETE FROM scans WHERE scan_id=?",
                                   [(s,) for s in gone])
            self._conn.executemany("DELETE FROM findings WHERE scan_id=?",
                                   [(s,) for s in gone])
        self._conn.commit()
        return n

    def _upsert(self, rep: LoadedReport, mtime: float) -> None:
        c = rep.counts_by_disposition()
        self._conn.execute(
            "INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rep.scan_id, rep.created_at_utc, rep.verdict, rep.profile_name,
             rep.budget_tier, rep.sha256, len(rep.findings), c.get("accept", 0),
             c.get("review", 0), c.get("quarantine", 0), None, mtime))
        self._conn.execute("DELETE FROM findings WHERE scan_id=?", (rep.scan_id,))
        self._conn.executemany(
            "INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(rep.scan_id, str(f["finding_id"]), str(f.get("detector_id", "")),
              str(f.get("target_type", "")), str(f.get("target_ref", "")),
              str(f.get("severity", "info")), float(f.get("confidence", 0.0)),
              str(f.get("disposition", "accept")), str(f.get("disposition_rule", "")),
              str(f.get("nature", "indeterminate")), str(f.get("attack_class", "")),
              str(f.get("availability", "OK")),
              f.get("exclusion_reason"), str(f.get("reason", "")))
             for f in rep.findings])

    def _upsert_unreadable(self, scan_id: str, reason: str, mtime: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (scan_id, "", "", "", "", "", 0, 0, 0, 0, reason, mtime))
        self._conn.execute("DELETE FROM findings WHERE scan_id=?", (scan_id,))

    # -- reading -------------------------------------------------------------------------
    def scans(self) -> list[ScanRow]:
        rows = self._conn.execute(
            "SELECT * FROM scans ORDER BY created_at_utc DESC, scan_id DESC").fetchall()
        return [ScanRow(r["scan_id"], r["created_at_utc"] or "", r["verdict"] or "",
                        r["profile_name"] or "", r["budget_tier"] or "",
                        r["report_sha256"] or "", r["n_findings"] or 0, r["n_accept"] or 0,
                        r["n_review"] or 0, r["n_quarantine"] or 0, r["unreadable_reason"])
                for r in rows]

    def scan(self, scan_id: str) -> ScanRow | None:
        return next((s for s in self.scans() if s.scan_id == scan_id), None)

    def finding_ids(self, scan_id: str, *, disposition: str | None = None,
                    nature: str | None = None, module: str | None = None,
                    availability: str | None = None,
                    limit: int | None = None, offset: int = 0) -> list[str]:
        """Filtered, severity-ordered finding ids. Pagination happens HERE, not in Python:
        §9's 10k-finding budget is met by never materialising the other 9,900."""
        sql = ["SELECT finding_id, severity FROM findings WHERE scan_id=?"]
        args: list[Any] = [scan_id]
        for column, value in (("disposition", disposition), ("nature", nature),
                              ("availability", availability)):
            if value:
                sql.append(f"AND {column}=?")
                args.append(value)
        if module:
            sql.append("AND detector_id LIKE ?")
            args.append(f"{module}.%")
        sql.append("ORDER BY CASE severity")
        for sev, rank in _SEV_RANK.items():
            sql.append(f" WHEN '{sev}' THEN {rank}")
        sql.append(" ELSE 99 END, confidence DESC, finding_id")
        if limit is not None:
            sql.append("LIMIT ? OFFSET ?")
            args += [limit, offset]
        return [r["finding_id"] for r in self._conn.execute(" ".join(sql), args)]

    def count(self, scan_id: str, **filters: Any) -> int:
        sql = ["SELECT COUNT(*) AS n FROM findings WHERE scan_id=?"]
        args: list[Any] = [scan_id]
        for column in ("disposition", "nature", "availability"):
            if filters.get(column):
                sql.append(f"AND {column}=?")
                args.append(filters[column])
        if filters.get("module"):
            sql.append("AND detector_id LIKE ?")
            args.append(f"{filters['module']}.%")
        return int(self._conn.execute(" ".join(sql), args).fetchone()["n"])

    def modules(self, scan_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT detector_id FROM findings WHERE scan_id=?", (scan_id,))
        return sorted({str(r["detector_id"]).split(".")[0] for r in rows if r["detector_id"]})

    def close(self) -> None:
        self._conn.close()


def rebuild(path: Path, reports_dir: Path) -> ReportIndex:
    """§8: `index.db` corrupt or missing -> deleted and rebuilt. No data lost."""
    Path(path).unlink(missing_ok=True)
    idx = ReportIndex(path, reports_dir)
    idx.refresh()
    return idx


def severity_sorted(findings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(findings, key=lambda f: (_SEV_RANK.get(str(f.get("severity")), 99),
                                           -float(f.get("confidence", 0.0)),
                                           str(f.get("finding_id"))))


__all__ = ["INDEX_VERSION", "SEVERITY_ORDER", "ReportIndex", "ScanRow", "rebuild",
           "severity_sorted"]
