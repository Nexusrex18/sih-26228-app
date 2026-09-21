"""Running `cva remediate` as an allowlisted subprocess (plan §7.5, S11).

The `Remediator` interface and the `cva remediate` command are Backend's
(`backend_plan.md` §5.13). This is the human workflow around them, and it must not overclaim
— including about its own availability. `cva remediate` is not in this build, so
`entry_point_available()` reports that as a first-class fact and the UI says so, rather than
offering a button that fails.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The UI never uses these words about a remediated artefact (plan §7.5, wording rule).
#: A clean re-scan is a statement about what was checked, not about the data.
FORBIDDEN_WORDS = ("cleaned", "clean data", "fixed", "safe", "sanitised", "sanitized",
                   "purged", "verified safe")


@dataclass
class Job:
    job_id: str
    scan_id: str
    target_type: str
    target_ref: str
    requested_by: str
    state: str = "queued"            # queued | running | done | failed | unavailable
    started_at: float = 0.0
    finished_at: float = 0.0
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    artefact_path: str = ""
    artefact_digest: str = ""
    manifest: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""

    @property
    def running(self) -> bool:
        return self.state in ("queued", "running")

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at) if self.started_at else 0.0


def _executable() -> list[str] | None:
    found = shutil.which("cva")
    if found:
        return [found]
    return [sys.executable, "-m", "cva.cli"]


def entry_point_available() -> tuple[bool, str]:
    """Is `cva remediate` a subcommand this build has?

    Asked by running `cva --help` and looking for the subcommand, not by importing anything:
    the answer has to be about the command the job would actually run.
    """
    argv = _executable()
    if argv is None:
        return False, "the `cva` command is not on PATH"
    try:
        proc = subprocess.run([*argv, "--help"], capture_output=True, text=True,
                              timeout=30, shell=False, check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"`cva --help` could not be run: {e}"
    if "remediate" in (proc.stdout + proc.stderr):
        return True, ""
    return False, (
        "`cva remediate` is not a subcommand of this build. Remediation is Backend's "
        "component (backend_plan.md §5.13, ADR-004's amendment); this dashboard is the "
        "workflow around it. Until the command exists, no remediation can be requested "
        "here — and this page says so rather than offering an action that would fail.")


class JobRunner:
    """In-process, one job at a time per request. Small on purpose: this is a local tool."""

    def __init__(self, out_dir: Path, timeout_s: int = 3600) -> None:
        self.out_dir = Path(out_dir)
        self.timeout_s = timeout_s
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, scan_id: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if scan_id:
            jobs = [j for j in jobs if j.scan_id == scan_id]
        return sorted(jobs, key=lambda j: j.started_at, reverse=True)

    def start(self, *, scan_id: str, target_type: str, target_ref: str,
              requested_by: str, dataset: Path) -> Job:
        job = Job(job_id=uuid.uuid4().hex, scan_id=scan_id, target_type=target_type,
                  target_ref=target_ref, requested_by=requested_by)
        available, why = entry_point_available()
        if not available:
            job.state = "unavailable"
            job.detail = why
            with self._lock:
                self._jobs[job.job_id] = job
            return job

        out = self.out_dir / f"remediated-{job.job_id[:12]}"
        argv = [*(_executable() or []), "remediate", "--scan-id", scan_id,
                "--dataset", str(dataset), "--exclude-target-type", target_type,
                "--exclude-target-ref", target_ref, "--out", str(out), "--json"]
        job.started_at = time.time()
        job.state = "running"
        with self._lock:
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job, argv), daemon=True).start()
        return job

    def _run(self, job: Job, argv: list[str]) -> None:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout_s, shell=False, check=False)
        except subprocess.TimeoutExpired:
            job.state, job.detail = "failed", f"timed out after {self.timeout_s} s"
            job.finished_at = time.time()
            return
        except OSError as e:
            job.state, job.detail = "failed", f"could not start: {e}"
            job.finished_at = time.time()
            return
        # Captured output is DISPLAYED ESCAPED and never interpreted (S11).
        job.stdout, job.stderr = proc.stdout[:100_000], proc.stderr[:100_000]
        job.returncode = proc.returncode
        job.finished_at = time.time()
        if proc.returncode != 0:
            job.state = "failed"
            job.detail = f"cva remediate exited {proc.returncode}"
            return
        try:
            payload = json.loads(proc.stdout)
            job.artefact_path = str(payload.get("artefact_path", ""))
            job.artefact_digest = str(payload.get("artefact_digest", ""))
            job.manifest = list(payload.get("manifest") or [])
        except (json.JSONDecodeError, AttributeError):
            job.detail = "cva remediate succeeded but its output could not be parsed"
        job.state = "done"


def contains_forbidden_wording(text: str) -> list[str]:
    """Used by a test, and by anything that renders remediation prose.

    The rule is not decorative. `cva remediate` produces a NEW artefact with its own digest
    that must be RE-SCANNED to be assessed (ADR-004: "its output is re-assessed rather than
    trusted"). Saying "cleaned" claims the opposite of what the tool did.
    """
    low = text.lower()
    return [w for w in FORBIDDEN_WORDS if w in low]


__all__ = ["FORBIDDEN_WORDS", "Job", "JobRunner", "contains_forbidden_wording",
           "entry_point_available"]
