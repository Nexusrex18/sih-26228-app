"""The reviewed half of the coverage statement (`docs/coverage-standing.yaml`).

Backend's generator walks the registry and can only ever state what the registry supports.
That is the point of it, and it is also its limit: no registry walk discovers that the
signing key could have been held by an adversary from the start, or that contributor
aggregation has only been demonstrated on synthetic assignment. Those are human statements,
they live in one small reviewed file, and this module merges them into the generated
statement so a report carries both halves or says which half is missing.

Two rules shape the implementation:

  * **The file is never required, and its absence is never silent.** A bundle assembled
    without it still produces a report; that report says the reviewed half was not found,
    rather than printing a shorter statement that reads like a cleaner one.
  * **Nothing here can subtract.** The merge only appends. A hand-edited file cannot delete
    a generated limitation, so no edit to this file can make a scan look better than the
    registry says it is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Where the reviewed file lives in a source checkout. `parents[2]` is the repo root
#: (`cva/report/standing.py` -> `cva/report` -> `cva` -> root).
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "docs" / "coverage-standing.yaml"

#: An operator running from an installed wheel, or a test, points at another file with this.
ENV_VAR = "CVA_COVERAGE_STANDING"

MISSING = ("The reviewed half of the coverage statement (docs/coverage-standing.yaml) was "
           "not found for this build, so only the generated half is printed above. What a "
           "human would have declared about this system's assumptions is absent, which is "
           "not the same as there being nothing to declare.")

UNREADABLE = ("The reviewed half of the coverage statement (docs/coverage-standing.yaml) "
              "could not be read: {error}. Only the generated half is printed above.")


@dataclass(frozen=True)
class StandingFile:
    """What the reviewed file says, plus whether it was there at all."""

    found: bool
    path: Path
    invariant: str = ""
    limitations: tuple[str, ...] = ()
    reviewed_by: tuple[str, ...] = ()
    reviewed_at: str = ""
    error: str = ""
    ids: tuple[str, ...] = field(default=())

    def lines(self) -> list[str]:
        """The strings a report prints, in the order it prints them.

        The invariant leads because everything else is a special case of it. The review
        line is last and is printed even when the reviewers are unnamed: a standing
        statement nobody has signed is a fact about the statement.
        """
        if not self.found:
            return [UNREADABLE.format(error=self.error) if self.error else MISSING]
        out = [self.invariant] if self.invariant else []
        out += list(self.limitations)
        out.append(self._review_line())
        return out

    def _review_line(self) -> str:
        who = ", ".join(self.reviewed_by) if self.reviewed_by else "nobody recorded"
        when = self.reviewed_at or "no date recorded"
        return (f"The standing limitations above come from {self.path.name}, last reviewed "
                f"by {who} on {when}. They are reviewed by hand; the rest of this coverage "
                f"statement is generated from the detector registry.")


def resolve_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else DEFAULT_PATH


def load(path: Path | str | None = None) -> StandingFile:
    """Read the reviewed file. Never raises: a report is produced either way."""
    target = resolve_path(path)
    if not target.is_file():
        return StandingFile(found=False, path=target)
    try:
        import yaml

        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as e:  # unreadable, malformed, or PyYAML absent
        return StandingFile(found=False, path=target, error=f"{type(e).__name__}: {e}")
    if not isinstance(data, dict):
        return StandingFile(found=False, path=target,
                            error="the file does not contain a YAML mapping")

    entries = data.get("limitations") or []
    texts: list[str] = []
    ids: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict):
            text = str(entry.get("text") or "").strip()
            entry_id = str(entry.get("id") or "").strip()
        else:
            text, entry_id = str(entry).strip(), ""
        if text:
            texts.append(_flatten(text))
            ids.append(entry_id)

    reviewers = data.get("reviewed_by") or []
    if isinstance(reviewers, str):
        reviewers = [reviewers]

    return StandingFile(
        found=True,
        path=target,
        invariant=_flatten(str(data.get("invariant") or "").strip()),
        limitations=tuple(texts),
        reviewed_by=tuple(str(r) for r in reviewers),
        reviewed_at=str(data.get("reviewed_at") or ""),
        ids=tuple(ids),
    )


def _flatten(text: str) -> str:
    """YAML folded scalars keep their trailing newline and any hard-wrapped indentation.

    Report consumers print a limitation as one paragraph — the Jinja view, the React view
    and `coverage.md` all do — so the wrapping the file needs to stay reviewable must not
    survive into the output.
    """
    return " ".join(text.split())


__all__ = ["DEFAULT_PATH", "ENV_VAR", "MISSING", "StandingFile", "load", "resolve_path"]
