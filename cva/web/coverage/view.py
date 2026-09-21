"""Presenting the GENERATED coverage statement (plan §7.6).

Three groups, never two: covered by a check that ran / covered by one that was UNAVAILABLE
or DEGRADED, with the reason / in the taxonomy and covered by nothing. The third group is
the one that makes the statement honest, and it is never collapsed into the second.

Nothing here computes a coverage row. Backend's generator walks the registry; this reshapes
what it produced into something a reader can hold in their head (D-E2).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoverageRow:
    attack_class: str
    checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageGroups:
    assessed: tuple[CoverageRow, ...] = ()
    not_assessed: tuple[CoverageRow, ...] = ()
    never_covered: tuple[str, ...] = ()
    operational_reports: tuple[str, ...] = ()
    standing_limitations: tuple[str, ...] = ()

    @property
    def total_attack_classes(self) -> int:
        return len(self.assessed) + len(self.not_assessed) + len(self.never_covered)

    @property
    def assessed_fraction(self) -> float:
        total = self.total_attack_classes
        return len(self.assessed) / total if total else 0.0


def coverage_groups(coverage: Mapping[str, Any]) -> CoverageGroups:
    def rows(section: Any) -> tuple[CoverageRow, ...]:
        if not isinstance(section, Mapping):
            return ()
        return tuple(CoverageRow(str(k), tuple(str(c) for c in (v or [])))
                     for k, v in sorted(section.items()))
    return CoverageGroups(
        assessed=rows(coverage.get("assessed")),
        not_assessed=rows(coverage.get("not_assessed")),
        never_covered=tuple(sorted(str(c) for c in (coverage.get("never_covered") or []))),
        operational_reports=tuple(sorted(
            str(c) for c in (coverage.get("operational_reports") or []))),
        standing_limitations=tuple(str(s) for s in
                                   (coverage.get("standing_limitations") or [])))


@dataclass(frozen=True)
class CoverageDiff:
    """Two scans side by side. The point is that coverage is PER SCAN — a smaller statement
    on a black-box scan is correct, and the reader should be able to see exactly why."""

    only_left: tuple[str, ...] = ()
    only_right: tuple[str, ...] = ()
    both: tuple[str, ...] = ()
    lost_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def shrank(self) -> bool:
        return bool(self.only_left)


def compare_coverage(left: Mapping[str, Any], right: Mapping[str, Any]) -> CoverageDiff:
    la = set((left.get("assessed") or {}))
    ra = set((right.get("assessed") or {}))
    right_not = right.get("not_assessed") or {}
    return CoverageDiff(
        only_left=tuple(sorted(la - ra)),
        only_right=tuple(sorted(ra - la)),
        both=tuple(sorted(la & ra)),
        lost_reasons={cls: tuple(str(c) for c in (right_not.get(cls) or []))
                      for cls in sorted(la - ra)})


__all__ = ["CoverageDiff", "CoverageGroups", "CoverageRow", "compare_coverage",
           "coverage_groups"]
