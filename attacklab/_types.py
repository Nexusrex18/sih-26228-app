"""The real core types the attack lab builds.

``Dataset`` is a CONTRACT-ONLY dataset: it exposes exactly the frozen ``Dataset`` Protocol in
``cva/core/types.py`` (``samples``, ``categories``, ``annotations()``, ``capabilities()``) and nothing
else, delegating the two methods to the loaders' real ``InMemoryDataset`` so the behaviour is the real one.

It deliberately does NOT offer ``sample(id)``, ``category_name(id)`` or ``__len__``. The concrete
``InMemoryDataset`` gained those conveniences after Module A was first written against a stand-in that
had them, and five of six detectors crashed on a real dataset because of it. The Protocol — the frozen
contract — still does not promise them, so any test dataset that offered them could hide a detector
that depends on them. Everything built or scanned in tests therefore exercises the contract, not a
convenience of one implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cva.core.types import Category, ContributorSource, Label, Sample  # noqa: F401
from cva.loaders.datasets.base import InMemoryDataset


@dataclass
class Dataset:
    samples: list[Sample] = field(default_factory=list)
    categories: list[Category] = field(default_factory=list)
    root: Path | None = None
    notes: list = field(default_factory=list)

    def _real(self) -> InMemoryDataset:
        return InMemoryDataset(samples=self.samples, categories=self.categories,
                               root=self.root, notes=self.notes)

    def annotations(self):
        return self._real().annotations()

    def capabilities(self):
        return self._real().capabilities()


def strict(ds: Any) -> Dataset:
    """Re-wrap any dataset (e.g. what a real loader returned) as the contract-only type."""
    return Dataset(list(ds.samples), list(ds.categories), getattr(ds, "root", None),
                   list(getattr(ds, "notes", [])))
