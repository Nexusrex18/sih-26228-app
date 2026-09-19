"""The real core types the attack lab builds — no stand-ins.

``Dataset`` here is the loaders' concrete ``InMemoryDataset``: exactly what ``COCOLoader`` /
``YOLOLoader`` return, and deliberately WITHOUT ``sample()``, ``category_name()`` or ``__len__``.
Anything built or scanned in tests therefore exercises the frozen contract, not conveniences of a
test double.
"""
from cva.core.types import Category, ContributorSource, Label, Sample  # noqa: F401
from cva.loaders.datasets.base import InMemoryDataset as Dataset  # noqa: F401
