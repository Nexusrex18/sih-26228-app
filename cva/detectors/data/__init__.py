"""Module A — training-data integrity detectors (PS §2.2.1).

Importing this package populates ``DATA_REGISTRY`` — Module A's OWN registry, deliberately NOT
``cva.detectors.base.REGISTRY``, which is Module B's and is walked with ``.check(model, ctx)``.
A ``.detect(dataset, embeddings, model)`` detector in that dict would break Module B's scan.
"""
from .base import DATA_REGISTRY  # noqa: F401
from . import registry  # noqa: F401  — imports every detector module (registration side effect)
