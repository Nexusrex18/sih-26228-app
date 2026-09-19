"""Module A — training-data integrity detectors (PS §2.2.1).

Importing this package registers every detector in ``cva.core.registry.DETECTOR_REGISTRY``.
"""
from . import registry  # noqa: F401  — imports every detector module (registration side effect)
from .registry import MODULE_A_DETECTORS  # noqa: F401
