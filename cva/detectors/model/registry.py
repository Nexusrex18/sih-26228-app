"""Explicit registration for Module B.

ADR-008 rejects dynamic discovery as over-engineering for this team size, so registration
is explicit and in one place. Importing this module is what puts Module B's checks into
the registries; `cva/core/` must never import it (CI invariant 2).
"""
from cva.detectors.base import DETECTOR_REGISTRY, REGISTRY

from . import (  # noqa: F401
               anomalous,
               data_consistency,
               fingerprint,
               graph_structure,
               intrinsic_probes,
               neural_cleanse,
               strip,
               universal_margin,
               weight_digest,
               weight_stats,
)

__all__ = ["REGISTRY", "DETECTOR_REGISTRY", "MODULE_B_CHECKS", "MODULE_B_DETECTORS"]

MODULE_B_CHECKS = [
    "model.weight_digest", "model.fingerprint", "model.strip", "model.weight_stats",
    "model.neural_cleanse", "model.intrinsic_probes", "model.graph_structure",
    "model.anomalous",
    # Beyond the ratified plan: added after measurement showed model.strip cannot answer
    # the model-level question from clean probes alone. See NOTES.md.
    "model.universal_margin",
]
MODULE_B_DETECTORS = ["model.data_consistency"]
