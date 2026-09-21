"""Test helpers. A proper package module (not ``conftest``) so it can never collide with Module B's
``tests/detectors/model/conftest.py``."""
from __future__ import annotations

import numpy as np

from cva.core.capability import Availability, CapabilitySet
from cva.core.context import CheckContext


def make_ctx(params: dict | None = None, detector_id: str | None = None, out_dir=None, seed: int = 0,
             scan_id: str = "test-scan", profile: dict | None = None) -> CheckContext:
    prof = dict(profile or {})
    if params is not None and detector_id is not None:
        prof[detector_id] = params
    return CheckContext(profile=prof, out_dir=out_dir, scan_id=scan_id, rng_seed=seed)


def detect(cls, dataset, embeddings=None, model=None, params=None, out_dir=None, seed=0, profile=None):
    """What the orchestrator does: ZERO-ARG construction, then ``detect(dataset, embeddings, model,
    ctx)`` — every setting arrives through ``ctx``."""
    ctx = make_ctx(params, cls.id, out_dir, seed, profile=profile)
    return cls().detect(dataset, embeddings, model, ctx)


def resolve(detector_cls, dataset, extra=()):
    """The (Backend-owned) capability resolution done before ``detect()``. Test-only."""
    caps = CapabilitySet.union(dataset.capabilities(), CapabilitySet(frozenset(extra)))
    return caps.resolve(set(detector_cls.requires), set(detector_cls.optional))


def only(findings, availability=Availability.OK):
    return [f for f in findings if f.availability == availability]


def reference_matrix(ref_dataset, ref_embeddings) -> np.ndarray:
    """The ``(M, d)`` array ``data.ood`` reads from ``ctx.profile['reference_embeddings']``."""
    return ref_embeddings.vectors([s.sample_id for s in ref_dataset.samples])
