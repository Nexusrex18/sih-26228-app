"""Fixtures for Module A: a synthetic clean dataset, four synthetic contributors, and the stub
embedding backbone. Session-scoped — generating images is the slow part."""
from __future__ import annotations

import dataclasses

import pytest

from attacklab.contributor_metadata import assign_contributors
from attacklab.synth_dataset import make_clean_dataset
from cva.core.capability import Availability, CapabilitySet, Capability
from cva.detectors.data._stub_types import stub_embeddings

N_CLEAN = 240


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("module_a")


@pytest.fixture(scope="session")
def clean(workdir):
    """240 images, 4 classes, 4 contributors (40/30/20/10), sidecar-attributed."""
    ds = make_clean_dataset(workdir / "clean", seed=11, n=N_CLEAN, n_classes=4)
    ds, _ = assign_contributors(ds, workdir / "clean", seed=11)
    return ds


@pytest.fixture(scope="session")
def clean_emb(clean):
    return stub_embeddings(clean)


def resolve(detector_cls, dataset, extra=()):
    """What the (Backend-owned) orchestrator does before calling ``detect()``: resolve
    requires/optional against the capability set. Test-only — Module A never writes this."""
    caps = CapabilitySet(frozenset(dataset.capabilities()) | frozenset(extra))
    return caps.resolve(set(detector_cls.requires), set(detector_cls.optional))


def only(findings, availability=Availability.OK):
    return [f for f in findings if f.availability == availability]


# ---- trigger / model fixtures (slow: each trains a tiny CNN once per session) -------------
@pytest.fixture(scope="session")
def poisoned(workdir, clean):
    from _triggers import inject_trigger
    ds, trig = inject_trigger(clean, workdir / "trig", seed=1, n=40, target_class=0, contributor="B")
    return ds, trig


@pytest.fixture(scope="session")
def poisoned_model(poisoned):
    from _triggers import train_handle
    return train_handle(poisoned[0], seed=0, model_id="poisoned")


@pytest.fixture(scope="session")
def clean_model(clean):
    from _triggers import train_handle
    return train_handle(clean, seed=0, model_id="clean")
