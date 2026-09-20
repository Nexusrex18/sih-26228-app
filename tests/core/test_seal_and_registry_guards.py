"""Three guards that had no caller and no test, and were findings for exactly that reason.

  * item 13 — the recorded profile must match its own recorded hash, because the scan record
    seals that hash and `report.json` ships the profile beside it. A per-run value written
    back into the hashed dict made the tamper-evident trail attest something that was never
    produced.
  * item 14 — an `attack_class` not in `core/taxonomy.py` is a STARTUP ERROR (plan §9.5),
    not a warning. `taxonomy.require()` has always documented that the registry calls it;
    until this change nothing did, and the undeclared string was filtered out of both
    coverage rows and swept into `operational_reports`.
  * item 15 — a second class claiming a registered id used to win silently, while the plan
    row kept claiming the id was assessed.

Each test asserts the guard from the OUTSIDE (through `scan`/the decorators), so removing
the guard fails the test rather than only changing an internal.
"""
from __future__ import annotations

import json

import pytest

from cva.core.capability import Availability, Capability, CapabilitySet
from cva.core.orchestrator import profile_hash_of, scan
from cva.core.registry import DuplicateCheckId, register, register_detector
from cva.core.runcontext import RunContext
from cva.core.taxonomy import UnknownAttackClass
from cva.core.types import Finding, Severity


class FakeModel:
    model_id = "fake-001"
    fmt = "onnx"
    num_classes = 4

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(frozenset({Capability.MODEL_PREDICT}), ())


class RankingProbe:
    """Stands in for `model.intrinsic_probes`: emits the ranking the orchestrator harvests."""

    id = "model.intrinsic_probes"
    version = "0.0.1"
    requires: frozenset = frozenset()
    optional: frozenset = frozenset()
    attack_classes = frozenset({"weight_anomaly"})

    def check(self, model, ctx):
        return [Finding(detector_id=self.id, detector_version=self.version,
                        target_type="model", target_ref=model.model_id,
                        severity=Severity.INFO, confidence=0.0, reason="ranking",
                        attack_class="weight_anomaly",
                        produced_by="ranking=" + json.dumps([3, 1, 0, 2]),
                        availability=Availability.OK)]


class RankingReader:
    """Stands in for Neural Cleanse: records what the ranking actually arrived as."""

    id = "stub.reader"
    version = "0.0.1"
    requires: frozenset = frozenset()
    optional: frozenset = frozenset()
    attack_classes = frozenset({"backdoor_trigger"})
    seen: list = []

    def check(self, model, ctx):
        RankingReader.seen.append(ctx.run_state.get("nc_class_order"))
        return []


def test_item13_the_recorded_profile_matches_its_own_recorded_hash():
    RankingReader.seen = []
    res = scan(FakeModel(), RunContext(), "deep",
               registries=({RankingProbe.id: RankingProbe,
                            RankingReader.id: RankingReader}, {}))
    # The invariant the seal depends on: `:626` writes `result.profile_hash` into the scan
    # record while `report.json` ships `result.profile`. If these two disagree, the
    # tamper-evident trail attests a hash of a profile that does not exist.
    assert profile_hash_of(res.profile) == res.profile_hash
    assert "nc_class_order" not in res.profile
    # And the ranking still reaches the check that needs it — the fix must not be a deletion.
    assert RankingReader.seen == [[3, 1, 0, 2]]


def test_item13_a_check_cannot_reach_the_hashed_profile_through_the_context():
    """`ctx.profile` is still handed out (checks read thresholds from it), so the guarantee
    is that the orchestrator writes per-run data to `run_state`, not that the dict is frozen.
    Assert the two are distinct objects, or "write to run_state" means nothing."""
    seen: dict = {}

    class Peek(RankingReader):
        id = "stub.peek"

        def check(self, model, ctx):
            seen["profile"] = ctx.profile
            seen["run_state"] = ctx.run_state
            return []

    res = scan(FakeModel(), RunContext(), "deep", registries=({Peek.id: Peek}, {}))
    assert seen["run_state"] is not seen["profile"]
    assert profile_hash_of(res.profile) == res.profile_hash


def test_item14_an_undeclared_attack_class_is_a_startup_error():
    with pytest.raises(UnknownAttackClass) as exc:
        @register
        class Undeclared:
            id = "stub.undeclared"
            version = "0.0.1"
            requires: frozenset = frozenset()
            optional: frozenset = frozenset()
            attack_classes = frozenset({"not_in_taxonomy"})

            def check(self, model, ctx):
                return []

    assert "not_in_taxonomy" in str(exc.value)
    from cva.core.registry import REGISTRY
    assert "stub.undeclared" not in REGISTRY, "a rejected class must not be half-registered"


def test_item14_the_detector_decorator_checks_the_same_list():
    with pytest.raises(UnknownAttackClass):
        @register_detector
        class UndeclaredDetector:
            id = "stub.undeclared_detector"
            version = "0.0.1"
            requires: frozenset = frozenset()
            optional: frozenset = frozenset()
            attack_classes = frozenset({"label_flipping", "typo_flipping"})

            def detect(self, dataset, embeddings, model, ctx):
                return []


def test_item14_every_class_the_tree_registers_is_declared():
    """The import IS the startup check. If any shipped detector declares a string that is not
    in `core/taxonomy.py`, importing its registry module raises — which is the point."""
    import cva.detectors.data.registry  # noqa: F401
    import cva.detectors.model.registry  # noqa: F401
    from cva.core.registry import DETECTOR_REGISTRY, REGISTRY
    from cva.core.taxonomy import TAXONOMY
    for reg in (REGISTRY, DETECTOR_REGISTRY):
        for cid, cls in reg.items():
            for ac in cls.attack_classes:
                assert ac in TAXONOMY, f"{cid} declares undeclared attack_class {ac!r}"


def test_item15_a_second_class_may_not_claim_a_registered_id():
    @register
    class First:
        id = "stub.contested"
        version = "0.0.1"
        requires: frozenset = frozenset()
        optional: frozenset = frozenset()
        attack_classes = frozenset({"weight_anomaly"})

        def check(self, model, ctx):
            return []

    try:
        with pytest.raises(DuplicateCheckId) as exc:
            @register
            class Second:
                id = "stub.contested"
                version = "0.0.2"
                requires: frozenset = frozenset()
                optional: frozenset = frozenset()
                attack_classes = frozenset({"weight_anomaly"})

                def check(self, model, ctx):
                    return []

        assert "stub.contested" in str(exc.value)
        from cva.core.registry import REGISTRY
        assert REGISTRY["stub.contested"] is First, "the incumbent must survive"
    finally:
        from cva.core.registry import REGISTRY
        REGISTRY.pop("stub.contested", None)


def test_item15_re_registering_the_same_class_is_not_a_collision():
    """A module reimported under `importlib.reload` produces a new class object for the same
    source. That is the same check registering again, not two checks fighting over an id."""
    class Same:
        id = "stub.idempotent"
        version = "0.0.1"
        requires: frozenset = frozenset()
        optional: frozenset = frozenset()
        attack_classes = frozenset({"weight_anomaly"})

        def check(self, model, ctx):
            return []

    try:
        register(Same)
        register(Same)                       # the identical object
        clone = type("Same", (Same,), {})    # same module and qualname, new object
        clone.__qualname__ = Same.__qualname__
        register(clone)
    finally:
        from cva.core.registry import REGISTRY
        REGISTRY.pop("stub.idempotent", None)
