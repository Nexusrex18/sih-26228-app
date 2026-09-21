"""RunContext — the third source of the CapabilitySet.

The capability enum spans three sources and no single object can answer for all of them:
a ModelHandle cannot know whether a reference clean set was supplied or a signing key
exists. Probing here is active like everywhere else — a RunContext reporting
REFERENCE_CLEAN_SET must have actually opened it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cva.core.capability import Capability, CapabilitySet
from cva.core.ledger import (
    AuditLedger,
    InferenceLedgerSource,
    NullAuditLedger,
    NullInferenceLedger,
)
from cva.core.model import ModelBattery


@dataclass
class RunContext:
    probes_x: np.ndarray | None = None
    probes_y: np.ndarray | None = None
    suspect_x: np.ndarray | None = None
    battery: ModelBattery | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    out_dir: Path | None = None
    seed: int = 0
    # NOT Optional. The Null stubs ARE the honest default — each reports no capability,
    # so `SIGNING_KEY` and `INFERENCE_LEDGER` are absent from the scan's CapabilitySet and
    # every consumer sees the truth. Typing them `| None` would push a `not None` guard
    # onto every call site including the orchestrator's last step, for a state that must
    # never exist.
    audit_ledger: AuditLedger = field(default_factory=NullAuditLedger)
    inference_ledger: InferenceLedgerSource = field(default_factory=NullInferenceLedger)
    code_commit: str = "unknown"
    dataset: Any = None
    # A second, operator-supplied dataset known to be clean. It does two jobs, and the second
    # one was missing until item 4: B7's absolute-rate baseline (the orchestrator runs the
    # data detectors over it and hands the risk engine only a flag count), AND the reference
    # the data-side checks compare against.
    #
    # It GRANTS `REFERENCE_CLEAN_SET`. `backend_plan.md:904` types that capability
    # `reference_probes: Dataset | None  # REFERENCE_CLEAN_SET — probe IMAGES`, and §12
    # (:1685) says `data.ood` and every `drift.distribution` test resolve UNAVAILABLE for
    # want of it — both of which are dataset-side checks that no array of model probes could
    # ever satisfy. An earlier comment here asserted the capability "means `probes_x`, a
    # model-side array"; that was the code contradicting Plan, and Plan wins. Granting from
    # either source is what makes `data.ood`'s declared `requires` true instead of a row that
    # resolves OK and then returns `not_performed` on every real scan.
    reference_dataset: Any = None
    # A fitted `CalibrationSet` from the benchmark, or None. A scan has no labels of its own,
    # so with None the report says `calibration: null` rather than inventing a curve.
    calibration: Any = None

    def capabilities(self) -> CapabilitySet:
        caps, notes = set(), []
        # Two sources, one capability: a model-side probe ARRAY and a data-side reference
        # DATASET. A check declaring `REFERENCE_CLEAN_SET` must therefore still say which
        # form it needs when only the other one is present — see the guards in
        # `model.neural_cleanse` and `model.anomalous`, which return an unavailable row
        # rather than crashing on a probe array that was never supplied.
        if self.probes_x is not None and len(self.probes_x):
            caps.add(Capability.REFERENCE_CLEAN_SET)
        elif getattr(self.reference_dataset, "samples", None):
            caps.add(Capability.REFERENCE_CLEAN_SET)
        else:
            notes.append((Capability.REFERENCE_CLEAN_SET,
                          "no clean probe set supplied: neither a probe array nor a "
                          "--reference-dataset"))
        if self.suspect_x is not None and len(self.suspect_x):
            caps.add(Capability.SUSPECT_INPUTS)
        else:
            notes.append((Capability.SUSPECT_INPUTS,
                          "no suspect input set supplied — only known-clean probes"))
        if self.battery and self.battery.models:
            caps.add(Capability.REFERENCE_MODEL_BATTERY)
        else:
            notes.append((Capability.REFERENCE_MODEL_BATTERY,
                          "no reference model battery supplied — organisers provide none, "
                          "so this is the expected case"))
        if self.battery and self.battery.manifest:
            caps.add(Capability.REFERENCE_MANIFEST)
        else:
            notes.append((Capability.REFERENCE_MANIFEST,
                          "no manifest registered for this model"))

        # Delegated, never assumed: the ledger objects probe themselves by attempting the
        # operation, so a Null stub reports {} and a real one reports only what it opened.
        for ledger, cap, why in (
            (self.audit_ledger, Capability.SIGNING_KEY,
             "no audit ledger with a signing key — the scan record is not sealed"),
            (self.inference_ledger, Capability.INFERENCE_LEDGER,
             "no inference ledger supplied — prov.* checks resolve UNAVAILABLE"),
        ):
            if cap in set(ledger.capabilities()):
                caps.add(cap)
            else:
                notes.append((cap, why))

        if self.dataset is not None:
            ds = self.dataset.capabilities()
            caps |= set(ds.caps)
            notes.extend(ds.notes)
        return CapabilitySet(frozenset(caps), tuple(notes))


