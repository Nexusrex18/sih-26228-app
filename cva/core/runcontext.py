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

    def capabilities(self) -> CapabilitySet:
        caps, notes = set(), []
        if self.probes_x is not None and len(self.probes_x):
            caps.add(Capability.REFERENCE_CLEAN_SET)
        else:
            notes.append((Capability.REFERENCE_CLEAN_SET, "no clean probe set supplied"))
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
        return CapabilitySet(frozenset(caps), tuple(notes))


