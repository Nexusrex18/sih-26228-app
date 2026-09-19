"""CheckContext — what core hands a plug-in.

It lives in core/, not detectors/, because core CONSTRUCTS it. Defining it in detectors/
is what forced core to import detectors and break CI invariant 2 — caught by
tests/boundaries/test_import_invariants.py the first time that test ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cva.core.model import ModelBattery


@dataclass
class CheckContext:
    probes_x: Any = None                    # np.ndarray, REFERENCE_CLEAN_SET
    probes_y: Any = None
    suspect_x: Any = None
    battery: ModelBattery | None = None     # REFERENCE_MODEL_BATTERY / REFERENCE_MANIFEST
    profile: dict = field(default_factory=dict)
    scan_id: str = ""
    out_dir: Any = None
    rng_seed: int = 0

    def opt(self, key: str, default: Any) -> Any:
        return self.profile.get(key, default)


