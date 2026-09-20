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
    profile: dict[str, Any] = field(default_factory=dict)
    scan_id: str = ""
    out_dir: Any = None
    rng_seed: int = 0
    # Per-run data one check derives for another — the intrinsic-probe class ranking that
    # Neural Cleanse and universal_margin consume. It is NOT profile content: `profile` is
    # hashed into `profile_hash` before the first check runs, and writing a derived value
    # back into it made the recorded profile disagree with its own recorded hash, so the
    # tamper-evident scan record attested a hash of something that was never shipped.
    # Anything a check derives at run time belongs here, where nothing hashes it.
    run_state: dict[str, Any] = field(default_factory=dict)
    # Typed reference embeddings for the known-clean set (`--reference-dataset`), or None.
    # `REFERENCE_CLEAN_SET` is the model-side probe arrays and stays that; this is the data
    # side, and `data.ood` reads it here rather than smuggling it through `profile`.
    reference_embeddings: Any = None

    def opt(self, key: str, default: Any) -> Any:
        return self.profile.get(key, default)


