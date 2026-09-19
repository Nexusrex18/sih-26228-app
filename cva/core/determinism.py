"""Reproducibility harness — backend_plan.md §2.9.

Three seeds, `PYTHONHASHSEED` read FROM THE ENVIRONMENT, deterministic torch algorithms, and
ONNX Runtime pinned to one thread with sequential execution.

`PYTHONHASHSEED` cannot be set from inside a running interpreter: hash randomisation is
fixed at start-up, so assigning `os.environ["PYTHONHASHSEED"]` afterwards changes nothing
for this process and would only make a report claim something untrue. `ensure_hashseed`
therefore re-executes the interpreter with the variable set.

`CUBLAS_WORKSPACE_CONFIG` is deliberately absent: it is a GPU-overlay setting and has no
place in the CPU profile.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any

ENV_FLAG = "CVA_DETERMINISTIC"


def armed() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def hashseed() -> str | None:
    return os.environ.get("PYTHONHASHSEED")


def ensure_hashseed(seed: int) -> None:
    """Re-exec once with PYTHONHASHSEED=<seed>. A no-op when it is already set."""
    if os.environ.get("PYTHONHASHSEED") is not None:
        return
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


def arm(seed: int) -> None:
    """Seed everything a scan touches and pin the sources of run-to-run drift."""
    os.environ[ENV_FLAG] = "1"
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def ort_session_options() -> Any:
    """`SessionOptions` for a verification path; `None` when the harness is not armed, so
    unarmed scans keep ONNX Runtime's own threading and are not slowed for nothing."""
    if not armed():
        return None
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def recorded_seeds(scan_seed: int) -> dict[str, int]:
    """What the report may truthfully claim was seeded."""
    seeds = {"scan": scan_seed}
    if armed():
        seeds.update({"python": scan_seed, "numpy": scan_seed % (2**32), "torch": scan_seed})
    return seeds
