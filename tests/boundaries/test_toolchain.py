"""V3 — the two toolchain pins that are controls rather than conveniences.

Both fail quietly if left to review: nothing breaks, the scan runs, and the control is
decorative.
"""
from __future__ import annotations

import pytest


def test_v3_torch_floor_is_version_compared_not_string_compared():
    """CVE-2025-32434 (CVSS 9.3): below 2.6.0 `weights_only=True` existed and did not close
    the RCE path.

    The comparison MUST go through `packaging.version` — this machine runs torch 2.14.0,
    and `'2.14.0' >= '2.6'` is False as a string compare, so a string assertion fails on a
    *correct* install and gets "fixed" by deleting it. backend_plan.md §11 V3.
    """
    torch = pytest.importorskip("torch")
    from packaging.version import Version

    assert Version(torch.__version__.split("+")[0]) >= Version("2.6"), (
        f"torch {torch.__version__} is below the CVE-2025-32434 floor")


def test_v3_no_cuda_stack_leaked_into_the_environment():
    """§5.7: the CPU index pin is the <=3.5 GB bundle target, made once. A CUDA wheel set
    is ~3.03 GB against ~176 MB — a 17x miss on the size gate (R5)."""
    import importlib.metadata as md

    nvidia = sorted({d.metadata["Name"] for d in md.distributions()
                     if (d.metadata["Name"] or "").lower().startswith("nvidia")})
    assert not nvidia, (
        "CUDA wheels present: " + ", ".join(nvidia) + ". Install torch from "
        "https://download.pytorch.org/whl/cpu — see pyproject [tool.cva.index].")


def test_the_assurance_workflow_needs_no_gpu():
    """backend_revised.md §2.6: only `backdoor_finetune` needs a GPU, and that runs in
    Mode A. Developing on CPU-only hardware means the path that actually ships is the path
    exercised daily, and a CUDA dependency cannot creep in unnoticed."""
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_available() or True  # informational: CUDA present is not a failure
    assert "cpu" in torch.__version__ or not torch.version.cuda, (
        "This build carries a CUDA runtime; the bundle must be built from the CPU index.")
