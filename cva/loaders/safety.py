"""S1 — hash and safety-gate the artefact BEFORE opening it.

Per the orchestrator control flow in backend_plan.md §6.3, this runs first, before any
loader touches the file. The order is the point: hash the bytes before opening, so the
ledger entry exists even if the load kills the process.

`torch.load` executes arbitrary code during unpickling and the model supplier is untrusted
by premise. CVE-2025-32434 (CVSS 9.3) — `weights_only=True` existed from 2.4 but did not
fully close the path until 2.6.0.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

TORCH_MIN = (2, 6, 0)
STANDARD_ONNX_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


@dataclass(frozen=True)
class SafetyReport:
    path: Path
    sha256: str
    size_bytes: int
    safe_to_load: bool
    reasons: tuple[str, ...]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_torch_version() -> None:
    import torch
    v = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:3])
    if v < TORCH_MIN:
        raise RuntimeError(
            f"torch {torch.__version__} < 2.6.0 — weights_only=True does not fully close "
            "CVE-2025-32434. Refusing to load an untrusted checkpoint.")


def prescan(path: Path, max_bytes: int = 4 << 30) -> SafetyReport:
    """Hash first, then decide whether opening is safe at all."""
    digest = sha256_file(path)
    size = path.stat().st_size
    reasons: list[str] = []
    safe = True

    if size > max_bytes:
        safe = False
        reasons.append(f"file is {size/1e9:.1f} GB, above the {max_bytes/1e9:.0f} GB cap")

    if path.suffix == ".onnx":
        try:
            import onnx
            m = onnx.load(str(path), load_external_data=False)
            custom = sorted({n.op_type for n in m.graph.node
                             if n.domain not in STANDARD_ONNX_DOMAINS})
            if custom:
                safe = False
                reasons.append(
                    f"non-standard ONNX operators {custom} — a custom op can load a shared "
                    "library at session-creation time")
        except Exception as exc:
            safe = False
            reasons.append(f"ONNX graph could not be parsed for a safety prescan: {exc}")

    if path.suffix in {".pt", ".pth"}:
        reasons.append("loaded with weights_only=True on torch>=2.6.0 (CVE-2025-32434)")

    return SafetyReport(path, digest, size, safe, tuple(reasons))
