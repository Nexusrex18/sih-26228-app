from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cva.core.capability import Capability, CapabilitySet


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_weights(weights: dict[str, np.ndarray]) -> str:
    """Order-independent digest over tensor bytes; stable across save formats."""
    h = hashlib.sha256()
    for name in sorted(weights):
        h.update(name.encode())
        h.update(np.ascontiguousarray(weights[name], dtype=np.float32).tobytes())
    return h.hexdigest()


def arch_hash_of(tokens: Sequence[str]) -> str:
    """§9.2 — the STRUCTURE digest Module C's seal binds, owned by the loaders.

    It is deliberately not `weight_digest`'s sibling but its complement. `weight_digest`
    answers "are these the same numbers?"; this answers "is this the same shape of model?",
    and the substitution check needs both to tell a re-export apart from a swap. A digest
    mixing the two could only ever say "something changed", which is the one answer that
    triggers an investigation without narrowing it.

    So the tokens must carry NO parameter VALUES and no name that a re-export can
    legitimately rewrite — ONNX renames intermediate tensors on every export, and a
    structure hash that moved when a name moved would report a substitution for a file
    round-tripped through the same exporter twice.
    """
    h = hashlib.sha256()
    for tok in tokens:
        h.update(tok.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ProbeLog:
    """Records what a capability probe actually attempted, so the report can say why."""

    def __init__(self) -> None:
        self.caps: set[Capability] = set()
        self.notes: list[tuple[Capability, str]] = []

    def ok(self, cap: Capability) -> None:
        self.caps.add(cap)

    def absent(self, cap: Capability, why: str) -> None:
        self.notes.append((cap, why))

    def result(self) -> CapabilitySet:
        return CapabilitySet(frozenset(self.caps), tuple(self.notes))
