"""Deterministic keys, clocks and chains for the C3+ tests."""
from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

from cva.provenance.seal.chain import MemoryChain
from cva.provenance.seal.keys import EnvKeyProvider

from ._fixtures import BODIES, MANIFEST

SEED_A = bytes(range(32))
SEED_B = bytes(range(100, 132))


def provider(seed: bytes = SEED_A) -> EnvKeyProvider:
    return EnvKeyProvider("K", environ={"K": base64.b64encode(seed).decode()})


def clock():
    t = [datetime(2026, 9, 19, 2, 0, 0, tzinfo=UTC)]

    def tick() -> datetime:
        t[0] += timedelta(milliseconds=7)
        return t[0]
    return tick


def rng():
    n = [0]

    def draw(k: int) -> bytes:
        n[0] += 1
        return n[0].to_bytes(k, "big")
    return draw


def manifest_for(key, **kw) -> dict:
    return {**MANIFEST, "key_id": key.key_id, **kw}


def make_chain(n_after_genesis: int, seed: bytes = SEED_A, types=("inference",), **manifest_kw) -> MemoryChain:
    key = provider(seed)
    c = MemoryChain(key, clock=clock(), rng=rng())
    c.append("genesis", {"deployment_manifest": manifest_for(key, **manifest_kw)})
    for i in range(n_after_genesis):
        t = types[i % len(types)]
        c.append(t, BODIES[t])
    return c


def ledger_keys(key) -> dict[str, bytes]:
    return {key.key_id: key.public_key}
