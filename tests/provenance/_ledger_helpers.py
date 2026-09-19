"""Builders for ledger / Sealer tests: deterministic clock, rng and keys, one shared per fixture."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cva.provenance.seal.keys import TrustKey, TrustRoot
from cva.provenance.seal.records import genesis_prev_hash
from cva.provenance.seal.sealer import Sealer, SealPolicy
from cva.provenance.seal.store import SealedLedger

from ._chain_helpers import SEED_A, clock, provider, rng

MANIFEST = {"device_id": "jetson-07", "unit": "alpha-coy", "profile_hash": "7" * 64}
MODEL = {"id": "resnet50-v3", "weights_sha256": "ab" * 32, "arch_hash": "cd" * 32, "format": "onnx"}
CONFIG = {"preprocess_spec": {"mean_e6": [485000, 456000, 406000], "std_e6": [229000, 224000, 225000]},
          "postprocess_spec": {"conf_thr_e6": 250000, "nms_iou_e6": 450000},
          "runtime": "onnxruntime 1.17.1 / CPUExecutionProvider", "version_pins_hash": "ef" * 32,
          "code_commit": "0" * 40}
CLASSIFY = {"task": "classify", "top": [{"cls": 7, "conf": 0.993118}]}
DETECT_RAW = {"task": "detect", "detections": [
    {"cls": 3, "conf": 0.871204, "box": [18.8125, 13.75, 51.875, 37.671875]},
    {"cls": 1, "conf": 0.31, "box": [100.0, 100.0, 140.5, 160.25]},
    {"cls": 1, "conf": 0.12, "box": [5.0, 5.0, 9.0, 9.0]}]}
DETECT_FILTERED = {"task": "detect", "detections": DETECT_RAW["detections"][:2],
                   "filter": {"conf_thr": 0.25, "nms_iou": 0.45}}


class Env:
    """A ready ledger + Sealer in `tmp_path`, plus everything a test needs to poke at it."""

    def __init__(self, tmp: Path, *, durability: str = "per_record", checkpoint_every: int = 1000,
                 policy: SealPolicy | None = None, seed: bytes = SEED_A, background_flush: bool = True,
                 **manifest_kw: Any) -> None:
        self.tmp = tmp
        self.key = provider(seed)
        self.clock, self.rng = clock(), rng()
        self.ledger_path = tmp / "ledger.db"
        led = SealedLedger.init_ledger(self.ledger_path, self.key, {**MANIFEST, "checkpoint_every": checkpoint_every,
                                                                    **manifest_kw}, durability=durability,
                                       group_n=100, group_ms=50, clock=self.clock, rng=self.rng)
        self.trust = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(self.key.key_id, self.key.public_key, "ledger"),))
        led.close()
        self.policy = policy
        self.background_flush = background_flush
        self.sealer = self.open()
        self.model = self.sealer.register_model(**MODEL)
        self.config = self.sealer.register_config(**CONFIG)

    def open(self) -> Sealer:
        return Sealer.open(self.ledger_path, key=self.key, trust_root=self.trust,
                           policy=self.policy, clock=self.clock, rng=self.rng, background_flush=self.background_flush)

    def reopen(self) -> Sealer:
        self.sealer.close()
        self.sealer = self.open()
        self.model = self.sealer.register_model(**MODEL)
        self.config = self.sealer.register_config(**CONFIG)
        return self.sealer

    def seal(self, i: int = 0, **kw: Any) -> Any:
        buf = kw.pop("buf", bytes([i % 256]) * 512 + i.to_bytes(4, "big"))
        return self.sealer.seal(buf, self.model, self.config, output=kw.pop("output", CLASSIFY),
                                dims=kw.pop("dims", (64, 64)), **kw)

    def keys(self) -> dict[str, bytes]:
        return {self.key.key_id: self.key.public_key}

    def close(self) -> None:
        self.sealer.close()


# --- an in-memory, fully valid chain for verifier tests (genesis -> registration -> inferences) ---------------

def valid_chain(n_inferences: int = 8, *, seed: bytes = SEED_A, checkpoint_every: int = 1000, rng_fn=None,
                extra: list[tuple[str, dict]] | None = None):
    """Returns (MemoryChain, TrustRoot, key). `extra` records are appended after the inferences."""
    from cva.provenance.seal.chain import MemoryChain

    from ._chain_helpers import manifest_for
    from ._fixtures import BODIES
    key = provider(seed)
    chain = MemoryChain(key, clock=clock(), rng=rng_fn or rng())
    chain.append("genesis", {"deployment_manifest": manifest_for(key, checkpoint_every=checkpoint_every)})
    chain.append("model_registration", BODIES["model_registration"])
    for _ in range(n_inferences):
        chain.append("inference", BODIES["inference"])
    for rtype, body in extra or []:
        chain.append(rtype, body)
    trust = TrustRoot(genesis_prev_hash(chain.records[0]["deployment_manifest"]),
                      (TrustKey(key.key_id, key.public_key, "ledger"),))
    return chain, trust, key


def export_bytes(chain) -> bytes:
    return b"".join(x + b"\n" for x in chain.stored)
