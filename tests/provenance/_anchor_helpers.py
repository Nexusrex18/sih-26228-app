"""A real ledger with a full anchoring setup: ledger key, a witness, a boundary device, a trust root that lists them."""
from __future__ import annotations

from pathlib import Path

from cva.provenance.seal.keys import TrustKey, TrustRoot
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import export_records

from ._chain_helpers import provider
from ._ledger_helpers import Env

WITNESS_SEED = bytes(range(200, 232))
BOUNDARY_SEED = bytes(range(150, 182))
NEW_SEED = bytes(range(50, 82))
EVIL_SEED = bytes(range(90, 122))


class AnchorEnv(Env):
    def __init__(self, tmp: Path, **kw):
        super().__init__(tmp, **kw)
        self.witness, self.boundary = provider(WITNESS_SEED), provider(BOUNDARY_SEED)
        self.trust = TrustRoot(self.trust.deployment_manifest_hash, (
            *self.trust.keys, TrustKey(self.witness.key_id, self.witness.public_key, "witness"),
            TrustKey(self.boundary.key_id, self.boundary.public_key, "boundary")))

    def anchor(self, name: str = "a0.json", **kw):
        """Flush, then export an anchor through a second handle on the ledger (as `cva-seal anchor export` would)."""
        from cva.provenance.seal.anchor import export_anchor
        self.sealer.flush() if hasattr(self.sealer, "flush") else None
        led = SealedLedger.open(self.ledger_path, key=self.key, clock=self.clock, rng=self.rng, background_flush=False)
        try:
            return export_anchor(led, self.tmp / name, **kw)
        finally:
            led.close()

    def export(self, name: str = "x.jsonl") -> Path:
        self.sealer.close()
        out = self.tmp / name
        export_records(self.ledger_path, out)
        return out
