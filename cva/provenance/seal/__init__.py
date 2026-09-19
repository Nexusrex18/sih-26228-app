"""Standalone seal SDK (Mode C). Allowed imports: stdlib, `cryptography`, `rfc8785`, itself.

Public API (plan §6). Gate C5 adds the standalone verifier, C7 anchoring and rotation.
"""
from .chain import check_record, verify_chain
from .errors import (
                     KeyNotConfigured,
                     LedgerBusy,
                     LedgerCorrupt,
                     LedgerNotInitialised,
                     LedgerUnavailable,
                     NonCanonical,
                     SealError,
                     SealMissing,
                     SigningFailed,
)
from .keys import (
                     EnvKeyProvider,
                     FileKeyProvider,
                     KeyProvider,
                     TrustRoot,
                     generate_keypair,
                     load_trust_root,
                     parse_trust_root,
)
from .sealer import Receipt, Sealer, SealPolicy
from .store import SealedLedger

__all__ = [
    "EnvKeyProvider", "FileKeyProvider", "KeyNotConfigured", "KeyProvider", "LedgerBusy", "LedgerCorrupt",
    "LedgerNotInitialised", "LedgerUnavailable", "NonCanonical", "Receipt", "SealError", "SealMissing",
    "SealPolicy", "SealedLedger", "Sealer", "SigningFailed", "TrustRoot", "check_record", "generate_keypair",
    "load_trust_root", "parse_trust_root", "verify_chain",
]
