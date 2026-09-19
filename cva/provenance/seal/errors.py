"""Exceptions for the seal SDK. More are added by the gate that needs them
(LedgerUnavailable/LedgerCorrupt at C4, KeyNotConfigured at C3, ...)."""
from __future__ import annotations


class SealError(Exception):
    """Base class for every error the seal raises deliberately."""


class NonCanonical(SealError, ValueError):
    """Input rejected by the canonical profile (plan §5.1). The SDK never "fixes" it silently.

    `code` is what a verifier reports: "malformed_record" for anything that is not a valid profile
    object, "non_canonical_encoding" for a valid object whose bytes are not the canonical bytes.
    """

    def __init__(self, message: str, *, code: str = "malformed_record", path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path
        self.detail = message


class InvalidRecord(SealError, ValueError):
    """A record whose fields do not satisfy its type's in-code schema (plan §7.3)."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.detail = message


class QuantiseError(SealError, ValueError):
    """A value that cannot be quantised to a hashable integer."""


class NonFiniteValue(QuantiseError):
    """NaN or ±inf reached quantise(). The Sealer's default is to seal a marker instead (plan §8)."""
