"""Payload addressing (plan §7.7): quantised output objects and preprocessing specs.

A payload is addressed by `sha256:<hex>` of its exact canonical bytes. Payloads live in the ledger
database itself (`SealedLedger.get_payload` / `put_payloads`), in a `payloads` table that is NOT part of
the hash chain: what protects a payload is that the record commits to its hash, and reading always
re-verifies the address. They are inserted in the SAME TRANSACTION as the record that references them,
so a record can never point at a payload that does not exist, and a payload can never be lost while its
record survives (one fsync covers both).

This module holds only the addressing helpers.
"""
from __future__ import annotations

import hashlib
import re

_REF = re.compile(r"sha256:([0-9a-f]{64})")


def ref_for(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hex_of(ref: str) -> str:
    m = _REF.fullmatch(ref)
    if not m:
        raise ValueError(f"not a payload reference: {ref!r}")
    return m.group(1)
