"""Content-addressed payload store (plan §7.7): quantised output objects and preprocessing specs.

Layout: `<root>/<hex[0:2]>/<hex[2:]>.json`, address `sha256:<hex>` = SHA-256 of the exact stored bytes.
Write-once (`O_EXCL`), fsync'd (file and directory). NOT part of the chain: what protects a payload is
that the ledger record commits to its hash. It can be lost or corrupted without breaking CHAIN
verification — `prov.recompute` then reports `payload_missing` (DEGRADED), never a false pass — and
reading always re-verifies the address.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path

from .errors import PayloadCorrupt, PayloadMissing

_REF = re.compile(r"sha256:([0-9a-f]{64})")


def ref_for(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hex_of(ref: str) -> str:
    m = _REF.fullmatch(ref)
    if not m:
        raise ValueError(f"not a payload reference: {ref!r}")
    return m.group(1)


class PayloadStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._pending: set[Path] = set()          # files and directories written but not yet fsynced
        self._lock = threading.Lock()

    def path_for(self, ref: str) -> Path:
        h = hex_of(ref)
        return self.root / h[:2] / f"{h[2:]}.json"

    def put(self, data: bytes, *, sync: bool = True) -> str:
        """Store `data` and return its address. Idempotent: identical content is a no-op; an existing
        file whose bytes do NOT match its own address is corruption and is reported, not overwritten.

        `sync=True` fsyncs the file and its directory before returning (the strongest form; costs about
        one fsync each). `sync=False` writes now and defers the fsync to `sync_pending()` — used under
        the ledger's `group_commit` mode, where the ledger's own durability window already applies and
        must not be defeated by per-payload fsyncs (measured: two payload fsyncs cost more than the
        ledger commit itself).
        """
        ref = ref_for(data)
        p = self.path_for(ref)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if p.read_bytes() != data:
                raise PayloadCorrupt(f"{ref}: existing file does not match its address") from None
            return ref
        try:
            os.write(fd, data)
            if sync:
                os.fsync(fd)
        finally:
            os.close(fd)
        if sync:
            _fsync_path(p.parent)
        else:
            with self._lock:
                self._pending.update((p, p.parent))
        return ref

    def sync_pending(self) -> int:
        """fsync every payload (and directory) written with `sync=False`. Returns how many paths."""
        with self._lock:
            todo, self._pending = list(self._pending), set()
        for path in todo:
            _fsync_path(path)
        return len(todo)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def get(self, ref: str) -> bytes:
        p = self.path_for(ref)
        try:
            data = p.read_bytes()
        except FileNotFoundError:
            raise PayloadMissing(f"{ref} is not in the payload store") from None
        if ref_for(data) != ref:
            raise PayloadCorrupt(f"{ref}: stored bytes hash to {ref_for(data)}")
        return data

    def has(self, ref: str) -> bool:
        return self.path_for(ref).is_file()


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
