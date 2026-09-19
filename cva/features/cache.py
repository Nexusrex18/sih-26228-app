"""The feature cache, keyed `(content_sha256, extractor_id, extractor_version)`.

**The version in the key is a CORRECTNESS property, not a speed one.** A cache keyed on
content alone returns last week's vectors after the extraction procedure changes, and the
report then attributes findings to a backbone that did not produce them. So a changed
`extractor_version` must MISS, and V8 asserts exactly that — a hit on the second run and a
miss when the version moves.

`content_sha256` is the hash of the IMAGE BYTES, never of the path. Two datasets that ship
the same image under different filenames share the entry; one path whose bytes changed
does not, which is the property that matters when the dataset under assessment is the
thing suspected of having been tampered with.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(os.environ.get("CVA_CACHE_DIR", ".cache/features"))


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_key(content: str, extractor_id: str, extractor_version: str) -> str:
    """One key from all three parts. Hashed rather than concatenated because
    `extractor_id` pins a checkpoint digest and is 70+ characters — a path built from the
    raw triple overflows filesystem name limits on the first real backbone."""
    h = hashlib.sha256()
    for part in (content, extractor_id, extractor_version):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")          # length-unambiguous: "ab"+"c" must not equal "a"+"bc"
    return h.hexdigest()


@dataclass
class FeatureCache:
    root: Path = field(default_factory=lambda: DEFAULT_ROOT)
    hits: int = 0
    misses: int = 0
    writes: int = 0
    enabled: bool = True

    def _path(self, key: str) -> Path:
        # Two-level fan-out: one flat directory with 100k entries is slow to list and
        # unpleasant to inspect by hand, and inspecting it by hand is how a cache bug
        # gets found.
        return self.root / key[:2] / key[2:4] / f"{key}.npz"

    def get(self, content: str, extractor_id: str, extractor_version: str
            ) -> dict[str, np.ndarray] | None:
        if not self.enabled:
            self.misses += 1
            return None
        p = self._path(cache_key(content, extractor_id, extractor_version))
        if not p.is_file():
            self.misses += 1
            return None
        try:
            with np.load(p) as z:
                out = {k: z[k] for k in z.files}
        except Exception:
            # A corrupt entry is a miss, not a crash — but it is never silently reused.
            self.misses += 1
            p.unlink(missing_ok=True)
            return None
        self.hits += 1
        return out

    def put(self, content: str, extractor_id: str, extractor_version: str,
            arrays: dict[str, np.ndarray]) -> None:
        if not self.enabled:
            return
        p = self._path(cache_key(content, extractor_id, extractor_version))
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a half-written .npz from an interrupted scan must never be
        # read back as a hit on the next one.
        tmp = p.with_suffix(".npz.part")
        # Through an open handle: np.savez(path) appends `.npz` to a name not ending in
        # it, writing `<key>.npz.part.npz` and leaving the rename below nothing to move.
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)  # type: ignore[arg-type]
        tmp.replace(p)
        self.writes += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
