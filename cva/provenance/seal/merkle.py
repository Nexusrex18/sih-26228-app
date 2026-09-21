"""RFC 6962 Merkle tree, exactly (plan §5.8, §7.4; decisions D4/D7).

    leaf hash      SHA-256(0x00 ‖ leaf_data)
    internal node  SHA-256(0x01 ‖ left ‖ right)
    MTH([])        SHA-256("")
    MTH(D[n>1])    node(MTH(D[0:k]), MTH(D[k:n]))       k = largest power of two STRICTLY < n

Never duplicate the last node at an odd level: that is CVE-2012-2459 (two distinct trees, one root).
RFC 6962 was obsoleted by RFC 9162 (CT v2); the SHA-256 tree definition is identical, and the
iterative proof verifiers below follow RFC 9162 §2.1.3.2 (inclusion) and §2.1.4.2 (consistency) —
cite both wherever the spec is quoted.

Storage: the tree keeps every COMPLETED PERFECT SUBTREE, keyed (level, index), behind a `NodeStore`.
Appending finishes any subtrees the new leaf completes (amortised O(1) hashes); `root(size)`,
inclusion and consistency proofs assemble their answers from O(log n) stored nodes. The store is a
CACHE, exactly like a derived column: the verifier never trusts it and recomputes from the records
with `mth()` (C4/C5). Leaf data for the ledger is the full signed record bytes (D7).
"""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from typing import Protocol

from .constants import DOMAIN_LEAF, DOMAIN_NODE
from .errors import MerkleError

HASH_LEN = 32
EMPTY_ROOT = hashlib.sha256(b"").digest()      # e3b0c442…b855, MTH of the empty tree


def leaf_hash(leaf_data: bytes) -> bytes:
    return hashlib.sha256(DOMAIN_LEAF + leaf_data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(DOMAIN_NODE + left + right).digest()


def split(n: int) -> int:
    """Largest power of two strictly less than n (n >= 2)."""
    if n < 2:
        raise MerkleError(f"split() needs n >= 2, got {n}")
    return 1 << ((n - 1).bit_length() - 1)


def _is_hash(b: object) -> bool:
    return isinstance(b, (bytes, bytearray)) and len(b) == HASH_LEN


def mth(leaf_hashes: Sequence[bytes]) -> bytes:
    """MTH over leaf HASHES, in O(n) with no store and no recursion: fold a stack of perfect subtrees.
    This is what the verifier uses to recompute a root straight from the records."""
    stack: list[tuple[int, bytes]] = []
    for h in leaf_hashes:
        level, cur = 0, h
        while stack and stack[-1][0] == level:
            cur = node_hash(stack.pop()[1], cur)
            level += 1
        stack.append((level, cur))
    if not stack:
        return EMPTY_ROOT
    root = stack[-1][1]
    for _, h in reversed(stack[:-1]):
        root = node_hash(h, root)
    return root


class RootAccumulator:
    """Streaming MTH: append leaf hashes one at a time and read the root of everything appended so far in
    O(log n), holding only the perfect subtrees of the current prefix. The verifier uses it to check every
    checkpoint's `root_hash` as it walks the ledger — O(n) in total, not O(n) per checkpoint."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, bytes]] = []
        self._size = 0

    def size(self) -> int:
        return self._size

    def append(self, leaf_h: bytes) -> None:
        level, cur = 0, bytes(leaf_h)
        while self._stack and self._stack[-1][0] == level:
            cur = node_hash(self._stack.pop()[1], cur)
            level += 1
        self._stack.append((level, cur))
        self._size += 1

    def root(self) -> bytes:
        if not self._stack:
            return EMPTY_ROOT
        root = self._stack[-1][1]
        for _, h in reversed(self._stack[:-1]):
            root = node_hash(h, root)
        return root


class NodeStore(Protocol):
    """Where completed perfect subtrees live. `put` must be idempotent (SQLite: INSERT OR IGNORE)."""

    def get(self, level: int, idx: int) -> bytes | None: ...
    def put(self, level: int, idx: int, h: bytes) -> None: ...


class DictNodeStore:
    """In-memory store, and the reference for the SQLite one added at C4."""

    def __init__(self) -> None:
        self._d: dict[tuple[int, int], bytes] = {}

    def get(self, level: int, idx: int) -> bytes | None:
        return self._d.get((level, idx))

    def put(self, level: int, idx: int, h: bytes) -> None:
        self._d.setdefault((level, idx), h)

    def __len__(self) -> int:
        return len(self._d)


class MerkleTree:
    """Append-only RFC 6962 tree over leaf hashes."""

    def __init__(self, store: NodeStore | None = None, size: int = 0) -> None:
        """`size` is the number of leaves already in `store` (the caller knows — for the ledger it is
        the record count). A fresh tree is `MerkleTree()`."""
        if size < 0:
            raise MerkleError("size must be >= 0")
        self._store: NodeStore = store if store is not None else DictNodeStore()
        self._size = size

    @classmethod
    def from_leaves(cls, leaf_hashes: Sequence[bytes], store: NodeStore | None = None) -> MerkleTree:
        """Build (or rebuild a cache) from leaf hashes — the recovery path for a stale store."""
        t = cls(store)
        for h in leaf_hashes:
            t.append(h)
        return t

    def size(self) -> int:
        return self._size

    def node(self, level: int, idx: int) -> bytes | None:
        """A stored completed subtree hash (for cross-checking against published node vectors)."""
        return self._store.get(level, idx)

    def append(self, leaf_h: bytes) -> int:
        """Add a leaf HASH (use `leaf_hash(data)`); returns its index."""
        if not _is_hash(leaf_h):
            raise MerkleError(f"leaf hash must be {HASH_LEN} bytes")
        n = self._size
        level, i, cur = 0, n, bytes(leaf_h)
        self._store.put(0, i, cur)
        while i & 1:                       # i is a right child: its parent is now complete
            left = self._get(level, i - 1)
            cur = node_hash(left, cur)
            level, i = level + 1, i >> 1
            self._store.put(level, i, cur)
        self._size = n + 1
        return n

    def leaf(self, index: int) -> bytes:
        if not 0 <= index < self._size:
            raise MerkleError(f"leaf index {index} outside tree of size {self._size}")
        return self._get(0, index)

    def root(self, size: int | None = None) -> bytes:
        """MTH of the first `size` leaves (default: all)."""
        n = self._size if size is None else size
        self._check_size(n)
        return EMPTY_ROOT if n == 0 else self._range_hash(0, n)

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        """RFC 6962 §2.1.1 audit path for leaf `index` in the tree of the first `size` leaves."""
        n = self._size if size is None else size
        self._check_size(n)
        if not 0 <= index < n:
            raise MerkleError(f"leaf index {index} outside tree of size {n}")
        return self._path(index, 0, n)

    def consistency_proof(self, m: int, n: int | None = None) -> list[bytes]:
        """RFC 6962 §2.1.2: proof that the tree of size `m` is a prefix of the tree of size `n`."""
        n = self._size if n is None else n
        self._check_size(n)
        if not 1 <= m <= n:
            raise MerkleError(f"need 1 <= m <= n, got m={m}, n={n}")
        return [] if m == n else self._sub(m, 0, n, True)

    # -- internals ------------------------------------------------------------------------------

    def _check_size(self, n: int) -> None:
        if not 0 <= n <= self._size:
            raise MerkleError(f"size {n} outside 0..{self._size}")

    def _get(self, level: int, idx: int) -> bytes:
        h = self._store.get(level, idx)
        if h is None:
            raise MerkleError(f"node ({level},{idx}) missing from the store — cache corrupt; rebuild "
                              "it from the records")
        return h

    def _range_hash(self, start: int, n: int) -> bytes:
        """MTH of leaves [start, start+n). Aligned perfect ranges are single stored nodes."""
        if n & (n - 1) == 0 and start % n == 0:
            return self._get(n.bit_length() - 1, start // n)
        k = split(n)
        return node_hash(self._range_hash(start, k), self._range_hash(start + k, n - k))

    def _path(self, m: int, start: int, n: int) -> list[bytes]:
        if n == 1:
            return []
        k = split(n)
        if m < k:
            return [*self._path(m, start, k), self._range_hash(start + k, n - k)]
        return [*self._path(m - k, start + k, n - k), self._range_hash(start, k)]

    def _sub(self, m: int, start: int, n: int, b: bool) -> list[bytes]:
        if m == n:
            return [] if b else [self._range_hash(start, n)]
        k = split(n)
        if m <= k:
            return [*self._sub(m, start, k, b), self._range_hash(start + k, n - k)]
        return [*self._sub(m - k, start + k, n - k, False), self._range_hash(start, k)]


def verify_inclusion(leaf_h: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes) -> bool:
    """RFC 9162 §2.1.3.2. Returns False (never raises) for any malformed or non-matching input.

    The proof binds `leaf_h` to `root`, NOT to `size`: a proof can verify under a nearby size whose
    path has the same shape (e.g. leaf 0 of a 3-leaf tree "verifies" at size 4). Take `size` from the
    same SIGNED checkpoint/anchor that gave you `root` — a checkpoint signs (tree_size, root_hash)
    together — never from whoever presented the proof.
    """
    if not (_is_hash(leaf_h) and _is_hash(root) and all(_is_hash(p) for p in proof)):
        return False
    if index < 0 or size < 1 or index >= size:
        return False
    fn, sn = index, size - 1
    r = bytes(leaf_h)
    for p in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(bytes(p), r)
            if not fn & 1:
                while fn & 1 == 0 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, bytes(p))
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(r, bytes(root))


def verify_consistency(m: int, n: int, root_m: bytes, root_n: bytes, proof: Sequence[bytes]) -> bool:
    """RFC 9162 §2.1.4.2: `root_m` (size m) is a prefix of `root_n` (size n). Never raises."""
    if not (_is_hash(root_m) and _is_hash(root_n) and all(_is_hash(p) for p in proof)):
        return False
    if m < 1 or m > n:
        return False
    if m == n:
        return not proof and hmac.compare_digest(bytes(root_m), bytes(root_n))
    if not proof:
        return False
    pr = [bytes(p) for p in proof]
    if m & (m - 1) == 0:                  # m is a power of two: its root is the first proof element
        pr = [bytes(root_m), *pr]
    fn, sn = m - 1, n - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = pr[0]
    for c in pr[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            if not fn & 1:
                while fn & 1 == 0 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(fr, bytes(root_m)) and hmac.compare_digest(sr, bytes(root_n))
