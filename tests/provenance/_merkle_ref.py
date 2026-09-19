"""Independent reference for the Merkle tests — deliberately NOT sharing code with `merkle.py`.

  * naive RECURSIVE mth / path / consistency (RFC 6962 §2.1, plan Appendix A) over slices, versus the
    product's stored-node, iterative implementation
  * RECURSIVE verifiers that rebuild a root by mirroring the proof's recursive structure, versus the
    product's ITERATIVE RFC 9162 verifiers (§2.1.3.2 / §2.1.4.2)
  * `dup_last_mth`: the classic buggy tree (duplicate the last node at an odd level), kept only so a
    test can show that the forgery it enables (CVE-2012-2459) is real and that we are not exposed.
"""
from __future__ import annotations

import hashlib


def H(b): return hashlib.sha256(b).digest()
EMPTY = H(b"")


def leaf(d: bytes) -> bytes:
    return H(b"\x00" + d)


def node(left: bytes, right: bytes) -> bytes:
    return H(b"\x01" + left + right)


def split(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def mth(leaves: list[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return EMPTY
    if n == 1:
        return leaves[0]
    k = split(n)
    return node(mth(leaves[:k]), mth(leaves[k:]))


def path(m: int, leaves: list[bytes]) -> list[bytes]:
    n = len(leaves)
    if n == 1:
        return []
    k = split(n)
    if m < k:
        return path(m, leaves[:k]) + [mth(leaves[k:])]
    return path(m - k, leaves[k:]) + [mth(leaves[:k])]


def _subproof(m: int, leaves: list[bytes], b: bool) -> list[bytes]:
    n = len(leaves)
    if m == n:
        return [] if b else [mth(leaves)]
    k = split(n)
    if m <= k:
        return _subproof(m, leaves[:k], b) + [mth(leaves[k:])]
    return _subproof(m - k, leaves[k:], False) + [mth(leaves[:k])]


def cons(m: int, leaves: list[bytes]) -> list[bytes]:
    return [] if m == len(leaves) else _subproof(m, leaves, True)


def dup_last_mth(leaves: list[bytes]) -> bytes:
    """The BUGGY construction: pair up, and if a level is odd, duplicate its last node."""
    if not leaves:
        return EMPTY
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


# --- recursive verifiers (mirror the structure of the proofs) ---------------------------------------

def recursive_inclusion_root(m: int, n: int, leaf_h: bytes, proof: list[bytes]) -> bytes | None:
    """Rebuild the root from an audit path by recursing exactly as `path()` built it. None if the
    proof length does not match the recursion."""
    p = list(proof)

    def rec(m: int, n: int) -> bytes | None:
        if n == 1:
            return leaf_h
        if not p:
            return None
        x = p.pop()                                  # outermost sibling is the LAST element
        k = split(n)
        if m < k:
            sub = rec(m, k)
            return None if sub is None else node(sub, x)
        sub = rec(m - k, n - k)
        return None if sub is None else node(x, sub)

    out = rec(m, n)
    return out if not p else None


def recursive_consistency_roots(m: int, n: int, root_m: bytes,
                                proof: list[bytes]) -> tuple[bytes, bytes] | None:
    """Rebuild (old_root, new_root) from a consistency proof by mirroring `_subproof`."""
    if m == n:
        return (root_m, root_m) if not proof else None
    p = list(proof)

    def rec(m: int, n: int, b: bool) -> tuple[bytes, bytes] | None:
        if m == n:
            if b:
                return root_m, root_m
            if not p:
                return None
            h = p.pop()
            return h, h
        if not p:
            return None
        x = p.pop()
        k = split(n)
        if m <= k:
            sub = rec(m, k, b)
            return None if sub is None else (sub[0], node(sub[1], x))
        sub = rec(m - k, n - k, False)
        return None if sub is None else (node(x, sub[0]), node(x, sub[1]))

    out = rec(m, n, True)
    return out if not p else None


def leaves_for(n: int, salt: bytes = b"") -> list[bytes]:
    return [leaf(salt + i.to_bytes(4, "big")) for i in range(n)]
