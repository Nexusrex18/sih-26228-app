"""Property-based Merkle tests (plan §11.3): the interesting failures are the ones nobody wrote an
example for."""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cva.provenance.seal.merkle import (
    MerkleTree,
    leaf_hash,
    mth,
    verify_consistency,
    verify_inclusion,
)

from . import _merkle_ref as ref

_LEAF_DATA = st.lists(st.binary(max_size=40), min_size=1, max_size=120)
_SET = {"max_examples": 200, "suppress_health_check": [HealthCheck.too_slow], "deadline": None}


@settings(**_SET)
@given(_LEAF_DATA)
def test_tree_root_equals_reference_and_storeless_fold(data):
    leaves = [leaf_hash(d) for d in data]
    t = MerkleTree.from_leaves(leaves)
    assert t.root() == ref.mth(leaves) == mth(leaves)


@settings(**_SET)
@given(_LEAF_DATA, st.data())
def test_any_inclusion_proof_verifies_and_dies_on_any_single_bit_change(data, draw):
    leaves = [leaf_hash(d) for d in data]
    t = MerkleTree.from_leaves(leaves)
    n = len(leaves)
    i = draw.draw(st.integers(0, n - 1))
    proof, root = t.inclusion_proof(i), t.root()
    assert verify_inclusion(leaves[i], i, n, proof, root)
    if proof:
        pos = draw.draw(st.integers(0, len(proof) - 1))
        byte = draw.draw(st.integers(0, 31))
        bit = draw.draw(st.integers(0, 7))
        edited = [bytearray(p) for p in proof]
        edited[pos][byte] ^= 1 << bit
        assert not verify_inclusion(leaves[i], i, n, [bytes(e) for e in edited], root)
    edited_root = bytearray(root)
    edited_root[draw.draw(st.integers(0, 31))] ^= 1 << draw.draw(st.integers(0, 7))
    assert not verify_inclusion(leaves[i], i, n, proof, bytes(edited_root))


@settings(**_SET)
@given(_LEAF_DATA, st.data())
def test_any_consistency_proof_verifies_and_matches_the_reference(data, draw):
    leaves = [leaf_hash(d) for d in data]
    t = MerkleTree.from_leaves(leaves)
    n = len(leaves)
    m = draw.draw(st.integers(1, n))
    proof = t.consistency_proof(m, n)
    assert proof == ref.cons(m, leaves)
    assert verify_consistency(m, n, t.root(m), t.root(n), proof)
    assert ref.recursive_consistency_roots(m, n, t.root(m), proof) == (t.root(m), t.root(n))


@settings(**_SET)
@given(_LEAF_DATA, st.data())
def test_editing_any_one_leaf_changes_the_root_and_breaks_consistency_with_the_old_tree(data, draw):
    leaves = [leaf_hash(d) for d in data]
    n = len(leaves)
    i = draw.draw(st.integers(0, n - 1))
    forged = list(leaves)
    forged[i] = leaf_hash(data[i] + b"!")
    assert mth(forged) != mth(leaves)
    if n > 1 and i < n - 1:                                   # an edit strictly inside the older prefix
        m = draw.draw(st.integers(i + 1, n))
        good, bad = MerkleTree.from_leaves(leaves), MerkleTree.from_leaves(forged)
        assert not verify_consistency(m, n, good.root(m), bad.root(n), bad.consistency_proof(m, n))


@settings(**_SET)
@given(_LEAF_DATA)
def test_append_only_roots_of_earlier_sizes_are_stable(data):
    leaves = [leaf_hash(d) for d in data]
    t = MerkleTree()
    roots = []
    for lh in leaves:
        t.append(lh)
        roots.append(t.root())
    assert [t.root(n) for n in range(1, len(leaves) + 1)] == roots


@settings(**_SET)
@given(st.lists(st.binary(min_size=1, max_size=20), min_size=1, max_size=60))
def test_a_repeated_last_leaf_never_collides_with_the_original(data):
    leaves = [leaf_hash(d) for d in data]
    assert mth(leaves) != mth([*leaves, leaves[-1]])
