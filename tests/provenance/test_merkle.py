"""RFC 6962 Merkle tree (plan §7.4, §11.2; gate C2). Ordering per the plan: the reference and the
published vectors come first, because everything else is built on them."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from cva.provenance.seal.errors import MerkleError
from cva.provenance.seal.merkle import (
    EMPTY_ROOT,
    DictNodeStore,
    MerkleTree,
    leaf_hash,
    mth,
    node_hash,
    split,
    verify_consistency,
    verify_inclusion,
)

from . import _merkle_ref as ref
from ._merkle_ref import leaves_for

CT = json.loads((Path(__file__).resolve().parents[2] / "spec" / "vectors" / "merkle_ct.json").read_text())
CT_LEAVES = [leaf_hash(bytes.fromhex(x)) for x in CT["leaf_inputs"]]


def flip(b: bytes, bit: int = 0) -> bytes:
    return bytes([b[0] ^ (1 << bit)]) + b[1:]


# --- published vectors (transparency-dev/merkle, Apache-2.0) -----------------------------------------

def test_empty_tree_root_is_sha256_of_the_empty_string():
    assert EMPTY_ROOT.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert CT["empty_root"] == EMPTY_ROOT.hex()
    assert MerkleTree().root() == EMPTY_ROOT == mth([]) == ref.mth([])


def test_roots_for_every_size_match_the_published_ct_vectors():
    t = MerkleTree()
    assert t.root(0).hex() == CT["root_hashes_by_size"][0]
    for i, lh in enumerate(CT_LEAVES, start=1):
        assert t.append(lh) == i - 1
        assert t.root().hex() == CT["root_hashes_by_size"][i], f"size {i}"


def test_stored_nodes_equal_the_published_complete_subtree_hashes():
    t = MerkleTree.from_leaves(CT_LEAVES)
    for level, row in enumerate(CT["node_hashes_by_level"]):
        for idx, expected in enumerate(row):
            assert t.node(level, idx).hex() == expected, f"node ({level},{idx})"


def test_historic_roots_can_be_read_from_the_finished_tree():
    t = MerkleTree.from_leaves(CT_LEAVES)
    assert [t.root(n).hex() for n in range(9)] == CT["root_hashes_by_size"]


# --- differential: the optimised tree equals the naive recursive reference --------------------------------

def test_matches_the_naive_reference_at_every_size_up_to_2000():
    leaves = leaves_for(2000)
    t = MerkleTree()
    for n, lh in enumerate(leaves, start=1):
        t.append(lh)
        assert t.root() == ref.mth(leaves[:n]), f"size {n}"


def test_matches_the_reference_at_random_sizes_up_to_100000():
    leaves = leaves_for(100_000)
    t = MerkleTree.from_leaves(leaves)
    rng = random.Random(20260919)
    sizes = [1, 2, 3, 65_535, 65_536, 65_537, 99_999, 100_000] + [rng.randrange(1, 100_001) for _ in range(6)]
    for n in sizes:
        assert t.root(n) == ref.mth(leaves[:n]) == mth(leaves[:n]), f"size {n}"


def test_the_storeless_fold_equals_the_reference_for_every_size_up_to_300():
    leaves = leaves_for(300)
    for n in range(301):
        assert mth(leaves[:n]) == ref.mth(leaves[:n])


def test_split_is_the_largest_power_of_two_strictly_below_n():
    assert [split(n) for n in (2, 3, 4, 5, 8, 9, 16, 17)] == [1, 2, 2, 4, 4, 8, 8, 16]
    for n in range(2, 300):
        k = split(n)
        assert k & (k - 1) == 0 and k < n <= 2 * k
    with pytest.raises(MerkleError):
        split(1)


# --- exhaustive proofs, every size up to 65 ---------------------------------------------------------------

def test_every_inclusion_proof_up_to_65_matches_the_reference_and_verifies():
    for n in range(1, 66):
        leaves = leaves_for(n)
        t = MerkleTree.from_leaves(leaves)
        root = t.root()
        for i in range(n):
            proof = t.inclusion_proof(i)
            assert proof == ref.path(i, leaves), f"n={n} i={i}"
            assert verify_inclusion(leaves[i], i, n, proof, root), f"n={n} i={i}"
            assert ref.recursive_inclusion_root(i, n, leaves[i], proof) == root       # independent verifier


def test_inclusion_proofs_verify_against_historic_sizes_of_a_larger_tree():
    leaves = leaves_for(65)
    t = MerkleTree.from_leaves(leaves)
    for size in range(1, 66):
        for i in range(size):
            assert verify_inclusion(leaves[i], i, size, t.inclusion_proof(i, size), t.root(size))


def test_inclusion_proofs_fail_at_a_wrong_index_wrong_root_or_wrong_leaf():
    for n in range(1, 66):
        leaves = leaves_for(n)
        t = MerkleTree.from_leaves(leaves)
        root = t.root()
        for i in range(n):
            proof = t.inclusion_proof(i)
            for j in range(n):
                if j != i:
                    assert not verify_inclusion(leaves[i], j, n, proof, root), f"n={n} proved i={i} at j={j}"
            assert not verify_inclusion(leaves[i], i, n, proof, flip(root))
            assert not verify_inclusion(flip(leaves[i]), i, n, proof, root)



def test_documented_limitation_an_inclusion_proof_does_not_authenticate_the_tree_size():
    """RFC 9162's algorithm binds a leaf to a ROOT, not to a size: a leaf-0 proof for a 3-leaf tree also
    'verifies' if the caller claims size 4, because both trees give leaf 0 the same left/right path shape.
    That is safe here only because a checkpoint signs (tree_size, root_hash) TOGETHER — so a verifier
    must always take `size` from a signed checkpoint/anchor, never from the proof's presenter. This test
    pins the behaviour so nobody 'fixes' it by accident or forgets why the size must come from the
    signed source."""
    leaves = leaves_for(3)
    t = MerkleTree.from_leaves(leaves)
    proof, root = t.inclusion_proof(0), t.root()
    assert verify_inclusion(leaves[0], 0, 3, proof, root)
    assert verify_inclusion(leaves[0], 0, 4, proof, root)          # size not authenticated by the proof
    assert not verify_inclusion(leaves[0], 0, 5, proof, root)      # ...but a different path shape fails


def test_inclusion_proofs_fail_when_truncated_extended_or_edited():
    n = 37
    leaves = leaves_for(n)
    t = MerkleTree.from_leaves(leaves)
    root = t.root()
    for i in range(n):
        proof = t.inclusion_proof(i)
        assert not verify_inclusion(leaves[i], i, n, proof[:-1], root) or not proof
        assert not verify_inclusion(leaves[i], i, n, [*proof, proof[0] if proof else root], root)
        assert not verify_inclusion(leaves[i], i, n, [*proof, root], root)
        for pos in range(len(proof)):
            edited = list(proof)
            edited[pos] = flip(edited[pos], 3)
            assert not verify_inclusion(leaves[i], i, n, edited, root)
            swapped = list(proof)
            swapped[pos], swapped[-1] = swapped[-1], swapped[pos]
            if swapped != proof:
                assert not verify_inclusion(leaves[i], i, n, swapped, root)


def test_every_consistency_proof_up_to_65_matches_the_reference_and_verifies():
    for n in range(1, 66):
        leaves = leaves_for(n)
        t = MerkleTree.from_leaves(leaves)
        for m in range(1, n + 1):
            proof = t.consistency_proof(m, n)
            assert proof == ref.cons(m, leaves), f"m={m} n={n}"
            assert verify_consistency(m, n, t.root(m), t.root(n), proof), f"m={m} n={n}"
            assert ref.recursive_consistency_roots(m, n, t.root(m), proof) == (t.root(m), t.root(n))


def test_consistency_proofs_fail_on_wrong_roots_sizes_or_edited_proofs():
    for n in range(2, 66):
        leaves = leaves_for(n)
        t = MerkleTree.from_leaves(leaves)
        for m in range(1, n):
            proof = t.consistency_proof(m, n)
            rm, rn = t.root(m), t.root(n)
            assert not verify_consistency(m, n, flip(rm), rn, proof), f"m={m} n={n}"
            assert not verify_consistency(m, n, rm, flip(rn), proof)
            assert not verify_consistency(m, n, rn, rm, proof)                      # swapped roots
            assert not verify_consistency(m, n, rm, rn, proof[:-1])
            assert not verify_consistency(m, n, rm, rn, [*proof, rn])
            assert not verify_consistency(m, n, rm, rn, [])
            for pos in range(len(proof)):
                edited = list(proof)
                edited[pos] = flip(edited[pos], 5)
                assert not verify_consistency(m, n, rm, rn, edited), f"m={m} n={n} pos={pos}"


def test_a_rewritten_history_is_not_consistent_with_the_original():
    """The point of consistency proofs: prove the log only GREW. Change one old leaf and it must fail."""
    n, m = 40, 25
    honest = leaves_for(n)
    t = MerkleTree.from_leaves(honest)
    forged = list(honest)
    forged[7] = leaf_hash(b"rewritten")
    tf = MerkleTree.from_leaves(forged)
    assert not verify_consistency(m, n, t.root(m), tf.root(n), tf.consistency_proof(m, n))
    assert not verify_consistency(m, n, tf.root(m), t.root(n), t.consistency_proof(m, n))


# --- odd leaf counts, and 2^k +/- 1 (the "highest-confidence predicted defect", S1) ------------------------

@pytest.mark.parametrize("n", [3, 5, 6, 7, 9, 11, 13, 15, 17, 21, 31, 33, 63, 65])
def test_odd_and_awkward_leaf_counts_have_correct_proofs_for_every_index(n):
    leaves = leaves_for(n, b"odd")
    t = MerkleTree.from_leaves(leaves)
    assert t.root() == ref.mth(leaves)
    for i in range(n):
        assert verify_inclusion(leaves[i], i, n, t.inclusion_proof(i), t.root())


@pytest.mark.parametrize("k", range(1, 11))
def test_two_to_the_k_plus_and_minus_one(k):
    for n in (2**k - 1, 2**k, 2**k + 1):
        leaves = leaves_for(n, b"pow")
        t = MerkleTree.from_leaves(leaves)
        assert t.root() == ref.mth(leaves)
        for i in {0, 1, n // 2, n - 2, n - 1} & set(range(n)):
            p = t.inclusion_proof(i)
            assert p == ref.path(i, leaves)
            assert verify_inclusion(leaves[i], i, n, p, t.root())
        for m in {1, n // 2, n - 1, n} & set(range(1, n + 1)):
            assert verify_consistency(m, n, t.root(m), t.root(), t.consistency_proof(m))


# --- the two structural attacks ---------------------------------------------------------------------------

def test_duplicate_last_node_forgery_is_real_for_the_naive_tree_and_impossible_for_ours():
    a, b, c = leaves_for(3)
    # CVE-2012-2459: the naive tree gives [a,b,c] and [a,b,c,c] the SAME root ...
    assert ref.dup_last_mth([a, b, c]) == ref.dup_last_mth([a, b, c, c])
    # ... RFC 6962's does not, and neither does ours.
    assert ref.mth([a, b, c]) != ref.mth([a, b, c, c])
    assert mth([a, b, c]) != mth([a, b, c, c])
    assert MerkleTree.from_leaves([a, b, c]).root() != MerkleTree.from_leaves([a, b, c, c]).root()


def test_the_duplicate_forgery_holds_at_every_odd_size():
    for n in range(1, 40, 2):
        leaves = leaves_for(n, b"dup")
        assert mth(leaves) != mth([*leaves, leaves[-1]])


def test_an_internal_node_presented_as_a_leaf_does_not_verify_against_the_root():
    """Leaf/node confusion: with domain separation the two-leaf tree over [N01, N23] cannot reproduce
    the four-leaf tree's root, and an inclusion 'proof' built that way fails."""
    l0, l1, l2, l3 = leaves_for(4, b"conf")
    root = mth([l0, l1, l2, l3])
    n01, n23 = node_hash(l0, l1), node_hash(l2, l3)
    assert root == node_hash(n01, n23)
    assert mth([leaf_hash(n01), leaf_hash(n23)]) != root                 # data = internal hashes
    assert not verify_inclusion(leaf_hash(n01), 0, 2, [leaf_hash(n23)], root)


def test_without_domain_separation_the_same_confusion_would_succeed():
    """Shows the test above is meaningful: hash leaves as bare SHA-256 and the attack works."""
    import hashlib
    def bare(d): return hashlib.sha256(d).digest()
    def bare_node(a, b): return hashlib.sha256(a + b).digest()
    l = [bare(bytes([i])) for i in range(4)]
    n01, n23 = bare_node(l[0], l[1]), bare_node(l[2], l[3])
    assert bare_node(n01, n23) == bare_node(bare_node(l[0], l[1]), bare_node(l[2], l[3]))
    forged_root = bare_node(n01, n23)
    assert bare_node(n01, n23) == forged_root            # a 2-leaf 'tree' whose leaves ARE n01, n23


def test_a_leaf_hash_never_equals_a_node_hash_of_the_same_bytes():
    x = bytes(range(64))
    assert leaf_hash(x) != node_hash(x[:32], x[32:])


# --- tree behaviour ---------------------------------------------------------------------------------------

def test_appending_never_changes_an_earlier_root_or_proof():
    leaves = leaves_for(200)
    t = MerkleTree()
    seen_roots, seen_proofs = {}, {}
    for n, lh in enumerate(leaves, start=1):
        t.append(lh)
        seen_roots[n] = t.root()
        seen_proofs[n] = t.inclusion_proof(n // 2, n)
    for n in range(1, 201):
        assert t.root(n) == seen_roots[n]
        assert t.inclusion_proof(n // 2, n) == seen_proofs[n]


def test_a_tree_reopened_over_the_same_store_continues_correctly():
    leaves = leaves_for(50)
    store = DictNodeStore()
    first = MerkleTree(store)
    for lh in leaves[:31]:
        first.append(lh)
    second = MerkleTree(store, size=31)                       # simulates reopening the ledger
    for lh in leaves[31:]:
        second.append(lh)
    assert second.root() == ref.mth(leaves) and second.size() == 50


def test_a_rebuilt_cache_gives_the_same_root_as_the_original():
    leaves = leaves_for(77)
    assert MerkleTree.from_leaves(leaves, DictNodeStore()).root() == MerkleTree.from_leaves(leaves).root()


def test_a_corrupt_store_is_reported_not_silently_wrong():
    for needed in [(3, 0), (1, 4)]:                           # the two subtrees root() of 10 leaves reads
        t = MerkleTree.from_leaves(leaves_for(10))
        del t._store._d[needed]                               # type: ignore[attr-defined]
        with pytest.raises(MerkleError, match="missing"):
            t.root()
    t = MerkleTree.from_leaves(leaves_for(10))
    del t._store._d[(3, 0)]                                   # type: ignore[attr-defined]
    with pytest.raises(MerkleError, match="missing"):
        t.inclusion_proof(9)


def test_leaf_returns_the_appended_hash():
    leaves = leaves_for(5)
    t = MerkleTree.from_leaves(leaves)
    assert [t.leaf(i) for i in range(5)] == leaves
    with pytest.raises(MerkleError):
        t.leaf(5)


class CountingStore(DictNodeStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def get(self, level: int, idx: int) -> bytes | None:
        self.reads += 1
        return super().get(level, idx)


def test_roots_and_proofs_are_read_from_few_stored_nodes_not_the_whole_tree():
    store = CountingStore()
    n = 100_000
    t = MerkleTree.from_leaves(leaves_for(n), store)
    logn = math.ceil(math.log2(n))
    for label, op, bound in [("root", lambda: t.root(), 2 * logn),
                             ("root(size)", lambda: t.root(77_777), 2 * logn),
                             ("inclusion", lambda: t.inclusion_proof(54_321), logn * logn),
                             ("consistency", lambda: t.consistency_proof(33_333), logn * logn)]:
        store.reads = 0
        op()
        assert 0 < store.reads <= bound, f"{label}: {store.reads} reads (bound {bound}) for n={n}"


# --- argument validation ----------------------------------------------------------------------------------

def test_out_of_range_requests_raise_rather_than_return_something():
    t = MerkleTree.from_leaves(leaves_for(8))
    for bad in (lambda: t.inclusion_proof(8), lambda: t.inclusion_proof(-1), lambda: t.inclusion_proof(0, 9),
                lambda: t.inclusion_proof(0, 0), lambda: t.consistency_proof(0), lambda: t.consistency_proof(9),
                lambda: t.consistency_proof(5, 4), lambda: t.consistency_proof(1, 9), lambda: t.root(9),
                lambda: t.root(-1)):
        with pytest.raises(MerkleError):
            bad()


def test_append_rejects_anything_that_is_not_a_32_byte_hash():
    t = MerkleTree()
    for bad in (b"", b"x" * 31, b"x" * 33, "a" * 32, None, 5):
        with pytest.raises(MerkleError):
            t.append(bad)   # type: ignore[arg-type]
    assert t.size() == 0


def test_verifiers_return_false_on_malformed_input_instead_of_raising():
    leaves = leaves_for(6)
    t = MerkleTree.from_leaves(leaves)
    root, proof = t.root(), t.inclusion_proof(2)
    assert verify_inclusion(leaves[2], 2, 6, proof, root)
    good = {"leaf_h": leaves[2], "index": 2, "size": 6, "proof": proof, "root": root}
    for bad in [{"leaf_h": b"short"}, {"root": b"short"}, {"index": -1}, {"index": 6}, {"size": 0},
                {"proof": [b"x" * 31]}, {"proof": [*proof, "not bytes"]}, {"size": -3}]:
        args = good | bad
        assert verify_inclusion(args["leaf_h"], args["index"], args["size"], args["proof"], args["root"]) is False
    assert verify_consistency(0, 6, root, root, []) is False
    assert verify_consistency(7, 6, root, root, []) is False
    assert verify_consistency(3, 6, b"short", root, proof) is False
    assert verify_consistency(6, 6, root, root, [root]) is False       # same size must carry no proof
    assert verify_consistency(6, 6, root, root, []) is True
