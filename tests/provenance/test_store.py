"""The SQLite ledger (plan §7.6; gate C4)."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from cva.provenance.seal.canonical import parse_strict
from cva.provenance.seal.chain import verify_chain
from cva.provenance.seal.errors import (
    LedgerCorrupt,
    LedgerNotInitialised,
    LedgerUnavailable,
    SealError,
    WrongKey,
)
from cva.provenance.seal.merkle import leaf_hash, mth, verify_consistency, verify_inclusion
from cva.provenance.seal.store import SealedLedger

from ._chain_helpers import SEED_A, SEED_B, clock, provider, rng
from ._fixtures import BODIES

REPO = Path(__file__).resolve().parents[2]
MANIFEST = {"device_id": "jetson-07", "unit": "alpha", "profile_hash": "7" * 64}


def new_ledger(tmp_path, *, checkpoint_every=1000, durability="per_record", seed=SEED_A, name="l.db") -> SealedLedger:
    return SealedLedger.init_ledger(tmp_path / name, provider(seed), {**MANIFEST, "checkpoint_every": checkpoint_every},
                                    durability=durability, clock=clock(), rng=rng())


def fill(led: SealedLedger, n: int, rtype="inference") -> None:
    for _ in range(n):
        led.append_typed(rtype, BODIES[rtype])


def keys(seed=SEED_A):
    k = provider(seed)
    return {k.key_id: k.public_key}


def raw_db(path):
    return sqlite3.connect(path, isolation_level=None)


# --- creation and opening ------------------------------------------------------------------------------

def test_init_writes_a_genesis_that_commits_to_the_manifest_and_key(tmp_path):
    led = new_ledger(tmp_path)
    g = led.genesis
    assert g["type"] == "genesis" and g["seq"] == 0 and led.size() == 1
    assert g["deployment_manifest"]["key_id"] == provider().key_id == led.genesis_key_id
    assert led.deployment_manifest["device_id"] == "jetson-07" and led.checkpoint_every == 1000
    assert verify_chain(list(led.stored_records()), ledger_keys=keys()).ok


def test_init_refuses_to_reinitialise_an_existing_file(tmp_path):
    new_ledger(tmp_path).close()
    with pytest.raises(SealError, match="refusing to re-initialise"):
        new_ledger(tmp_path)


def test_a_ledger_is_never_created_implicitly_by_open(tmp_path):
    with pytest.raises(LedgerNotInitialised, match="cva-seal init"):
        SealedLedger.open(tmp_path / "absent.db", key=provider())
    assert not (tmp_path / "absent.db").exists()


def test_a_writer_must_present_the_ledgers_own_key(tmp_path):
    new_ledger(tmp_path).close()
    with pytest.raises(WrongKey, match="not this ledger's active signing key"):
        SealedLedger.open(tmp_path / "l.db", key=provider(SEED_B))
    with pytest.raises(WrongKey, match="needs its signing key"):
        SealedLedger.open(tmp_path / "l.db")


def test_the_same_file_can_be_reopened_and_continues_the_chain(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 5)
    led.close()
    again = SealedLedger.open(tmp_path / "l.db", key=provider(), clock=clock())      # real CSPRNG nonces
    assert again.size() == 6
    again.append_typed("inference", BODIES["inference"])
    assert verify_chain(list(again.stored_records()), ledger_keys=keys()).ok


def test_durability_and_the_loss_window_are_recorded_in_meta(tmp_path):
    per = new_ledger(tmp_path, name="a.db")
    grp = SealedLedger.init_ledger(tmp_path / "b.db", provider(), {**MANIFEST, "checkpoint_every": 10},
                                   durability="group_commit", group_n=25, group_ms=40)
    meta = dict(raw_db(tmp_path / "a.db").execute("SELECT k, v FROM meta").fetchall())
    assert per.durability == "per_record" and meta["loss_window"] == "0 records"
    meta2 = dict(raw_db(tmp_path / "b.db").execute("SELECT k, v FROM meta").fetchall())
    assert grp.durability == "group_commit" and meta2["loss_window"] == "up to 25 records / 40 ms"
    grp.close()


def test_the_database_runs_in_wal_mode_with_full_sync_for_per_record(tmp_path):
    led = new_ledger(tmp_path)
    assert led._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert led._conn.execute("PRAGMA synchronous").fetchone()[0] == 2                # FULL


# --- append: chain, checkpoints, atomicity --------------------------------------------------------------

def test_appended_records_form_a_verifiable_chain_with_gapless_seq(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 40)
    recs = list(led.records())
    assert [r["seq"] for r in recs] == list(range(41))
    assert verify_chain(list(led.stored_records()), ledger_keys=keys()).ok


def test_checkpoints_are_written_every_n_records_and_commit_to_the_true_tree(tmp_path):
    led = new_ledger(tmp_path, checkpoint_every=7)
    fill(led, 30)
    recs = list(led.records())
    cps = [r for r in recs if r["type"] == "checkpoint"]
    assert len(cps) >= 4
    leaves = [leaf_hash(d) for d in led.stored_records()]
    for cp in cps:
        size = cp["checkpoint"]["tree_size"]
        assert size == cp["seq"]                                   # leaves BEFORE this record
        assert cp["checkpoint"]["root_hash"] == mth(leaves[:size]).hex()      # recomputed from the bytes


def test_checkpoint_roots_are_consistent_with_each_other(tmp_path):
    """The point of one cumulative tree (D4): every checkpoint is provably a prefix of the next."""
    led = new_ledger(tmp_path, checkpoint_every=5)
    fill(led, 40)
    cps = [r["checkpoint"] for r in led.records() if r["type"] == "checkpoint"]
    tree = led.merkle_tree()
    for a, b in zip(cps, cps[1:], strict=False):
        proof = tree.consistency_proof(a["tree_size"], b["tree_size"])
        assert verify_consistency(a["tree_size"], b["tree_size"], bytes.fromhex(a["root_hash"]),
                                  bytes.fromhex(b["root_hash"]), proof)


def test_a_checkpoint_never_triggers_another_checkpoint(tmp_path):
    led = new_ledger(tmp_path, checkpoint_every=1)          # pathological: would loop forever if it did
    fill(led, 3)
    types = [r["type"] for r in led.records()]
    assert types == ["genesis", "inference", "checkpoint", "inference", "checkpoint", "inference", "checkpoint"]


def test_the_leaf_of_every_record_is_the_full_signed_stored_bytes(tmp_path):
    """D7: the tree covers the signature."""
    led = new_ledger(tmp_path)
    fill(led, 9)
    tree = led.merkle_tree()
    for i, data in enumerate(led.stored_records()):
        assert tree.leaf(i) == leaf_hash(data)
        assert verify_inclusion(leaf_hash(data), i, led.size(), tree.inclusion_proof(i), tree.root())


def test_the_stored_cache_root_equals_a_root_recomputed_from_the_records(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 33)
    assert led.verify_merkle_cache()
    assert led.root() == mth([leaf_hash(d) for d in led.stored_records()])


def test_append_many_is_atomic_all_or_nothing(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 3)
    before = led.size()
    with pytest.raises(ValueError):
        led.append_many([("inference", BODIES["inference"]), ("inference", {"input": {}})])   # 2nd is invalid
    assert led.size() == before                                     # the first was rolled back too
    assert verify_chain(list(led.stored_records()), ledger_keys=keys()).ok
    led.append_many([("inference", BODIES["inference"]), ("inference", BODIES["inference"])])
    assert led.size() == before + 2


def test_a_rejected_append_leaves_no_open_transaction_and_no_trace(tmp_path):
    led = new_ledger(tmp_path)
    with pytest.raises(ValueError):
        led.append_typed("inference", {"bogus": 1})
    assert not led._conn.in_transaction
    led.append_typed("inference", BODIES["inference"])
    assert led.size() == 2


def test_append_returns_what_was_written_including_triggered_checkpoints(tmp_path):
    led = new_ledger(tmp_path, checkpoint_every=3)
    w = led.append_many([("inference", BODIES["inference"]), ("inference", BODIES["inference"])])
    assert [x.type for x in w] == ["inference", "inference", "checkpoint"]
    assert [x.seq for x in w] == [1, 2, 3]
    assert all(len(x.record_hash) == 64 for x in w)


def test_audit_append_accepts_only_scan_and_analyst_records(tmp_path):
    led = new_ledger(tmp_path)
    rid = led.append({"type": "scan_record", **BODIES["scan_record"]})
    assert len(rid) == 64
    led.append({"type": "analyst_event", **BODIES["analyst_event"]})
    for rtype in ("genesis", "checkpoint", "key_rotation", "anchor_event", "degraded_marker", "inference"):
        with pytest.raises(SealError, match="may write only"):
            led.append({"type": rtype})
    assert [r["type"] for r in led.records()][-2:] == ["scan_record", "analyst_event"]


def test_a_read_only_handle_cannot_write(tmp_path):
    new_ledger(tmp_path).close()
    ro = SealedLedger.open(tmp_path / "l.db", read_only=True)
    with pytest.raises(LedgerUnavailable, match="read-only"):
        ro.append_typed("inference", BODIES["inference"])
    assert ro.size() == 1 and next(iter(ro.records()))["type"] == "genesis"
    ro.close()


# --- the append-only triggers (a guard against accidents, not a security control) ----------------------------

@pytest.mark.parametrize("sql", [
    "UPDATE records SET rec='x' WHERE seq=0", "DELETE FROM records WHERE seq=0",
    "UPDATE merkle_nodes SET hash=x'00'", "DELETE FROM merkle_nodes",
    "UPDATE meta SET v='x'", "DELETE FROM meta",
])
def test_triggers_reject_updates_and_deletes(tmp_path, sql):
    new_ledger(tmp_path).close()
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw_db(tmp_path / "l.db").execute(sql)


def test_a_trigger_is_not_a_security_control_anyone_with_file_access_can_drop_it(tmp_path):
    """Documented, and exactly what the tamper script does. The chain — not the trigger — detects this."""
    led = new_ledger(tmp_path)
    fill(led, 4)
    led.close()
    c = raw_db(tmp_path / "l.db")
    c.execute("DROP TRIGGER records_no_update")
    c.execute("UPDATE records SET rec = replace(rec, '\"seq\":2', '\"seq\":9') WHERE seq = 2")
    c.close()
    ro = SealedLedger.open(tmp_path / "l.db", read_only=True)
    assert not verify_chain(list(ro.stored_records()), ledger_keys=keys()).ok


def test_the_nonce_column_is_unique_as_a_writer_side_guard(tmp_path):
    led = new_ledger(tmp_path)
    led.append_typed("inference", BODIES["inference"])
    led._rng = lambda n: bytes(n)                                 # a broken rng: constant nonce
    led.append_typed("inference", BODIES["inference"])
    with pytest.raises(SealError, match="uniqueness guard"):
        led.append_typed("inference", BODIES["inference"])
    assert led.size() == 3


# --- derived columns and the cache ---------------------------------------------------------------------------

def test_derived_columns_agree_with_the_record_bytes(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 6)
    for seq, rtype, nonce, rec_hash, rec in raw_db(tmp_path / "l.db").execute("SELECT seq,type,nonce,rec_hash,rec FROM records"):
        parsed = parse_strict(rec.encode())
        assert (parsed["seq"], parsed["type"], parsed["nonce"]) == (seq, rtype, nonce)
        from cva.provenance.seal.records import record_hash
        assert bytes(rec_hash) == record_hash(parsed)


def test_a_tip_that_disagrees_with_its_derived_columns_refuses_to_open(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 3)
    led.close()
    c = raw_db(tmp_path / "l.db")
    c.execute("DROP TRIGGER records_no_update")
    c.execute("UPDATE records SET type='checkpoint' WHERE seq=3")
    c.close()
    with pytest.raises(LedgerCorrupt, match="derived columns"):
        SealedLedger.open(tmp_path / "l.db", key=provider())


def test_a_short_merkle_cache_fails_closed_and_can_be_rebuilt_deliberately(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 12)
    good_root = led.root()
    led.close()
    c = raw_db(tmp_path / "l.db")
    c.execute("DROP TRIGGER merkle_no_delete")
    c.execute("DELETE FROM merkle_nodes WHERE level=1 AND idx=2")
    c.close()
    with pytest.raises(LedgerCorrupt, match="rebuild_merkle_cache"):
        SealedLedger.open(tmp_path / "l.db", key=provider())
    repair = SealedLedger.open(tmp_path / "l.db", key=provider(), recover=False)
    repair.rebuild_merkle_cache()
    repair.close()
    fixed = SealedLedger.open(tmp_path / "l.db", key=provider())
    assert fixed.root() == good_root and fixed.verify_merkle_cache()


def test_a_wrong_but_complete_cache_is_caught_by_the_deep_check(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 9)
    led.close()
    c = raw_db(tmp_path / "l.db")
    c.execute("DROP TRIGGER merkle_no_update")
    (h,) = c.execute("SELECT hash FROM merkle_nodes WHERE level=3 AND idx=0").fetchone()
    c.execute("UPDATE merkle_nodes SET hash=? WHERE level=3 AND idx=0", (bytes([h[0] ^ 1]) + h[1:],))
    c.close()
    w = SealedLedger.open(tmp_path / "l.db", key=provider())        # cheap checks cannot see it ...
    assert not w.verify_merkle_cache()                               # ... the O(n) recomputation does


def test_an_unreadable_cache_cell_is_a_failed_check_not_a_crash(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 9)
    led.close()
    c = raw_db(tmp_path / "l.db")
    c.execute("DROP TRIGGER merkle_no_update")
    c.execute("UPDATE merkle_nodes SET hash = 'not a blob' WHERE level=3 AND idx=0")     # wrong TYPE in the cell
    c.close()
    w = SealedLedger.open(tmp_path / "l.db", key=provider())
    assert w.verify_merkle_cache() is False


@pytest.mark.parametrize("damage", ["garbage_header", "empty_file", "truncated"])
def test_a_damaged_database_file_is_reported_as_corrupt(tmp_path, damage):
    led = new_ledger(tmp_path)
    fill(led, 5)
    led.close()
    path = tmp_path / "l.db"
    if damage == "garbage_header":
        with open(path, "r+b") as f:
            f.write(b"\xde\xad" * 50)
    elif damage == "empty_file":
        path.write_bytes(b"")
    else:
        path.write_bytes(path.read_bytes()[:300])
    for extra in ("-wal", "-shm"):
        if os.path.exists(str(path) + extra):
            os.remove(str(path) + extra)
    with pytest.raises((LedgerCorrupt, LedgerNotInitialised)):
        SealedLedger.open(path, key=provider())


# --- capabilities are probed, not asserted --------------------------------------------------------------------

def test_a_writer_with_a_working_key_reports_signing_key_and_inference_ledger(tmp_path):
    led = new_ledger(tmp_path)
    assert led.capabilities() == {"SIGNING_KEY", "INFERENCE_LEDGER"}


def test_a_read_only_handle_reports_only_inference_ledger(tmp_path):
    new_ledger(tmp_path).close()
    ro = SealedLedger.open(tmp_path / "l.db", read_only=True)
    assert ro.capabilities() == {"INFERENCE_LEDGER"}


def test_signing_key_is_absent_when_the_key_cannot_produce_a_verifying_signature(tmp_path):
    led = new_ledger(tmp_path)
    led._key = _BrokenKey(led._key)
    assert led.capabilities() == {"INFERENCE_LEDGER"}


def test_a_closed_ledger_reports_no_capabilities(tmp_path):
    led = new_ledger(tmp_path)
    led.close()
    assert led.capabilities() == set()


class _BrokenKey:
    def __init__(self, real):
        self._real = real
        self.key_id, self.public_key, self.custody = real.key_id, real.public_key, real.custody

    def sign(self, message: bytes) -> bytes:
        return bytes(64)

    def self_test(self) -> bool:
        return False


def test_the_ledger_satisfies_backends_real_protocols_structurally(tmp_path):
    core_ledger = pytest.importorskip("cva.core.ledger")
    capability = pytest.importorskip("cva.core.capability")
    led = new_ledger(tmp_path)
    assert isinstance(led, core_ledger.AuditLedger) and isinstance(led, core_ledger.InferenceLedgerSource)
    # capabilities() returns names; Capability is a str-enum, so consumers' membership tests just work
    assert capability.Capability.SIGNING_KEY in led.capabilities()
    assert capability.Capability.INFERENCE_LEDGER in led.capabilities()
    assert capability.Capability.MODEL_PREDICT not in led.capabilities()
    body = core_ledger.scan_record("s-2026-09-19-0007", "b" * 64, "c" * 64, "0" * 40, {"critical": 1})
    assert len(led.append(body)) == 64                               # Backend's own builder, unchanged


def test_records_iterates_parsed_mappings(tmp_path):
    led = new_ledger(tmp_path)
    fill(led, 3)
    assert [r["type"] for r in led.records()] == ["genesis", "inference", "inference", "inference"]


# --- several processes, several restarts (C-11: nothing in memory to disagree after a reboot) -----------------

_WRITER = """
import sys, base64
sys.path.insert(0, {repo!r})
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.keys import EnvKeyProvider
from tests.provenance._fixtures import BODIES
key = EnvKeyProvider("K", environ={{"K": base64.b64encode(bytes(range(32))).decode()}})
led = SealedLedger.open({path!r}, key=key)
for _ in range({n}):
    led.append_typed("inference", BODIES["inference"])
led.close()
"""


def run_writers(path, procs, n):
    ps = [subprocess.Popen([sys.executable, "-c", _WRITER.format(repo=str(REPO), path=str(path), n=n)],
                           cwd=REPO, stderr=subprocess.PIPE, text=True) for _ in range(procs)]
    for p in ps:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err


def test_concurrent_writer_processes_keep_seq_gapless_and_the_chain_intact(tmp_path):
    """BEGIN IMMEDIATE takes the write lock up front, so two processes cannot read the same tip."""
    new_ledger(tmp_path, checkpoint_every=50).close()
    run_writers(tmp_path / "l.db", procs=4, n=60)
    led = SealedLedger.open(tmp_path / "l.db", read_only=True)
    recs = list(led.records())
    assert [r["seq"] for r in recs] == list(range(len(recs))) and len(recs) >= 1 + 4 * 60
    assert verify_chain(list(led.stored_records()), ledger_keys=keys()).ok
    assert len({r["nonce"] for r in recs}) == len(recs)
    rw = SealedLedger.open(tmp_path / "l.db", key=provider())
    assert rw.verify_merkle_cache()


def test_the_ledger_survives_repeated_process_restarts_with_no_false_alarm(tmp_path):
    """Three real restarts, each a fresh process resuming from the DB tip. There is deliberately no
    in-memory counter that could disagree after a reboot."""
    new_ledger(tmp_path, checkpoint_every=40).close()
    for _ in range(3):
        run_writers(tmp_path / "l.db", procs=1, n=75)
    led = SealedLedger.open(tmp_path / "l.db", read_only=True)
    assert verify_chain(list(led.stored_records()), ledger_keys=keys()).ok
    assert led.size() >= 1 + 3 * 75


def test_two_handles_in_one_process_serialise_correctly(tmp_path):
    a = new_ledger(tmp_path)
    a._rng = os.urandom
    b = SealedLedger.open(tmp_path / "l.db", key=provider(), clock=clock())
    for i in range(20):
        (a if i % 2 else b).append_typed("inference", BODIES["inference"])
    assert verify_chain(list(a.stored_records()), ledger_keys=keys()).ok
    assert a.size() == b.size() == 21


def test_a_stale_cached_tip_link_is_never_paired_with_a_refreshed_tip(tmp_path):
    """The writer caches link_hash(tip) to avoid re-hashing. If ANOTHER handle appends and this handle's tip()
    then refreshes the cached record, the cached link must be dropped with it — otherwise the next append would
    chain to the wrong link and corrupt the ledger."""
    a = new_ledger(tmp_path)
    a._rng = os.urandom
    b = SealedLedger.open(tmp_path / "l.db", key=provider(), clock=clock())
    a.append_typed("inference", BODIES["inference"])          # a caches its tip + link
    b.append_typed("inference", BODIES["inference"])          # b moves the tip on
    assert a.tip()["seq"] == 2 and a._tip_link is None        # refresh drops the stale link
    a.append_typed("inference", BODIES["inference"])          # would chain wrongly if the link were reused
    assert verify_chain(list(a.stored_records()), ledger_keys=keys()).ok
    assert a._tip_link is not None and a._tip_link == __import__("cva.provenance.seal.chain", fromlist=["x"]).link_hash(a.tip())


def test_the_cached_link_always_equals_a_freshly_computed_one(tmp_path):
    from cva.provenance.seal.chain import link_hash
    led = new_ledger(tmp_path, checkpoint_every=4)
    for _ in range(25):
        led.append_typed("inference", BODIES["inference"])
        assert led._tip_link == link_hash(led.tip())
