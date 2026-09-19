"""Failure injection (plan §8, decision C-14; gate C4).

Every way the ledger can become unwritable must end in ONE of two states: fail-closed (the caller is
told, and must not release the output) or fail-open with a DECLARED hole. Never a silent gap, never a
half-written record. "A declared hole is evidence; an undeclared hole is a lie by omission."
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from cva.provenance.seal.chain import verify_chain
from cva.provenance.seal.errors import (
    LedgerCorrupt,
    LedgerUnavailable,
    SealError,
    SealMissing,
    SigningFailed,
)
from cva.provenance.seal.sealer import Sealer, SealPolicy

from ._ledger_helpers import CLASSIFY, Env

FAIL_OPEN = SealPolicy(on_ledger_failure="fail_open", allow_fail_open=True)
NO_LIMIT = 1_073_741_823
needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")


def fill_disk(env: Env) -> None:
    """Make the next page allocation fail with SQLITE_FULL — portable, no real disk needed."""
    conn = env.sealer.ledger._conn
    conn.execute(f"PRAGMA max_page_count = {conn.execute('PRAGMA page_count').fetchone()[0]}")


def free_disk(env: Env) -> None:
    env.sealer.ledger._conn.execute(f"PRAGMA max_page_count = {NO_LIMIT}")


def seal_until_full(env: Env, limit: int = 400) -> int:
    """Seal until the injected fault trips; returns how many succeeded first."""
    for i in range(limit):
        try:
            env.seal(i + 10_000)
        except LedgerUnavailable:
            return i
    raise AssertionError("the injected SQLITE_FULL never fired")


def chain_ok(env: Env) -> bool:
    return verify_chain(list(env.sealer.ledger.stored_records()), ledger_keys=env.keys()).ok


# --- fail-closed (the default) --------------------------------------------------------------------------------

def test_a_full_disk_fails_closed_and_leaves_no_partial_record(tmp_path):
    e = Env(tmp_path)
    for i in range(5):
        e.seal(i)
    fill_disk(e)
    sealed_before_failure = seal_until_full(e)
    size = e.sealer.ledger.size()
    assert size == 1 + 1 + 5 + sealed_before_failure                         # genesis + registration + inferences
    assert not e.sealer.ledger._conn.in_transaction                          # rolled back cleanly
    assert chain_ok(e) and e.sealer.ledger.verify_merkle_cache()
    with pytest.raises(LedgerUnavailable):                                   # and it keeps failing, loudly
        e.seal(99)
    assert e.sealer.ledger.size() == size
    e.close()


def test_the_ledger_recovers_the_moment_space_returns(tmp_path):
    e = Env(tmp_path)
    fill_disk(e)
    seal_until_full(e)
    free_disk(e)
    r = e.seal(1)
    assert r.sealed and chain_ok(e)
    e.close()


def test_a_rollback_after_sqlite_already_rolled_back_does_not_mask_the_real_error(tmp_path):
    """SQLITE_FULL rolls the transaction back itself; a follow-up ROLLBACK raises 'no transaction is
    active'. The error path must not assume one is open — the caller must see LedgerUnavailable, not that."""
    e = Env(tmp_path)
    fill_disk(e)
    with pytest.raises(LedgerUnavailable) as ei:
        for i in range(400):
            e.seal(i)
    assert "no transaction" not in str(ei.value).lower()
    e.close()


def test_guard_never_runs_the_release_step_when_the_ledger_is_unwritable(tmp_path):
    e = Env(tmp_path)
    fill_disk(e)
    released = []
    with pytest.raises(LedgerUnavailable):
        for i in range(400):
            with e.sealer.guard() as g:
                g.seal(bytes([i % 256]) * 300 + i.to_bytes(4, "big"), e.model, e.config, output=CLASSIFY, dims=(8, 8))
                released.append(i)                                           # only reached if the seal succeeded
    sealed = len(released)
    assert e.sealer.ledger.size() == 2 + sealed                              # every RELEASED output has a record
    e.close()


def test_a_read_only_database_fails_closed(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    e = Env(tmp_path)
    e.seal(0)
    e.close()
    e.ledger_path.chmod(0o444)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(str(e.ledger_path) + suffix):
            os.chmod(str(e.ledger_path) + suffix, 0o444)
    try:
        with pytest.raises(LedgerUnavailable):
            s = e.open()                                                     # open itself may refuse ...
            s.seal(b"frame", e.model, e.config, output=CLASSIFY, dims=(8, 8))       # ... or the first write must
    finally:
        e.ledger_path.chmod(0o644)


def test_a_corrupt_database_header_fails_closed_at_open(tmp_path):
    e = Env(tmp_path)
    e.seal(0)
    e.close()
    with open(e.ledger_path, "r+b") as f:
        f.write(b"\xff" * 100)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(str(e.ledger_path) + suffix):
            os.remove(str(e.ledger_path) + suffix)
    with pytest.raises(LedgerCorrupt):
        e.open()


def test_an_unplugged_key_fails_closed_with_a_signing_error(tmp_path):
    e = Env(tmp_path)
    e.seal(0)
    e.sealer.ledger._key = _DeadKey(e.key)
    size = e.sealer.ledger.size()
    with pytest.raises(SigningFailed, match="could not sign"):
        e.seal(1)
    assert isinstance(SigningFailed("x"), LedgerUnavailable) and e.sealer.ledger.size() == size
    e.close()


def test_an_unwritable_payload_store_fails_closed_and_appends_nothing(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    e = Env(tmp_path)
    e.seal(0)
    size = e.sealer.ledger.size()
    e.payload_dir.chmod(0o500)
    try:
        with pytest.raises(LedgerUnavailable, match="payload store"):
            e.seal(1, output={"task": "classify", "top": [{"cls": 5, "conf": 0.42}]})     # a NEW payload
    finally:
        e.payload_dir.chmod(0o755)
    assert e.sealer.ledger.size() == size
    e.close()


def test_a_record_never_references_a_payload_that_does_not_exist(tmp_path):
    from cva.provenance.seal.payloads import PayloadStore
    e = Env(tmp_path)
    for i in range(8):
        e.seal(i, output={"task": "classify", "top": [{"cls": i, "conf": 0.5}]})
    ps = PayloadStore(e.payload_dir)
    for rec in e.sealer.ledger.records():
        if rec["type"] == "inference":
            assert ps.has(rec["output"]["payload_ref"]) and ps.has("sha256:" + rec["output"]["raw_jcs_sha256"])
            assert ps.has(rec["config"]["preprocess_ref"])
    e.close()


class _DeadKey:
    def __init__(self, real):
        self.key_id, self.public_key, self.custody = real.key_id, real.public_key, real.custody

    def sign(self, message: bytes) -> bytes:
        raise OSError("hsm unplugged")

    def self_test(self) -> bool:
        return False


# --- fail-open (explicit opt-in) -------------------------------------------------------------------------------

def test_fail_open_needs_an_explicit_allow_and_a_key_that_can_sign(tmp_path):
    with pytest.raises(ValueError, match="allow_fail_open"):
        SealPolicy(on_ledger_failure="fail_open")
    e = Env(tmp_path)
    e.close()
    with pytest.raises(SealError, match="signable"):
        Sealer.open(e.ledger_path, key=_DeadKey(e.key), trust_root=e.trust, payload_dir=e.payload_dir,
                    policy=FAIL_OPEN)


def test_under_fail_open_an_unwritable_ledger_returns_an_honest_unsealed_receipt(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    e.seal(0)
    fill_disk(e)
    n_ok = seal_until_full_open(e)
    r = e.seal(500)
    assert r.sealed is False and r.seq is None and r.record_hash is None and r.durable is False
    assert e.sealer._gap is not None and e.sealer._gap.count >= 2 and n_ok >= 0
    e.close()


def seal_until_full_open(env: Env, limit: int = 400) -> int:
    """Under fail-open nothing raises, so detect the fault by the first unsealed receipt."""
    for i in range(limit):
        if not env.seal(i + 10_000).sealed:
            return i
    raise AssertionError("the injected SQLITE_FULL never fired")


def test_unsealed_inferences_are_counted_and_their_input_hashes_spilled(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    fill_disk(e)
    seal_until_full_open(e)
    frames = [b"unsealed-frame-%d" % i for i in range(3)]
    for f in frames:
        assert e.sealer.seal(f, e.model, e.config, output=CLASSIFY, dims=(8, 8)).sealed is False
    spill = Path(str(e.ledger_path) + ".spill").read_text().splitlines()
    hashes = [ln.split(" ")[1] for ln in spill]
    for f in frames:
        assert hashlib.sha256(f).hexdigest() in hashes
    assert e.sealer._gap.count == len(spill)
    assert chain_ok(e)                                                       # nothing bogus reached the chain
    e.close()


def test_recovery_writes_a_signed_degraded_marker_that_declares_exactly_the_hole(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    e.seal(0)
    fill_disk(e)
    seal_until_full_open(e)
    for i in range(4):
        e.seal(600 + i)
    gap_count = e.sealer._gap.count
    spill_path = Path(str(e.ledger_path) + ".spill")
    spill_sha = hashlib.sha256(spill_path.read_bytes()).hexdigest()
    free_disk(e)
    r = e.seal(999)
    assert r.sealed
    recs = list(e.sealer.ledger.records())
    marker = next(x for x in recs if x["type"] == "degraded_marker")
    assert marker["gap"]["reported_count"] == gap_count and marker["gap"]["spill_sha256"] == spill_sha
    assert marker["gap"]["first_unsealed_utc"] <= marker["gap"]["last_unsealed_utc"]
    assert marker["gap"]["reason"] == "ledger_unwritable"
    idx = recs.index(marker)
    assert recs[idx + 1]["type"] == "inference" and recs[idx + 1]["seq"] == r.seq       # marker BEFORE the next inference
    assert chain_ok(e)
    assert not spill_path.exists()                                            # retired ...
    assert len(list(tmp_path.glob("ledger.db.spill.sealed-*"))) == 1        # ... and kept as evidence
    assert e.sealer._gap is None
    e.close()


def test_the_marker_is_written_once_not_on_every_later_inference(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    fill_disk(e)
    seal_until_full_open(e)
    free_disk(e)
    for i in range(5):
        e.seal(i)
    assert sum(r["type"] == "degraded_marker" for r in e.sealer.ledger.records()) == 1
    e.close()


def test_a_second_outage_after_recovery_gets_its_own_marker(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    for _ in range(2):
        fill_disk(e)
        seal_until_full_open(e)
        free_disk(e)
        e.seal(1234)
    markers = [r for r in e.sealer.ledger.records() if r["type"] == "degraded_marker"]
    assert len(markers) == 2 and chain_ok(e)
    e.close()


def test_a_failure_while_writing_the_marker_keeps_the_gap_and_counts_the_new_inference(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    fill_disk(e)
    seal_until_full_open(e)
    before = e.sealer._gap.count
    assert e.seal(700).sealed is False                                        # still failing
    assert e.sealer._gap.count == before + 1
    e.close()


def test_a_process_that_died_while_failing_open_leaves_a_marker_on_the_next_start(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    fill_disk(e)
    seal_until_full_open(e)
    for i in range(3):
        e.seal(800 + i)
    lines = Path(str(e.ledger_path) + ".spill").read_text().splitlines()
    sha = hashlib.sha256(Path(str(e.ledger_path) + ".spill").read_bytes()).hexdigest()
    e.sealer.ledger.close()                                                   # "the process dies": no recovery ran
    again = e.open()                                                          # a fresh process, disk healthy
    marker = next(r for r in again.ledger.records() if r["type"] == "degraded_marker")
    assert marker["gap"]["reported_count"] == len(lines) and marker["gap"]["spill_sha256"] == sha
    assert not Path(str(e.ledger_path) + ".spill").exists()
    assert verify_chain(list(again.ledger.stored_records()), ledger_keys=e.keys()).ok
    again.close()


def test_recovering_the_same_orphan_spill_twice_writes_only_one_marker(tmp_path):
    """Crash after the marker was committed but before the spill was renamed: don't declare the hole twice."""
    e = Env(tmp_path, policy=FAIL_OPEN)
    fill_disk(e)
    seal_until_full_open(e)
    e.seal(900)
    spill = Path(str(e.ledger_path) + ".spill")
    data = spill.read_bytes()
    e.sealer.ledger.close()
    a = e.open()                                                              # recovery #1: writes marker, renames
    spill.write_bytes(data)                                                   # the rename "never happened"
    a.close()
    b = e.open()                                                              # recovery #2
    assert sum(r["type"] == "degraded_marker" for r in b.ledger.records()) == 1
    assert not spill.exists()
    b.close()


def test_when_the_spill_itself_cannot_be_written_the_marker_says_so(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    os.mkdir(str(e.ledger_path) + ".spill")                                   # a directory where the file should be
    fill_disk(e)
    seal_until_full_open(e)
    e.seal(1)
    e.seal(2)
    assert e.sealer._gap.spill_ok is False and e.sealer._gap.count >= 2
    os.rmdir(str(e.ledger_path) + ".spill")
    free_disk(e)
    e.seal(3)
    marker = next(r for r in e.sealer.ledger.records() if r["type"] == "degraded_marker")
    assert marker["gap"]["spill_sha256"] == "unavailable"                     # declared, not hidden
    assert marker["gap"]["reported_count"] >= 2
    e.close()


def test_an_empty_orphan_spill_is_just_cleaned_up(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    e.close()
    Path(str(e.ledger_path) + ".spill").write_bytes(b"")
    s = e.open()
    assert not any(r["type"] == "degraded_marker" for r in s.ledger.records())
    assert not Path(str(e.ledger_path) + ".spill").exists()
    s.close()


def test_a_signing_failure_under_fail_open_is_a_declared_gap_not_a_crash(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    e.seal(0)
    good = e.sealer.ledger._key
    e.sealer.ledger._key = _DeadKey(good)
    assert e.seal(1).sealed is False
    e.sealer.ledger._key = good
    e.seal(2)
    assert any(r["type"] == "degraded_marker" for r in e.sealer.ledger.records())
    e.close()


def test_guard_counts_a_fail_open_unsealed_inference_as_handled(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    fill_disk(e)
    seal_until_full_open(e)
    with e.sealer.guard() as g:                                               # must NOT raise SealMissing
        g.seal(b"frame", e.model, e.config, output=CLASSIFY, dims=(8, 8))
    e.close()


def test_guard_still_catches_a_missing_seal_under_fail_open(tmp_path):
    e = Env(tmp_path, policy=FAIL_OPEN)
    with pytest.raises(SealMissing), e.sealer.guard():
        pass
    e.close()


# --- group_commit: the loss window is a stated property, not a surprise -------------------------------------------

def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_group_commit_receipts_are_not_durable_until_a_flush(tmp_path):
    e = Env(tmp_path, durability="group_commit", background_flush=False)
    r = e.seal(0)
    assert r.sealed and r.durable is False and e.sealer.ledger.durable is False
    e.sealer.flush()
    assert e.sealer.ledger.durable is True
    e.close()


def test_group_commit_defers_payload_fsyncs_until_the_flush(tmp_path):
    e = Env(tmp_path, durability="group_commit", background_flush=False)
    ps = e.sealer._payloads
    for i in range(6):
        e.seal(i, output={"task": "classify", "top": [{"cls": i, "conf": 0.5}]})
    assert ps.pending > 0                                                     # written, not yet fsynced
    e.sealer.flush()
    assert ps.pending == 0
    e.close()


def test_per_record_never_defers_anything(tmp_path):
    e = Env(tmp_path, durability="per_record")
    e.seal(0, output={"task": "classify", "top": [{"cls": 3, "conf": 0.5}]})
    assert e.sealer._payloads.pending == 0 and e.sealer.ledger.durable is True
    e.close()


def test_the_background_flusher_makes_records_durable_within_the_window(tmp_path):
    e = Env(tmp_path, durability="group_commit", background_flush=True)
    for i in range(10):
        e.seal(i)
    assert wait_until(lambda: e.sealer.ledger.durable), "flusher never caught up within the group window"
    e.close()


def test_the_flusher_is_woken_early_by_a_full_group(tmp_path):
    """With the time bound out of play, reaching group_n must still trigger a flush of that group. (A few
    records committed after the flush's snapshot legitimately wait for the next group or the time bound.)"""
    e = Env(tmp_path, durability="group_commit", background_flush=True)
    led = e.sealer.ledger
    led.group_ms = 60_000
    time.sleep(0.15)                                                          # let the loop pick up the new timeout
    for i in range(101):                                                      # registration + 101 inferences >= group_n
        e.seal(i)
    assert wait_until(lambda: led._unflushed < led.group_n, timeout=5.0), "group_n did not trigger a flush"
    e.close()


def test_closing_stops_the_flusher_thread_and_flushes(tmp_path):
    e = Env(tmp_path, durability="group_commit", background_flush=True)
    thread = e.sealer.ledger._flusher
    assert thread is not None and thread.is_alive()
    e.seal(0)
    e.close()
    assert not thread.is_alive()
    assert Path(tmp_path / "ledger.db").exists()


def test_a_failed_background_flush_is_sticky_and_blocks_further_sealing(tmp_path):
    """If we can no longer make records durable, the loss window is unbounded: fail closed."""
    e = Env(tmp_path, durability="group_commit", background_flush=True)
    led = e.sealer.ledger

    def broken() -> None:
        raise OSError("disk gone")

    led._flush_hooks.append(broken)
    for i in range(101):
        try:
            e.seal(i)
        except LedgerUnavailable:
            break
    assert wait_until(lambda: led._flush_error is not None)
    with pytest.raises(LedgerUnavailable, match="background flush failed"):
        e.seal(5000)
    led._flush_hooks.remove(broken)
    led.flush()                                                               # storage recovered
    assert led._flush_error is None
    assert e.seal(5001).sealed
    e.close()


def test_a_group_commit_ledger_is_still_a_valid_chain(tmp_path):
    e = Env(tmp_path, durability="group_commit", background_flush=True)
    for i in range(60):
        e.seal(i)
    e.sealer.flush()
    assert chain_ok(e) and e.sealer.ledger.verify_merkle_cache()
    e.close()


def test_the_flusher_really_fsyncs_the_wal_and_the_deferred_payloads(tmp_path, monkeypatch):
    """Group commit is only honest if the flush actually reaches storage. Spy on the syscalls: the flush
    must open the `-wal` file and fsync it, and must fsync the deferred payload files."""
    from cva.provenance.seal import store as store_mod

    e = Env(tmp_path, durability="group_commit", background_flush=False)
    for i in range(5):
        e.seal(i, output={"task": "classify", "top": [{"cls": i, "conf": 0.5}]})
    assert e.sealer._payloads.pending > 0
    opened: dict[int, str] = {}
    synced: list[str] = []
    real_open, real_fsync = os.open, os.fsync

    def spy_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        opened[fd] = os.fspath(path)
        return fd

    def spy_fsync(fd):
        synced.append(opened.get(fd, "?"))
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    e.sealer.ledger._sync_now()
    assert any(p.endswith("-wal") for p in synced), f"the WAL was never fsynced: {synced}"
    assert any(str(e.payload_dir) in p for p in synced), "deferred payloads were never fsynced"
    assert store_mod is not None
    monkeypatch.undo()
    e.close()


def test_the_commit_hot_path_never_scans_the_table(tmp_path):
    """`COUNT(*)` walks the whole records table, so using it per commit makes sealing slower as the ledger
    grows. Nothing on the per-inference path may issue one (recovery-on-open legitimately does, once)."""
    e = Env(tmp_path)
    for i in range(3):
        e.seal(i)                                                            # warm-up incl. registration
    statements: list[str] = []
    e.sealer.ledger._conn.set_trace_callback(statements.append)
    for i in range(20, 25):
        e.seal(i)
    e.sealer.ledger._conn.set_trace_callback(None)
    assert statements, "the trace callback saw nothing"
    assert not [s for s in statements if "count(" in s.lower()], [s for s in statements if "count(" in s.lower()]
    e.close()
