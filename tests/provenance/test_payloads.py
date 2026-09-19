"""Payloads live in the ledger database (plan §7.7; decision C4-12).

A payload is addressed by the SHA-256 of its exact bytes, stored in a `payloads` table that is NOT part of
the hash chain, inserted in the SAME transaction as the record that references it, and re-verified on
every read.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3

import pytest

from cva.provenance.seal.chain import verify_chain
from cva.provenance.seal.errors import LedgerUnavailable, PayloadCorrupt, PayloadMissing
from cva.provenance.seal.payloads import hex_of, ref_for
from cva.provenance.seal.store import SealedLedger

from ._chain_helpers import SEED_A, clock, provider, rng
from ._fixtures import BODIES

MANIFEST = {"device_id": "jetson-07", "unit": "alpha", "profile_hash": "7" * 64, "checkpoint_every": 1000}


def new_ledger(tmp_path, durability="per_record") -> SealedLedger:
    return SealedLedger.init_ledger(tmp_path / "l.db", provider(SEED_A), MANIFEST, durability=durability,
                                    clock=clock(), rng=rng())


def raw(tmp_path):
    return sqlite3.connect(tmp_path / "l.db", isolation_level=None)


def keys():
    k = provider(SEED_A)
    return {k.key_id: k.public_key}


# --- addressing --------------------------------------------------------------------------------------

def test_a_payload_is_addressed_by_the_sha256_of_its_bytes(tmp_path):
    led = new_ledger(tmp_path)
    data = b'{"task":"classify"}'
    (ref,) = led.put_payloads([data])
    assert ref == "sha256:" + hashlib.sha256(data).hexdigest() == ref_for(data)
    assert led.get_payload(ref) == data and led.has_payload(ref) and led.payload_count() == 1


@pytest.mark.parametrize("bad", ["", "sha256:", "sha256:" + "A" * 64, "sha256:" + "a" * 63, "sha1:" + "a" * 64,
                                 "a" * 64, "sha256:../../etc/passwd"])
def test_references_must_be_exact_lowercase_sha256(tmp_path, bad):
    with pytest.raises(ValueError, match="not a payload reference"):
        hex_of(bad)
    with pytest.raises(ValueError):
        new_ledger(tmp_path).get_payload(bad)


def test_put_is_idempotent(tmp_path):
    led = new_ledger(tmp_path)
    a = led.put_payloads([b"same", b"other"])
    b = led.put_payloads([b"same"])
    assert a[0] == b[0] and led.payload_count() == 2


def test_the_same_content_inside_one_batch_is_stored_once(tmp_path):
    led = new_ledger(tmp_path)
    refs = led.put_payloads([b"dup", b"dup", b"dup"])
    assert len(set(refs)) == 1 and led.payload_count() == 1


# --- reading is always re-verified --------------------------------------------------------------------

def test_a_missing_payload_is_reported_as_missing_never_as_empty(tmp_path):
    with pytest.raises(PayloadMissing):
        new_ledger(tmp_path).get_payload("sha256:" + "0" * 64)


def test_an_edited_payload_row_is_detected_on_read(tmp_path):
    """T1b, database form: someone edits the stored output. The trigger is dropped (anyone with file access
    can) — the content address is what catches it."""
    led = new_ledger(tmp_path)
    (ref,) = led.put_payloads([b"honest payload"])
    led.close()
    c = raw(tmp_path)
    c.execute("DROP TRIGGER payloads_no_update")
    c.execute("UPDATE payloads SET data = ?", (b"edited payload",))
    c.close()
    ro = SealedLedger.open(tmp_path / "l.db", read_only=True)
    with pytest.raises(PayloadCorrupt, match="hash to"):
        ro.get_payload(ref)


def test_a_deleted_payload_row_reads_as_missing(tmp_path):
    led = new_ledger(tmp_path)
    (ref,) = led.put_payloads([b"gone soon"])
    led.close()
    c = raw(tmp_path)
    c.execute("DROP TRIGGER payloads_no_delete")
    c.execute("DELETE FROM payloads")
    c.close()
    with pytest.raises(PayloadMissing):
        SealedLedger.open(tmp_path / "l.db", read_only=True).get_payload(ref)


def test_a_different_body_at_an_existing_address_is_corruption_not_overwritten(tmp_path):
    led = new_ledger(tmp_path)
    (ref,) = led.put_payloads([b"honest payload"])
    led._conn.execute("DROP TRIGGER payloads_no_update")
    led._conn.execute("UPDATE payloads SET data = ?", (b"edited payload",))
    with pytest.raises(PayloadCorrupt, match="do not match"):
        led.put_payloads([b"honest payload"])
    assert bytes(led._conn.execute("SELECT data FROM payloads").fetchone()[0]) == b"edited payload"   # evidence kept
    assert ref.startswith("sha256:")


@pytest.mark.parametrize("sql", ["UPDATE payloads SET data = x'00'", "DELETE FROM payloads"])
def test_the_table_is_append_only_like_the_rest(tmp_path, sql):
    led = new_ledger(tmp_path)
    led.put_payloads([b"x"])
    led.close()
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw(tmp_path).execute(sql)


def test_the_payload_table_is_not_part_of_the_chain(tmp_path):
    """Chain verification needs no payloads, and losing every payload does not break it — recompute then
    reports payload_missing (DEGRADED), never a false pass."""
    led = new_ledger(tmp_path)
    led.append_many([("inference", BODIES["inference"])], payloads=[b"raw", b"fine"])
    led.close()
    c = raw(tmp_path)
    c.execute("DROP TRIGGER payloads_no_delete")
    c.execute("DELETE FROM payloads")
    c.close()
    ro = SealedLedger.open(tmp_path / "l.db", read_only=True)
    assert ro.payload_count() == 0
    assert verify_chain(list(ro.stored_records()), ledger_keys=keys()).ok


def test_a_read_only_handle_can_read_payloads_but_not_write_them(tmp_path):
    led = new_ledger(tmp_path)
    (ref,) = led.put_payloads([b"readable"])
    led.close()
    ro = SealedLedger.open(tmp_path / "l.db", read_only=True)
    assert ro.get_payload(ref) == b"readable"
    with pytest.raises(LedgerUnavailable, match="read-only"):
        ro.put_payloads([b"nope"])


# --- the point of putting them in the database: atomicity ------------------------------------------------

def test_a_payload_and_its_record_are_committed_together(tmp_path):
    led = new_ledger(tmp_path)
    w = led.append_many([("inference", BODIES["inference"])], payloads=[b"raw output", b"fine output"])
    assert led.has_payload(ref_for(b"raw output")) and led.has_payload(ref_for(b"fine output"))
    assert w[0].type == "inference" and led.size() == 2


def test_a_record_that_fails_validation_rolls_its_payloads_back(tmp_path):
    led = new_ledger(tmp_path)
    before = led.payload_count()
    with pytest.raises(ValueError):
        led.append_many([("inference", {"bogus": 1})], payloads=[b"orphan-if-not-atomic"])
    assert led.payload_count() == before and not led.has_payload(ref_for(b"orphan-if-not-atomic"))
    assert not led._conn.in_transaction


def test_a_full_disk_loses_the_payload_and_the_record_together(tmp_path):
    led = new_ledger(tmp_path)
    led.append_typed("inference", BODIES["inference"])
    conn = led._conn
    conn.execute(f"PRAGMA max_page_count = {conn.execute('PRAGMA page_count').fetchone()[0]}")
    size, pays = led.size(), led.payload_count()
    with pytest.raises(LedgerUnavailable):
        led.append_many([("inference", BODIES["inference"])], payloads=[os.urandom(300_000)])
    assert led.size() == size and led.payload_count() == pays              # neither half survived
    conn.execute("PRAGMA max_page_count = 1073741823")
    led.append_many([("inference", BODIES["inference"])], payloads=[b"now it fits"])
    assert led.has_payload(ref_for(b"now it fits"))


def test_payloads_written_with_a_record_share_its_durability(tmp_path):
    """Under group_commit there is ONE window for records and payloads alike: nothing is fsynced per payload."""
    led = new_ledger(tmp_path, durability="group_commit")
    led.append_many([("inference", BODIES["inference"])], payloads=[b"a", b"b"])
    assert led.durable is False
    led.flush()
    assert led.durable is True
    led.close()
    reopened = SealedLedger.open(tmp_path / "l.db", read_only=True)
    assert reopened.has_payload(ref_for(b"a")) and reopened.has_payload(ref_for(b"b"))


def test_large_payloads_round_trip_exactly(tmp_path):
    led = new_ledger(tmp_path)
    big = os.urandom(2_000_000)                      # a raw pre-filter output with thousands of boxes
    (ref,) = led.put_payloads([big])
    assert led.get_payload(ref) == big


def test_the_store_version_marks_the_payload_schema(tmp_path):
    led = new_ledger(tmp_path)
    (v,) = raw(tmp_path).execute("SELECT v FROM meta WHERE k='store_version'").fetchone()
    assert v == "cva-seal-store/2"
    led.close()
