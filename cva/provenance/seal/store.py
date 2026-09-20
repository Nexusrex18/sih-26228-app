"""The SQLite ledger (plan §7.6; decisions D4, D7, D8).

`records.rec` is the SINGLE SOURCE OF TRUTH: the canonical bytes of the signed record. `type`, `nonce`
and `rec_hash` are DERIVED columns kept only for indexing; the verifier cross-checks them against `rec`
and treats a disagreement as tampering (`derived_column_mismatch`, C5). Never trust a derived column.

Append-only is enforced twice, and the two protect different things: the SQLite triggers protect
against ACCIDENT; the hash chain protects against INTENT. A trigger is not a security control —
anyone with file access can `DROP TRIGGER`, and the tamper script does exactly that. Do not describe
triggers as tamper-proof anywhere.

`append` is one transaction: `BEGIN IMMEDIATE` (so two processes cannot both read the same tip) →
read tip → seal_next → INSERT → Merkle nodes → maybe a checkpoint → COMMIT. Anything that goes wrong
rolls the WHOLE transaction back and surfaces as `LedgerUnavailable`; a half-written record is never
left behind.

The Merkle node table is a CACHE (the verifier recomputes from the records). It is insert-only like
everything else; a damaged cache is rebuilt by dropping and recreating the whole table, an explicit
maintenance act, never silently.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, NamedTuple
from urllib.parse import quote

from .canonical import parse_strict
from .chain import link_from, link_hash, rotation_body, seal_next
from .errors import (
    InvalidRecord,
    LedgerBusy,
    LedgerCorrupt,
    LedgerNotInitialised,
    LedgerUnavailable,
    MerkleError,
    NonCanonical,
    PayloadCorrupt,
    PayloadMissing,
    SealError,
    WrongKey,
)
from .keys import KeyProvider
from .merkle import MerkleTree, leaf_hash, mth
from .payloads import hex_of, ref_for
from .records import new_nonce, record_hash, validate_record

STORE_VERSION = "cva-seal-store/2"
DURABILITY_MODES = ("per_record", "group_commit")
AUDIT_APPEND_TYPES = ("scan_record", "analyst_event")     # what `append()` (the AuditLedger protocol) may write

_SCHEMA = """
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE records (
    seq INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    nonce TEXT NOT NULL UNIQUE,      -- writer-side guard only; the verifier scopes nonces itself (D9)
    rec_hash BLOB NOT NULL,
    rec TEXT NOT NULL                -- JCS(signed record): the single source of truth
);
CREATE TABLE merkle_nodes (level INTEGER NOT NULL, idx INTEGER NOT NULL, hash BLOB NOT NULL,
                           PRIMARY KEY (level, idx));
CREATE TABLE payloads (            -- NOT part of the chain: protected by the record's commitment to its hash
    hash BLOB PRIMARY KEY,         -- SHA-256 of `data` (the content address)
    data BLOB NOT NULL
);
"""
_TRIGGERS = """
CREATE TRIGGER records_no_update BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER records_no_delete BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER merkle_no_update BEFORE UPDATE ON merkle_nodes BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER merkle_no_delete BEFORE DELETE ON merkle_nodes BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER payloads_no_update BEFORE UPDATE ON payloads BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER payloads_no_delete BEFORE DELETE ON payloads BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER meta_no_update BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER meta_no_delete BEFORE DELETE ON meta BEGIN SELECT RAISE(ABORT, 'append-only'); END;
"""


def _classify(e: BaseException) -> LedgerUnavailable:
    """Map a storage-layer exception onto the ledger's failure vocabulary."""
    msg = str(e)
    low = msg.lower()
    if isinstance(e, sqlite3.OperationalError) and ("locked" in low or "busy" in low):
        return LedgerBusy(f"ledger busy: {msg}")
    if isinstance(e, sqlite3.DatabaseError) and ("not a database" in low or "malformed" in low or "corrupt" in low
                                                  or "no such table" in low):
        return LedgerCorrupt(f"ledger file is damaged: {msg}")
    return LedgerUnavailable(f"ledger cannot be written: {type(e).__name__}: {msg}")


class SqliteNodeStore:
    """`NodeStore` over the `merkle_nodes` table, inside whatever transaction is open."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._c = conn

    def get(self, level: int, idx: int) -> bytes | None:
        try:
            row = self._c.execute("SELECT hash FROM merkle_nodes WHERE level=? AND idx=?", (level, idx)).fetchone()
            return None if row is None else bytes(row[0])
        except (sqlite3.Error, TypeError) as e:                 # e.g. a cell that is not a BLOB
            raise MerkleError(f"node ({level},{idx}) is unreadable in the cache: {e}") from None

    def put(self, level: int, idx: int, h: bytes) -> None:
        self._c.execute("INSERT OR IGNORE INTO merkle_nodes(level, idx, hash) VALUES (?,?,?)", (level, idx, h))


class Written(NamedTuple):
    """One record an append wrote: its record_hash (hex), its seq and its type."""

    record_hash: str
    seq: int
    type: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SealedLedger:
    """An append-only, hash-chained, signed ledger on SQLite. One instance = one connection = one
    (writer or read-only) role. Safe to share between threads (calls are serialised by a lock); use one
    instance per process.

    It satisfies BOTH of Backend's ledger protocols structurally (`core/ledger.py`, plan §3.1) without
    importing them — the seal must stay standalone:
      * `AuditLedger`            append(record) -> id, capabilities()   (a writer with a key)
      * `InferenceLedgerSource`  records() -> iterator, capabilities()  (any opened ledger)
    `capabilities()` is ACTIVE: it probes and returns capability NAMES (the strings `Capability`'s
    members equal); `SIGNING_KEY` only if the key loaded AND produced a signature that verified.
    """

    def __init__(self, conn: sqlite3.Connection, path: str, key: KeyProvider | None, *, read_only: bool,
                 clock: Callable[[], datetime], rng: Callable[[int], bytes] | None,
                 require_genesis: bool = True) -> None:
        self._conn = conn
        self.path = path
        self._key = key
        self.read_only = read_only
        self._clock = clock
        self._rng = rng
        self._lock = threading.RLock()
        self._closed = False
        self._tip: dict[str, Any] | None = None
        self._tip_link: str | None = None            # link_hash(self._tip), cached: the writer just made it
        self._unflushed = 0
        self._last_flush = time.monotonic()
        self._flush_hooks: list[Callable[[], object]] = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._flusher: threading.Thread | None = None
        self._flush_error: BaseException | None = None
        meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        self.durability = meta.get("durability", "per_record")
        self.group_n = int(meta.get("group_n", "100"))
        self.group_ms = int(meta.get("group_ms", "50"))
        self.checkpoint_every = int(meta["checkpoint_every"])
        self.genesis: dict[str, Any] = {}
        self.deployment_manifest: dict[str, Any] = {}
        self.genesis_key_id: str = ""
        self.active_key_id: str = ""                 # the key that signs the NEXT record (genesis key, or the last rotation's)
        if require_genesis:
            self._load_genesis()

    def _load_genesis(self) -> None:
        genesis = self._read_record(0)
        if genesis is None or genesis["type"] != "genesis":
            raise LedgerNotInitialised(f"{self.path}: no genesis record")
        self.genesis = genesis
        self.deployment_manifest = genesis["deployment_manifest"]
        self.genesis_key_id = genesis["key_id"]
        self.active_key_id = self._current_key_id()

    def _current_key_id(self) -> str:
        row = self._conn.execute("SELECT seq FROM records WHERE type='key_rotation' ORDER BY seq DESC LIMIT 1").fetchone()
        if row is None:
            return self.genesis_key_id
        rot = self._read_record(row[0])
        return str(rot["rotation"]["new_key_id"]) if rot else self.genesis_key_id

    @contextmanager
    def _snapshot(self) -> Iterator[None]:
        """A consistent READ snapshot: every statement inside sees the database as of one instant, so a
        concurrent writer cannot slip a record between (say) reading the tip and counting the rows.
        A no-op when a transaction is already open."""
        with self._lock:
            if self._conn.in_transaction:
                yield
                return
            try:
                self._conn.execute("BEGIN")
            except sqlite3.Error as e:
                raise _classify(e) from None
            try:
                yield
            finally:
                try:
                    self._conn.execute("COMMIT")
                except sqlite3.Error:
                    pass

    # -- opening / creating --------------------------------------------------------------------

    @staticmethod
    def _connect(path: str, mode: str) -> sqlite3.Connection:
        uri = f"file:{quote(os.path.abspath(path))}?mode={mode}"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0, isolation_level=None, check_same_thread=False)
        except sqlite3.OperationalError as e:
            if not os.path.exists(path):
                raise LedgerNotInitialised(f"no ledger at {path}; create one with `cva-seal init`") from None
            raise _classify(e) from None
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @classmethod
    def init_ledger(cls, path: str | os.PathLike[str], key: KeyProvider, manifest: Mapping[str, Any], *,
                    durability: str = "per_record", group_n: int = 100, group_ms: int = 50,
                    clock: Callable[[], datetime] | None = None,
                    rng: Callable[[int], bytes] | None = None) -> SealedLedger:
        """Create a NEW ledger and write its genesis record. `manifest` is the deployment manifest
        (device_id, unit, profile_hash, checkpoint_every[, spec]); its `key_id` is filled from `key`.
        Refuses to touch an existing file — a ledger is initialised exactly once."""
        p = os.fspath(path)
        if durability not in DURABILITY_MODES:
            raise ValueError(f"durability must be one of {DURABILITY_MODES}")
        if os.path.exists(p):
            raise SealError(f"{p} already exists; refusing to re-initialise a ledger")
        man = {"spec": "cva-seal/1", **dict(manifest), "key_id": key.key_id}
        clock = clock or _utc_now
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            conn = sqlite3.connect(p, timeout=5.0, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.executescript(_SCHEMA + _TRIGGERS)
            conn.executemany("INSERT INTO meta(k, v) VALUES (?,?)", [
                ("store_version", STORE_VERSION), ("durability", durability), ("group_n", str(group_n)),
                ("group_ms", str(group_ms)), ("checkpoint_every", str(man["checkpoint_every"])),
                ("loss_window", "0 records" if durability == "per_record"
                 else f"up to {group_n} records / {group_ms} ms")])
        except (sqlite3.Error, OSError) as e:
            raise _classify(e) from None
        conn.close()
        led = cls.open(p, key=key, clock=clock, rng=rng, _require_genesis=False)
        led.append_typed("genesis", {"deployment_manifest": man})
        led._load_genesis()
        led._conn.execute("PRAGMA wal_checkpoint(FULL)")
        return led

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, key: KeyProvider | None = None, read_only: bool = False,
             clock: Callable[[], datetime] | None = None,
             rng: Callable[[int], bytes] | None = None, background_flush: bool = True,
             recover: bool = True, _require_genesis: bool = True) -> SealedLedger:
        """Open an existing ledger. Read-only when `read_only` (the field ledger under audit); a writer
        needs a key that IS the ledger's signing key. There is no implicit creation. `recover=False`
        skips the on-open coherence checks — for maintenance only (e.g. `rebuild_merkle_cache()` on a
        ledger whose cache the checks rejected)."""
        p = os.fspath(path)
        conn = cls._connect(p, "ro" if read_only else "rw")
        try:
            row = conn.execute("SELECT v FROM meta WHERE k='store_version'").fetchone()
        except sqlite3.DatabaseError as e:
            conn.close()
            raise _classify(e) from None
        if row is None or row[0] != STORE_VERSION:
            conn.close()
            raise LedgerCorrupt(f"{p}: not a {STORE_VERSION} ledger")
        if not read_only:
            if key is None:
                conn.close()
                raise WrongKey("a writable ledger needs its signing key")
            try:
                dur = conn.execute("SELECT v FROM meta WHERE k='durability'").fetchone()[0]
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute(f"PRAGMA synchronous = {'FULL' if dur == 'per_record' else 'NORMAL'}")
            except sqlite3.Error as e:
                conn.close()
                raise _classify(e) from None
        try:
            led = cls(conn, p, key, read_only=read_only, clock=clock or _utc_now, rng=rng,
                      require_genesis=_require_genesis)
        except (sqlite3.DatabaseError, NonCanonical, InvalidRecord, KeyError) as e:
            conn.close()
            raise (_classify(e) if isinstance(e, sqlite3.DatabaseError) else LedgerCorrupt(f"{p}: {e}")) from None
        except SealError:
            conn.close()
            raise
        if not read_only and _require_genesis:
            assert key is not None
            if key.key_id != led.active_key_id:
                led.close()
                raise WrongKey(f"key {key.key_id[:16]}… is not this ledger's active signing key "
                               f"({led.active_key_id[:16]}…)")
            if recover:
                led._recover()
            if led.durability == "group_commit" and background_flush:
                led._start_flusher()
        return led

    # -- reading ---------------------------------------------------------------------------------

    def _read_record(self, seq: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT rec FROM records WHERE seq=?", (seq,)).fetchone()
        return None if row is None else parse_strict(row[0].encode("ascii"))

    def size(self) -> int:
        """Number of records — O(log n): seq is contiguous from 0, so it is the last seq + 1. (`COUNT(*)`
        scans the table, and this is called on the hot path once per commit.) `_recover` separately
        checks that the row COUNT agrees, which is what proves the sequence really is gapless."""
        with self._lock:
            row = self._conn.execute("SELECT seq FROM records ORDER BY seq DESC LIMIT 1").fetchone()
            return 0 if row is None else int(row[0]) + 1

    def _row_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def tip(self) -> dict[str, Any] | None:
        """The last record (a parsed dict), or None for an empty database."""
        with self._lock:
            row = self._conn.execute("SELECT seq FROM records ORDER BY seq DESC LIMIT 1").fetchone()
            if row is None:
                return None
            if self._tip is not None and self._tip["seq"] == row[0]:
                return self._tip
            self._tip = self._read_record(row[0])
            self._tip_link = None                     # the cached link belonged to the OLD tip
            return self._tip

    def stored_records(self) -> Iterator[bytes]:
        """The exact stored bytes of every record, oldest first — the leaf data of the Merkle tree and
        what an export writes line by line."""
        cur = self._conn.execute("SELECT rec FROM records ORDER BY seq")
        for (rec,) in cur:
            yield rec.encode("ascii")

    def records(self) -> Iterator[Mapping[str, Any]]:
        """`InferenceLedgerSource.records()`: every record as a parsed mapping."""
        for data in self.stored_records():
            yield parse_strict(data)

    def registered_models(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT rec FROM records WHERE type='model_registration' ORDER BY seq").fetchall()
        return [parse_strict(r[0].encode("ascii")) for r in rows]

    def has_degraded_marker_for(self, spill_sha256: str) -> bool:
        pattern = f'%"spill_sha256":"{spill_sha256}"%'
        return self._conn.execute("SELECT 1 FROM records WHERE type='degraded_marker' AND rec LIKE ? LIMIT 1",
                                  (pattern,)).fetchone() is not None

    def merkle_tree(self) -> MerkleTree:
        return MerkleTree(SqliteNodeStore(self._conn), size=self.size())

    def root(self, size: int | None = None) -> bytes:
        with self._snapshot():
            return self.merkle_tree().root(size)

    def capabilities(self) -> set[str]:
        """ACTIVE probe. `INFERENCE_LEDGER` iff the database opens and its tip parses;
        `SIGNING_KEY` iff this is a writer whose key produced a signature that verified."""
        caps: set[str] = set()
        try:
            with self._lock:
                if self.tip() is not None:
                    caps.add("INFERENCE_LEDGER")
                self._conn.execute("SELECT 1").fetchone()
        except Exception:                                       # noqa: BLE001 - probing: any failure = absent
            return set()
        if not self.read_only and self._key is not None:
            self_test = getattr(self._key, "self_test", None)
            ok = self_test() if callable(self_test) else False
            if ok:
                caps.add("SIGNING_KEY")
        return caps

    # -- writing ---------------------------------------------------------------------------------

    def append(self, record: Mapping[str, Any]) -> str:
        """`AuditLedger.append`: write a typed record given as `{"type": ..., <section>: {...}}`
        (e.g. Backend's `scan_record(...)`). Least privilege: only `scan_record` and `analyst_event`
        may be written this way; every other type has a dedicated path (genesis, checkpoints, anchors,
        rotations, markers are this seat's)."""
        rtype = record.get("type")
        if rtype not in AUDIT_APPEND_TYPES:
            raise SealError(f"append() may write only {AUDIT_APPEND_TYPES}, not {rtype!r}")
        body = {k: v for k, v in record.items() if k != "type"}
        return self.append_typed(str(rtype), body)[0].record_hash

    def append_typed(self, rtype: str, body: Mapping[str, Any]) -> list[Written]:
        """Append one record. Returns what was written, including any checkpoint it triggered."""
        return self.append_many([(rtype, body)])

    def append_many(self, items: Sequence[tuple[str, Mapping[str, Any] | Callable[[int], Mapping[str, Any]]]],
                    payloads: Sequence[bytes] = (), *, rotate_to: KeyProvider | None = None) -> list[Written]:
        """Append several records — and the payloads they reference — ATOMICALLY (one transaction): all
        are written or none are. Returns a `Written(record_hash, seq, type)` for every record written,
        checkpoints included, in order. A body may be a callable taking the record's seq (for records that must
        state their own position — a rotation's `effective_seq`); a `key_rotation` needs `rotate_to`, the
        incoming key, which signs every record after it — including a checkpoint the rotation triggers.
        Payloads go in first, in the same transaction, so a record never
        references a payload that does not exist and one fsync covers both."""
        if self.read_only or self._key is None:
            raise LedgerUnavailable("this ledger handle is read-only or has no signing key")
        key = self._key
        with self._lock:
            if self._closed:
                raise LedgerUnavailable("ledger is closed")
            if self._flush_error is not None:
                raise LedgerUnavailable(f"background flush failed, durability is no longer guaranteed: "
                                        f"{self._flush_error} (call flush() once the storage recovers)")
            written: list[Written] = []
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as e:
                raise _classify(e) from None
            try:
                self._insert_payloads(payloads)
                row = self._conn.execute("SELECT seq FROM records ORDER BY seq DESC LIMIT 1").fetchone()
                prev_link: str | None = None
                if row is None:
                    prev = None
                elif self._tip is not None and self._tip["seq"] == row[0]:
                    prev, prev_link = self._tip, self._tip_link
                else:
                    prev = self._read_record(row[0])
                count = 0 if row is None else row[0] + 1
                tree = MerkleTree(SqliteNodeStore(self._conn), size=count)
                queue: list[tuple[str, Mapping[str, Any] | Callable[[int], Mapping[str, Any]]]] = list(items)
                while queue:
                    rtype, body = queue.pop(0)
                    if callable(body):
                        if prev is None:
                            raise SealError("no genesis record to follow")
                        body = body(prev["seq"] + 1)
                    if rtype == "key_rotation" and (rotate_to is None or
                                                    body["rotation"]["new_key_id"] != rotate_to.key_id):
                        raise SealError("a key_rotation record needs the incoming key it names (rotate_to)")
                    nonce = new_nonce(self._rng) if self._rng else new_nonce()
                    signed, data = seal_next(rtype, body, key=key, prev=prev, now=self._clock(), nonce=nonce,
                                             prev_link=prev_link)
                    rh = record_hash(signed)
                    self._conn.execute("INSERT INTO records(seq, type, nonce, rec_hash, rec) VALUES (?,?,?,?,?)",
                                       (signed["seq"], rtype, nonce, rh, data.decode("ascii")))
                    tree.append(leaf_hash(data))
                    written.append(Written(rh.hex(), signed["seq"], rtype))
                    prev, count = signed, count + 1
                    prev_link = link_from(rh, signed["signature"])
                    if rtype == "key_rotation":
                        assert rotate_to is not None
                        key = rotate_to                            # effective_seq == seq + 1: the next record is theirs
                    if rtype not in ("checkpoint", "genesis") and count % self.checkpoint_every == 0:
                        queue.insert(0, ("checkpoint", {"checkpoint": {"tree_size": count,
                                                                       "root_hash": tree.root(count).hex()}}))
                self._conn.execute("COMMIT")
            except BaseException as e:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass                                        # SQLite may already have rolled back (e.g. SQLITE_FULL)
                if isinstance(e, (SealError, ValueError, TypeError)) and not isinstance(e, LedgerUnavailable):
                    raise                                       # a caller/programming error, not a ledger failure
                if isinstance(e, sqlite3.IntegrityError):
                    raise SealError(f"record rejected by the ledger's uniqueness guard: {e}") from None
                if isinstance(e, (sqlite3.Error, OSError)):
                    raise _classify(e) from None
                raise
            self._tip, self._tip_link = prev, prev_link
            self._key, self.active_key_id = key, key.key_id
            self._after_commit(len(written))
            return written

    # -- key rotation, checkpoints, anchors (gate C7) ----------------------------------------------------

    def rotate_key(self, new_key: KeyProvider) -> Written:
        """Hand signing over to `new_key`: append a `key_rotation` record signed by the OUTGOING key, carrying
        the incoming public key and the incoming key's proof-of-possession, then sign with `new_key` from the
        next record. Durable before it returns — a rotation that only exists in a buffer would leave the next
        process opening the ledger with the wrong key."""
        if self._key is None:
            raise LedgerUnavailable("this ledger handle is read-only or has no signing key")
        out, key = self._key, new_key
        if key.key_id == out.key_id:
            raise SealError("cannot rotate to the key that is already active")

        def body(seq: int) -> Mapping[str, Any]:
            return rotation_body(out, key, seq)
        written = self.append_many([("key_rotation", body)], rotate_to=new_key)
        self.flush()
        return written[0]

    def checkpoint_now(self) -> Written:
        """Append a signed checkpoint at the CURRENT size, whatever the cadence — an anchor needs one at the tip."""
        with self._lock:
            n = self.size()
            item = ("checkpoint", {"checkpoint": {"tree_size": n, "root_hash": self.root(n).hex()}})
            return self.append_many([item])[0]

    def latest_checkpoint(self) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT seq FROM records WHERE type='checkpoint' ORDER BY seq DESC LIMIT 1").fetchone()
        return None if row is None else self._read_record(row[0])

    def records_after(self, seq: int) -> list[str]:
        """Types of the records after `seq`, in order."""
        return [str(r[0]) for r in self._conn.execute("SELECT type FROM records WHERE seq>? ORDER BY seq", (seq,))]

    # -- payloads ----------------------------------------------------------------------------------

    def _insert_payloads(self, payloads: Sequence[bytes]) -> None:
        """Inside an open transaction. Idempotent (same content, same address); if the address already
        holds DIFFERENT bytes that is corruption and is reported, never overwritten."""
        for data in payloads:
            h = hashlib.sha256(data).digest()
            cur = self._conn.execute("INSERT OR IGNORE INTO payloads(hash, data) VALUES (?,?)", (h, data))
            if cur.rowcount == 0:
                existing = self._conn.execute("SELECT data FROM payloads WHERE hash=?", (h,)).fetchone()
                if existing is None or bytes(existing[0]) != data:
                    raise PayloadCorrupt(f"{ref_for(data)}: the stored bytes do not match their address")

    def put_payloads(self, payloads: Sequence[bytes]) -> list[str]:
        """Store payloads in their own transaction and return their addresses (used for the once-per-load
        preprocessing specs; per-inference payloads travel with their record in `append_many`)."""
        if self.read_only or self._key is None:
            raise LedgerUnavailable("this ledger handle is read-only or has no signing key")
        with self._lock:
            if self._closed:
                raise LedgerUnavailable("ledger is closed")
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as e:
                raise _classify(e) from None
            try:
                self._insert_payloads(payloads)
                self._conn.execute("COMMIT")
            except BaseException as e:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(e, sqlite3.Error):
                    raise _classify(e) from None
                raise
            self._after_commit(0 if self.durability != "group_commit" else 1)
            return [ref_for(d) for d in payloads]

    def get_payload(self, ref: str) -> bytes:
        """The payload at `ref`, re-verified against its address. `PayloadMissing` if absent,
        `PayloadCorrupt` if the stored bytes no longer hash to it — never a silent wrong answer."""
        h = bytes.fromhex(hex_of(ref))
        with self._lock:
            try:
                row = self._conn.execute("SELECT data FROM payloads WHERE hash=?", (h,)).fetchone()
            except sqlite3.Error as e:
                raise _classify(e) from None
        if row is None:
            raise PayloadMissing(f"{ref} is not in the ledger's payload table")
        data = bytes(row[0])
        if hashlib.sha256(data).digest() != h:
            raise PayloadCorrupt(f"{ref}: stored bytes hash to {ref_for(data)}")
        return data

    def has_payload(self, ref: str) -> bool:
        h = bytes.fromhex(hex_of(ref))
        with self._lock:
            return self._conn.execute("SELECT 1 FROM payloads WHERE hash=?", (h,)).fetchone() is not None

    def payload_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM payloads").fetchone()[0])

    def _after_commit(self, n: int) -> None:
        """Group-commit accounting. Under `group_commit` a commit is acknowledged before it is fsynced;
        the background flusher makes it durable within `group_n` records or `group_ms` ms, whichever
        comes first. Without the flusher (tests) call `flush()` yourself."""
        if self.durability != "group_commit":
            return
        self._unflushed += n
        if self._unflushed >= self.group_n:
            self._wake.set()

    def _start_flusher(self) -> None:
        self._flusher = threading.Thread(target=self._flush_loop, name="cva-seal-flush", daemon=True)
        self._flusher.start()

    def _flush_loop(self) -> None:
        """Runs OFF the request path, so no commit ever pays for a group's fsyncs. It does not take the
        ledger lock: it fsyncs the payload files, then the `-wal` file by path (all committed frames live
        there until a checkpoint, and fsync on any descriptor flushes the file's cached pages)."""
        while not self._stop.is_set():
            self._wake.wait(timeout=self.group_ms / 1000)
            self._wake.clear()
            with self._lock:
                pending = self._unflushed
            if pending == 0:
                continue
            try:
                self._sync_now()
            except BaseException as e:                       # noqa: BLE001 - surfaced on the next commit
                self._flush_error = e
                continue
            with self._lock:
                self._unflushed = max(0, self._unflushed - pending)
                self._last_flush = time.monotonic()

    def _sync_now(self) -> None:
        for hook in self._flush_hooks:
            hook()
        wal = self.path + "-wal"
        if os.path.exists(wal):
            fd = os.open(wal, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def add_flush_hook(self, fn: Callable[[], object]) -> None:
        """Run `fn` at every flush, BEFORE the WAL is synced (the Sealer syncs deferred payloads here,
        so payloads are never less durable than the records that reference them)."""
        self._flush_hooks.append(fn)

    def flush(self) -> None:
        """Make every committed record durable NOW, synchronously. Clears a sticky background-flush error
        only if this succeeds."""
        with self._lock:
            if self.read_only or self._closed:
                return
            pending = self._unflushed
            try:
                for hook in self._flush_hooks:
                    hook()
                self._conn.execute("PRAGMA wal_checkpoint(FULL)")
            except (sqlite3.Error, OSError) as e:
                raise _classify(e) from None
            self._unflushed = max(0, self._unflushed - pending)
            self._last_flush = time.monotonic()
            self._flush_error = None

    @property
    def durable(self) -> bool:
        """True when everything committed so far has been fsynced."""
        return self.durability == "per_record" or self._unflushed == 0

    # -- integrity of the derived cache ----------------------------------------------------------

    def _recover(self) -> None:
        """On open (writers only): confirm the tip is coherent and the Merkle cache has the right shape.
        A cache that is short of nodes is `LedgerCorrupt` — fail closed, never silently repaired; use
        `rebuild_merkle_cache()` deliberately."""
        with self._snapshot():
            self._recover_checks()

    def _recover_checks(self) -> None:
        tip = self.tip()
        if tip is None:
            raise LedgerNotInitialised(f"{self.path}: empty ledger")
        row = self._conn.execute("SELECT type, nonce, rec_hash FROM records WHERE seq=?", (tip["seq"],)).fetchone()
        if row is None or row[0] != tip["type"] or row[1] != tip["nonce"] or bytes(row[2]) != record_hash(tip):
            raise LedgerCorrupt(f"{self.path}: tip record disagrees with its derived columns")
        n = self._row_count()
        if tip["seq"] != n - 1:
            raise LedgerCorrupt(f"{self.path}: {n} records but the tip is seq {tip['seq']} (gap or duplicate)")
        expected, level = 0, 0
        while (n >> level) > 0:
            expected += n >> level
            level += 1
        have = int(self._conn.execute("SELECT COUNT(*) FROM merkle_nodes").fetchone()[0])
        if have != expected:
            raise LedgerCorrupt(f"{self.path}: merkle cache holds {have} nodes, expected {expected} for "
                                f"{n} records — run rebuild_merkle_cache()")

    def verify_merkle_cache(self) -> bool:
        """Deep, O(n) check: does the cached root equal a root recomputed from the record bytes? An
        unreadable or missing cache node counts as "does not verify"."""
        with self._snapshot():
            leaves = [leaf_hash(d) for d in self.stored_records()]
            if not leaves:
                return True
            try:
                return self.merkle_tree().root() == mth(leaves)
            except MerkleError:
                return False

    def rebuild_merkle_cache(self) -> None:
        """Explicit maintenance: drop and recreate the derived `merkle_nodes` table from the records."""
        if self.read_only:
            raise LedgerUnavailable("read-only handle")
        with self._lock:
            leaves = [leaf_hash(d) for d in self.stored_records()]
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("DROP TABLE merkle_nodes")
                self._conn.execute("CREATE TABLE merkle_nodes (level INTEGER NOT NULL, idx INTEGER NOT NULL, "
                                   "hash BLOB NOT NULL, PRIMARY KEY (level, idx))")
                self._conn.execute("CREATE TRIGGER merkle_no_update BEFORE UPDATE ON merkle_nodes "
                                   "BEGIN SELECT RAISE(ABORT, 'append-only'); END")
                self._conn.execute("CREATE TRIGGER merkle_no_delete BEFORE DELETE ON merkle_nodes "
                                   "BEGIN SELECT RAISE(ABORT, 'append-only'); END")
                tree = MerkleTree(SqliteNodeStore(self._conn))
                for h in leaves:
                    tree.append(h)
                self._conn.execute("COMMIT")
            except BaseException as e:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(e, sqlite3.Error):
                    raise _classify(e) from None
                raise

    def chain_ok(self) -> bool:
        """Cheap self-check used by tests: the tip's link matches what the next record would carry."""
        tip = self.tip()
        return tip is not None and validate_and_link(tip)

    # -- lifecycle -------------------------------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._flusher is not None:
            self._flusher.join(timeout=5)
        with self._lock:
            if self._closed:
                return
            if not self.read_only:
                try:
                    for hook in self._flush_hooks:
                        hook()
                except OSError:
                    pass
            if not self.read_only:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
            self._closed = True
            self._conn.close()

    def __enter__(self) -> SealedLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<SealedLedger {self.path} {'ro' if self.read_only else 'rw'} records={self.size()}>"


def validate_and_link(record: Mapping[str, Any]) -> bool:
    try:
        validate_record(record, signed=True)
        link_hash(record)
    except (InvalidRecord, ValueError):
        return False
    return True


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
