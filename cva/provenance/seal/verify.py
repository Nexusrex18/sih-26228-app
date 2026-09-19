"""The standalone ledger verifier (plan §7.8, gate C5).

Imports nothing from `core/` (invariant 4): it reads a ledger — the SQLite file the Sealer writes, or its
strict JSONL export — and a trust root, and returns a `VerifyReport` of plain dataclasses. Mapping those
onto Backend's `Finding` happens in `checks/ledger_verify.py`, which MAY import `core/`.

The decision procedure, per record, is normative — its ORDER decides which finding names the cause when
one physical change trips several checks:

    1 PARSE       strict parse                  -> malformed_record | non_canonical_encoding
    2 DERIVED     type/nonce/rec_hash/seq columns agree with the record   -> derived_column_mismatch
    3 SEQ         seq is where the position says it should be
                    a gap whose missing numbers are NOT found later       -> record_delete
                    a gap whose missing numbers DO appear later            -> record_reorder
                    a duplicate of an earlier record, byte-identical       -> record_replay
                    a different record validly signed at a used seq        -> ledger_fork
    4 KEY         signed by the active key      -> key_unauthorised
                    (`adversarial` ONLY if the key is in the trust root AND its signature verifies;
                     an unknown key_id is indistinguishable from a flipped bit -> `indeterminate`)
    5 SIGNATURE   Ed25519 over TAG ‖ canonical bytes, under the claimed key   -> record_edit
    6 LINK        prev_record_hash == link(previous record)                   -> chain_broken
    7 NONCE       unseen in the current anchoring interval                    -> nonce_reuse
    8 TYPE RULES  genesis binding · degraded marker (declared gap) · anchor event · checkpoint root
    9 TIME        created_at going backwards    -> clock_regression  (INFO — never a tamper signal)

Then: expected-count reconciliation, model registration vs use, input hashes, payload hashes.

WHAT IS REPORTED, AND HOW
  * The FIRST break is reported precisely. Its downstream consequences — the successor's link, the later
    checkpoints whose roots changed, the partner of a swapped pair — are folded into that finding as one
    `cascade_suppressed` count. A thousand findings after one flipped bit is noise, and an analyst who sees
    noise stops reading. A LATER, independent break is still reported as its own finding.
  * The *fact* that the ledger differs from what the key holder signed is certain: it is arithmetic. The
    *label* (edit vs replace vs reorder) is the verifier's best reading of which check failed first, so
    every finding carries `primary_check`, the check that carries the certainty. `nature` is `indeterminate`
    for anything that could equally be storage corruption.
  * Some things are NOT visible in-band and are never guessed at: tail truncation, selective logging, and a
    history rewritten by the holder of the signing key. They are asserted as non-detections in the tests.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from .anchor import AnchorResult, load_anchor, verify_anchor
from .canonical import parse_strict
from .chain import link_from, rotation_pop_valid, verify_signature
from .errors import AnchorError, InvalidRecord, LedgerUnreadable, NonCanonical
from .keys import TrustRoot, load_trust_root
from .merkle import RootAccumulator, leaf_hash
from .records import genesis_prev_hash, record_hash, validate_record

SQLITE_MAGIC = b"SQLite format 3\x00"
SEVERITY_ORDER = ("info", "low", "medium", "high", "critical")

# class -> (severity, default nature). Every class that must quarantine is `high` or `critical`
# (backend_plan §7.9's D1 floor); a certain tamper class left at `medium` would be silently downgraded.
CLASS_PROFILE: dict[str, tuple[str, str]] = {
    "record_edit": ("critical", "indeterminate"),
    "record_replace": ("high", "indeterminate"),
    "record_replay": ("high", "indeterminate"),
    "record_delete": ("high", "indeterminate"),
    "record_reorder": ("high", "indeterminate"),
    "input_swap": ("critical", "indeterminate"),
    "model_swap": ("critical", "indeterminate"),
    "output_payload_mismatch": ("high", "indeterminate"),
    "output_mismatch": ("critical", "indeterminate"),
    "boundary_flip": ("medium", "indeterminate"),
    "tail_truncation": ("high", "indeterminate"),
    "ledger_fork": ("critical", "adversarial"),
    "key_unauthorised": ("high", "indeterminate"),
    "anchor_invalid": ("medium", "indeterminate"),
    "chain_broken": ("high", "indeterminate"),
    "nonce_reuse": ("high", "indeterminate"),
    "genesis_mismatch": ("high", "indeterminate"),
    "checkpoint_mismatch": ("high", "indeterminate"),
    "ledger_incomplete": ("high", "indeterminate"),
    "non_canonical_encoding": ("high", "indeterminate"),
    "derived_column_mismatch": ("high", "indeterminate"),
    "malformed_record": ("high", "indeterminate"),
    "degraded_gap": ("low", "quality"),
    "clock_regression": ("info", "quality"),
}
# Findings whose cause changes the bytes of an already-hashed record, so later Merkle roots stop matching.
TREE_TAINTING = {"record_edit", "record_delete", "record_reorder", "record_replay", "ledger_fork",
                 "malformed_record", "non_canonical_encoding", "key_unauthorised"}


@dataclass
class VerifyFinding:
    attack_class: str
    seq: int | None                 # None only for findings about the ledger as a whole
    position: int | None
    severity: str
    nature: str
    reason_base: str
    primary_check: str
    evidence: tuple[tuple[str, str, Any], ...] = ()          # (kind, caption, data)
    cascade_suppressed: int = 0
    cascade_note: str = ""

    @property
    def reason(self) -> str:
        tail = ""
        if self.cascade_suppressed:
            tail = (f" {self.cascade_note or 'Consequences of this break'}: {self.cascade_suppressed} further "
                    f"item(s) folded into this finding (cascade suppressed).")
        return self.reason_base + tail

    @property
    def target_ref(self) -> str:
        return f"seq:{self.seq}" if self.seq is not None else "ledger"


@dataclass
class VerifyReport:
    findings: list[VerifyFinding] = field(default_factory=list)
    records_checked: int = 0
    count_by_type: Counter[str] = field(default_factory=Counter)
    checkpoints_verified: int = 0
    anchors_in_chain: int = 0
    anchors_verified: int = 0                   # external anchor artefacts that verified AND match this ledger
    anchors_invalid: int = 0                    # supplied anchors that could not be used (bad signature, malformed)
    anchor_results: list[AnchorResult] = field(default_factory=list)
    anchored_records: int = 0                   # leading records fixed by the strongest verified anchor
    attested_not_after: str | None = None       # witness-bounded time for those records (boundary clock, not host)
    rotations: int = 0
    active_key_id: str | None = None
    last_anchor_seq: int | None = None
    unwitnessed_records: int = 0                # records after the last anchor: the window that still trusts the key holder
    declared_gaps: int = 0
    payloads_checked: int = 0
    payloads_missing: int = 0
    inputs_checked: int = 0
    inputs_unavailable: int = 0
    durability: str | None = None
    loss_window: str | None = None
    source_kind: str = ""
    limitations: list[str] = field(default_factory=list)
    verified_seqs: set[int] = field(default_factory=set)      # records whose signature verified (recompute's eligible set)

    @property
    def clean(self) -> bool:
        """No finding above `info`. (A declared `degraded_gap` is a true report and is `low`, so a ledger
        with a declared hole is not 'clean' — it is 'declared'.)"""
        return all(f.severity == "info" for f in self.findings)

    def by_class(self, attack_class: str) -> list[VerifyFinding]:
        return [f for f in self.findings if f.attack_class == attack_class]

    def classes(self) -> list[str]:
        return [f.attack_class for f in self.findings]

    @property
    def worst(self) -> str:
        return max((f.severity for f in self.findings), key=SEVERITY_ORDER.index, default="info")


# --- reading a ledger ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """One stored record exactly as found. `*_col` fields exist only for a SQLite source: they are the
    DERIVED columns, cross-checked against the record and never trusted."""

    data: bytes
    seq_col: int | None = None
    type_col: str | None = None
    nonce_col: str | None = None
    hash_col: bytes | None = None
    format_error: str | None = None


class SqliteSource:
    kind = "sqlite"

    def __init__(self, path: str | os.PathLike[str]) -> None:
        p = os.fspath(path)
        try:
            self._conn = sqlite3.connect(f"file:{quote(os.path.abspath(p))}?mode=ro", uri=True, timeout=5.0)
            self._conn.text_factory = bytes           # never let a damaged TEXT cell raise: keep the raw bytes
            self._conn.execute("SELECT 1 FROM records LIMIT 1").fetchall()
        except sqlite3.Error as e:
            raise LedgerUnreadable(f"{p}: cannot be read as a ledger ({e})") from None
        self.meta: dict[str, str] = {}
        try:
            self.meta = {bytes(k).decode("latin-1"): bytes(v).decode("latin-1")
                         for k, v in self._conn.execute("SELECT k, v FROM meta").fetchall()}
        except sqlite3.Error:
            pass

    def rows(self) -> Iterable[Row]:
        try:
            cur = self._conn.execute("SELECT seq, type, nonce, rec_hash, rec FROM records ORDER BY rowid").fetchall()
        except sqlite3.Error as e:
            raise LedgerUnreadable(f"records cannot be read: {e}") from None
        for seq, rtype, nonce, rec_hash, rec in cur:
            yield Row(_as_bytes(rec), seq if isinstance(seq, int) else None,
                      _as_bytes(rtype).decode("latin-1"), _as_bytes(nonce).decode("latin-1"),
                      _as_bytes(rec_hash))

    def payload(self, address_hex: str) -> tuple[bool, bytes | None]:
        """(a row exists at this address, its data). A missing table reads as 'no payloads', not an error."""
        try:
            row = self._conn.execute("SELECT data FROM payloads WHERE hash=?", (bytes.fromhex(address_hex),)).fetchone()
        except sqlite3.Error:
            return False, None
        return (row is not None), (None if row is None else _as_bytes(row[0]))

    def close(self) -> None:
        self._conn.close()


class JsonlSource:
    """A ledger export: EXACTLY the stored record bytes, one per line, every line — including the last —
    ending in `\\n`, no blank lines, no BOM. The export format is strict because the 'no undetected
    single-bit change' claim only holds if a flipped separator is itself a finding."""

    kind = "jsonl"
    meta: dict[str, str] = {}

    def __init__(self, data: bytes, payloads: Mapping[str, bytes] | None = None) -> None:
        self._data = data
        self._payloads = payloads

    def rows(self) -> Iterable[Row]:
        if not self._data:
            return
        lines = self._data.split(b"\n")
        unterminated = lines[-1] != b""
        if not unterminated:
            lines.pop()                                # the empty string after the final "\n"
        for i, line in enumerate(lines):
            err = "the last line is not terminated by a newline" if (unterminated and i == len(lines) - 1) else None
            yield Row(line, format_error=err)

    def payload(self, address_hex: str) -> tuple[bool, bytes | None]:
        if self._payloads is None:
            return False, None
        data = self._payloads.get(address_hex)
        return (data is not None), data

    def close(self) -> None:
        return None


def _as_bytes(v: Any) -> bytes:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    return b"" if v is None else str(v).encode("utf-8", "replace")


def open_source(source: Any, payloads: Mapping[str, bytes] | None = None) -> SqliteSource | JsonlSource:
    """Open a ledger from a path (SQLite file or JSONL export) or from export bytes. `payloads` (address hex ->
    bytes) supplies the payloads a JSONL export does not carry; a SQLite ledger has its own. A prepared source
    is returned as is — but note `verify_ledger` CLOSES the source it is given, so pass a path to reuse one."""
    if isinstance(source, (SqliteSource, JsonlSource)):
        return source
    if isinstance(source, (bytes, bytearray)):
        return JsonlSource(bytes(source), payloads)
    p = os.fspath(source)
    try:
        with open(p, "rb") as f:
            head = f.read(len(SQLITE_MAGIC))
            if head == SQLITE_MAGIC:
                return SqliteSource(p)
            rest = f.read()
    except OSError as e:
        raise LedgerUnreadable(f"{p}: {e.strerror}") from None
    return JsonlSource(head + rest, payloads)


def export_payloads(ledger_path: str | os.PathLike[str], out_path: str | os.PathLike[str]) -> int:
    """Write the payload SIDECAR for an export: one JSON object per line, `{"address": <sha256 hex>, "b64": ...}`.
    A JSONL export carries records only; without this the payload checks (`output_payload_mismatch`) and
    `prov.recompute` have nothing to compare against. The address is what the record committed to, so the
    sidecar needs no trust of its own: a wrong payload simply fails to hash to its address."""
    src = SqliteSource(ledger_path)
    try:
        n = 0
        with open(out_path, "wb") as f:
            for h, data in src._conn.execute("SELECT hash, data FROM payloads ORDER BY hash").fetchall():
                f.write(json.dumps({"address": _as_bytes(h).hex(), "b64": base64.b64encode(_as_bytes(data)).decode("ascii")},
                                   separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n")
                n += 1
        return n
    except sqlite3.Error as e:
        raise LedgerUnreadable(f"payloads cannot be read: {e}") from None
    finally:
        src.close()


def load_payloads(path: str | os.PathLike[str]) -> dict[str, bytes]:
    """Read a payload sidecar (address hex -> bytes). Strict: a malformed line raises `LedgerUnreadable`."""
    out: dict[str, bytes] = {}
    try:
        with open(path, "rb") as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    if set(obj) != {"address", "b64"}:
                        raise ValueError("keys")
                    out[str(obj["address"])] = base64.b64decode(obj["b64"], validate=True)
                except (ValueError, TypeError, KeyError):
                    raise LedgerUnreadable(f"{path}: line {i} is not a payload sidecar entry") from None
    except OSError as e:
        raise LedgerUnreadable(f"{path}: {e.strerror}") from None
    return out


def export_records(ledger_path: str | os.PathLike[str], out_path: str | os.PathLike[str]) -> int:
    """Write the strict JSONL export (each line = the stored record bytes verbatim, `\\n`-terminated).
    Returns the number of records. This is what makes verification possible with the live database absent."""
    src = SqliteSource(ledger_path)
    try:
        n = 0
        with open(out_path, "wb") as f:
            for row in src.rows():
                f.write(row.data + b"\n")
                n += 1
        return n
    finally:
        src.close()


# --- helpers ----------------------------------------------------------------------------------------

def _short(key_id: str) -> str:
    return f"{key_id[:8]}…{key_id[-4:]}"


def _snippet(data: bytes, limit: int = 1500) -> str:
    return data[:limit].decode("ascii", "backslashreplace") + ("…" if len(data) > limit else "")


def _describe(rec: Mapping[str, Any]) -> str:
    model = rec.get("model", {}).get("id") if isinstance(rec.get("model"), dict) else None
    extra = f", model {model}" if model else ""
    return f"seq {rec['seq']} ({rec['type']}, {rec['created_at_utc']}{extra})"


# --- memoised, pure per-record work ------------------------------------------------------------------
# Both functions are pure functions of the exact stored bytes, so caching them can never go stale; it just
# means re-verifying a ledger (a dashboard re-check, a fuzz sweep) does not re-parse, re-canonicalise and
# re-verify records that did not change. Bounded, so a million-record ledger cannot exhaust memory.

@lru_cache(maxsize=16384)
def _analyse(data: bytes) -> tuple[dict[str, Any] | None, tuple[str, str] | None, str | None, bytes | None]:
    """(record, None, link, record_hash) for a valid stored record, or (None, (code, detail), None, None)."""
    try:
        rec = parse_strict(data)
    except NonCanonical as e:
        return None, (e.code, e.detail), None, None
    try:
        validate_record(rec, signed=True)
    except InvalidRecord as e:
        return None, ("malformed_record", e.detail if e.path == "$" else f"{e.path}: {e.detail}"), None, None
    rh = record_hash(rec)
    return rec, None, link_from(rh, rec["signature"]), rh


@lru_cache(maxsize=16384)
def _sig_valid(data: bytes, public_key: bytes) -> bool:
    rec = _analyse(data)[0]
    return rec is not None and verify_signature(rec, public_key)


# --- the verifier -----------------------------------------------------------------------------------

def verify_ledger(source: Any, *, trust_root: TrustRoot | str | os.PathLike[str],
                  expect_deployment: str | Mapping[str, Any] | None = None,
                  reference_manifest: Mapping[str, Iterable[str]] | None = None,
                  input_resolver: Callable[[Mapping[str, Any]], bytes | None] | None = None,
                  expected_count: int | None = None,
                  anchors: Iterable[Any] | None = None,
                  payloads: Mapping[str, bytes] | None = None) -> VerifyReport:
    """Verify a ledger. See the module docstring for the procedure.

    `anchors` are external anchor artefacts (dicts, bytes, paths, or `logbook_anchor(...)` dicts): each is
    verified against the trust root and then against the ledger — a tree size beyond the ledger's length is
    `tail_truncation` (D11), a root that does not match is `ledger_fork` (or is folded into the tamper that
    explains it). Without anchors those two are invisible in-band, and the report says so.

    `reference_manifest` maps model id -> the weight digests that are legitimate for it (an EXTERNAL
    registry; without one, model changes are only visible as in-chain re-registrations, reported `info`).
    `input_resolver(record)` returns the bytes of the input image a record claims, or None if unavailable.
    `expected_count` is the operator's independent count of sealed inferences (selective-logging check).
    """
    tr = trust_root if isinstance(trust_root, TrustRoot) else load_trust_root(trust_root)
    src = open_source(source, payloads)
    try:
        return _Verifier(src, tr, expect_deployment, reference_manifest, input_resolver, expected_count,
                         list(anchors or ())).run()
    finally:
        src.close()


class _Verifier:
    def __init__(self, src: SqliteSource | JsonlSource, tr: TrustRoot, expect_deployment: Any,
                 reference_manifest: Mapping[str, Iterable[str]] | None,
                 input_resolver: Callable[[Mapping[str, Any]], bytes | None] | None,
                 expected_count: int | None, anchors: list[Any]) -> None:
        self.src, self.tr = src, tr
        self.anchors = anchors
        # key_id -> (public key, role). Trust-root keys, plus keys introduced by a VERIFIED rotation.
        self.keys: dict[str, tuple[bytes, str]] = {k.key_id: (k.public_key, k.role) for k in tr.keys}
        self.rotated_in: dict[str, bytes] = {}
        self.retired: set[str] = set()
        self.active_key: str | None = None
        self.rejected: dict[str, VerifyFinding] = {}     # key ids introduced by a rotation that did not verify
        self.checkpoints: dict[int, dict[str, Any]] = {}
        self.prefix_roots: dict[int, str] = {}
        self.expect_deployment = expect_deployment
        self.manifest = None if reference_manifest is None else {k: set(v) for k, v in reference_manifest.items()}
        self.input_resolver = input_resolver
        self.expected_count = expected_count
        self.report = VerifyReport(source_kind=src.kind)
        self.report.durability = src.meta.get("durability")
        self.report.loss_window = src.meta.get("loss_window")
        self.last_primary: VerifyFinding | None = None
        self.last_primary_pos = -10
        self.tree_taint: VerifyFinding | None = None

    # -- emit ----------------------------------------------------------------------------------------

    def emit(self, cls: str, rec: Mapping[str, Any] | None, pos: int, reason: str, check: str, *,
             evidence: tuple[tuple[str, str, Any], ...] = (), nature: str | None = None,
             severity: str | None = None, primary: bool = True, cascade_note: str = "",
             seq: int | None = None) -> VerifyFinding:
        sev, nat = CLASS_PROFILE[cls]
        if seq is None:
            seq = int(rec["seq"]) if rec is not None and isinstance(rec.get("seq"), int) else pos
        f = VerifyFinding(cls, seq, pos, severity or sev, nature or nat, reason, check, evidence,
                          cascade_note=cascade_note)
        self.report.findings.append(f)
        if primary:
            self.last_primary, self.last_primary_pos = f, pos
            if cls in TREE_TAINTING and self.tree_taint is None:
                self.tree_taint = f
        return f

    def cascade(self, owner: VerifyFinding | None, note: str = "") -> bool:
        if owner is None:
            return False
        owner.cascade_suppressed += 1
        if note and not owner.cascade_note:
            owner.cascade_note = note
        return True

    # -- the walk ---------------------------------------------------------------------------------------

    def run(self) -> VerifyReport:
        rows = list(self.src.rows())
        rep = self.report
        rep.records_checked = len(rows)
        if not rows:
            self.emit("genesis_mismatch", None, 0, "The ledger holds no records. A ledger must begin with a "
                      "genesis record; there is nothing to verify, which is not the same as verified.",
                      "genesis_binding", primary=False)
            return rep

        parsed: list[tuple[dict[str, Any] | None, tuple[str, str] | None, str | None, bytes | None]] = []
        seq_positions: dict[int, list[int]] = {}
        for p, row in enumerate(rows):
            rec, err, link, rh = self._parse(row)
            parsed.append((rec, err, link, rh))
            if rec is not None:
                seq_positions.setdefault(rec["seq"], []).append(p)
                rep.count_by_type[rec["type"]] += 1

        acc = RootAccumulator()
        shift, shift_before_reorder = 0, 0
        pending: dict[int, VerifyFinding] = {}
        prev_link: str | None = None
        nonces: set[str] = set()
        last_created: str | None = None
        registrations: dict[str, dict[str, Any]] = {}
        sealed_ok: list[tuple[int, dict[str, Any]]] = []          # rows whose signature verified

        needed = self._anchor_sizes()
        for p, (rec, err, link, rh) in enumerate(parsed):
            row = rows[p]
            if p in needed:
                self.prefix_roots[p] = acc.root().hex()
            try:
                if rec is None:                                   # 1 PARSE
                    code, detail = err  # type: ignore[misc]
                    reason = (f"The record at position {p} is not a valid stored record: {detail}."
                              if code == "malformed_record" else
                              f"The record at position {p} is valid JSON but not in canonical form: {detail}. "
                              "A stored record is exactly its canonical bytes; re-serialisation is itself a finding.")
                    self.emit(code, None, p, reason, "parse" if code == "malformed_record" else "canonical_form",
                              evidence=(("json", "the stored bytes at this position", _snippet(row.data)),))
                    prev_link = None
                    continue

                # 2 DERIVED
                bad = self._derived_mismatch(row, rec, rh)
                if bad:
                    self.emit("derived_column_mismatch", rec, p,
                              f"Record {_describe(rec)}: the indexed column(s) {', '.join(bad)} disagree with the "
                              "record's own bytes. The record is the source of truth; a derived column that "
                              "disagrees was edited independently of it.", "derived_columns",
                              evidence=(("json", "the stored record", _snippet(row.data)),))

                # 3 SEQ
                expected = p + shift
                suppress_link = False
                is_replay_copy = False
                if rec["seq"] != expected:
                    if rec["seq"] in pending:
                        owner = pending.pop(rec["seq"])
                        self.cascade(owner, "The displaced record(s) that this reordering explains")
                        self.last_primary, self.last_primary_pos = owner, p
                        suppress_link = True
                        if not pending:
                            shift = shift_before_reorder
                    elif rec["seq"] > expected:
                        missing = list(range(expected, rec["seq"]))
                        if all(any(q > p for q in seq_positions.get(m, [])) for m in missing):
                            f = self.emit("record_reorder", rec, p,
                                          f"Record {_describe(rec)} appears at position {p}, ahead of seq "
                                          f"{missing[0]}–{missing[-1]}, which occur later in the ledger: records are out "
                                          "of order. Certain, not statistical.", "seq_position",
                                          evidence=(("table", "position vs seq", {"position": p, "seq": rec["seq"],
                                                                                 "expected": expected}),))
                            shift_before_reorder = shift
                            for m in missing:
                                pending[m] = f
                            shift = rec["seq"] - p
                        else:
                            f = self.emit("record_delete", rec, p,
                                          f"{len(missing)} record(s) are missing from the ledger: seq "
                                          f"{missing[0]}{'' if len(missing) == 1 else '–' + str(missing[-1])} "
                                          f"absent before {_describe(rec)}. Sequence numbers are gapless; a gap is "
                                          "a deletion. Certain, not statistical.", "seq_position",
                                          evidence=(("table", "gap", {"missing_seq": missing, "found_at_position": p}),),
                                          cascade_note="The link that named the deleted record", seq=missing[0])
                            del f
                            shift = rec["seq"] - p
                            suppress_link = True
                    else:                                                  # rec.seq < expected: behind
                        earlier = [q for q in seq_positions.get(rec["seq"], []) if q < p]
                        if earlier:
                            same = rows[earlier[0]].data == row.data
                            forked = (not same and self.active_key is not None and self._sig_ok(row.data, self.active_key))
                            if same:
                                f = self.emit("record_replay", rec, p,
                                              f"Record {_describe(rec)} is a byte-identical copy of the record already "
                                              f"at position {earlier[0]}: a replay. Certain, not statistical.",
                                              "seq_position", evidence=(("hash", "record hash (identical)",
                                                                         {"first": earlier[0], "copy": p}),))
                            elif forked:
                                f = self.emit("ledger_fork", rec, p,
                                              f"Two different records carry seq {rec['seq']} and BOTH verify under the "
                                              f"active key: the key holder signed two histories. Certain.",
                                              "ed25519_signature", nature="adversarial")
                            else:
                                f = self.emit("record_reorder", rec, p,
                                              f"Record {_describe(rec)} reuses a sequence number already held by the "
                                              f"record at position {earlier[0]} but is not a valid copy of it.",
                                              "seq_position")
                            f.cascade_note = "The record's link, which cannot match its new predecessor"
                            is_replay_copy = same          # a copy repeats its nonce by definition: one cause, one finding
                            shift -= 1
                            suppress_link = True
                        else:
                            f = self.emit("record_reorder", rec, p,
                                          f"Record {_describe(rec)} is behind its place (position {p}, expected seq "
                                          f"{expected}): records are out of order.", "seq_position")
                            suppress_link = True

                if p == 0 and rec["type"] != "genesis":
                    self.emit("genesis_mismatch", rec, p,
                              f"The first record is {_describe(rec)}, not a genesis record: the history does not "
                              "begin where a ledger must.", "genesis_binding")

                # 4 KEY (and 5 SIGNATURE)
                sig_ok = False
                if self.active_key is None and p == 0:
                    self.active_key = rec["key_id"]
                if p == 0 and rec["key_id"] not in self.tr.ledger_keys():
                    self.emit("key_unauthorised", rec, p,
                              f"The genesis record is signed by key {_short(rec['key_id'])}, which is not a ledger key "
                              "in the trust root shipped with this deployment.", "key_authorised")
                elif rec["key_id"] in self.rejected:
                    # signed by the incoming key of a rotation that did not verify: the same cause, already reported
                    self.cascade(self.rejected[rec["key_id"]], "Later records signed by the key that rotation named")
                    self.last_primary, self.last_primary_pos = self.rejected[rec["key_id"]], p
                elif rec["key_id"] != self.active_key:
                    known = self.pub(rec["key_id"])
                    if known is not None and _sig_valid(row.data, known):
                        role = self.keys[rec["key_id"]][1]
                        origin = ("rotated-out (retired)" if rec["key_id"] in self.retired else
                                  "trust-root" if rec["key_id"] in {k.key_id for k in self.tr.keys} else "known")
                        f = self.emit("key_unauthorised", rec, p,
                                      f"Record {_describe(rec)} verifies under key {_short(rec['key_id'])}, a {origin} "
                                      f"'{role}' key that is not authorised to sign ledger records here (the active key is "
                                      f"{_short(self.active_key or '')}). A holder of that key wrote this. Certain.",
                                      "key_authorised", nature="adversarial")
                        self._reject_rotation_target(rec, f)
                    elif known is not None:
                        self.emit("record_edit", rec, p,
                                  f"Record {_describe(rec)} names key {_short(rec['key_id'])} but does not verify under "
                                  "it: the record was changed or re-keyed after signing.", "ed25519_signature")
                    else:
                        f = self.emit("key_unauthorised", rec, p,
                                  f"Record {_describe(rec)} is signed under key {_short(rec['key_id'])}, which is not "
                                  "in the trust root. The arithmetic cannot say whether an attacker signed it with "
                                  "their own key or a bit of key_id was damaged, so the cause is left indeterminate.",
                                  "key_authorised")
                        self._reject_rotation_target(rec, f)
                else:
                    sig_ok = self._sig_ok(row.data, self.active_key)
                    prev_edit = self.last_primary
                    if (not sig_ok and prev_edit is not None and prev_edit.attack_class == "record_edit"
                            and self.last_primary_pos == p - 1):
                        # a RUN of records that fail their signatures (a renumbering, a bulk rewrite): one
                        # finding with a count, not one per record — and the run is still fully counted
                        self.cascade(prev_edit, "Consecutive records that also fail their signatures")
                        self.last_primary_pos = p
                        prev_edit.reason_base = prev_edit.reason_base.split(" [run:")[0] + f" [run: through seq {rec['seq']}]"
                    elif not sig_ok:
                        self.emit("record_edit", rec, p,
                                  f"Record {_describe(rec)} does not verify under key {_short(rec['key_id'])} (the "
                                  "active ledger key): the stored bytes differ from what was signed. Certain that the "
                                  "record is not what the key holder signed; the arithmetic does not say why (an edit, "
                                  "or storage corruption).", "ed25519_signature",
                                  evidence=(("json", "the stored record", _snippet(row.data)),))

                # 6 LINK
                if p == 0:
                    if rec["prev_record_hash"] != self.tr.deployment_manifest_hash:
                        self.emit("genesis_mismatch", rec, p,
                                  "The genesis record commits to a different deployment than the trust root: this "
                                  "history belongs to another device or unit (its manifest hash "
                                  f"{rec['prev_record_hash'][:12]}… ≠ the trust root's "
                                  f"{self.tr.deployment_manifest_hash[:12]}…). Certain.", "genesis_binding",
                                  evidence=(("hash", "deployment manifest hash", {
                                      "ledger": rec["prev_record_hash"], "trust_root": self.tr.deployment_manifest_hash}),))
                    self._check_expect_deployment(rec)
                elif prev_link is None or rec["prev_record_hash"] != prev_link:
                    adjacent = self.last_primary is not None and self.last_primary_pos in (p, p - 1)
                    if suppress_link or adjacent:
                        self.cascade(self.last_primary, "The chain link that names the record that is no longer there")
                    else:
                        self.emit("chain_broken", rec, p,
                                  f"Record {_describe(rec)} does not chain to the record before it: its "
                                  "prev_record_hash does not match the previous record's link (which covers that "
                                  "record's signature). Certain.", "hash_link",
                                  evidence=(("hash", "chain link", {"stored": rec["prev_record_hash"],
                                                                    "recomputed": prev_link}),))

                if sig_ok:
                    sealed_ok.append((p, rec))
                    # 7 NONCE
                    if rec["nonce"] in nonces and not is_replay_copy:
                        self.emit("nonce_reuse", rec, p,
                                  f"Record {_describe(rec)} reuses nonce {rec['nonce'][:8]}… already used in this "
                                  "anchoring interval. Each record's nonce must be unique. Certain.", "nonce_unique")
                    nonces.add(rec["nonce"])
                    # 8 TYPE RULES
                    self._type_rules(rec, p, acc, nonces, registrations)
                # 9 TIME
                if last_created is not None and rec["created_at_utc"] < last_created:
                    self.emit("clock_regression", rec, p,
                              f"The host clock stepped backwards at {_describe(rec)} ({rec['created_at_utc']} after "
                              f"{last_created}). Wall-clock time is untrusted and never used for ordering; recorded "
                              "for information only.", "clock_order", primary=False)
                last_created = rec["created_at_utc"]
                prev_link = link
            finally:
                acc.append(leaf_hash(row.data))

        if len(rows) in needed:
            self.prefix_roots[len(rows)] = acc.root().hex()
        rep.verified_seqs = {rec["seq"] for _, rec in sealed_ok}
        rep.active_key_id = self.active_key
        self._finish(rows, sealed_ok)
        return rep

    # -- anchors ----------------------------------------------------------------------------------------

    def _anchor_sizes(self) -> set[int]:
        """Parse the supplied anchors up front and return the tree sizes whose ledger prefix root is needed."""
        self._loaded: list[tuple[int, dict[str, Any] | None, str]] = []
        sizes: set[int] = set()
        for i, raw in enumerate(self.anchors):
            try:
                a = load_anchor(raw)
                size = int(a["tree_size"]) if a.get("kind") == "logbook" else int(a["checkpoint"]["checkpoint"]["tree_size"])
                if size < 0:
                    raise ValueError("negative tree size")
            except (AnchorError, KeyError, TypeError, ValueError) as e:
                self._loaded.append((i, None, str(e) or type(e).__name__))
                continue
            self._loaded.append((i, a, ""))
            sizes.add(size)
        return sizes

    def _tainted_before(self, size: int) -> bool:
        """Did a byte-changing finding occur inside the first `size` records — i.e. does it explain an anchor mismatch?"""
        t = self.tree_taint
        return t is not None and t.position is not None and t.position < size

    def _check_anchors(self, rows: list[Row]) -> None:
        rep, n = self.report, len(rows)
        for i, a, err in self._loaded:
            label = f"anchor #{i + 1}"
            if a is None:
                rep.anchors_invalid += 1
                self.emit("anchor_invalid", None, n, f"{label} cannot be used: {err}. It says nothing about the ledger "
                          "either way; without a usable anchor, tail truncation and a rewritten history stay invisible.",
                          "anchor_verify", primary=False, seq=None)
                continue
            try:
                res = verify_anchor(a, self.tr, extra_ledger_keys=self.rotated_in)
            except AnchorError as e:
                rep.anchors_invalid += 1
                self.emit("anchor_invalid", None, n, f"{label} cannot be used: {e}", "anchor_verify", primary=False, seq=None)
                continue
            rep.anchor_results.append(res)
            if not res.valid:
                rep.anchors_invalid += 1
                self.emit("anchor_invalid", None, n, f"{label} did not verify, so it is not used: "
                          + "; ".join(res.problems) + ". An anchor that fails its own signatures says nothing about "
                          "the ledger either way.", "anchor_verify", primary=False, seq=None,
                          evidence=(("json", "anchor problems", res.problems),))
                continue
            t = res.tree_size
            if n < res.records_covered:
                self.emit("tail_truncation", None, n,
                          f"{label} fixes the first {res.records_covered} records (tree size {t}"
                          f"{', plus its checkpoint' if res.checkpoint_bytes else ''}), but the ledger holds only {n}: "
                          f"at least {res.records_covered - n} record(s) sealed at the tail are missing. Certain "
                          "relative to the anchor; the chain alone could not show this.", "anchor_size",
                          primary=False, evidence=(("table", "anchor vs ledger", {
                              "anchor_tree_size": t, "anchor_records": res.records_covered, "ledger_records": n}),))
                continue
            root = self.prefix_roots.get(t)
            if root != res.root_hash:
                if self._tainted_before(t):
                    self.cascade(self.tree_taint, "Anchors whose fixed root no longer matches the ledger")
                else:
                    self.emit("ledger_fork", None, n,
                              f"{label} fixes root {res.root_hash[:12]}… for the first {t} records, but this ledger's "
                              f"first {t} records hash to {(root or '')[:12]}…, and every record in it verifies: two "
                              "different, individually valid histories exist, and only the key holder could have "
                              "signed both. Certain.", "anchor_root", primary=False, nature="adversarial",
                              evidence=(("hash", "merkle root at the anchored size", {
                                  "anchor": res.root_hash, "ledger": root, "tree_size": t}),))
                continue
            if res.checkpoint_bytes is not None and rows[t].data != res.checkpoint_bytes:
                if self._tainted_before(t + 1):
                    self.cascade(self.tree_taint, "Anchors whose checkpoint record no longer matches the ledger")
                else:
                    self.emit("ledger_fork", None, n,
                              f"{label}'s prefix matches, but the checkpoint record it carries is not the record at "
                              f"index {t} of this ledger: the checkpoint was re-signed or replaced. Certain.",
                              "anchor_checkpoint", primary=False, nature="adversarial")
                continue
            rep.anchors_verified += 1
            if res.records_covered > rep.anchored_records:
                rep.anchored_records = res.records_covered
                rep.attested_not_after = res.time_bound_utc
            elif res.records_covered == rep.anchored_records and res.time_bound_utc:
                rep.attested_not_after = min(filter(None, [rep.attested_not_after, res.time_bound_utc]))
        if rep.anchors_verified:
            rep.unwitnessed_records = max(0, n - rep.anchored_records)

    # -- pieces -----------------------------------------------------------------------------------------

    def _parse(self, row: Row) -> tuple[dict[str, Any] | None, tuple[str, str] | None, str | None, bytes | None]:
        if row.format_error:
            return None, ("malformed_record", row.format_error), None, None
        return _analyse(row.data)

    @staticmethod
    def _derived_mismatch(row: Row, rec: Mapping[str, Any], rh: bytes | None) -> list[str]:
        if row.type_col is None:
            return []
        bad = []
        if row.seq_col != rec["seq"]:
            bad.append("seq")
        if row.type_col != rec["type"]:
            bad.append("type")
        if row.nonce_col != rec["nonce"]:
            bad.append("nonce")
        if row.hash_col != rh:
            bad.append("rec_hash")
        return bad

    def pub(self, key_id: str | None) -> bytes | None:
        entry = None if key_id is None else self.keys.get(key_id)
        return None if entry is None else entry[0]

    def _sig_ok(self, data: bytes, key_id: str | None) -> bool:
        pub = self.pub(key_id)
        return pub is not None and _sig_valid(data, pub)

    def _check_expect_deployment(self, genesis: Mapping[str, Any]) -> None:
        exp = self.expect_deployment
        if exp is None:
            return
        want = exp if isinstance(exp, str) else genesis_prev_hash(exp)
        if genesis["prev_record_hash"] != want:
            self.emit("genesis_mismatch", genesis, 0,
                      "The genesis record commits to a different deployment than the one expected "
                      f"({genesis['prev_record_hash'][:12]}… ≠ {want[:12]}…). Certain.", "genesis_binding",
                      primary=False)

    def _type_rules(self, rec: dict[str, Any], p: int, acc: RootAccumulator, nonces: set[str],
                    registrations: dict[str, dict[str, Any]]) -> None:
        rep, t = self.report, rec["type"]
        if t == "checkpoint":
            cp = rec["checkpoint"]
            root_now = acc.root().hex()
            if cp["tree_size"] == acc.size() and cp["root_hash"] == root_now:
                rep.checkpoints_verified += 1
                self.checkpoints[rec["seq"]] = cp
            elif self.tree_taint is not None:
                self.cascade(self.tree_taint, "Later Merkle checkpoints whose roots no longer match")
            else:
                self.emit("checkpoint_mismatch", rec, p,
                          f"Checkpoint {_describe(rec)} claims tree size {cp['tree_size']} with root "
                          f"{cp['root_hash'][:12]}…, but the {acc.size()} records before it hash to "
                          f"{root_now[:12]}…. Certain.", "checkpoint_root",
                          evidence=(("hash", "merkle root", {"checkpoint": cp["root_hash"], "recomputed": root_now,
                                                            "tree_size_claimed": cp["tree_size"],
                                                            "tree_size_actual": acc.size()}),))
        elif t == "anchor_event":
            a = rec["anchor"]
            seen = self.checkpoints.get(a["checkpoint_seq"])
            if (seen is None or (seen["tree_size"], seen["root_hash"]) != (a["tree_size"], a["root_hash"])):
                if self.tree_taint is not None:
                    self.cascade(self.tree_taint, "Later anchor events whose checkpoints no longer match")
                else:
                    self.emit("checkpoint_mismatch", rec, p,
                              f"Anchor event {_describe(rec)} names a checkpoint (seq {a['checkpoint_seq']}, tree size "
                              f"{a['tree_size']}) that the ledger does not contain as stated. Certain.", "anchor_event",
                              primary=False, evidence=(("json", "anchor event", a),))
            rep.anchors_in_chain += 1
            rep.last_anchor_seq = rec["seq"]
            nonces.clear()                                         # D9: an anchor closes the nonce window
            nonces.add(rec["nonce"])
        elif t == "degraded_marker":
            g = rec["gap"]
            rep.declared_gaps += 1
            self.emit("degraded_gap", rec, p,
                      f"The ledger DECLARES a hole: {g['reported_count']} inference(s) between {g['first_unsealed_utc']} "
                      f"and {g['last_unsealed_utc']} could not be sealed (reason: {g['reason']}; spill file digest "
                      f"{g['spill_sha256'][:12]}…). A declared hole is evidence, not tampering.", "declared_gap",
                      primary=False, evidence=(("json", "declared gap", g),))
        elif t == "model_registration":
            self._registration(rec, p, registrations)
        elif t == "inference":
            self._inference_model(rec, p, registrations)
        elif t == "key_rotation":
            self._rotation(rec, p)

    def _reject_rotation_target(self, rec: dict[str, Any], owner: VerifyFinding) -> None:
        """A `key_rotation` that the active key did not sign never takes effect, so records signed by the key it
        names would each look unauthorised: they are the SAME finding, folded into this one."""
        if rec["type"] == "key_rotation":
            self.rejected[rec["rotation"]["new_key_id"]] = owner

    def _rotation(self, rec: dict[str, Any], p: int) -> None:
        """A rotation is valid iff the ACTIVE key signed it (already established: only records that passed the key
        and signature checks reach here) and the incoming key proves possession. Only then does the incoming key
        become active — at `effective_seq`, which the schema pins to seq + 1."""
        rot = rec["rotation"]
        if not rotation_pop_valid(rec):
            f = self.emit("key_unauthorised", rec, p,
                          f"Key rotation {_describe(rec)} names incoming key {_short(rot['new_key_id'])}, but that key's "
                          "proof-of-possession does not verify: whoever signed this rotation could not show that the "
                          "new key is held by the party rotating to it. The rotation is not applied.",
                          "rotation_pop", evidence=(("json", "rotation", rot),))
            self.rejected[rot["new_key_id"]] = f
            return
        self.keys[rot["new_key_id"]] = (bytes.fromhex(rot["new_public_key"]), "ledger")
        self.rotated_in[rot["new_key_id"]] = bytes.fromhex(rot["new_public_key"])
        if self.active_key is not None:
            self.retired.add(self.active_key)
        self.active_key = rot["new_key_id"]
        self.report.rotations += 1

    def _registration(self, rec: dict[str, Any], p: int, registrations: dict[str, dict[str, Any]]) -> None:
        m = rec["model"]
        old = registrations.get(m["id"])
        registrations[m["id"]] = m
        if self.manifest is not None:
            allowed = self.manifest.get(m["id"])
            if allowed is None or m["weights_sha256"] not in allowed:
                self.emit("model_swap", rec, p,
                          f"Model '{m['id']}' was registered with weights {m['weights_sha256'][:12]}…, which the "
                          "reference manifest does not list for it. Certain that these are not the registered weights.",
                          "model_registration", evidence=(("hash", "weights digest", {
                              "ledger": m["weights_sha256"], "reference": sorted(allowed) if allowed else None}),))
        elif old is not None and old["weights_sha256"] != m["weights_sha256"]:
            self.emit("model_swap", rec, p,
                      f"Model '{m['id']}' was re-registered with different weights ({old['weights_sha256'][:12]}… → "
                      f"{m['weights_sha256'][:12]}…). That is a legitimate reload or a swap; without an external "
                      "reference manifest the ledger alone cannot tell which. Informational.", "model_registration",
                      severity="info", primary=False)

    def _inference_model(self, rec: dict[str, Any], p: int, registrations: dict[str, dict[str, Any]]) -> None:
        m = rec["model"]
        reg = registrations.get(m["id"])
        if reg is None:
            self.emit("model_swap", rec, p, f"Inference {_describe(rec)} uses model '{m['id']}', but no model "
                      "registration for it precedes it in the ledger.", "model_registration")
        elif (m["weights_sha256"], m["arch_hash"], m["format"]) != (reg["weights_sha256"], reg["arch_hash"], reg["format"]):
            self.emit("model_swap", rec, p,
                      f"Inference {_describe(rec)} records weights {m['weights_sha256'][:12]}… but the model "
                      f"registration in effect for '{m['id']}' names {reg['weights_sha256'][:12]}…. Certain.",
                      "model_registration", evidence=(("hash", "weights digest", {
                          "inference": m["weights_sha256"], "registered": reg["weights_sha256"]}),))

    def _finish(self, rows: list[Row], sealed_ok: list[tuple[int, dict[str, Any]]]) -> None:
        rep = self.report
        n = len(rows)
        rep.unwitnessed_records = n if rep.last_anchor_seq is None else max(0, n - 1 - rep.last_anchor_seq)
        self._check_anchors(rows)

        if self.expected_count is not None:
            got = rep.count_by_type["inference"]
            if got != self.expected_count:
                self.emit("ledger_incomplete", None, n,
                          f"The ledger holds {got} sealed inference record(s) but the operator's independent count "
                          f"is {self.expected_count}: {abs(self.expected_count - got)} "
                          f"{'unaccounted for' if got < self.expected_count else 'more than expected'}. "
                          "A chain cannot show records that were never sealed; only an independent counter can.",
                          "count_reconcile", primary=False)

        for p, rec in sealed_ok:
            if rec["type"] != "inference":
                continue
            if self.input_resolver is not None:
                data = self.input_resolver(rec)
                if data is None:
                    rep.inputs_unavailable += 1
                else:
                    rep.inputs_checked += 1
                    actual = hashlib.sha256(data).hexdigest()
                    if actual != rec["input"]["sha256"]:
                        self.emit("input_swap", rec, p,
                                  f"The input image for {_describe(rec)} does not match the hash sealed with the "
                                  f"record (sealed {rec['input']['sha256'][:12]}…, found {actual[:12]}…): the "
                                  "image was swapped after sealing. The hash comparison, not recompute, carries this "
                                  "certainty.", "input_hash", primary=False,
                                  evidence=(("hash", "input sha256", {"sealed": rec["input"]["sha256"], "found": actual}),))
            self._payloads(rec, p)

    def _payloads(self, rec: dict[str, Any], p: int) -> None:
        rep = self.report
        out = rec["output"]
        if out["payload_ref"] != "sha256:" + out["jcs_sha256"]:
            self.emit("output_payload_mismatch", rec, p,
                      f"{_describe(rec)}: payload_ref does not point at the payload whose hash the record commits to "
                      "(output.jcs_sha256). The record contradicts itself.", "payload_hash", primary=False)
        # raw == decision for an unfiltered output: one address, checked (and reported) once
        targets: dict[str, str] = {out["jcs_sha256"]: "decision output"}
        targets.setdefault(out["raw_jcs_sha256"], "raw output")
        for addr, label in targets.items():
            exists, data = self.src.payload(addr)
            if not exists:
                rep.payloads_missing += 1
                continue
            rep.payloads_checked += 1
            if data is None or hashlib.sha256(data).hexdigest() != addr:
                self.emit("output_payload_mismatch", rec, p,
                          f"The stored {label} payload for {_describe(rec)} does not hash to the value sealed in the "
                          f"record ({addr[:12]}…): the stored output was edited after sealing. Certain.",
                          "payload_hash", primary=False,
                          evidence=(("hash", f"{label} payload", {"sealed": addr,
                                                                  "stored": hashlib.sha256(data or b"").hexdigest()}),))
