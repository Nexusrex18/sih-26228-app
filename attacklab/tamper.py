"""Inference-record tamper lab (Module C plan §10; gate C5).

Each scenario attacks a COPY of a real ledger the way someone with file access would: it opens the SQLite
file, drops the append-only triggers (they protect against accidents, not intent), and edits rows — or, for
the export form, edits lines. It is the only thing that makes "tamper-evident" a measured claim instead of
a sentence, and it exists to test this module's own output, so it lives with the Crypto seat.

REPRODUCIBILITY (PS §2.3): a scenario is `(id, seed)`. The clean ledger is built from deterministic keys, a
deterministic clock and a seeded nonce stream; every choice inside a scenario (which record to hit) comes
from `random.Random(seed)`. Same id + same seed => byte-identical tampered artefact, twice — asserted on a
canonical logical dump of the artefact (`dump_artefact`), because a SQLite file's bytes also depend on
page-allocation history that is not part of the ledger.

Two of these scenarios are deliberately tests of what is NOT detectable in-band (T4c tail truncation, T12
selective logging): the expected outcome is silence, and it is asserted, so the limitation in the coverage
statement is a measured fact rather than prose.

Gate C7 adds the two that need anchoring and key rotation — T8 (a fork, caught only against an anchor from the
other branch) and T10 (rotation abuse, three ways) — and gives T4c its second half: silent without an anchor,
`tail_truncation` with one.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import shutil
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cva.provenance.seal.canonical import canonical_bytes
from cva.provenance.seal.chain import MemoryChain, link_hash, seal_next
from cva.provenance.seal.constants import TAG_RECORD
from cva.provenance.seal.keys import EnvKeyProvider, TrustKey, TrustRoot
from cva.provenance.seal.records import SECTIONS, build_record, canon, genesis_prev_hash
from cva.provenance.seal.sealer import Sealer
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import VerifyReport, export_records, verify_ledger

DEVICE = {"device_id": "tamper-lab-01", "unit": "lab", "profile_hash": "7" * 64}
T0 = datetime(2026, 9, 19, 2, 0, 0, tzinfo=UTC)
MODEL_ID = "resnet50-v3"
TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _read_records(path: Path) -> list[dict[str, Any]]:
    c = sqlite3.connect(path)
    try:
        return [json.loads(r[0]) for r in c.execute("SELECT rec FROM records ORDER BY rowid")]
    finally:
        c.close()


# --- deterministic building blocks ------------------------------------------------------------------

def make_key(label: str) -> EnvKeyProvider:
    seed = hashlib.sha256(f"cva-tamper-lab/{label}".encode()).digest()
    return EnvKeyProvider("K", environ={"K": base64.b64encode(seed).decode()})


def _clock() -> Callable[[], datetime]:
    t = [T0]

    def tick() -> datetime:
        t[0] += timedelta(milliseconds=7)
        return t[0]
    return tick


def _rng(seed: int) -> Callable[[int], bytes]:
    r = random.Random(seed)
    return lambda n: r.randbytes(n)


def _weights(label: str) -> str:
    return hashlib.sha256(f"weights/{label}".encode()).hexdigest()


@dataclass
class Fixture:
    """A clean, verified ledger and everything needed to attack and re-verify it."""

    directory: Path
    ledger: Path
    trust: TrustRoot
    key: EnvKeyProvider
    attacker: EnvKeyProvider
    inputs: dict[int, bytes]                   # seq -> the input image bytes that seq sealed
    n_inferences: int
    seed: int
    checkpoint_every: int
    swap_registration_seq: int | None = None
    anchor: Path | None = None                 # an anchor exported after the last inference (covers the whole ledger)
    key_after: EnvKeyProvider | None = None    # the key that took over at `rotated_at_seq`, if the ledger was rotated
    rotated_at_seq: int | None = None

    def records(self) -> list[dict[str, Any]]:
        c = sqlite3.connect(self.ledger)
        try:
            return [json.loads(r[0]) for r in c.execute("SELECT rec FROM records ORDER BY rowid")]
        finally:
            c.close()

    def seqs(self, rtype: str) -> list[int]:
        return [r["seq"] for r in self.records() if r["type"] == rtype]


def build_clean_ledger(directory: str | Path, *, n: int = 60, seed: int = 1, checkpoint_every: int = 16,
                       model_swap_at: int | None = None, anchor: bool = False, diverge_at: int | None = None,
                       rotate_at: int | None = None) -> Fixture:
    """Seal `n` deterministic inferences. `model_swap_at=m` re-registers the SAME model id with different
    weights after `m` inferences — a legitimate-looking reload (T7a). `anchor` exports an anchor after the last
    inference. `diverge_at=k` seals different inputs from inference k on: the same key, manifest, clock and
    nonces, so the first k inferences are byte-identical to the undiverged build and the histories fork
    there (T8). `rotate_at=m` rotates the signing key after m inferences."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    key, attacker = make_key(f"ledger/{seed}"), make_key(f"attacker/{seed}")
    clock, rng = _clock(), _rng(seed)
    path = d / "ledger.db"
    led = SealedLedger.init_ledger(path, key, {**DEVICE, "checkpoint_every": checkpoint_every},
                                   clock=clock, rng=rng)
    trust = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    r = random.Random(seed * 7919)
    inputs: dict[int, bytes] = {}
    key_after = make_key(f"ledger-rotated/{seed}") if rotate_at is not None else None
    rotated_at_seq: int | None = None
    with Sealer.open(path, key=key, trust_root=trust, clock=clock, rng=rng) as s:
        model = s.register_model(id=MODEL_ID, weights_sha256=_weights("A"), arch_hash=_weights("arch"), format="onnx")
        cfg = s.register_config(preprocess_spec={"mean_e6": [485000, 456000, 406000]},
                                postprocess_spec={"conf_thr_e6": 250000}, runtime="lab 1.0",
                                version_pins_hash=_weights("pins"), code_commit="0" * 40)
        for i in range(n):
            if model_swap_at is not None and i == model_swap_at:
                model = s.register_model(id=MODEL_ID, weights_sha256=_weights("B"), arch_hash=_weights("arch"),
                                         format="onnx")
            if rotate_at is not None and i == rotate_at:
                assert key_after is not None
                s.rotate_key(key_after)
                rotated_at_seq = max(x["seq"] for x in _read_records(path) if x["type"] == "key_rotation")
            forked = diverge_at is not None and i >= diverge_at
            frame = hashlib.sha256(f"frame/{seed}/{i}{'/fork' if forked else ''}".encode()).digest() * 8
            out = {"task": "classify", "top": [{"cls": r.randrange(10), "conf": round(r.uniform(0.5, 0.99), 6)}]}
            rec = s.seal(frame, model, cfg, output=out, dims=(64, 64))
            assert rec.seq is not None
            inputs[rec.seq] = frame
    anchor_path: Path | None = None
    if anchor:
        from cva.provenance.seal.anchor import export_anchor
        with SealedLedger.open(path, key=key_after or key, clock=clock, rng=rng, background_flush=False) as led:
            anchor_path = d / "anchor.json"
            export_anchor(led, anchor_path, label="lab")
    fx = Fixture(d, path, trust, key, attacker, inputs, n, seed, checkpoint_every, anchor=anchor_path,
                 key_after=key_after, rotated_at_seq=rotated_at_seq)
    if model_swap_at is not None:
        regs = [x for x in fx.records() if x["type"] == "model_registration"]
        fx.swap_registration_seq = regs[-1]["seq"]
    return fx


# --- expectations and outcomes ---------------------------------------------------------------------

@dataclass(frozen=True)
class Expected:
    """What a correct verifier must report. `classes` is the EXACT ordered list of attack classes of the
    findings above `info` — no more, no fewer: an unrelated extra finding is a false alarm and a missing
    one is a miss, and both fail the matrix."""

    classes: tuple[str, ...]
    first_seq: int | None = None
    severity: str | None = None
    nature: str | None = None
    primary_check: str | None = None
    cascade_min: int = 0
    note: str = ""


@dataclass
class Outcome:
    scenario: str
    seed: int
    trust: TrustRoot
    db: Path | None
    export: Path | None
    verify_kwargs: dict[str, Any]
    expected: Expected
    reconcile: Expected | None = None            # the "only detectable against an independent count" scenarios:
    reconcile_kwargs: dict[str, Any] = field(default_factory=dict)      # ... what to pass to see it
    reconcile_more: tuple[tuple[str, Expected, dict[str, Any]], ...] = ()   # further (name, expectation, kwargs)

    def verify(self, form: str, **override: Any) -> VerifyReport:
        path = self.db if form == "db" else self.export
        assert path is not None, f"{self.scenario} has no {form} form"
        return verify_ledger(path, trust_root=self.trust, **{**self.verify_kwargs, **override})

    @property
    def forms(self) -> tuple[str, ...]:
        return tuple(f for f, p in (("db", self.db), ("export", self.export)) if p is not None)


# --- file-level attack primitives --------------------------------------------------------------------

class TamperDb:
    """A COPY of the ledger with the guards removed — exactly what an attacker with file access has."""

    def __init__(self, src: Path, dst: Path) -> None:
        shutil.copy(src, dst)
        self.path = dst
        self.c = sqlite3.connect(dst, isolation_level=None)
        for (name,) in self.c.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            self.c.execute(f"DROP TRIGGER {name}")

    def rec(self, seq: int) -> dict[str, Any]:
        (text,) = self.c.execute("SELECT rec FROM records WHERE seq=?", (seq,)).fetchone()
        return json.loads(text)                                        # type: ignore[no-any-return]

    def put(self, seq: int, obj: Mapping[str, Any], *, new_seq: int | None = None) -> None:
        """Rewrite a record AND its derived columns consistently — a competent attacker leaves no
        indexed-column disagreement (T15 is the sloppy one)."""
        unsigned = {k: v for k, v in obj.items() if k != "signature"}
        text = canonical_bytes(obj).decode("ascii")
        rec_hash = hashlib.sha256(canonical_bytes(unsigned)).digest()
        self.c.execute("UPDATE records SET seq=?, type=?, nonce=?, rec_hash=?, rec=? WHERE seq=?",
                       (new_seq if new_seq is not None else obj["seq"], obj["type"], obj["nonce"], rec_hash, text, seq))

    def raw(self, seq: int, text: str) -> None:
        self.c.execute("UPDATE records SET rec=? WHERE seq=?", (text, seq))

    def delete(self, seq: int) -> None:
        self.c.execute("DELETE FROM records WHERE seq=?", (seq,))

    def last_seq(self) -> int:
        return int(self.c.execute("SELECT MAX(seq) FROM records").fetchone()[0])

    def close(self) -> None:
        self.c.close()


def _flip_hex(h: str, at: int = -1) -> str:
    i = at % len(h)
    return h[:i] + "0123456789abcdef"[("0123456789abcdef".index(h[i]) + 1) % 16] + h[i + 1:]


def _stamp(s: str) -> datetime:
    return datetime.strptime(s, TS_FORMAT).replace(tzinfo=UTC)


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().split(b"\n")[:-1]


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"".join(x + b"\n" for x in lines))


def _pick(fx: Fixture, rng: random.Random, *, need_next: int = 0, of_type: str = "inference") -> int:
    seqs = fx.seqs(of_type)
    last = max(fx.seqs(of_type) + fx.seqs("checkpoint"))
    ok = [s for s in seqs if s + need_next <= last and s >= 3 and all(x in seqs for x in range(s, s + need_next))]
    return rng.choice(ok)


def _tampered_db(fx: Fixture, workdir: Path) -> TamperDb:
    workdir.mkdir(parents=True, exist_ok=True)
    return TamperDb(fx.ledger, workdir / "tampered.db")


def _finish_db(fx: Fixture, t: TamperDb, workdir: Path, name: str, seed: int, expected: Expected, *,
               kwargs: dict[str, Any] | None = None, export: bool = True, trust: TrustRoot | None = None,
               reconcile: Expected | None = None, reconcile_kwargs: dict[str, Any] | None = None,
               reconcile_more: tuple[tuple[str, Expected, dict[str, Any]], ...] = ()) -> Outcome:
    t.close()
    exp = workdir / "tampered.jsonl"
    if export:
        export_records(t.path, exp)
    return Outcome(name, seed, trust or fx.trust, t.path, exp if export else None, kwargs or {}, expected,
                   reconcile, reconcile_kwargs or {}, reconcile_more)


def _finish_export(fx: Fixture, lines: list[bytes], workdir: Path, name: str, seed: int, expected: Expected, *,
                   kwargs: dict[str, Any] | None = None, trust: TrustRoot | None = None) -> Outcome:
    workdir.mkdir(parents=True, exist_ok=True)
    exp = workdir / "tampered.jsonl"
    _write_lines(exp, lines)
    return Outcome(name, seed, trust or fx.trust, None, exp, kwargs or {}, expected)


def _clean_export(fx: Fixture, workdir: Path) -> list[bytes]:
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / "clean.jsonl"
    export_records(fx.ledger, tmp)
    return _lines(tmp)


# --- the scenarios ----------------------------------------------------------------------------------

Scenario = Callable[[Fixture, Path, int], Outcome]


def t1a_edit_output(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Change one field of record k. The signature no longer verifies; the successor's link and every later
    checkpoint follow from that, so they are folded into the one finding."""
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    rec = t.rec(k)
    rec["output"]["jcs_sha256"] = _flip_hex(rec["output"]["jcs_sha256"])
    t.put(k, rec)
    return _finish_db(fx, t, w, "T1a", seed, Expected(("record_edit",), k, "critical", "indeterminate",
                                                        "ed25519_signature", cascade_min=1))


def t1b_edit_payload(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Edit the STORED output, leave the signed record alone: the payload no longer hashes to what the
    signature covers. (Database form only — an export carries no payloads.)"""
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for r in fx.records():
        if r["type"] == "inference":
            counts[r["output"]["jcs_sha256"]] = counts.get(r["output"]["jcs_sha256"], 0) + 1
    k = rng.choice([r["seq"] for r in fx.records() if r["type"] == "inference"
                    and counts[r["output"]["jcs_sha256"]] == 1])
    t = _tampered_db(fx, w)
    addr = bytes.fromhex(t.rec(k)["output"]["jcs_sha256"])
    (data,) = t.c.execute("SELECT data FROM payloads WHERE hash=?", (addr,)).fetchone()
    t.c.execute("UPDATE payloads SET data=? WHERE hash=?", (bytes(data) + b" ", addr))
    return _finish_db(fx, t, w, "T1b", seed, Expected(("output_payload_mismatch",), k, "high", "indeterminate",
                                                        "payload_hash"), export=False)


def t1c_edit_and_relink(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Edit k AND rewrite k+1's link to match: the link is INSIDE k+1's signed bytes, so k+1 fails its own
    signature. This is exactly why the chain covers the signature (S4)."""
    k = _pick(fx, random.Random(seed), need_next=2)
    t = _tampered_db(fx, w)
    rec = t.rec(k)
    rec["output"]["jcs_sha256"] = _flip_hex(rec["output"]["jcs_sha256"])
    t.put(k, rec)
    nxt = t.rec(k + 1)
    from cva.provenance.seal.chain import link_from
    from cva.provenance.seal.records import record_hash
    nxt["prev_record_hash"] = link_from(record_hash(rec), rec["signature"])
    t.put(k + 1, nxt)
    return _finish_db(fx, t, w, "T1c", seed, Expected(("record_edit",), k, "critical", "indeterminate",
                                                        "ed25519_signature", cascade_min=1,
                                                        note="k and k+1 both fail their signatures: one run"))


def t2a_replace_with_attacker_key(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Replace record k with a perfectly well-formed record signed by the attacker's OWN key."""
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    old, prev = t.rec(k), t.rec(k - 1)
    forged, _ = seal_next("inference", {s: old[s] for s in SECTIONS["inference"]}, key=fx.attacker, prev=prev,
                          now=_stamp(old["created_at_utc"]), nonce=old["nonce"])
    t.put(k, forged)
    return _finish_db(fx, t, w, "T2a", seed, Expected(("key_unauthorised",), k, "high", "indeterminate",
                                                        "key_authorised", cascade_min=1,
                                                        note="an unknown key is indistinguishable from a flipped "
                                                             "key_id bit, so this is never called adversarial"))


def t2b_forged_signature_under_real_key_id(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Claim the legitimate key_id but sign with the attacker's key."""
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    old, prev = t.rec(k), t.rec(k - 1)
    unsigned = build_record("inference", seq=k, prev_record_hash=link_hash(prev), key_id=fx.key.key_id,
                            created_at_utc=old["created_at_utc"], nonce=old["nonce"],
                            body={s: old[s] for s in SECTIONS["inference"]})
    sig = fx.attacker.sign(TAG_RECORD + canon(unsigned))
    t.put(k, {**unsigned, "signature": sig.hex()})
    return _finish_db(fx, t, w, "T2b", seed, Expected(("record_edit",), k, "critical", "indeterminate",
                                                        "ed25519_signature", cascade_min=1))


def t3a_replay_copy_at_tip(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Append a byte-identical copy of an old record. (Export form: a database cannot hold two rows with one
    seq, so the DB variant is a different — and louder — attack, tested separately.)"""
    j = _pick(fx, random.Random(seed))
    lines = _clean_export(fx, w)
    lines.append(lines[j])
    return _finish_export(fx, lines, w, "T3a", seed, Expected(("record_replay",), j, "high", "indeterminate",
                                                                "seq_position", cascade_min=1))


def t3b_replay_with_seq_rewritten(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Replay, but rewrite the copy's seq to tip+1 so it looks in place. Rewriting seq breaks the signature."""
    j = _pick(fx, random.Random(seed))
    lines = _clean_export(fx, w)
    rec = json.loads(lines[j])
    rec["seq"] = len(lines)
    lines.append(canonical_bytes(rec))
    return _finish_export(fx, lines, w, "T3b", seed, Expected(("record_edit",), len(lines) - 1, "critical",
                                                                "indeterminate", "ed25519_signature",
                                                                cascade_min=1))


def t4a_delete_interior(fx: Fixture, w: Path, seed: int) -> Outcome:
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    t.delete(k)
    return _finish_db(fx, t, w, "T4a", seed, Expected(("record_delete",), k, "high", "indeterminate",
                                                        "seq_position", cascade_min=1))


def t4b_delete_and_renumber(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Delete k and renumber every successor to close the gap. Every renumbered record now fails its own
    signature — one run, one finding, fully counted."""
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    last = t.last_seq()
    t.delete(k)
    for s in range(k + 1, last + 1):
        rec = t.rec(s)
        rec["seq"] = s - 1
        t.put(s, rec, new_seq=s - 1)
    return _finish_db(fx, t, w, "T4b", seed, Expected(("record_edit",), k, "critical", "indeterminate",
                                                        "ed25519_signature", cascade_min=1))


def t4c_tail_truncation(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Delete the LAST m records. Nothing follows them to disagree: the chain is perfectly intact. NOT
    detectable in-band — only against an independent count (or, from C7, an anchor). The expected outcome
    without one is silence, and that is the point."""
    assert fx.anchor is not None, "build the fixture with anchor=True"
    m = random.Random(seed).randrange(3, 8)
    t = _tampered_db(fx, w)
    last = t.last_seq()
    for s in range(last - m + 1, last + 1):
        t.delete(s)
    dropped_inferences = sum(1 for r in fx.records() if r["type"] == "inference" and r["seq"] > last - m)
    return _finish_db(fx, t, w, "T4c", seed,
                      Expected((), note="undetectable in-band: no successor exists to disagree"),
                      reconcile=Expected(("ledger_incomplete",), None, "high", "indeterminate", "count_reconcile",
                                         note=f"{dropped_inferences} inference(s) gone"),
                      reconcile_kwargs={"expected_count": fx.n_inferences},
                      reconcile_more=(("anchor", Expected(("tail_truncation",), None, "high", "indeterminate",
                                                          "anchor_size", note="the anchor fixes records that are gone"),
                                       {"anchors": [fx.anchor]}),))


def t5a_reorder(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Swap two records. (Export form: in a database the seq column IS the row order, so a reorder there
    necessarily disagrees with the record — a louder variant, tested separately.)"""
    rng = random.Random(seed)
    j = _pick(fx, rng, need_next=1)
    k = rng.choice([s for s in fx.seqs("inference") if s > j + 2])
    lines = _clean_export(fx, w)
    lines[j], lines[k] = lines[k], lines[j]
    return _finish_export(fx, lines, w, "T5a", seed, Expected(("record_reorder",), k, "high", "indeterminate",
                                                                 "seq_position", cascade_min=1))


def t5b_reorder_and_renumber(fx: Fixture, w: Path, seed: int) -> Outcome:
    rng = random.Random(seed)
    j = _pick(fx, rng, need_next=1)
    k = rng.choice([s for s in fx.seqs("inference") if s > j + 3])
    lines = _clean_export(fx, w)
    lines[j], lines[k] = lines[k], lines[j]
    for pos in (j, k):
        rec = json.loads(lines[pos])
        rec["seq"] = pos
        lines[pos] = canonical_bytes(rec)
    return _finish_export(fx, lines, w, "T5b", seed, Expected(("record_edit", "record_edit"), j, "critical",
                                                                "indeterminate", "ed25519_signature"))


def t6_swap_input_image(fx: Fixture, w: Path, seed: int) -> Outcome:
    """The ledger is untouched; the image it points at is replaced. The HASH COMPARISON carries the certainty."""
    k = random.Random(seed).choice(fx.seqs("inference"))
    inputs = dict(fx.inputs)
    inputs[k] = b"a different image entirely" * 20
    t = _tampered_db(fx, w)
    return _finish_db(fx, t, w, "T6", seed, Expected(("input_swap",), k, "critical", "indeterminate", "input_hash"),
                      kwargs={"input_resolver": lambda rec: inputs.get(rec["seq"])})


def t7a_swap_model(fx: Fixture, w: Path, seed: int) -> Outcome:
    """A different model is loaded; the ledger honestly records the new digest under the SAME id. Against an
    external reference manifest that is a swap (critical). Without one, the ledger alone cannot tell a
    legitimate reload from a swap, and says so at `info`."""
    assert fx.swap_registration_seq is not None, "build the fixture with model_swap_at"
    t = _tampered_db(fx, w)
    return _finish_db(fx, t, w, "T7a", seed,
                      Expected(("model_swap",), fx.swap_registration_seq, "critical", "indeterminate",
                               "model_registration"),
                      kwargs={"reference_manifest": {MODEL_ID: {_weights("A")}}})


def t7b_edit_ledger_to_show_old_digest(fx: Fixture, w: Path, seed: int) -> Outcome:
    """After the swap, an attacker edits an inference to claim the OLD digest. That breaks its signature."""
    assert fx.swap_registration_seq is not None
    after = [s for s in fx.seqs("inference") if s > fx.swap_registration_seq + 1]
    k = random.Random(seed).choice(after[:-1])
    t = _tampered_db(fx, w)
    rec = t.rec(k)
    rec["model"]["weights_sha256"] = _weights("A")
    t.put(k, rec)
    return _finish_db(fx, t, w, "T7b", seed, Expected(("record_edit",), k, "critical", "indeterminate",
                                                        "ed25519_signature", cascade_min=1))


def t9_genesis_splice(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Device A's perfectly valid history presented as device B's."""
    t = _tampered_db(fx, w)
    other = genesis_prev_hash({**DEVICE, "device_id": "tamper-lab-99", "key_id": fx.key.key_id,
                               "checkpoint_every": fx.checkpoint_every, "spec": "cva-seal/1"})
    other_trust = TrustRoot(other, fx.trust.keys)
    return _finish_db(fx, t, w, "T9", seed, Expected(("genesis_mismatch",), 0, "high", "indeterminate",
                                                       "genesis_binding"), trust=other_trust)


def t11_nonce_reuse(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Two validly signed records carrying one nonce in the same anchoring interval. Needs the signing key,
    so the ledger is rebuilt by the key holder; export form only (a database's UNIQUE column would refuse)."""
    w.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    dup_at = rng.randrange(8, 20)
    key = fx.key
    nonces = random.Random(seed ^ 0xBEEF)
    state = {"n": 0, "dup": None}

    def draw(k: int) -> bytes:
        state["n"] += 1
        if state["n"] == dup_at:
            state["dup"] = nonces.randbytes(k)
            return state["dup"]                                            # type: ignore[return-value]
        if state["n"] == dup_at + 4:
            return state["dup"]                                            # type: ignore[return-value]
        return nonces.randbytes(k)

    chain = MemoryChain(key, clock=_clock(), rng=draw)
    chain.append("genesis", {"deployment_manifest": {**DEVICE, "key_id": key.key_id,
                                                      "checkpoint_every": 1000, "spec": "cva-seal/1"}})
    reg, body = fx.records()[1], fx.records()[2]                          # a registration, then inferences using it
    chain.append("model_registration", {"model": reg["model"], "config": reg["config"]})
    for _ in range(30):
        chain.append("inference", {s: body[s] for s in SECTIONS["inference"]})
    trust = TrustRoot(genesis_prev_hash(chain.records[0]["deployment_manifest"]),
                      (TrustKey(key.key_id, key.public_key, "ledger"),))
    second = dup_at + 4 - 1                                               # draw d is seq d-1 (genesis is draw 1)
    return _finish_export(fx, list(chain.stored), w, "T11", seed,
                          Expected(("nonce_reuse",), second, "high", "indeterminate", "nonce_unique"), trust=trust)


def t12_selective_logging(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Run N inferences, seal N-m. The chain is perfect AND incomplete; in-band undetectable. Only an
    independent counter (`expected_count`) sees it."""
    m = random.Random(seed).randrange(2, 7)
    t = _tampered_db(fx, w)
    return _finish_db(fx, t, w, "T12", seed, Expected((), note="undetectable in-band"),
                      reconcile=Expected(("ledger_incomplete",), None, "high", "indeterminate", "count_reconcile"),
                      reconcile_kwargs={"expected_count": fx.n_inferences + m})


def t13_noncanonical_encoding(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Re-serialise a record with a different key order: the same logical record, different bytes."""
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    rec = t.rec(k)
    t.raw(k, json.dumps(dict(reversed(list(rec.items()))), separators=(",", ":")))
    return _finish_db(fx, t, w, "T13", seed, Expected(("non_canonical_encoding",), k, "high", "indeterminate",
                                                        "canonical_form", cascade_min=1))


def t14_uppercase_hex(fx: Fixture, w: Path, seed: int) -> Outcome:
    """`deadbeef` and `DEADBEEF` are the same bytes; only one spelling may exist."""
    k = _pick(fx, random.Random(seed), need_next=1)
    t = _tampered_db(fx, w)
    (text,) = t.c.execute("SELECT rec FROM records WHERE seq=?", (k,)).fetchone()
    marker = '"prev_record_hash":"'
    i = text.index(marker) + len(marker)
    t.raw(k, text[:i] + text[i:i + 64].upper() + text[i + 64:])
    return _finish_db(fx, t, w, "T14", seed, Expected(("malformed_record",), k, "high", "indeterminate",
                                                        "parse", cascade_min=1))


def t15_derived_column_only(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Edit only the indexed columns, never the record. Database form only."""
    rng = random.Random(seed)
    k = _pick(fx, rng)
    t = _tampered_db(fx, w)
    col = rng.choice(["type", "nonce", "rec_hash"])
    val: Any = {"type": "checkpoint", "nonce": "f" * 32, "rec_hash": bytes(32)}[col]
    t.c.execute(f"UPDATE records SET {col}=? WHERE seq=?", (val, k))
    return _finish_db(fx, t, w, "T15", seed, Expected(("derived_column_mismatch",), k, "high", "indeterminate",
                                                        "derived_columns", note=f"column {col}"), export=False)


def t8_ledger_fork(fx: Fixture, w: Path, seed: int) -> Outcome:
    """Two valid histories that diverge at k, both signed by the key holder. Each is perfectly consistent, so
    neither verifies as anything but clean ON ITS OWN; an anchor taken from the honest branch is what shows the
    other one is not the history that was witnessed."""
    assert fx.anchor is not None, "build the fixture with anchor=True"
    k = random.Random(seed).randrange(10, fx.n_inferences - 5)
    forked = build_clean_ledger(w / "fork", n=fx.n_inferences, seed=fx.seed, checkpoint_every=fx.checkpoint_every,
                                anchor=True, diverge_at=k)
    exp = w / "tampered.jsonl"
    export_records(forked.ledger, exp)
    clean_alone = Expected((), note="each branch is internally valid: nothing in-band distinguishes them")
    return Outcome("T8", seed, fx.trust, forked.ledger, exp, {"anchors": [fx.anchor]},
                   Expected(("ledger_fork",), None, "critical", "adversarial", "anchor_root"),
                   reconcile=clean_alone, reconcile_kwargs={"anchors": []})


def _append_forged(t: TamperDb, key: EnvKeyProvider, rtype: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Append a record signed by `key` at the tip of an attacker's copy, keeping every derived column consistent."""
    last = t.rec(t.last_seq())
    signed, _ = seal_next(rtype, body, key=key, prev=last, now=_stamp(last["created_at_utc"]) + timedelta(seconds=1),
                          nonce=hashlib.sha256(f"forged/{last['seq']}/{rtype}".encode()).hexdigest()[:32])
    unsigned = {k: v for k, v in signed.items() if k != "signature"}
    t.c.execute("INSERT INTO records(seq, type, nonce, rec_hash, rec) VALUES (?,?,?,?,?)",
                (signed["seq"], rtype, signed["nonce"], hashlib.sha256(canonical_bytes(unsigned)).digest(),
                 canonical_bytes(signed).decode("ascii")))
    return signed


def t10a_rotation_by_unknown_key(fx: Fixture, w: Path, seed: int) -> Outcome:
    """The attacker signs a `key_rotation` with THEIR key — which is not the active key — naming a second key of
    theirs, then continues writing as that second key. The key state machine refuses the rotation; the records
    after it are the same cause and fold into it."""
    from cva.provenance.seal.chain import rotation_body
    t = _tampered_db(fx, w)
    a1, a2 = fx.attacker, make_key(f"attacker2/{seed}")
    seq = t.last_seq() + 1
    _append_forged(t, a1, "key_rotation", rotation_body(a1, a2, seq))
    for _ in range(3):
        _append_forged(t, a2, "inference", {s: t.rec(3)[s] for s in SECTIONS["inference"]})
    return _finish_db(fx, t, w, "T10a", seed, Expected(("key_unauthorised",), seq, "high", "indeterminate",
                                                         "key_authorised", cascade_min=3,
                                                         note="an unknown key is never called adversarial"))


def t10b_rotation_without_proof_of_possession(fx: Fixture, w: Path, seed: int) -> Outcome:
    """The ACTIVE key signs a rotation to a key whose holder never proved possession (the PoP is another key's).
    Whoever holds the outgoing key cannot smuggle in a key they cannot show they hold... or hand the ledger to
    a key nobody consented to. The rotation is not applied; later records by that key fold into the finding."""
    from cva.provenance.seal.chain import rotation_body
    t = _tampered_db(fx, w)
    newk = make_key(f"unproven/{seed}")
    seq = t.last_seq() + 1
    body = rotation_body(fx.key, newk, seq)
    body["rotation"]["new_key_pop"] = fx.attacker.sign(b"not the proof").hex()
    _append_forged(t, fx.key, "key_rotation", body)
    for _ in range(2):
        _append_forged(t, newk, "inference", {s: t.rec(3)[s] for s in SECTIONS["inference"]})
    return _finish_db(fx, t, w, "T10b", seed, Expected(("key_unauthorised",), seq, "high", "indeterminate",
                                                         "rotation_pop", cascade_min=2))


def t10c_retired_key_resumes(fx: Fixture, w: Path, seed: int) -> Outcome:
    """After a legitimate rotation the OLD key's holder appends a record. The key is known and its signature
    verifies, so this one IS attributable: adversarial. (A retired key is exactly what a stolen backup holds.)"""
    assert fx.rotated_at_seq is not None, "build the fixture with rotate_at"
    t = _tampered_db(fx, w)
    seq = t.last_seq() + 1
    _append_forged(t, fx.key, "inference", {s: t.rec(t.last_seq() - 3)[s] for s in SECTIONS["inference"]
                                            if s in t.rec(t.last_seq() - 3)})
    return _finish_db(fx, t, w, "T10c", seed, Expected(("key_unauthorised",), seq, "high", "adversarial",
                                                         "key_authorised"))


SCENARIOS: dict[str, Scenario] = {
    "T1a": t1a_edit_output, "T1b": t1b_edit_payload, "T1c": t1c_edit_and_relink,
    "T2a": t2a_replace_with_attacker_key, "T2b": t2b_forged_signature_under_real_key_id,
    "T3a": t3a_replay_copy_at_tip, "T3b": t3b_replay_with_seq_rewritten,
    "T4a": t4a_delete_interior, "T4b": t4b_delete_and_renumber, "T4c": t4c_tail_truncation,
    "T5a": t5a_reorder, "T5b": t5b_reorder_and_renumber, "T6": t6_swap_input_image,
    "T7a": t7a_swap_model, "T7b": t7b_edit_ledger_to_show_old_digest, "T8": t8_ledger_fork, "T9": t9_genesis_splice,
    "T10a": t10a_rotation_by_unknown_key, "T10b": t10b_rotation_without_proof_of_possession,
    "T10c": t10c_retired_key_resumes,
    "T11": t11_nonce_reuse, "T12": t12_selective_logging, "T13": t13_noncanonical_encoding,
    "T14": t14_uppercase_hex, "T15": t15_derived_column_only,
}
NEEDS_MODEL_SWAP = {"T7a", "T7b"}
NEEDS_ANCHOR = {"T4c", "T8"}
NEEDS_ROTATION = {"T10c"}


def run_scenario(scenario_id: str, directory: str | Path, *, seed: int = 1, n: int = 60,
                 checkpoint_every: int = 16) -> tuple[Fixture, Outcome]:
    """Build the clean ledger for `seed` and apply one scenario to a copy of it."""
    d = Path(directory)
    fx = build_clean_ledger(d / "clean", n=n, seed=seed, checkpoint_every=checkpoint_every,
                            model_swap_at=n // 2 if scenario_id in NEEDS_MODEL_SWAP else None,
                            anchor=scenario_id in NEEDS_ANCHOR,
                            rotate_at=n // 2 if scenario_id in NEEDS_ROTATION else None)
    return fx, SCENARIOS[scenario_id](fx, d / "attack", seed)


def dump_artefact(outcome: Outcome) -> bytes:
    """A canonical LOGICAL dump of the tampered artefact — every row of every table for a database, the file
    bytes for an export. Two runs of one (scenario, seed) must produce identical dumps."""
    parts: list[bytes] = []
    if outcome.db is not None:
        c = sqlite3.connect(outcome.db)
        try:
            for table, order in (("meta", "k"), ("records", "seq"), ("payloads", "hash"), ("merkle_nodes", "level, idx")):
                parts.append(f"## {table}\n".encode())
                for row in c.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                    parts.append(repr(row).encode() + b"\n")
        finally:
            c.close()
    if outcome.export is not None:
        parts.append(b"## export\n" + outcome.export.read_bytes())
    return b"".join(parts)
