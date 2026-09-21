"""The strongest single test in the module (plan §11.4): no undetected single-bit change ANYWHERE in an export.

Because every stored byte is exactly canonical, uppercase hex is rejected, and the export format itself is
strict (one record per line, every line newline-terminated, no blank lines, no BOM), EVERY byte of the export
is semantically load-bearing — so this is a total claim, not a sampled one.

Efficiency, honestly stated: a flip inside record i can only surface at record i (or, for a newline byte, at
i+1), so the exhaustive sweep verifies the export PREFIX through that record instead of the whole file. A
separate sample verifies the WHOLE file, to show the two agree.
"""
from __future__ import annotations

import random

import pytest

from cva.provenance.seal.verify import export_records, verify_ledger

from ._fixtures import BODIES
from ._ledger_helpers import Env

MARKER = {"gap": {"first_unsealed_utc": "2026-09-19T03:00:00.000000Z", "last_unsealed_utc": "2026-09-19T03:00:05.000000Z",
                  "reported_count": 12, "reason": "ledger_unwritable", "spill_sha256": "a" * 64}}


def signature(report):
    """What the verifier concluded, order-preserving. A mutation is 'detected' iff this changes."""
    return [(f.attack_class, f.seq, f.severity, f.primary_check, f.cascade_suppressed) for f in report.findings]


def build_export(tmp_path, n_inferences, checkpoint_every):
    e = Env(tmp_path, checkpoint_every=checkpoint_every)
    for i in range(n_inferences):
        e.seal(i)
    led = e.sealer.ledger
    led.append({"type": "scan_record", **BODIES["scan_record"]})
    led.append({"type": "analyst_event", **BODIES["analyst_event"]})
    led.append_typed("degraded_marker", MARKER)
    e.close()
    export_records(e.ledger_path, tmp_path / "ledger.jsonl")
    return (tmp_path / "ledger.jsonl").read_bytes(), e.trust


def flip(data: bytes, byte: int, bit: int) -> bytes:
    b = bytearray(data)
    b[byte] ^= 1 << bit
    return bytes(b)


def test_every_bit_of_every_byte_of_a_small_export_is_load_bearing(tmp_path):
    data, trust = build_export(tmp_path, n_inferences=3, checkpoint_every=6)
    starts = [0]
    for i, ch in enumerate(data):
        if ch == 0x0A and i + 1 < len(data):
            starts.append(i + 1)
    ends = [s - 1 for s in starts[1:]] + [len(data) - 1]              # index of each line's "\n"
    types = {__import__("json").loads(line)["type"] for line in data.split(b"\n")[:-1]}
    assert {"genesis", "model_registration", "inference", "checkpoint", "scan_record", "analyst_event",
            "degraded_marker"} <= types, types                       # every record kind is under test
    baseline = {cut: signature(verify_ledger(data[:cut], trust_root=trust)) for cut in set(ends[k] + 1 for k in range(len(ends)))}
    undetected: list[tuple[int, int]] = []
    total = 0
    line_of = []
    for li, (s, e) in enumerate(zip(starts, ends, strict=True)):
        line_of.extend([li] * (e - s + 1))
    for i in range(len(data)):
        li = line_of[i]
        cut = ends[min(li + 1, len(ends) - 1)] + 1                     # through this line AND the next
        for bit in range(8):
            mutated = flip(data, i, bit)[:cut]
            total += 1
            if signature(verify_ledger(mutated, trust_root=trust)) == baseline[cut]:
                undetected.append((i, bit))
    assert total == len(data) * 8 and total > 15_000
    assert undetected == [], f"{len(undetected)} undetected single-bit changes, e.g. {undetected[:5]}"


def test_a_sampled_sweep_over_the_WHOLE_file_agrees_with_the_prefix_sweep(tmp_path):
    data, trust = build_export(tmp_path, n_inferences=25, checkpoint_every=8)
    base = signature(verify_ledger(data, trust_root=trust))
    rng = random.Random(20260919)
    for _ in range(250):
        i, bit = rng.randrange(len(data)), rng.randrange(8)
        assert signature(verify_ledger(flip(data, i, bit), trust_root=trust)) != base, f"undetected: byte {i} bit {bit}"


def test_flipping_any_single_bit_of_the_database_records_is_also_caught(tmp_path):
    """The database form: flip a bit in a record's stored TEXT."""
    import sqlite3
    e = Env(tmp_path, checkpoint_every=6)
    for i in range(4):
        e.seal(i)
    e.close()
    base = signature(verify_ledger(e.ledger_path, trust_root=e.trust))
    c = sqlite3.connect(e.ledger_path, isolation_level=None)
    c.execute("DROP TRIGGER records_no_update")
    rows = c.execute("SELECT seq, rec FROM records").fetchall()
    rng = random.Random(3)
    for _ in range(120):
        seq, rec = rng.choice(rows)
        raw = bytearray(rec.encode())
        raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(7)          # bits 0-6 keep it valid ASCII text
        c.execute("UPDATE records SET rec=? WHERE seq=?", (bytes(raw).decode("ascii", "replace"), seq))
        assert signature(verify_ledger(e.ledger_path, trust_root=e.trust)) != base, f"seq {seq} undetected"
        c.execute("UPDATE records SET rec=? WHERE seq=?", (rec, seq))                 # restore
    c.close()


@pytest.mark.parametrize("byte_edit", ["final_newline_removed", "final_newline_flipped", "leading_bom"])
def test_the_structural_bytes_of_the_export_are_themselves_checked(tmp_path, byte_edit):
    data, trust = build_export(tmp_path, n_inferences=3, checkpoint_every=50)
    base = signature(verify_ledger(data, trust_root=trust))
    mutated = {"final_newline_removed": data[:-1], "final_newline_flipped": data[:-1] + b"\x0b",
               "leading_bom": b"\xef\xbb\xbf" + data}[byte_edit]
    assert signature(verify_ledger(mutated, trust_root=trust)) != base
