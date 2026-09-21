"""The single-bit sweep, extended to what C7 adds (plan §11.4): a rotated ledger, and the anchor artefact itself.

  * every bit of every byte of a ROTATED, ANCHORED export is load-bearing — a flip changes what the verifier
    concludes (with the anchor supplied, the truncated prefixes also report `tail_truncation`, so the baseline is
    recomputed per prefix exactly as in the C5 sweep);
  * every bit of an anchor file (cosigned and attested) either fails to parse or fails to verify: there is no
    single-bit edit of an anchor that leaves it valid.
"""
from __future__ import annotations

import json

from cva.provenance.seal.anchor import AnchorError, attest, cosign, encode_anchor, verify_anchor
from cva.provenance.seal.verify import verify_ledger

from ._anchor_helpers import NEW_SEED, AnchorEnv
from ._chain_helpers import provider


def signature(report):
    return [(f.attack_class, f.seq, f.severity, f.primary_check, f.cascade_suppressed) for f in report.findings]


def flip(data: bytes, byte: int, bit: int) -> bytes:
    b = bytearray(data)
    b[byte] ^= 1 << bit
    return bytes(b)


def build(tmp_path):
    e = AnchorEnv(tmp_path, checkpoint_every=5)
    for i in range(2):
        e.seal(i)
    e.sealer.rotate_key(provider(NEW_SEED))
    for i in range(3, 5):
        e.seal(i)
    e.sealer.close()
    from cva.provenance.seal.anchor import export_anchor
    from cva.provenance.seal.store import SealedLedger
    led = SealedLedger.open(e.ledger_path, key=provider(NEW_SEED), clock=e.clock, rng=e.rng, background_flush=False)
    anchor = export_anchor(led, tmp_path / "a.json")
    led.close()
    from cva.provenance.seal.verify import export_records
    export_records(e.ledger_path, tmp_path / "x.jsonl")
    return (tmp_path / "x.jsonl").read_bytes(), e, anchor


def test_every_bit_of_a_rotated_anchored_export_is_load_bearing(tmp_path):
    data, e, anchor = build(tmp_path)
    types = [json.loads(x)["type"] for x in data.split(b"\n")[:-1]]
    assert {"key_rotation", "checkpoint", "anchor_event", "inference"} <= set(types)
    starts = [0] + [i + 1 for i, ch in enumerate(data) if ch == 0x0A and i + 1 < len(data)]
    ends = [s - 1 for s in starts[1:]] + [len(data) - 1]
    line_of = []
    for li, (s, en) in enumerate(zip(starts, ends, strict=True)):
        line_of.extend([li] * (en - s + 1))
    baseline = {en + 1: signature(verify_ledger(data[:en + 1], trust_root=e.trust, anchors=[anchor])) for en in ends}
    undetected = []
    for i in range(len(data)):
        cut = ends[min(line_of[i] + 1, len(ends) - 1)] + 1
        for bit in range(8):
            if signature(verify_ledger(flip(data, i, bit)[:cut], trust_root=e.trust, anchors=[anchor])) == baseline[cut]:
                undetected.append((i, bit))
    assert len(data) * 8 > 9_000
    assert undetected == [], f"{len(undetected)} undetected single-bit changes, e.g. {undetected[:5]}"


def test_no_single_bit_edit_of_a_cosigned_attested_anchor_leaves_it_valid(tmp_path):
    _, e, anchor = build(tmp_path)
    full = attest(cosign(anchor, e.witness), e.boundary, "2026-09-19T03:00:00.000000Z")
    data = encode_anchor(full)
    ledger_key = provider(NEW_SEED)
    from cva.provenance.seal.keys import TrustKey, TrustRoot
    trust = TrustRoot(e.trust.deployment_manifest_hash, (*e.trust.keys, TrustKey(ledger_key.key_id, ledger_key.public_key, "ledger")))
    assert verify_anchor(data, trust).valid
    still_valid = []
    for i in range(len(data)):
        for bit in range(8):
            try:
                r = verify_anchor(flip(data, i, bit), trust)
            except AnchorError:
                continue
            if r.valid:
                still_valid.append((i, bit))
    assert still_valid == [], f"{len(still_valid)} bit flips left the anchor valid, e.g. {still_valid[:5]}"
