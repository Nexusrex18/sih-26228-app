"""Documented limitations, each with a test that CONSTRUCTS the attack and asserts the verifier does not see
it (Module C plan §10.2, §14). Passing here does not mean the system is weak: it means we know exactly where
the arithmetic stops, and the coverage statement says the same thing this file proves.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from cva.provenance.seal import verify as verify_mod
from cva.provenance.seal.chain import MemoryChain
from cva.provenance.seal.records import genesis_prev_hash
from cva.provenance.seal.verify import verify_ledger

from ._chain_helpers import clock, manifest_for, provider, rng
from ._fixtures import BODIES
from ._ledger_helpers import export_bytes, valid_chain


def test_L1_a_holder_of_the_signing_key_from_the_start_can_forge_a_wholly_consistent_history():
    """The mitigation is an HSM (a key that cannot be copied) and the anchoring ceremony — deployment
    decisions, not something the arithmetic can supply."""
    real, trust, key = valid_chain(6)
    forged = MemoryChain(key, clock=clock(), rng=rng())
    forged.append("genesis", {"deployment_manifest": manifest_for(key)})
    forged.append("model_registration", BODIES["model_registration"])
    for _ in range(6):
        body = {**BODIES["inference"], "output": {**BODIES["inference"]["output"], "jcs_sha256": "9" * 64,
                                                   "payload_ref": "sha256:" + "9" * 64}}
        forged.append("inference", body)
    assert forged.stored != real.stored
    assert verify_ledger(export_bytes(forged), trust_root=trust).clean


def test_L2_a_perfectly_sealed_record_of_a_doctored_input_is_still_a_perfectly_sealed_record():
    """Module C proves a record was produced by our pipeline and has not been altered SINCE. It cannot
    attest that the input was genuine BEFORE it entered — that is Modules A and D's problem."""
    chain, trust, _ = valid_chain(2)
    doctored = b"the sensor was fed a doctored image"
    body = {**BODIES["inference"], "input": {**BODIES["inference"]["input"], "sha256": hashlib.sha256(doctored).hexdigest()}}
    chain.append("inference", body)
    last = chain.records[-1]["seq"]
    r = verify_ledger(export_bytes(chain), trust_root=trust,
                      input_resolver=lambda rec: doctored if rec["seq"] == last else None)
    assert r.clean and r.inputs_checked == 1              # the doctored image matches its seal perfectly


def test_L3_backdating_by_the_key_holder_is_invisible_beyond_a_sync_boundary():
    """created_at is the host clock: untrusted, never used for ordering. A holder can write any time they
    like. Absolute time is bounded only where a witness statement exists (from C7)."""
    key = provider()
    stamps = iter([datetime(2026, 9, 19, 2, 0, i, tzinfo=UTC) for i in range(3)] + [datetime(2001, 1, 1, tzinfo=UTC)] * 4)
    chain = MemoryChain(key, clock=lambda: next(stamps))
    chain.append("genesis", {"deployment_manifest": manifest_for(key)})
    chain.append("model_registration", BODIES["model_registration"])
    for _ in range(4):
        chain.append("inference", BODIES["inference"])
    from cva.provenance.seal.keys import TrustKey, TrustRoot
    trust = TrustRoot(genesis_prev_hash(chain.records[0]["deployment_manifest"]), (TrustKey(key.key_id, key.public_key, "ledger"),))
    r = verify_ledger(export_bytes(chain), trust_root=trust)
    assert r.clean and {f.attack_class for f in r.findings} == {"clock_regression"}     # information only


def test_L4_a_compromised_verifier_can_be_made_to_say_anything(monkeypatch):
    """Compromise of the assurance tool itself is outside the model: nothing in the ledger can defend against a
    verifier that lies. Its integrity is verified out of band (the vendoring manifest, the independent verifier).
    The layering does help against a PARTIALLY doctored tool, which is worth pinning."""
    import json

    from cva.provenance.seal.canonical import canonical_bytes
    chain, trust, _ = valid_chain(4)
    lines = list(chain.stored)
    rec = json.loads(lines[3])
    rec["output"]["jcs_sha256"] = "0" * 64
    lines[3] = canonical_bytes(rec)
    data = b"".join(x + b"\n" for x in lines)
    assert not verify_ledger(data, trust_root=trust).clean                    # the honest verifier sees it
    monkeypatch.setattr(verify_mod, "verify_signature", lambda rec, pub: True)
    verify_mod._sig_valid.cache_clear()
    partial = verify_ledger(data, trust_root=trust)                           # a verifier whose SIGNATURE check lies ...
    assert not partial.clean and "chain_broken" in partial.classes()          # ... is still caught by the LINK
    monkeypatch.setattr(verify_mod._Verifier, "run", lambda self: verify_mod.VerifyReport())
    assert verify_ledger(data, trust_root=trust).clean                        # a fully gutted one says whatever it likes
    verify_mod._sig_valid.cache_clear()


def test_L5_a_power_cut_under_group_commit_leaves_a_valid_prefix_with_no_trace():
    """The loss window is printed in every verify report (meta.loss_window) because the chain cannot show it."""
    chain, trust, _ = valid_chain(10)
    for keep in (4, 7, 11):
        assert verify_ledger(b"".join(x + b"\n" for x in chain.stored[:keep]), trust_root=trust).clean
