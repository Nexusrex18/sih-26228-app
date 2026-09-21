"""Key rotation (plan §5.10; gate C7): outgoing-signed, incoming proves possession, verifier state machine."""
from __future__ import annotations

import pytest

from cva.provenance.seal.anchor import verify_anchor
from cva.provenance.seal.chain import rotation_body
from cva.provenance.seal.errors import SealError, WrongKey
from cva.provenance.seal.sealer import Sealer
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import export_records, verify_ledger

from ._anchor_helpers import EVIL_SEED, NEW_SEED, AnchorEnv
from ._chain_helpers import provider
from ._ledger_helpers import CONFIG, MODEL, Env


def classes(rep):
    return [f.attack_class for f in rep.findings if f.severity != "info"]


NEW = provider(NEW_SEED)


@pytest.fixture
def env(tmp_path):
    e = AnchorEnv(tmp_path, checkpoint_every=1000)
    for i in range(5):
        e.seal(i)
    return e


def rotate_and_seal(env, n_after=5, new=NEW):
    env.sealer.rotate_key(new)
    for i in range(100, 100 + n_after):
        env.seal(i)


# --- the happy path --------------------------------------------------------------------------------------

def test_a_rotated_ledger_verifies_clean_with_only_the_genesis_key_in_the_trust_root(env):
    rotate_and_seal(env)
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust)
    assert rep.clean and rep.rotations == 1 and rep.active_key_id == NEW.key_id
    assert NEW.key_id not in {k.key_id for k in env.trust.keys}


def test_records_before_the_rotation_are_the_old_keys_and_after_it_the_new_keys(env):
    rotate_and_seal(env)
    env.sealer.close()
    import json
    import sqlite3
    c = sqlite3.connect(env.ledger_path)
    recs = [json.loads(r[0]) for r in c.execute("SELECT rec FROM records ORDER BY seq")]
    c.close()
    rot = next(r for r in recs if r["type"] == "key_rotation")
    assert rot["key_id"] == env.key.key_id and rot["rotation"]["new_key_id"] == NEW.key_id
    assert rot["rotation"]["effective_seq"] == rot["seq"] + 1
    assert all(r["key_id"] == env.key.key_id for r in recs if r["seq"] <= rot["seq"])
    assert all(r["key_id"] == NEW.key_id for r in recs if r["seq"] > rot["seq"])


def test_the_ledger_reopens_with_the_new_key_and_refuses_the_retired_one(env):
    rotate_and_seal(env, 2)
    env.sealer.close()
    with pytest.raises(WrongKey, match="active signing key"):
        SealedLedger.open(env.ledger_path, key=env.key)
    led = SealedLedger.open(env.ledger_path, key=NEW)
    assert led.active_key_id == NEW.key_id and led.genesis_key_id == env.key.key_id
    led.close()
    s = Sealer.open(env.ledger_path, key=NEW, trust_root=env.trust, clock=env.clock, rng=env.rng)
    m, c = s.register_model(**MODEL), s.register_config(**CONFIG)
    s.seal(b"after reopen" * 40, m, c, output={"task": "classify", "top": [{"cls": 1, "conf": 0.5}]}, dims=(8, 8))
    s.close()
    assert verify_ledger(env.ledger_path, trust_root=env.trust).clean


def test_a_checkpoint_triggered_by_the_rotation_itself_is_signed_by_the_new_key(tmp_path):
    """The record right after a rotation is the new key's — including a cadence checkpoint that lands there."""
    for pre in range(2, 9):                       # sweep the cadence so a checkpoint falls right after the rotation
        d = tmp_path / f"p{pre}"
        d.mkdir()
        e = AnchorEnv(d, checkpoint_every=4)
        for i in range(pre):
            e.seal(i)
        e.sealer.rotate_key(NEW)
        for i in range(6):
            e.seal(50 + i)
        e.sealer.close()
        rep = verify_ledger(e.ledger_path, trust_root=e.trust)
        assert rep.clean and rep.rotations == 1, (pre, [f.attack_class for f in rep.findings])
        assert rep.checkpoints_verified >= 2


def test_two_successive_rotations_form_a_chain_of_custody(env):
    rotate_and_seal(env, 3)
    third = provider(bytes(range(70, 102)))
    env.sealer.rotate_key(third)
    for i in range(200, 204):
        env.seal(i)
    env.sealer.close()
    rep = verify_ledger(env.ledger_path, trust_root=env.trust)
    assert rep.clean and rep.rotations == 2 and rep.active_key_id == third.key_id


def test_rotation_survives_export_verification_and_reports_the_same(env):
    rotate_and_seal(env)
    exp = env.export()
    rep = verify_ledger(exp, trust_root=env.trust)
    assert rep.clean and rep.rotations == 1


# --- refusals ---------------------------------------------------------------------------------------------

def test_rotating_to_the_key_that_is_already_active_is_refused(env):
    with pytest.raises(SealError, match="already active"):
        env.sealer.rotate_key(env.key)


def test_a_key_that_cannot_sign_is_never_rotated_to(env):
    class Broken:
        custody = "file"
        key_id = NEW.key_id
        public_key = NEW.public_key

        def sign(self, m):
            raise RuntimeError("hsm gone")

        def self_test(self):
            return False
    with pytest.raises(SealError, match="cannot produce a verifying signature"):
        env.sealer.rotate_key(Broken())
    env.seal(9)                                                # the ledger is still writable by the OLD key
    env.sealer.close()
    assert verify_ledger(env.ledger_path, trust_root=env.trust).rotations == 0


def test_a_read_only_handle_cannot_rotate(env):
    env.sealer.close()
    with SealedLedger.open(env.ledger_path, read_only=True) as led:
        with pytest.raises(SealError):
            led.rotate_key(NEW)


# --- the verifier's key state machine ---------------------------------------------------------------------------

def test_the_retired_key_signing_after_its_rotation_is_attributed_adversarial(tmp_path):
    from attacklab.tamper import run_scenario
    _, o = run_scenario("T10c", tmp_path, seed=2)
    rep = o.verify("db")
    f = [x for x in rep.findings if x.severity != "info"]
    assert [x.attack_class for x in f] == ["key_unauthorised"] and f[0].nature == "adversarial"
    assert "rotated-out (retired)" in f[0].reason


def test_a_rotation_edited_after_signing_is_a_record_edit_not_a_key_finding(env):
    rotate_and_seal(env, 3)
    env.sealer.close()
    from attacklab.tamper import TamperDb
    t = TamperDb(env.ledger_path, env.tmp / "t.db")
    seq = int(t.c.execute("SELECT seq FROM records WHERE type='key_rotation'").fetchone()[0])
    rec = t.rec(seq)
    rec["rotation"]["effective_seq"] = seq + 5                                # would delay the takeover
    t.put(seq, rec)
    t.close()
    rep = verify_ledger(env.tmp / "t.db", trust_root=env.trust)
    assert classes(rep)[0] in ("malformed_record", "record_edit")


def test_the_rotation_pop_is_bound_to_this_rotation_so_it_cannot_be_lifted_from_another(env):
    """An attacker who has seen one rotation's PoP cannot reuse it at a different position."""
    from attacklab.tamper import TamperDb, _append_forged
    env.sealer.close()
    t = TamperDb(env.ledger_path, env.tmp / "t.db")
    seq = t.last_seq() + 1
    good = rotation_body(env.key, NEW, seq + 7)                              # a PoP for a DIFFERENT effective_seq
    body = {"rotation": {**good["rotation"], "effective_seq": seq + 1}}
    _append_forged(t, env.key, "key_rotation", body)
    t.close()
    rep = verify_ledger(env.tmp / "t.db", trust_root=env.trust)
    assert classes(rep) == ["key_unauthorised"] and rep.findings[0].primary_check == "rotation_pop"


def test_a_ledger_rotated_to_an_attackers_key_by_the_holder_is_the_declared_limit(env):
    """Emergency revocation is NOT covered (plan §5.10): a compromised key can validly rotate to a key its thief
    controls. Nothing in-band distinguishes that from a real rotation — which is why it is declared, and tested."""
    evil = provider(EVIL_SEED)
    rotate_and_seal(env, 3, new=evil)
    env.sealer.close()
    assert verify_ledger(env.ledger_path, trust_root=env.trust).clean


def test_a_rotated_ledger_presented_under_another_deployments_trust_root_is_one_genesis_finding(env):
    rotate_and_seal(env)
    env.sealer.close()
    from cva.provenance.seal.keys import TrustRoot
    other = TrustRoot("9" * 64, env.trust.keys)
    rep = verify_ledger(env.ledger_path, trust_root=other)
    assert classes(rep) == ["genesis_mismatch"]


# --- anchors across a rotation --------------------------------------------------------------------------------------------

def test_an_anchor_signed_by_the_rotated_in_key_needs_the_ledger_to_be_trusted(env):
    rotate_and_seal(env, 3)
    env.sealer.close()
    led = SealedLedger.open(env.ledger_path, key=NEW, clock=env.clock, rng=env.rng, background_flush=False)
    from cva.provenance.seal.anchor import export_anchor
    a = export_anchor(led, env.tmp / "post.json")
    led.close()
    assert a["checkpoint"]["key_id"] == NEW.key_id
    alone = verify_anchor(a, env.trust)
    assert not alone.valid and "verify the anchor together with the ledger" in alone.problems[0]
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[a])
    assert rep.clean and rep.anchors_verified == 1


def test_an_anchor_signed_by_a_key_the_ledger_never_rotated_to_is_not_accepted(env):
    """Rotated-in keys are learned only from a VERIFIED rotation in this ledger's own chain."""
    rotate_and_seal(env, 2)
    env.sealer.close()
    stray = provider(bytes(range(120, 152)))
    from datetime import UTC, datetime

    from cva.provenance.seal.anchor import build_anchor
    from cva.provenance.seal.chain import seal_next
    with SealedLedger.open(env.ledger_path, key=NEW, clock=env.clock, rng=env.rng, background_flush=False) as led:
        led.checkpoint_now()
        cp = led.latest_checkpoint()
    from cva.provenance.seal.chain import sign_record
    forged = sign_record({k: v for k, v in cp.items() if k not in ("signature", "key_id")} | {"key_id": stray.key_id}, stray)
    rep = verify_ledger(env.ledger_path, trust_root=env.trust, anchors=[build_anchor(forged)])
    assert classes(rep) == ["anchor_invalid"]
    assert seal_next and datetime and UTC


def test_key_rotation_is_visible_in_the_chain_summary(env):
    rotate_and_seal(env)
    env.sealer.close()
    from cva.provenance.checks.ledger_verify import LedgerVerify
    out = LedgerVerify().verify(env.ledger_path, env.trust, scan_id="s", produced_by="t")
    summary = next(f for f in out if f.attack_class == "ledger_verified")
    assert summary.evidence[0].data["key_rotations"] == 1
    assert export_records is not None and Env is not None
