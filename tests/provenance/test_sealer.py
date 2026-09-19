"""The Sealer SDK (plan §7.7; gate C4)."""
from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from cva.provenance.seal.canonical import canonical_bytes
from cva.provenance.seal.chain import verify_chain
from cva.provenance.seal.errors import (
    KeyNotConfigured,
    SealError,
    SealMissing,
    TrustRootError,
    WrongKey,
)
from cva.provenance.seal.keys import TrustKey, TrustRoot
from cva.provenance.seal.outputs import build_output_objects
from cva.provenance.seal.payloads import PayloadStore
from cva.provenance.seal.records import genesis_prev_hash
from cva.provenance.seal.sealer import Sealer, SealPolicy

from ._chain_helpers import SEED_B, provider
from ._ledger_helpers import CLASSIFY, CONFIG, DETECT_FILTERED, DETECT_RAW, MODEL, Env


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def inference_records(env):
    return [r for r in env.sealer.ledger.records() if r["type"] == "inference"]


# --- opening: no implicit key, trust root bound to this ledger --------------------------------------------

def test_open_without_a_key_raises_key_not_configured_and_creates_nothing(tmp_path):
    """Deferred here from C3: there is no fallback, no auto-generate, no development mode."""
    e = Env(tmp_path)
    e.close()
    before = sorted(p.name for p in tmp_path.rglob("*"))
    with pytest.raises(KeyNotConfigured, match="keygen"):
        Sealer.open(e.ledger_path, key=None, trust_root=e.trust, payload_dir=e.payload_dir)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_open_needs_an_existing_ledger_and_never_creates_one(tmp_path):
    from cva.provenance.seal.errors import LedgerNotInitialised
    e = Env(tmp_path)
    e.close()
    with pytest.raises(LedgerNotInitialised):
        Sealer.open(tmp_path / "nothing.db", key=e.key, trust_root=e.trust, payload_dir=tmp_path / "pl")
    assert not (tmp_path / "nothing.db").exists()


def test_open_rejects_a_key_that_is_not_the_ledgers_key(tmp_path):
    e = Env(tmp_path)
    e.close()
    with pytest.raises(WrongKey):
        Sealer.open(e.ledger_path, key=provider(SEED_B), trust_root=e.trust, payload_dir=e.payload_dir)


def test_open_rejects_a_trust_root_for_a_different_deployment(tmp_path):
    e = Env(tmp_path)
    e.close()
    other = TrustRoot("f" * 64, e.trust.keys)
    with pytest.raises(TrustRootError, match="different deployment manifest"):
        Sealer.open(e.ledger_path, key=e.key, trust_root=other, payload_dir=e.payload_dir)


def test_open_rejects_a_key_missing_from_the_trust_root(tmp_path):
    e = Env(tmp_path)
    e.close()
    other = provider(SEED_B)
    tr = TrustRoot(e.trust.deployment_manifest_hash, (TrustKey(other.key_id, other.public_key, "ledger"),))
    with pytest.raises(TrustRootError, match="not a ledger key"):
        Sealer.open(e.ledger_path, key=e.key, trust_root=tr, payload_dir=e.payload_dir)


def test_a_witness_key_in_the_trust_root_is_not_a_ledger_key(tmp_path):
    e = Env(tmp_path)
    e.close()
    tr = TrustRoot(e.trust.deployment_manifest_hash, (TrustKey(e.key.key_id, e.key.public_key, "witness"),
                                                      TrustKey(provider(SEED_B).key_id, provider(SEED_B).public_key, "ledger")))
    with pytest.raises(TrustRootError):
        Sealer.open(e.ledger_path, key=e.key, trust_root=tr, payload_dir=e.payload_dir)


def test_trust_root_can_be_given_as_a_file_path(tmp_path):
    e = Env(tmp_path)
    e.close()
    path = tmp_path / "trust_root.json"
    path.write_bytes(e.trust.to_bytes() + b"\n")
    s = Sealer.open(e.ledger_path, key=e.key, trust_root=path, payload_dir=e.payload_dir, clock=e.clock, rng=e.rng)
    assert s.capabilities() == {"SIGNING_KEY", "INFERENCE_LEDGER"}
    s.close()


def test_a_failed_open_does_not_leave_the_ledger_locked_open(tmp_path):
    e = Env(tmp_path)
    e.close()
    for _ in range(3):
        with pytest.raises(TrustRootError):
            Sealer.open(e.ledger_path, key=e.key, trust_root=TrustRoot("f" * 64, e.trust.keys), payload_dir=e.payload_dir)
    s = Sealer.open(e.ledger_path, key=e.key, trust_root=e.trust, payload_dir=e.payload_dir, clock=e.clock, rng=e.rng)
    s.close()


# --- registration ----------------------------------------------------------------------------------------

def test_register_config_stores_the_quantised_specs_and_returns_content_addresses(env):
    ps = PayloadStore(env.payload_dir)
    pre = canonical_bytes(CONFIG["preprocess_spec"], max_bytes=None)
    assert env.config.preprocess_hash == hashlib.sha256(pre).hexdigest()
    assert env.config.preprocess_ref == "sha256:" + env.config.preprocess_hash
    assert ps.get(env.config.preprocess_ref) == pre                        # prov.recompute can read it back
    assert ps.get("sha256:" + env.config.postprocess_hash) == canonical_bytes(CONFIG["postprocess_spec"], max_bytes=None)


def test_preprocessing_specs_with_floats_are_refused(env):
    from cva.provenance.seal.errors import NonCanonical
    with pytest.raises(NonCanonical, match="float"):
        env.sealer.register_config(**{**CONFIG, "preprocess_spec": {"mean": [0.485]}})


@pytest.mark.parametrize("field,bad", [("weights_sha256", "AB" * 32), ("arch_hash", "x"), ("format", "ONNX"), ("id", "")])
def test_register_model_validates_its_fields_up_front(env, field, bad):
    from cva.provenance.seal.errors import InvalidRecord
    with pytest.raises(InvalidRecord):
        env.sealer.register_model(**{**MODEL, field: bad})


def test_the_model_registration_is_written_once_and_before_the_first_inference(env):
    for i in range(5):
        env.seal(i)
    types = [r["type"] for r in env.sealer.ledger.records()]
    assert types[:3] == ["genesis", "model_registration", "inference"]
    assert Counter(types)["model_registration"] == 1
    reg = next(r for r in env.sealer.ledger.records() if r["type"] == "model_registration")
    assert reg["model"]["weights_sha256"] == MODEL["weights_sha256"]


def test_a_reloaded_model_with_new_weights_is_registered_again(env):
    env.seal(0)
    env.model = env.sealer.register_model(**{**MODEL, "weights_sha256": "12" * 32})
    env.seal(1)
    regs = [r for r in env.sealer.ledger.records() if r["type"] == "model_registration"]
    assert [r["model"]["weights_sha256"] for r in regs] == [MODEL["weights_sha256"], "12" * 32]


def test_registrations_survive_a_reopen_and_are_not_duplicated(env):
    env.seal(0)
    env.reopen()
    env.seal(1)
    assert Counter(r["type"] for r in env.sealer.ledger.records())["model_registration"] == 1


# --- binding the input (13-C4: no time-of-check/time-of-use gap) ------------------------------------------

def test_bind_input_hashes_the_bytes_and_returns_that_exact_object(env):
    buf = b"\x89PNG-ish encoded frame" * 100
    b = env.sealer.bind_input(buf, source_kind="encoded_file", dims=(1920, 1080))
    assert b.sha256 == hashlib.sha256(buf).hexdigest() and b.data is buf
    assert b.source_kind == "encoded_file" and b.dims == (1920, 1080) and b.phash is None
    assert b.phash_omitted_reason == "not_computed"


def test_a_mutable_buffer_is_snapshotted_so_it_cannot_change_between_hash_and_inference(env):
    buf = bytearray(b"frame-A" * 200)
    b = env.sealer.bind_input(buf, source_kind="encoded_file", dims=(64, 64))
    hashed = b.sha256
    buf[:7] = b"frame-B"                                                     # an attacker swaps the frame
    assert isinstance(b.data, bytes) and b.data.startswith(b"frame-A")
    assert hashlib.sha256(b.data).hexdigest() == hashed                       # what the model sees == what was hashed


def test_a_memoryview_is_snapshotted_too(env):
    src = bytearray(b"x" * 1000)
    b = env.sealer.bind_input(memoryview(src), source_kind="encoded_file", dims=(10, 10))
    src[0] = 0
    assert b.data[0] == ord("x")


def test_a_binding_is_frozen(env):
    b = env.sealer.bind_input(b"x", source_kind="encoded_file", dims=(1, 1))
    with pytest.raises(AttributeError):
        b.sha256 = "0" * 64                                                   # type: ignore[misc]


@pytest.mark.parametrize("kw", [
    {"source_kind": "decoded"}, {"dims": (0, 5)}, {"dims": (5,)}, {"dims": (5, 5, 5)}, {"dims": (1.5, 2)},
    {"dims": (True, 2)}, {"phash_omitted_reason": "because"},
    {"phash": {"algo": "phash64-v1", "value": "0" * 16}, "phash_omitted_reason": "not_computed"},
])
def test_bad_binding_arguments_are_rejected(env, kw):
    args = {"source_kind": "encoded_file", "dims": (64, 64), **kw}
    with pytest.raises(ValueError):
        env.sealer.bind_input(b"x", **args)


def test_a_supplied_phash_is_recorded_and_the_omitted_reason_is_absent(env):
    ph = {"algo": "phash64-v1", "value": "0123456789abcdef"}
    env.sealer.seal(b"frame", env.model, env.config, output=CLASSIFY, dims=(8, 8), phash=ph)
    rec = inference_records(env)[0]
    assert rec["input"]["phash"] == ph and "phash_omitted_reason" not in rec["input"]


def test_an_omitted_phash_is_declared_with_a_reason(env):
    env.seal(0)
    rec = inference_records(env)[0]
    assert rec["input"]["phash"] is None and rec["input"]["phash_omitted_reason"] == "not_computed"


def test_the_model_input_tensor_source_kind_is_recorded(env):
    env.sealer.seal(b"\x00" * 150_000, env.model, env.config, output=CLASSIFY, dims=(224, 224),
                    source_kind="model_input_tensor")
    assert inference_records(env)[0]["input"]["source_kind"] == "model_input_tensor"


# --- what commit writes -----------------------------------------------------------------------------------

def test_the_record_binds_input_model_config_and_output_hashes(env):
    buf = b"the encoded frame" * 50
    r = env.sealer.seal(buf, env.model, env.config, output=DETECT_RAW, filtered=DETECT_FILTERED, dims=(1920, 1080))
    rec = inference_records(env)[0]
    assert r.sealed and r.seq == rec["seq"] and r.record_hash
    assert rec["input"]["sha256"] == hashlib.sha256(buf).hexdigest() and rec["input"]["dims"] == [1920, 1080]
    assert rec["model"] == {"id": MODEL["id"], "weights_sha256": MODEL["weights_sha256"], "format": "onnx",
                            "arch_hash": MODEL["arch_hash"]}
    assert rec["config"]["preprocess_hash"] == env.config.preprocess_hash
    assert rec["config"]["runtime"] == CONFIG["runtime"] and rec["config"]["code_commit"] == "0" * 40


def test_the_three_output_hashes_are_the_hashes_of_the_three_quantised_objects(env):
    env.seal(0, output=DETECT_RAW, filtered=DETECT_FILTERED)
    out = inference_records(env)[0]["output"]
    raw, fine, coarse = (canonical_bytes(o, max_bytes=None) for o in build_output_objects(DETECT_RAW, DETECT_FILTERED))
    assert out["raw_jcs_sha256"] == hashlib.sha256(raw).hexdigest()
    assert out["jcs_sha256"] == hashlib.sha256(fine).hexdigest()
    assert out["decision_sha256"] == hashlib.sha256(coarse).hexdigest()
    assert len({out["raw_jcs_sha256"], out["jcs_sha256"], out["decision_sha256"]}) == 3


def test_both_payloads_are_stored_and_the_record_references_the_filtered_one(env):
    env.seal(0, output=DETECT_RAW, filtered=DETECT_FILTERED)
    out = inference_records(env)[0]["output"]
    ps = PayloadStore(env.payload_dir)
    assert out["payload_ref"] == "sha256:" + out["jcs_sha256"]
    assert hashlib.sha256(ps.get(out["payload_ref"])).hexdigest() == out["jcs_sha256"]
    assert hashlib.sha256(ps.get("sha256:" + out["raw_jcs_sha256"])).hexdigest() == out["raw_jcs_sha256"]


def test_the_raw_payload_shows_detections_the_filter_dropped(env):
    """D15 / 13-C2: sealing only the filtered output would hide the dropped detection."""
    import json
    env.seal(0, output=DETECT_RAW, filtered=DETECT_FILTERED)
    out = inference_records(env)[0]["output"]
    ps = PayloadStore(env.payload_dir)
    assert len(json.loads(ps.get("sha256:" + out["raw_jcs_sha256"]))["detections"]) == 3
    assert len(json.loads(ps.get(out["payload_ref"]))["detections"]) == 2


def test_identical_raw_and_filtered_output_share_one_payload_file(env):
    env.seal(0, output=CLASSIFY)
    out = inference_records(env)[0]["output"]
    assert out["jcs_sha256"] == out["raw_jcs_sha256"]


def test_a_non_finite_output_is_sealed_as_a_marker(env):
    env.seal(0, output={"task": "classify", "top": [{"cls": 1, "conf": float("nan")}]})
    out = inference_records(env)[0]["output"]
    assert out["jcs_sha256"] == hashlib.sha256(canonical_bytes({"nonfinite": True})).hexdigest()


def test_non_finite_output_can_be_configured_to_raise(tmp_path):
    from cva.provenance.seal.errors import NonFiniteValue
    e = Env(tmp_path, policy=SealPolicy(on_nonfinite="raise"))
    with pytest.raises(NonFiniteValue):
        e.seal(0, output={"task": "classify", "top": [{"cls": 1, "conf": float("inf")}]})
    assert e.sealer.ledger.size() == 1                                       # nothing half-written
    e.close()


def test_a_malformed_output_is_a_caller_error_not_a_ledger_failure(tmp_path):
    e = Env(tmp_path, policy=SealPolicy(on_ledger_failure="fail_open", allow_fail_open=True))
    with pytest.raises(ValueError):
        e.seal(0, output={"task": "classify"})                               # never silently 'unsealed'
    assert e.sealer._gap is None
    e.close()


def test_the_receipt_reports_position_and_durability(env):
    r = env.seal(0)
    assert r.sealed and r.durable and r.tree_size_after == env.sealer.ledger.size()
    assert r.seq == env.sealer.ledger.size() - 1 and len(r.record_hash) == 64


def test_the_appendix_b_round_trip(env):
    """Plan Appendix B: seal 300, verify clean, per-type counts, a stable Merkle root."""
    for i in range(300):
        env.seal(i, output={"task": "classify", "top": [{"cls": i % 10, "conf": 0.9}]})
    led = env.sealer.ledger
    assert verify_chain(list(led.stored_records()), ledger_keys=env.keys()).ok
    counts = Counter(r["type"] for r in led.records())
    assert counts["inference"] == 300 and counts["genesis"] == 1 and counts["model_registration"] == 1
    assert led.verify_merkle_cache()


def test_the_seal_shortcut_equals_bind_then_commit(env):
    buf = b"same frame" * 30
    r1 = env.sealer.seal(buf, env.model, env.config, output=CLASSIFY, dims=(8, 8))
    b = env.sealer.bind_input(buf, source_kind="encoded_file", dims=(8, 8))
    r2 = env.sealer.commit(b, env.model, env.config, output=CLASSIFY)
    a, c = inference_records(env)
    assert a["input"] == c["input"] and a["output"] == c["output"] and r1.seq + 1 == r2.seq


# --- guard(): releasing without sealing is hard to do by accident ----------------------------------------------

def test_guard_lets_the_block_finish_when_it_sealed(env):
    released = []
    with env.sealer.guard() as g:
        b = g.bind_input(b"frame" * 20, source_kind="encoded_file", dims=(8, 8))
        g.commit(b, env.model, env.config, output=CLASSIFY)
        released.append("output")
    assert released == ["output"]


def test_guard_raises_when_the_block_forgot_to_seal(env):
    with pytest.raises(SealMissing), env.sealer.guard():
        pass
    with pytest.raises(SealMissing), env.sealer.guard() as g:
        g.bind_input(b"frame", source_kind="encoded_file", dims=(8, 8))     # bound but never committed


def test_guard_does_not_mask_the_blocks_own_exception(env):
    with pytest.raises(RuntimeError, match="model crashed"), env.sealer.guard():
        raise RuntimeError("model crashed")


def test_guard_seal_shortcut_counts_as_sealing(env):
    with env.sealer.guard() as g:
        g.seal(b"frame" * 10, env.model, env.config, output=CLASSIFY, dims=(8, 8))
    assert len(inference_records(env)) == 1


# --- policy ------------------------------------------------------------------------------------------------

def test_policy_defaults_are_fail_closed_with_a_seal_marker_for_non_finite():
    p = SealPolicy()
    assert (p.on_ledger_failure, p.allow_fail_open, p.on_nonfinite) == ("fail_closed", False, "seal_marker")


@pytest.mark.parametrize("kw", [{"on_ledger_failure": "shrug"}, {"on_nonfinite": "ignore"},
                                {"on_ledger_failure": "fail_open"}])
def test_policy_rejects_nonsense_and_silent_fail_open(kw):
    with pytest.raises(ValueError):
        SealPolicy(**kw)


def test_context_manager_closes_the_ledger(tmp_path):
    e = Env(tmp_path)
    e.sealer.close()
    with Sealer.open(e.ledger_path, key=e.key, trust_root=e.trust, payload_dir=e.payload_dir, clock=e.clock,
                     rng=e.rng) as s:
        assert s.capabilities()
    assert s.capabilities() == set()
    e.sealer = s


def test_seal_error_is_the_common_base_of_everything_the_sdk_raises():
    for exc in (KeyNotConfigured, WrongKey, SealMissing, TrustRootError):
        assert issubclass(exc, SealError)


def test_genesis_prev_hash_is_what_the_trust_root_pins(env):
    assert env.trust.deployment_manifest_hash == genesis_prev_hash(env.sealer.ledger.deployment_manifest)
