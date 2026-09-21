"""Typed records, header rules, genesis (D5), in-code schemas (plan §5.3–§5.7; gate C1)."""
from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta, timezone

import pytest

from cva.provenance.seal import constants as C
from cva.provenance.seal.canonical import canonical_bytes, parse_strict
from cva.provenance.seal.errors import InvalidRecord
from cva.provenance.seal.records import (
    RECORD_TYPES,
    SECTIONS,
    build_record,
    canon,
    format_utc,
    genesis_prev_hash,
    new_nonce,
    record_hash,
    stored_bytes,
    validate_record,
)

from ._fixtures import (
    ANALYST_OVERRIDE,
    BODIES,
    CONFIG,
    INPUT,
    KEY_ID,
    MANIFEST,
    MODEL,
    NONCE,
    PUB,
    SIG,
    TS,
    H,
    make_record,
)

ALL = list(RECORD_TYPES)


def mutate(record: dict, path: str, value) -> dict:
    """Return a copy with `path` (dotted) set to `value`; the sentinel DELETE removes the key."""
    r = copy.deepcopy(record)
    *parents, leaf = path.split(".")
    node = r
    for p in parents:
        node = node[p]
    if value is DELETE:
        del node[leaf]
    else:
        node[leaf] = value
    return r


DELETE = object()


# --- constants -----------------------------------------------------------------------------------

def test_domain_bytes_are_distinct_and_are_the_documented_values():
    assert (C.DOMAIN_LEAF, C.DOMAIN_NODE, C.DOMAIN_LINK, C.DOMAIN_MANIFEST) == (b"\x00", b"\x01", b"\x02", b"\x03")
    assert len({C.DOMAIN_LEAF, C.DOMAIN_NODE, C.DOMAIN_LINK, C.DOMAIN_MANIFEST}) == 4


def test_signature_tags_are_the_exact_documented_strings_and_distinct():
    assert C.TAG_RECORD == b"cva-seal/1 record\n"
    assert C.TAG_CHECKPOINT == b"cva-seal/1 checkpoint\n"
    assert C.TAG_COSIGN == b"cva-seal/1 cosign\n"
    assert C.TAG_ROTATION_POP == b"cva-seal/1 rotation-pop\n"
    assert C.TAG_WITNESS == b"cva-seal/1 witness\n"
    tags = [C.TAG_RECORD, C.TAG_CHECKPOINT, C.TAG_COSIGN, C.TAG_ROTATION_POP, C.TAG_WITNESS]
    assert len(set(tags)) == 5


def test_no_tag_or_domain_byte_can_collide_with_a_record_hash_input():
    """`record_hash` hashes JSON text, which starts with '{' (0x7B). No tag/domain starts with it."""
    for prefix in (C.DOMAIN_LEAF, C.DOMAIN_NODE, C.DOMAIN_LINK, C.DOMAIN_MANIFEST, C.TAG_RECORD):
        assert prefix[0] != 0x7B


# --- every record type builds, validates, and round-trips ------------------------------------------

def test_every_type_has_a_fixture_and_a_section_set():
    assert set(BODIES) == set(RECORD_TYPES) == set(SECTIONS)


@pytest.mark.parametrize("rtype", ALL)
def test_each_type_builds_and_validates(rtype):
    validate_record(make_record(rtype, signed=False), signed=False)
    validate_record(make_record(rtype), signed=True)


@pytest.mark.parametrize("rtype", ALL)
def test_stored_bytes_are_exactly_canonical_and_survive_strict_parsing(rtype):
    r = make_record(rtype)
    data = stored_bytes(r)
    assert parse_strict(data) == r
    assert data == canonical_bytes(r)
    assert data.endswith(b"}") and b"\n" not in data                  # no trailing newline, one line


@pytest.mark.parametrize("rtype", ALL)
def test_canon_excludes_the_signature_and_stored_bytes_include_it(rtype):
    r = make_record(rtype)
    assert b'"signature"' not in canon(r)
    assert b'"signature":"' + SIG.encode() + b'"' in stored_bytes(r)
    assert canon(r) == canon({k: v for k, v in r.items() if k != "signature"})


@pytest.mark.parametrize("rtype", ALL)
def test_record_hash_input_starts_with_an_opening_brace(rtype):
    r = make_record(rtype)
    assert canon(r)[0] == 0x7B
    assert record_hash(r) == hashlib.sha256(canon(r)).digest()


@pytest.mark.parametrize("rtype", ALL)
def test_no_float_or_stray_null_ever_appears_in_a_record(rtype):
    def walk(v, path=""):
        assert not isinstance(v, float), path
        if v is None:
            assert path == "$.input.phash", f"unexpected null at {path}"
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
    walk(make_record(rtype), "$")


@pytest.mark.parametrize("rtype", ALL)
def test_the_signature_field_is_required_only_when_signed(rtype):
    unsigned = make_record(rtype, signed=False)
    validate_record(unsigned, signed=False)
    with pytest.raises(InvalidRecord, match="signature"):
        validate_record(unsigned, signed=True)
    with pytest.raises(InvalidRecord, match="signature"):
        validate_record(make_record(rtype), signed=False)


# --- header rules ----------------------------------------------------------------------------------

@pytest.mark.parametrize("field,bad", [
    ("prev_record_hash", "A" * 64), ("prev_record_hash", "a" * 63), ("prev_record_hash", "a" * 65),
    ("prev_record_hash", "g" * 64), ("prev_record_hash", 5),
    ("key_id", "F" * 64), ("key_id", "f" * 32),
    ("nonce", "CD" * 16), ("nonce", "cd" * 15), ("nonce", "cd" * 17),
    ("signature", "AB" * 64), ("signature", "ab" * 63),
    ("v", "cva-seal/2"), ("v", ""), ("v", None),
    ("seq", -1), ("seq", 1.0), ("seq", True), ("seq", "5"), ("seq", C.INT_MAX + 1),
    ("type", "other"), ("type", None),
])
def test_header_field_violations_are_rejected(field, bad):
    with pytest.raises(InvalidRecord):
        validate_record(mutate(make_record("inference"), field, bad))


@pytest.mark.parametrize("field", ["v", "seq", "prev_record_hash", "key_id", "created_at_utc",
                                   "nonce", "signature", "input", "model", "config", "output"])
def test_a_missing_field_is_rejected(field):
    with pytest.raises(InvalidRecord, match="missing"):
        validate_record(mutate(make_record("inference"), field, DELETE))


def test_a_record_with_no_type_is_rejected():
    with pytest.raises(InvalidRecord, match="type"):
        validate_record(mutate(make_record("inference"), "type", DELETE))


def test_an_unknown_top_level_key_is_rejected():
    with pytest.raises(InvalidRecord, match="unknown"):
        validate_record({**make_record("inference"), "extra": 1})


def test_a_section_belonging_to_another_type_is_rejected():
    r = make_record("checkpoint")
    r["model"] = MODEL
    with pytest.raises(InvalidRecord, match="unknown"):
        validate_record(r)


@pytest.mark.parametrize("ts", [
    "2026-09-19T02:14:07Z",                       # no fractional part
    "2026-09-19T02:14:07.48291Z",                 # 5 digits
    "2026-09-19T02:14:07.4829131Z",               # 7 digits
    "2026-09-19T02:14:07.482913+00:00",           # offset instead of Z
    "2026-09-19 02:14:07.482913Z",                # space
    "2026-09-19t02:14:07.482913z",                # lowercase
    "2026-02-30T02:14:07.482913Z",                # impossible date
    "2026-09-19T25:14:07.482913Z",                # impossible hour
    "20260919T021407.482913Z", 5, None,
])
def test_timestamps_must_be_exactly_27_char_utc(ts):
    with pytest.raises(InvalidRecord):
        validate_record(mutate(make_record("inference"), "created_at_utc", ts))


def test_format_utc_is_27_chars_utc_with_six_fraction_digits():
    dt = datetime(2026, 9, 19, 2, 14, 7, 482913, tzinfo=UTC)
    assert format_utc(dt) == TS and len(format_utc(dt)) == C.TIMESTAMP_LEN
    assert format_utc(datetime(2026, 1, 1, tzinfo=UTC)) == "2026-01-01T00:00:00.000000Z"
    ist = timezone(timedelta(hours=5, minutes=30))
    assert format_utc(datetime(2026, 9, 19, 7, 44, 7, 482913, tzinfo=ist)) == TS       # converted to UTC


def test_format_utc_refuses_a_naive_datetime():
    with pytest.raises(ValueError, match="naive"):
        format_utc(datetime(2026, 9, 19, 2, 14, 7))  # noqa: DTZ001 - naive on purpose


def test_seq_zero_is_reserved_for_genesis():
    with pytest.raises(InvalidRecord, match="reserved for genesis"):
        validate_record(mutate(make_record("inference"), "seq", 0))


# --- section schemas -----------------------------------------------------------------------------------

BAD_SECTION_VALUES = [
    ("inference", "input.sha256", "A" * 64), ("inference", "input.sha256", "a" * 63),
    ("inference", "input.source_kind", "decoded_frame"), ("inference", "input.source_kind", None),
    ("inference", "input.dims", [1920]), ("inference", "input.dims", [1920, 1080, 3]),
    ("inference", "input.dims", [0, 1080]), ("inference", "input.dims", [1920.0, 1080]),
    ("inference", "input.dims", [True, 1080]), ("inference", "input.dims", "1920x1080"),
    ("inference", "input.extra", 1),
    ("inference", "model.id", ""), ("inference", "model.id", "café"), ("inference", "model.id", "x" * 129),
    ("inference", "model.weights_sha256", "z" * 64), ("inference", "model.format", "ONNX"),
    ("inference", "model.format", "on nx"), ("inference", "model.arch_hash", "a" * 40),
    ("inference", "config.preprocess_ref", "sha256:" + "A" * 64), ("inference", "config.preprocess_ref", "sha1:" + "a" * 64),
    ("inference", "config.preprocess_ref", H("a")), ("inference", "config.code_commit", "a" * 64),
    ("inference", "config.runtime", ""), ("inference", "config.runtime", "x" * 201),
    ("inference", "output.payload_ref", "payload"), ("inference", "output.jcs_sha256", "F" * 64),
    ("inference", "output.decision_sha256", None), ("inference", "output.raw_jcs_sha256", 7),
    ("checkpoint", "checkpoint.tree_size", -1), ("checkpoint", "checkpoint.tree_size", 1.5),
    ("checkpoint", "checkpoint.root_hash", "A" * 64),
    ("anchor_event", "anchor.medium", "cloud"), ("anchor_event", "anchor.cosigner_key_ids", H("9")),
    ("anchor_event", "anchor.cosigner_key_ids", ["A" * 64]), ("anchor_event", "anchor.cosigner_key_ids", [H("9"), H("9")]),
    ("anchor_event", "anchor.checkpoint_seq", 1002),         # anchor cannot precede... equals seq
    ("anchor_event", "anchor.label", "x" * 65),
    ("degraded_marker", "gap.spill_sha256", "Unavailable"), ("degraded_marker", "gap.reported_count", -1),
    ("degraded_marker", "gap.last_unsealed_utc", "2026-09-19T02:00:00.000000Z"),      # before first
    ("degraded_marker", "gap.reason", "Ledger Unwritable"),
    ("scan_record", "scan.finding_counts", [1, 2]), ("scan_record", "scan.finding_counts", {"Critical": 1}),
    ("scan_record", "scan.finding_counts", {"critical": 1.5}), ("scan_record", "scan.code_commit", "a" * 39),
    ("scan_record", "scan.scan_id", "has space"),
    ("genesis", "deployment_manifest.checkpoint_every", 0),
    ("genesis", "deployment_manifest.spec", "cva-seal/2"), ("genesis", "deployment_manifest.key_id", "A" * 64),
    ("genesis", "deployment_manifest.extra", "x"),
]


@pytest.mark.parametrize("rtype,path,bad", BAD_SECTION_VALUES,
                         ids=[f"{t}:{p}={str(b)[:18]}" for t, p, b in BAD_SECTION_VALUES])
def test_section_field_violations_are_rejected(rtype, path, bad):
    with pytest.raises(InvalidRecord):
        validate_record(mutate(make_record(rtype), path, bad))


@pytest.mark.parametrize("path", ["input.sha256", "input.dims", "model.id", "model.weights_sha256",
                                  "config.runtime", "config.preprocess_ref", "output.payload_ref",
                                  "output.decision_sha256"])
def test_every_required_section_field_is_required(path):
    with pytest.raises(InvalidRecord, match="missing"):
        validate_record(mutate(make_record("inference"), path, DELETE))


def test_input_phash_present_form_validates_and_forbids_the_omitted_reason():
    ok = mutate(mutate(make_record("inference"), "input.phash", {"algo": "phash64-v1", "value": "0123456789abcdef"}),
                "input.phash_omitted_reason", DELETE)
    validate_record(ok)
    with pytest.raises(InvalidRecord, match="must be absent"):
        validate_record(mutate(ok, "input.phash_omitted_reason", "not_computed"))


@pytest.mark.parametrize("bad", [{"algo": "phash64-v1", "value": "0123456789ABCDEF"},
                                 {"algo": "phash64-v1", "value": "0123"},
                                 {"algo": "dhash", "value": "0123456789abcdef"},
                                 {"algo": "phash64-v1"}, "0123456789abcdef"])
def test_input_phash_malformed_forms_are_rejected(bad):
    r = mutate(mutate(make_record("inference"), "input.phash", bad), "input.phash_omitted_reason", DELETE)
    with pytest.raises(InvalidRecord):
        validate_record(r)


def test_a_null_phash_must_declare_why_it_was_omitted():
    """D10: an omission is declared, not silent."""
    with pytest.raises(InvalidRecord, match="required when phash is null"):
        validate_record(mutate(make_record("inference"), "input.phash_omitted_reason", DELETE))
    for reason in ("not_computed", "sampled_out", "unsupported_source"):
        validate_record(mutate(make_record("inference"), "input.phash_omitted_reason", reason))
    with pytest.raises(InvalidRecord):
        validate_record(mutate(make_record("inference"), "input.phash_omitted_reason", "forgot"))


def test_both_input_source_kinds_are_accepted():
    for kind in ("encoded_file", "model_input_tensor"):
        validate_record(mutate(make_record("inference"), "input.source_kind", kind))


def test_spill_sha256_may_be_the_declared_literal_unavailable():
    validate_record(mutate(make_record("degraded_marker"), "gap.spill_sha256", "unavailable"))


# --- genesis (D5) --------------------------------------------------------------------------------------

def test_genesis_prev_hash_is_sha256_of_0x03_then_the_canonical_manifest():
    expected = hashlib.sha256(b"\x03" + canonical_bytes(MANIFEST)).hexdigest()
    assert genesis_prev_hash(MANIFEST) == expected
    assert make_record("genesis")["prev_record_hash"] == expected


def test_genesis_does_not_start_from_32_zero_bytes():
    r = mutate(make_record("genesis"), "prev_record_hash", "0" * 64)
    with pytest.raises(InvalidRecord, match="deployment manifest"):
        validate_record(r)


@pytest.mark.parametrize("field,other", [("device_id", "jetson-08"), ("key_id", "2" * 64),
                                         ("profile_hash", "8" * 64), ("unit", "bravo-coy"),
                                         ("checkpoint_every", 500)])
def test_any_manifest_difference_changes_the_genesis_hash(field, other):
    """A valid chain from device A cannot be presented as device B's: the hash commits to identity."""
    assert genesis_prev_hash({**MANIFEST, field: other}) != genesis_prev_hash(MANIFEST)


def test_a_genesis_spliced_onto_another_manifest_fails_validation():
    r = make_record("genesis")
    r["deployment_manifest"] = {**MANIFEST, "device_id": "jetson-99"}     # prev hash now stale
    with pytest.raises(InvalidRecord, match="commit to the deployment manifest"):
        validate_record(r)


def test_genesis_must_be_seq_zero_and_signed_by_the_key_its_manifest_names():
    with pytest.raises(InvalidRecord, match="seq 0"):
        validate_record(mutate(make_record("genesis"), "seq", 1))
    with pytest.raises(InvalidRecord, match="key its manifest names"):
        validate_record(mutate(make_record("genesis"), "key_id", "2" * 64))


def test_genesis_prev_hash_rejects_an_invalid_manifest():
    with pytest.raises(InvalidRecord):
        genesis_prev_hash({**MANIFEST, "key_id": "nope"})
    with pytest.raises(InvalidRecord):
        genesis_prev_hash({k: v for k, v in MANIFEST.items() if k != "unit"})


# --- key rotation arithmetic (§5.10) ---------------------------------------------------------------------

def test_key_rotation_new_key_id_must_be_the_hash_of_the_new_public_key():
    with pytest.raises(InvalidRecord, match="not SHA-256 of new_public_key"):
        validate_record(mutate(make_record("key_rotation"), "rotation.new_key_id", H("9")))
    assert make_record("key_rotation")["rotation"]["new_key_id"] == hashlib.sha256(bytes.fromhex(PUB)).hexdigest()


def test_key_rotation_takes_effect_on_the_very_next_record():
    with pytest.raises(InvalidRecord, match=r"seq \+ 1"):
        validate_record(mutate(make_record("key_rotation"), "rotation.effective_seq", 9))
    with pytest.raises(InvalidRecord):
        validate_record(mutate(make_record("key_rotation"), "rotation.effective_seq", 7))


# --- analyst_event (D14 / Module E D-E6) -------------------------------------------------------------------

def analyst(**kw):
    a = {**ANALYST_OVERRIDE, **kw}
    for k in [k for k, v in kw.items() if v is DELETE]:
        del a[k]
    return make_record("analyst_event", analyst=a)


def test_an_override_needs_everything_that_makes_it_auditable():
    validate_record(analyst())
    for missing in ("new_disposition", "reason_code", "expected_prev_seq", "justification"):
        with pytest.raises(InvalidRecord):
            validate_record(analyst(**{missing: DELETE}))


@pytest.mark.parametrize("empty", ["", " ", "   "])
def test_an_override_without_a_justification_is_rejected(empty):
    """The SDK-level guarantee behind the C9 gate: "an override without a justification is rejected"."""
    with pytest.raises(InvalidRecord, match="non-empty"):
        validate_record(analyst(justification=empty))


def test_other_actions_may_have_an_empty_justification():
    validate_record(analyst(action="acknowledge", justification="", new_disposition=DELETE,
                            reason_code=DELETE, expected_prev_seq=DELETE))


def test_approve_must_name_the_event_it_blesses():
    base = {"action": "approve", "new_disposition": DELETE, "reason_code": DELETE, "expected_prev_seq": DELETE}
    with pytest.raises(InvalidRecord, match="refs_seq"):
        validate_record(analyst(**base))
    validate_record(analyst(refs_seq=12, **base))


def test_assign_must_name_the_assignee():
    base = {"action": "assign", "new_disposition": DELETE, "reason_code": DELETE, "expected_prev_seq": DELETE}
    with pytest.raises(InvalidRecord, match="assignee"):
        validate_record(analyst(**base))
    validate_record(analyst(assignee="b.rao", **base))


@pytest.mark.parametrize("field,bad", [
    ("action", "delete"), ("reason_code", "because"), ("new_disposition", "trash"),
    ("target_type", "image"), ("role", "Analyst"), ("request_id", "9" * 31), ("request_id", "F" * 32),
    ("expected_prev_seq", -1), ("expected_prev_seq", 1.5), ("actor_id", "a sharma"),
    ("justification", "café"), ("justification", "x" * 8193),
])
def test_analyst_field_violations_are_rejected(field, bad):
    with pytest.raises(InvalidRecord):
        validate_record(analyst(**{field: bad}))


def test_reason_codes_are_a_closed_enum_covering_both_general_and_prov_sets():
    for code in ("quality_issue", "known_benign", "insufficient_evidence", "accepted_risk",
                 "superseded_by_rescan", "other_with_justification",
                 "known_restore", "test_data", "superseded_ledger", "false_positive_confirmed"):
        validate_record(analyst(reason_code=code))


def test_a_percent_encoded_justification_is_accepted_and_raw_unicode_is_not():
    from cva.provenance.seal.canonical import ascii_encode
    validate_record(analyst(justification=ascii_encode("Duplicate cluster — confirmed")))
    with pytest.raises(InvalidRecord):
        validate_record(analyst(justification="Duplicate cluster — confirmed"))


# --- build_record ----------------------------------------------------------------------------------------

def build(**kw):
    args = {"seq": 5, "prev_record_hash": H("2"), "key_id": KEY_ID, "created_at_utc": TS, "nonce": NONCE,
            "body": {"checkpoint": {"tree_size": 1, "root_hash": H("8")}}}
    args.update(kw)
    return build_record("checkpoint", **args)


def test_build_record_requires_exactly_the_sections_of_the_type():
    with pytest.raises(InvalidRecord, match="needs exactly"):
        build(body={})
    with pytest.raises(InvalidRecord, match="needs exactly"):
        build(body={"checkpoint": {"tree_size": 1, "root_hash": H("8")}, "model": MODEL})
    with pytest.raises(InvalidRecord, match="unknown record type"):
        build_record("nope", seq=1, prev_record_hash=H("2"), key_id=KEY_ID, created_at_utc=TS,
                     nonce=NONCE, body={})


def test_build_record_deep_copies_so_later_mutation_cannot_change_what_is_signed():
    body = {"checkpoint": {"tree_size": 1, "root_hash": H("8")}}
    r = build(body=body)
    before = canon(r)
    body["checkpoint"]["tree_size"] = 999
    assert canon(r) == before


def test_build_record_accepts_an_aware_datetime_and_rejects_bad_strings():
    assert build(created_at_utc=datetime(2026, 9, 19, 2, 14, 7, 482913, tzinfo=UTC))["created_at_utc"] == TS
    with pytest.raises(InvalidRecord):
        build(created_at_utc="yesterday")


def test_build_record_validates_before_returning():
    with pytest.raises(InvalidRecord):
        build(body={"checkpoint": {"tree_size": -1, "root_hash": H("8")}})
    with pytest.raises(InvalidRecord):
        build(nonce="not-hex")


def test_build_record_output_never_contains_a_signature():
    assert "signature" not in build()


def test_field_order_of_the_caller_does_not_change_the_canonical_bytes():
    a = make_record("inference")
    b = {k: a[k] for k in reversed(list(a))}
    assert stored_bytes(a) == stored_bytes(b)


# --- nonce -----------------------------------------------------------------------------------------------

def test_new_nonce_is_32_lowercase_hex_from_16_bytes():
    n = new_nonce()
    assert len(n) == 32 and n == n.lower() and int(n, 16) >= 0
    assert new_nonce(lambda k: bytes(range(k))) == "000102030405060708090a0b0c0d0e0f"


def test_new_nonce_rejects_a_short_or_long_rng():
    with pytest.raises(ValueError):
        new_nonce(lambda k: b"\x00" * 8)
    with pytest.raises(ValueError):
        new_nonce(lambda k: b"\x00" * 32)


def test_nonces_do_not_repeat_over_many_draws():
    assert len({new_nonce() for _ in range(5000)}) == 5000


def test_unused_fixture_names_stay_referenced():
    assert INPUT and CONFIG and SIG      # keeps the shared fixtures honest under linters
