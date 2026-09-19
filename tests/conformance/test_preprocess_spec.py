"""The DECLARED preprocessing spec (`cva.loaders.preprocess`) — plan §7.7 and §9.2.

`preprocess_hash` and `preprocess_ref` are two of the six fields Backend owes Module C's seal
record. These tests pin the three things a wrong implementation would get quietly right on the
happy path: the hash is over the QUANTISED spec with `floor(x*1e6+0.5)` (never `round()`), a
hostile spec is REJECTED with the field named rather than ignored, and `preprocess_ref` is the
name the evidence store really gives the bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from cva.core.quantise import _floor_half, canonical_bytes, q_e6, q_preprocess
from cva.core.scanid import EvidenceStore
from cva.loaders.detect import detect_and_load
from cva.loaders.models import load_model
from cva.loaders.preprocess import (
    PreprocessSpecError,
    attach_preprocess,
    load_preprocess_spec,
    preprocess_digest,
    preprocess_ref_of,
    quantise_spec,
    sidecar_path,
    validate_preprocess_spec,
)

IMAGENET = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "layout": "CHW",
            "dtype": "float32", "value_range": [0.0, 1.0], "input_shape": [3, 224, 224]}


def _spec(**over: Any) -> dict[str, Any]:
    d = json.loads(json.dumps(IMAGENET))
    d.update(over)
    return d


def _write(tmp_path: Path, obj: Any, name: str = "spec.json") -> Path:
    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return p


# --- the hash ---------------------------------------------------------------------------------

def test_the_hash_is_deterministic_and_independent_of_key_order(tmp_path):
    a = _write(tmp_path, IMAGENET, "a.json")
    reordered = dict(reversed(list(IMAGENET.items())))
    assert list(reordered) != list(IMAGENET)
    b = _write(tmp_path, reordered, "b.json")
    h1, blob1 = preprocess_digest(load_preprocess_spec(a))
    h2, blob2 = preprocess_digest(load_preprocess_spec(b))
    h3, _ = preprocess_digest(load_preprocess_spec(a))
    assert h1 == h2 == h3 and blob1 == blob2
    assert len(h1) == 64 and h1 == h1.lower()


def test_the_hash_is_sha256_of_the_canonical_bytes_of_the_quantised_spec():
    expected = {"quantisation": "e6", "mean": [485000, 456000, 406000],
                "std": [229000, 224000, 225000], "layout": "CHW", "dtype": "float32",
                "value_range": [0, 1000000], "input_shape": [3, 224, 224]}
    blob = canonical_bytes(expected)
    h, got = preprocess_digest(IMAGENET)
    assert got == blob
    assert h == hashlib.sha256(blob).hexdigest()
    assert quantise_spec(IMAGENET) == expected
    # No float survives into the hashed form.
    assert b"." not in got


def test_quantisation_is_floor_half_up_not_bankers_rounding():
    """`round()` is banker's: round(0.5) == 0, round(1.5) == 2, round(2.5) == 2. The normative
    rule is floor(x + 0.5), which is 1, 2, 3 — and is the same in every language."""
    for half, want in ((0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4), (-0.5, 0), (-1.5, -1)):
        assert _floor_half(half) == want, half
    assert round(0.5) == 0 and round(2.5) == 2, "premise: Python's round() is banker's"
    # ...and the same rule reaches the hashed integers. x * 1e6 is not always an EXACT half in
    # float64, so assert against the rule's own definition and against round() where they part.
    for x in (0.0000005, 0.0000015, 0.0000025, 0.0000035, 0.4999995, 0.5000005):
        assert q_e6(x) == math.floor(float(x) * 1_000_000.0 + 0.5), x
    assert q_e6(0.0000005) == 1
    assert q_e6(0.0000015) == 2
    assert q_e6(0.0000025) == 3
    assert round(0.0000025 * 1_000_000.0) == 2, "round() would have said 2 for the same half"


def test_a_change_below_the_quantum_changes_nothing_and_at_the_quantum_changes_the_hash():
    base = _spec(mean=[0.485, 0.456, 0.406])
    seventh = _spec(mean=[0.4850001, 0.456, 0.406])       # 7th decimal: absorbed by the quantum
    sixth = _spec(mean=[0.485001, 0.456, 0.406])          # 6th decimal: one quantum
    assert preprocess_digest(base)[0] == preprocess_digest(seventh)[0]
    assert preprocess_digest(base)[0] != preprocess_digest(sixth)[0]


def test_an_integer_and_the_same_float_hash_alike():
    """JSON `1` parses to int and `1.0` to float; the hash must not depend on which the
    operator typed."""
    ints = _spec(mean=[0, 0, 0], std=[1, 1, 1], value_range=[0, 1])
    floats = _spec(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], value_range=[0.0, 1.0])
    assert preprocess_digest(ints) == preprocess_digest(floats)


def test_a_negative_mean_is_not_clamped_to_zero():
    """`q_confidence` clamps to [0, 1e6]; using it here would hash a mean of -0.4 exactly like
    a mean of 0."""
    neg = _spec(mean=[-0.4, 0.0, 0.0])
    zero = _spec(mean=[0.0, 0.0, 0.0])
    assert quantise_spec(neg)["mean"] == [-400000, 0, 0]
    assert preprocess_digest(neg)[0] != preprocess_digest(zero)[0]
    assert q_preprocess(-0.4) == -400000, "the generic walker had the same clamp"


def test_every_declared_field_is_part_of_the_hash():
    base = preprocess_digest(IMAGENET)[0]
    for variant in (_spec(std=[0.229, 0.224, 0.226]), _spec(layout="HWC", input_shape=[224, 224, 3]),
                    _spec(dtype="float16"), _spec(value_range=[0.0, 255.0]),
                    _spec(input_shape=[3, 256, 256])):
        assert preprocess_digest(variant)[0] != base, variant


def test_the_quantised_form_matches_the_seals_own_canonicaliser():
    """Crypto's `register_config` takes an already-quantised spec and canonicalises it with its
    own profile (ASCII keys, integers only, no mixed arrays). If our quantised form were outside
    that profile the seal would refuse it, or hash different bytes."""
    canon = pytest.importorskip("cva.provenance.seal.canonical")
    seal_q = pytest.importorskip("cva.provenance.seal.quantise")
    q = quantise_spec(IMAGENET)
    assert canon.canonical_bytes(q, max_bytes=None) == preprocess_digest(IMAGENET)[1]
    for x in (0.485, -0.4, 0.0000005, 0.0000015, 0.0000025, 0.4999995, 1e5 + 0.5e-6):
        assert seal_q.q_e6(x) == q_e6(x), x


# --- rejection (S5 + in-code validation) -------------------------------------------------------

def _reject(tmp_path: Path, payload: Any, needle: str) -> None:
    with pytest.raises(PreprocessSpecError) as exc:
        load_preprocess_spec(_write(tmp_path, payload))
    assert needle in str(exc.value), str(exc.value)


@pytest.mark.parametrize("over, needle", [
    ({"mean": [0.5, 0.5, "0.5"]}, "mean[2]"),
    ({"mean": [0.5, True, 0.5]}, "mean[1]"),
    ({"std": [0.2, False, 0.2]}, "std[1]"),
    ({"mean": [0.5, None, 0.5]}, "mean[1]"),
    ({"std": [0.2, 0.2, 0.0]}, "std[2]"),
    ({"std": [0.2, -0.1, 0.2]}, "std[1]"),
    ({"mean": [0.5, 0.5]}, "mean has 2 entries but std has 3"),
    ({"mean": [0.5, 0.5], "std": [0.2, 0.2]}, "3 channels"),
    ({"mean": []}, "mean"),
    ({"mean": 0.5}, "mean"),
    ({"mean": [1e9, 0.5, 0.5]}, "mean[0]"),
    ({"layout": "NCHW"}, "layout"),
    ({"layout": 3}, "layout"),
    ({"dtype": "bfloat16"}, "dtype"),
    ({"value_range": [1.0, 0.0]}, "value_range"),
    ({"value_range": [0.0]}, "value_range"),
    ({"input_shape": [3, 224]}, "input_shape"),
    ({"input_shape": [3, 224.5, 224]}, "input_shape[1]"),
    ({"input_shape": [3, 0, 224]}, "input_shape[1]"),
    ({"input_shape": [True, 224, 224]}, "input_shape[0]"),
    ({"resize": 256}, "resize"),
], ids=lambda v: str(v)[:40])
def test_a_hostile_or_malformed_spec_is_rejected_and_the_field_is_named(tmp_path, over, needle):
    _reject(tmp_path, _spec(**over), needle)


def test_hwc_channels_are_the_last_axis():
    ok = _spec(layout="HWC", input_shape=[224, 224, 3])
    assert validate_preprocess_spec(ok)["input_shape"] == [224, 224, 3]
    with pytest.raises(PreprocessSpecError, match="has 224 channels"):
        # A CHW-shaped input_shape declared as HWC has 224 "channels".
        validate_preprocess_spec(_spec(layout="HWC", input_shape=[3, 224, 224]))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_non_finite_numbers_are_rejected(tmp_path, literal):
    text = ('{"mean": [LITERAL, 0.5, 0.5], "std": [0.2, 0.2, 0.2], "layout": "CHW", '
            '"dtype": "float32"}').replace("LITERAL", literal)
    with pytest.raises(PreprocessSpecError) as exc:
        load_preprocess_spec(_write(tmp_path, text))
    assert "mean" in str(exc.value) or "NaN" in str(exc.value) or "Infinity" in str(exc.value)


def test_a_missing_required_key_is_named(tmp_path):
    spec = _spec()
    del spec["dtype"]
    _reject(tmp_path, spec, "'dtype'")


def test_a_duplicate_key_is_rejected_because_parsers_disagree_about_which_wins(tmp_path):
    text = ('{"mean": [0, 0, 0], "std": [1, 1, 1], "std": [2, 2, 2], "layout": "CHW", '
            '"dtype": "float32"}')
    _reject(tmp_path, text, "duplicate key")


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"mean"', "3", "null"])
def test_a_top_level_that_is_not_an_object_is_rejected(tmp_path, payload):
    _reject(tmp_path, payload, "JSON object")


def test_not_json_at_all_is_rejected(tmp_path):
    _reject(tmp_path, "{not json", "not valid JSON")


def test_an_oversized_file_is_rejected_by_s5_before_it_is_parsed(tmp_path):
    big = ('{"mean": [0.5], "std": [0.5], "layout": "CHW", "dtype": "float32", "pad": "'
           + "x" * (70 << 10) + '"}')
    _reject(tmp_path, big, "[S5]")


def test_a_json_nesting_bomb_is_rejected_by_s5(tmp_path):
    bomb = "[" * 200 + "]" * 200
    _reject(tmp_path, bomb, "[S5]")
    _reject(tmp_path, '{"mean": ' + "[" * 12 + "0" + "]" * 12 + "}", "[S5]")


def test_a_missing_file_is_a_clear_error_not_a_silent_no_spec(tmp_path):
    with pytest.raises(PreprocessSpecError, match="not found"):
        load_preprocess_spec(tmp_path / "nope.json")


# --- the ref ------------------------------------------------------------------------------------

def test_the_ref_is_the_bare_name_the_evidence_store_returns(tmp_path):
    h, blob = preprocess_digest(IMAGENET)
    ref = preprocess_ref_of(h)
    assert ref == f"{h}.json"
    stored = EvidenceStore(tmp_path).put_bytes(blob, ".json")
    assert stored == ref
    assert EvidenceStore(tmp_path).path_for(ref).read_bytes() == blob


# --- attaching to a handle ------------------------------------------------------------------------

@pytest.fixture
def model_copy(artefacts, tmp_path) -> Path:
    """A private copy: the session-scoped `artefacts["onnx"]` is loaded by other tests, and a
    sidecar written beside it would leak into them."""
    dest = tmp_path / "models" / "m.onnx"
    dest.parent.mkdir()
    shutil.copy(artefacts["onnx"], dest)
    return dest


def test_a_sidecar_beside_the_model_is_found_by_name(model_copy):
    assert sidecar_path(model_copy).name == "m.onnx.preprocess.json"
    sidecar_path(model_copy).write_text(json.dumps(IMAGENET))
    h = detect_and_load(model_copy)
    digest, blob = preprocess_digest(IMAGENET)
    assert h.preprocess_hash == digest
    assert h.preprocess_ref == f"{digest}.json"
    assert h.preprocess_spec_bytes == blob


def test_load_model_finds_the_sidecar_too(model_copy):
    sidecar_path(model_copy).write_text(json.dumps(IMAGENET))
    assert load_model(model_copy).preprocess_hash == preprocess_digest(IMAGENET)[0]


def test_an_explicit_preprocess_wins_over_the_sidecar(model_copy, tmp_path):
    sidecar_path(model_copy).write_text(json.dumps(IMAGENET))
    other = _spec(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    explicit = _write(tmp_path, other, "elsewhere.json")
    h = detect_and_load(model_copy, preprocess=explicit)
    assert h.preprocess_hash == preprocess_digest(other)[0]
    assert h.preprocess_hash != preprocess_digest(IMAGENET)[0]


def test_an_explicit_preprocess_works_without_a_sidecar(model_copy, tmp_path):
    explicit = _write(tmp_path, IMAGENET, "given.json")
    assert not sidecar_path(model_copy).exists()
    assert detect_and_load(model_copy, preprocess=explicit).preprocess_hash \
        == preprocess_digest(IMAGENET)[0]


def test_with_no_spec_nothing_is_attached_and_the_model_loads_as_before(model_copy):
    h = detect_and_load(model_copy)
    assert h.fmt == "onnx" and h.num_classes == 2
    assert not hasattr(h, "preprocess_hash")
    assert not hasattr(h, "preprocess_ref") and not hasattr(h, "preprocess_spec_bytes")
    assert getattr(h, "preprocess_hash", None) is None
    # The pre-existing positional signature is unchanged.
    h2 = detect_and_load(model_copy, None, "renamed", True)
    assert h2.model_id == "renamed" and not hasattr(h2, "preprocess_hash")


def test_a_malformed_sidecar_stops_the_load_instead_of_being_ignored(model_copy):
    sidecar_path(model_copy).write_text(json.dumps(_spec(std=[0.2, 0.2, 0.0])))
    with pytest.raises(PreprocessSpecError, match=r"std\[2\]"):
        detect_and_load(model_copy)


def test_attach_preprocess_on_a_plain_object_and_the_no_spec_case(tmp_path):
    class Handle:
        pass

    h = Handle()
    assert attach_preprocess(h, tmp_path / "absent.onnx") is False
    assert not hasattr(h, "preprocess_hash")
    spec = _write(tmp_path, IMAGENET)
    assert attach_preprocess(h, None, spec) is True
    assert h.preprocess_ref == preprocess_ref_of(h.preprocess_hash)      # type: ignore[attr-defined]
