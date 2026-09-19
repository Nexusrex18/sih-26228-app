"""Canonical bytes and the strict parser (plan §5.1, §7.1; gate C1)."""
from __future__ import annotations

import json

import pytest
import rfc8785
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cva.provenance.seal.canonical import (
    ascii_decode,
    ascii_encode,
    canonical_bytes,
    parse_strict,
    validate_profile,
)
from cva.provenance.seal.constants import INT_MAX, MAX_DEPTH, MAX_RECORD_BYTES
from cva.provenance.seal.errors import NonCanonical

from ._subset_canon import ref_canon


def nest(depth: int) -> dict:
    o: dict = {"x": 1}
    for _ in range(depth - 1):
        o = {"k": o}
    return o


# --- the profile rejects each thing it forbids, naming the path ----------------------------------

@pytest.mark.parametrize("obj,needle", [
    ({"a": 1.5}, "float"),
    ({"a": float("nan")}, "float"),
    ({"a": float("inf")}, "float"),
    ({"a": INT_MAX + 1}, "outside"),
    ({"a": -INT_MAX - 1}, "outside"),
    ({"a": "café"}, "printable ASCII"),
    ({"a": "tab\there"}, "printable ASCII"),
    ({"a": "nl\n"}, "printable ASCII"),
    ({"a": "\x7f"}, "printable ASCII"),
    ({"A": 1}, "key"),
    ({"a-b": 1}, "key"),
    ({"a b": 1}, "key"),
    ({"": 1}, "key"),
    ({"é": 1}, "key"),
    ({1: 2}, "key"),
    ({"a": [1, "x"]}, "mixed"),
    ({"a": [True, 1]}, "mixed"),          # bool is not an int here
    ({"a": [None, None]}, "null inside an array"),
    ({"a": b"bytes"}, "unsupported"),
    ({"a": {1, 2}}, "unsupported"),
    ({"a": [1.0]}, "float"),
])
def test_profile_violations_are_rejected(obj, needle):
    with pytest.raises(NonCanonical) as e:
        canonical_bytes(obj)
    assert needle in str(e.value)
    assert e.value.code == "malformed_record"


def test_violation_message_names_the_json_path():
    with pytest.raises(NonCanonical) as e:
        canonical_bytes({"input": {"dims": [1, 2], "score": 0.5}})
    assert e.value.path == "$.input.score"


def test_top_level_must_be_an_object():
    for bad in ([1], "s", 1, None):
        with pytest.raises(NonCanonical):
            canonical_bytes(bad)  # type: ignore[arg-type]


def test_depth_limit_is_exact():
    canonical_bytes(nest(MAX_DEPTH))                     # depth 8 accepted
    with pytest.raises(NonCanonical, match="nesting"):
        canonical_bytes(nest(MAX_DEPTH + 1))             # depth 9 rejected


def test_arrays_count_toward_depth():
    o: dict = {"a": [[[[[[[1]]]]]]]}                     # object + 7 arrays = 8
    canonical_bytes(o)
    with pytest.raises(NonCanonical, match="nesting"):
        canonical_bytes({"a": [[[[[[[[1]]]]]]]]})


def test_bool_int_str_and_object_arrays_are_accepted():
    for arr in ([1, 2], ["a", "b"], [True, False], [{"a": 1}, {"b": 2}], [[1], [2, 3]], []):
        canonical_bytes({"a": arr})


def test_boundary_integers_are_accepted():
    assert canonical_bytes({"a": INT_MAX, "b": -INT_MAX, "c": 0}) == b'{"a":9007199254740991,"b":-9007199254740991,"c":0}'


def test_tuples_canonicalise_like_lists():
    assert canonical_bytes({"a": (1, 2)}) == canonical_bytes({"a": [1, 2]})


def test_keys_are_sorted_as_in_rfc8785_appendix_c():
    address = {"name": "John Doe", "address": "2000 Sunset Boulevard", "city": "Los Angeles",
               "zip": "90001", "state": "CA"}
    keys = list(json.loads(canonical_bytes(address)))
    assert keys == ["address", "city", "name", "state", "zip"]


def test_size_cap_applies_to_records_but_not_to_payloads():
    big = {"a": "x" * (MAX_RECORD_BYTES + 1)}
    with pytest.raises(NonCanonical, match="cap"):
        canonical_bytes(big)
    assert len(canonical_bytes(big, max_bytes=None)) > MAX_RECORD_BYTES     # a raw payload is fine


def test_canonical_bytes_does_not_mutate_or_reorder_its_input():
    o = {"b": [1, 2], "a": {"z": 1, "y": 2}}
    snapshot = json.dumps(o)
    canonical_bytes(o)
    assert json.dumps(o) == snapshot


# --- differential: our profile, an independent reference, and the library agree -------------------

_ASCII = st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=12)
_KEYS = st.from_regex(r"[a-z0-9_]{1,6}", fullmatch=True)
_INTS = st.integers(-INT_MAX, INT_MAX)


def _objects(level: int):
    scalar = st.one_of(_ASCII, _INTS, st.booleans(), st.none())
    if level == 0:
        return st.dictionaries(_KEYS, scalar, max_size=4)
    sub = _objects(level - 1)
    arrays = st.one_of(st.lists(_INTS, max_size=4), st.lists(_ASCII, max_size=4),
                       st.lists(st.booleans(), max_size=4), st.lists(sub, max_size=3))
    return st.dictionaries(_KEYS, st.one_of(scalar, sub, arrays), max_size=4)


PROFILE_OBJECTS = _objects(2)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(PROFILE_OBJECTS)
def test_every_accepted_object_matches_rfc8785_and_the_independent_reference(obj):
    got = canonical_bytes(obj, max_bytes=None)
    assert got == rfc8785.dumps(obj)                       # subset of JCS: byte-identical
    assert got == ref_canon(obj).encode("ascii")           # and to a from-scratch implementation


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(PROFILE_OBJECTS)
def test_parse_strict_inverts_canonical_bytes(obj):
    data = canonical_bytes(obj, max_bytes=None)
    assert parse_strict(data, max_bytes=None) == obj
    assert canonical_bytes(parse_strict(data, max_bytes=None), max_bytes=None) == data


# --- parse_strict rejects what a tolerant parser would accept -------------------------------------

@pytest.mark.parametrize("data,code", [
    (b'{"a":1,"a":2}', "malformed_record"),                # duplicate key: the parser-differential attack
    (b'{"a":1,"b":2,"a":3}', "malformed_record"),
    (b'{"a":1.5}', "malformed_record"),
    (b'{"a":1e2}', "malformed_record"),
    (b'{"a":1.0}', "malformed_record"),
    (b'{"a":NaN}', "malformed_record"),
    (b'{"a":Infinity}', "malformed_record"),
    (b'{"a":-Infinity}', "malformed_record"),
    (b'{"a":9007199254740992}', "malformed_record"),       # 2**53
    (b'{"a":-9007199254740992}', "malformed_record"),
    (b'{"a":123456789012345678901234567890}', "malformed_record"),
    (b'{"a":"\xc3\xa9"}', "malformed_record"),             # non-ASCII byte
    (b'\xef\xbb\xbf{"a":1}', "malformed_record"),          # BOM
    (b'[1,2]', "malformed_record"),                        # top level not an object
    (b'"s"', "malformed_record"),
    (b'', "malformed_record"),
    (b'{"a":', "malformed_record"),                        # truncated
    (b'{"a":1}}', "malformed_record"),
    (b'{"A":1}', "malformed_record"),                      # key outside the profile
    (b'{"a":[1,"x"]}', "malformed_record"),                # mixed array
    (b'{"a":-0}', "non_canonical_encoding"),               # valid, but canonical form is 0
    (b'{"b":1,"a":2}', "non_canonical_encoding"),          # key order
    (b'{"a": 1}', "non_canonical_encoding"),               # whitespace
    (b' {"a":1}', "non_canonical_encoding"),
    (b'{"a":1}\n', "non_canonical_encoding"),              # trailing newline
    (b'{"a":"\\u0041"}', "non_canonical_encoding"),        # escape where the literal is canonical
    (b'{"a":"\\/"}', "non_canonical_encoding"),
    (b'{"a":"x\\ty"}', "malformed_record"),                # control char after unescape: profile violation
])
def test_parse_strict_rejects(data, code):
    with pytest.raises(NonCanonical) as e:
        parse_strict(data)
    assert e.value.code == code, str(e.value)


def test_parse_strict_rejects_pathological_nesting_without_crashing():
    with pytest.raises(NonCanonical):
        parse_strict(b"[" * 30_000)
    with pytest.raises(NonCanonical):
        parse_strict(b'{"a":' * 10_000 + b"1" + b"}" * 10_000)


def test_parse_strict_enforces_the_size_cap_before_parsing():
    with pytest.raises(NonCanonical, match="cap"):
        parse_strict(b'{"a":"' + b"x" * MAX_RECORD_BYTES + b'"}')


def test_parse_strict_accepts_bytearray_and_memoryview_inputs():
    assert parse_strict(bytearray(b'{"a":1}')) == {"a": 1}
    assert parse_strict(memoryview(b'{"a":1}')) == {"a": 1}   # type: ignore[arg-type]


def test_a_reordered_record_is_a_finding_not_a_parse_error():
    """T13: same logical record, different bytes -> `non_canonical_encoding`, distinct from malformed."""
    good = canonical_bytes({"a": 1, "b": 2})
    assert parse_strict(good) == {"a": 1, "b": 2}
    with pytest.raises(NonCanonical) as e:
        parse_strict(b'{"b":2,"a":1}')
    assert e.value.code == "non_canonical_encoding"


# --- ascii_encode / ascii_decode -------------------------------------------------------------------

@pytest.mark.parametrize("text,encoded", [
    ("plain text", "plain text"),
    ("100% sure", "100%25 sure"),
    ("café", "caf%C3%A9"),
    ("line1\nline2", "line1%0Aline2"),
    ("tab\t", "tab%09"),
    ("€", "%E2%82%AC"),
    ("\U0001f600", "%F0%9F%98%80"),
    ('quote " and \\ backslash', 'quote " and \\ backslash'),   # printable ASCII: left alone
    ("", ""),
    ("%", "%25"),
    ("%41", "%2541"),                                           # a literal "%41" must not decode to "A"
])
def test_ascii_encode_examples(text, encoded):
    assert ascii_encode(text) == encoded
    assert ascii_decode(encoded) == text


@settings(max_examples=300)
@given(st.text())
def test_ascii_encode_round_trips_and_lands_inside_the_profile(text):
    enc = ascii_encode(text)
    assert ascii_decode(enc) == text
    validate_profile({"justification": enc})                    # what the SDK would put in a record
    assert ascii_encode(ascii_decode(enc)) == enc               # one canonical spelling


def test_ascii_encode_uses_uppercase_hex_only():
    assert ascii_encode("ÿ") == "%C3%BF"


@pytest.mark.parametrize("bad", ["%c3%a9", "%", "%4", "%GG", "100%", "%C3%28", "x\ny", "café"])
def test_ascii_decode_is_strict(bad):
    with pytest.raises(NonCanonical):
        ascii_decode(bad)


def test_ascii_encode_rejects_a_lone_surrogate():
    with pytest.raises(NonCanonical):
        ascii_encode("bad \ud800 surrogate")
