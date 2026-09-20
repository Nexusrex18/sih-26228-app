"""The frozen wire-format vectors (plan §11.8; gate C8): regenerating them must reproduce the frozen file byte for
byte — here, and on any other interpreter that has the dependencies (set CVA_SEAL_OTHER_PYTHONS=/path/py:/path/py2).

WHAT THIS PROVES: determinism. The vectors are generated FROM the reference implementation, so a green result here does not
say the reference matches the specification — see the docstring of `spec/vectors/build_cva_seal_v1.py`. Conformance is
evidenced by an independent reader of the spec text, not by this file."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spec" / "vectors"))
import build_cva_seal_v1 as gen  # noqa: E402

FROZEN = ROOT / "spec/vectors/cva_seal_v1.json"
# The digest of the frozen file. Changing the wire format changes this on purpose — and needs a decision entry.
FROZEN_SHA256 = hashlib.sha256(FROZEN.read_bytes()).hexdigest()


def test_regenerating_the_vectors_reproduces_the_frozen_file_byte_for_byte():
    assert gen.encode(gen.build()) == FROZEN.read_bytes()


def test_two_builds_in_one_process_are_identical():
    assert gen.encode(gen.build()) == gen.encode(gen.build())


def test_the_frozen_file_covers_every_record_type_the_generator_can_make_and_the_features_c7_added():
    d = json.loads(FROZEN.read_text())
    types = {json.loads(r)["type"] for r in d["records"]}
    assert {"genesis", "model_registration", "inference", "checkpoint", "anchor_event", "key_rotation",
            "degraded_marker"} <= types
    assert d["anchor"]["cosignatures"] and d["anchor"]["attestations"] and d["inclusion_proof"]["path"]
    kinds = {json.loads(r)["input"]["source_kind"] for r in d["records"] if json.loads(r)["type"] == "inference"}
    assert kinds == {"encoded_file"}
    assert len({json.loads(r)["key_id"] for r in d["records"]}) == 2            # signed by two keys: the rotation is in it


def test_the_frozen_keys_are_what_the_seeds_say():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    for k in json.loads(FROZEN.read_text())["keys"].values():
        pub = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(k["seed"])).public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        assert pub.hex() == k["public_key"] and hashlib.sha256(pub).hexdigest() == k["key_id"]


def test_the_frozen_ledger_verifies_with_the_reference_verifier():
    from cva.provenance.seal.keys import parse_trust_root
    from cva.provenance.seal.verify import verify_ledger
    d = json.loads(FROZEN.read_text())
    rep = verify_ledger("".join(r + "\n" for r in d["records"]).encode(),
                        trust_root=parse_trust_root(json.dumps(d["trust_root"]).encode()), anchors=[d["anchor"]])
    assert rep.classes() == ["degraded_gap"] and rep.anchors_verified == 1 and rep.rotations == 1


def test_the_vectors_are_byte_identical_on_other_python_versions():
    # Not required under CVA_SEAL_REQUIRE_NATIVE: a CI matrix runs `spec/vectors/build_cva_seal_v1.py` once per Python
    # version (it exits 1 on any difference), which proves cross-version byte identity more directly than this test.
    if not os.environ.get("CVA_SEAL_OTHER_PYTHONS"):
        pytest.skip("set CVA_SEAL_OTHER_PYTHONS to run this on other interpreters (the CI matrix runs the generator per version instead)")
    for py in os.environ["CVA_SEAL_OTHER_PYTHONS"].split(":"):
        out = subprocess.run([py, str(ROOT / "spec/vectors/build_cva_seal_v1.py"), "--stdout"], capture_output=True, check=True)
        assert out.stdout == FROZEN.read_bytes(), f"{py} produced different bytes"
        ver = subprocess.run([py, str(ROOT / "spec/independent_verifier.py"), "-h"], capture_output=True)
        assert ver.returncode == 0
