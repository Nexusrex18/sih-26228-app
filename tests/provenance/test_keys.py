"""Keys, custody, the trust root (plan §5.10, §7.5; gate C3)."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import os
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from cva.provenance.seal.errors import (
    InvalidKeyMaterial,
    KeyExists,
    KeyNotConfigured,
    KeyPermissionError,
    TrustRootError,
)
from cva.provenance.seal.keys import (
    EnvKeyProvider,
    FileKeyProvider,
    Pkcs11KeyProvider,
    TrustKey,
    TrustRoot,
    generate_keypair,
    load_trust_root,
    parse_trust_root,
    verify_ed25519,
)

V = Path(__file__).resolve().parents[2] / "spec" / "vectors"
SEAL = Path(__file__).resolve().parents[2] / "cva" / "provenance" / "seal"
RFC8032 = json.loads((V / "rfc8032.json").read_text())["vectors"]
IDS = [v["name"] for v in RFC8032]


def write_key(path: Path, material: bytes, mode: int = 0o600) -> Path:
    path.write_bytes(material)
    path.chmod(mode)
    return path


def key_id(pub: bytes) -> str:
    return hashlib.sha256(pub).hexdigest()


# --- RFC 8032 published vectors, through OUR providers ----------------------------------------------------

@pytest.mark.parametrize("v", RFC8032, ids=IDS)
def test_rfc8032_vectors_through_a_file_provider_with_a_raw_seed(tmp_path, v):
    p = FileKeyProvider(write_key(tmp_path / "k", bytes.fromhex(v["secret_key"])))
    assert p.public_key.hex() == v["public_key"]
    assert p.sign(bytes.fromhex(v["message"])).hex() == v["signature"]          # deterministic: exact match
    assert p.key_id == key_id(bytes.fromhex(v["public_key"]))


@pytest.mark.parametrize("v", RFC8032, ids=IDS)
def test_rfc8032_vectors_through_a_file_provider_with_a_pem(tmp_path, v):
    pem = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(v["secret_key"])).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    p = FileKeyProvider(write_key(tmp_path / "k.pem", pem))
    assert p.public_key.hex() == v["public_key"]
    assert p.sign(bytes.fromhex(v["message"])).hex() == v["signature"]


@pytest.mark.parametrize("v", RFC8032, ids=IDS)
def test_rfc8032_vectors_through_an_env_provider(v):
    p = EnvKeyProvider("K", environ={"K": base64.b64encode(bytes.fromhex(v["secret_key"])).decode()})
    assert p.public_key.hex() == v["public_key"]
    assert p.sign(bytes.fromhex(v["message"])).hex() == v["signature"]


@pytest.mark.parametrize("v", RFC8032, ids=IDS)
def test_rfc8032_vectors_verify_and_reject_a_change(v):
    pub, msg, sig = (bytes.fromhex(v[k]) for k in ("public_key", "message", "signature"))
    assert verify_ed25519(pub, msg, sig)
    assert not verify_ed25519(pub, msg + b"x", sig)
    assert not verify_ed25519(pub, msg, bytes([sig[0] ^ 1]) + sig[1:])
    assert not verify_ed25519(bytes([pub[0] ^ 1]) + pub[1:], msg, sig)


def test_a_signature_with_a_non_canonical_S_is_rejected():
    """RFC 8032 §5.1.7: S must be < L. Adding L to S yields a second byte string that would verify under
    a lax library — a malleability channel. It must not (and the chain link covers the signature anyway)."""
    L = 2**252 + 27742317777372353535851937790883648493
    v = RFC8032[1]
    pub, msg, sig = (bytes.fromhex(v[k]) for k in ("public_key", "message", "signature"))
    s = int.from_bytes(sig[32:], "little")
    assert s < L and s + L < 2**256
    malleated = sig[:32] + (s + L).to_bytes(32, "little")
    assert malleated != sig and not verify_ed25519(pub, msg, malleated)


@pytest.mark.parametrize("pub,sig", [(b"", b""), (b"x" * 31, b"y" * 64), (b"x" * 32, b"y" * 63),
                                     (b"x" * 33, b"y" * 64), (bytes(32), bytes(64))])
def test_verify_ed25519_returns_false_on_malformed_input(pub, sig):
    assert verify_ed25519(pub, b"m", sig) is False


# --- file custody ---------------------------------------------------------------------------------------

def test_a_file_provider_loads_a_600_key_and_self_tests(tmp_path):
    p = FileKeyProvider(write_key(tmp_path / "k", bytes(range(32))))
    assert p.custody == "file" and p.self_test() and len(p.public_key) == 32


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_owner_only_modes_are_accepted(tmp_path, mode):
    FileKeyProvider(write_key(tmp_path / "k", bytes(32), mode))


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o666, 0o601])
def test_a_key_readable_by_group_or_other_is_refused(tmp_path, mode):
    with pytest.raises(KeyPermissionError, match="group/other"):
        FileKeyProvider(write_key(tmp_path / "k", bytes(32), mode))


def test_loose_permissions_are_allowed_only_explicitly_and_the_choice_is_logged(tmp_path, caplog):
    path = write_key(tmp_path / "k", bytes(32), 0o644)
    with caplog.at_level(logging.WARNING, logger="cva.provenance.seal"):
        FileKeyProvider(path, allow_loose_permissions=True)
    assert any("allow_loose_permissions" in r.message for r in caplog.records)


def test_a_missing_key_file_raises_and_creates_nothing(tmp_path):
    before = set(tmp_path.iterdir())
    with pytest.raises(KeyNotConfigured, match="keygen"):
        FileKeyProvider(tmp_path / "absent.key")
    assert set(tmp_path.iterdir()) == before


def test_a_directory_is_not_a_key(tmp_path):
    d = tmp_path / "dir"
    d.mkdir(mode=0o700)
    with pytest.raises((InvalidKeyMaterial, KeyNotConfigured)):
        FileKeyProvider(d)


def test_a_symlink_to_a_key_is_followed_and_its_target_mode_checked(tmp_path):
    real = write_key(tmp_path / "real", bytes(range(32)), 0o600)
    link = tmp_path / "link"
    link.symlink_to(real)
    assert FileKeyProvider(link).public_key == FileKeyProvider(real).public_key
    real.chmod(0o644)
    with pytest.raises(KeyPermissionError):
        FileKeyProvider(link)


@pytest.mark.parametrize("material", [b"", b"x" * 31, b"x" * 33, b"00" * 32, b"not a key\n", b"x" * 5000])
def test_key_files_of_the_wrong_shape_are_rejected(tmp_path, material):
    with pytest.raises(InvalidKeyMaterial):
        FileKeyProvider(write_key(tmp_path / "k", material))


def test_an_encrypted_pem_is_rejected_with_a_clear_message(tmp_path):
    enc = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.BestAvailableEncryption(b"pw"))
    with pytest.raises(InvalidKeyMaterial, match="encrypted"):
        FileKeyProvider(write_key(tmp_path / "k.pem", enc))


def test_a_non_ed25519_pem_is_rejected(tmp_path):
    rsa_pem = rsa.generate_private_key(65537, 2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    with pytest.raises(InvalidKeyMaterial, match="expected Ed25519"):
        FileKeyProvider(write_key(tmp_path / "k.pem", rsa_pem))


def test_a_garbage_pem_is_rejected(tmp_path):
    with pytest.raises(InvalidKeyMaterial):
        FileKeyProvider(write_key(tmp_path / "k.pem", b"-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"))


# --- environment custody ----------------------------------------------------------------------------------

def test_an_unset_or_empty_variable_raises_key_not_configured():
    for env in ({}, {"K": ""}):
        with pytest.raises(KeyNotConfigured, match="keygen"):
            EnvKeyProvider("K", environ=env)


@pytest.mark.parametrize("value", ["not base64!!", base64.b64encode(b"x" * 31).decode(),
                                   base64.b64encode(b"x" * 33).decode(), base64.b64encode(bytes(32)).decode() + "\n",
                                   " " + base64.b64encode(bytes(32)).decode(), "AAAA"])
def test_an_env_value_that_is_not_exactly_a_base64_seed_is_rejected(value):
    with pytest.raises(InvalidKeyMaterial):
        EnvKeyProvider("K", environ={"K": value})


def test_env_provider_reads_the_real_environment_by_default(monkeypatch):
    monkeypatch.setenv("CVA_TEST_KEY", base64.b64encode(bytes(range(32))).decode())
    assert EnvKeyProvider("CVA_TEST_KEY").custody == "env"
    monkeypatch.delenv("CVA_TEST_KEY")
    with pytest.raises(KeyNotConfigured):
        EnvKeyProvider("CVA_TEST_KEY")


# --- providers never leak or misreport ------------------------------------------------------------------

def test_repr_and_str_never_contain_key_material(tmp_path):
    seed = bytes(range(7, 39))
    for p in (FileKeyProvider(write_key(tmp_path / "k", seed)),
              EnvKeyProvider("K", environ={"K": base64.b64encode(seed).decode()})):
        for text in (repr(p), str(p)):
            assert seed.hex() not in text and base64.b64encode(seed).decode() not in text
            assert p.key_id[:16] in text and p.custody in text


def test_self_test_is_false_when_the_key_cannot_sign():
    class Broken(EnvKeyProvider):
        def sign(self, message: bytes) -> bytes:
            raise RuntimeError("hsm unplugged")

    class Wrong(EnvKeyProvider):
        def sign(self, message: bytes) -> bytes:
            return bytes(64)

    env = {"K": base64.b64encode(bytes(32)).decode()}
    assert EnvKeyProvider("K", environ=env).self_test() is True
    assert Broken("K", environ=env).self_test() is False
    assert Wrong("K", environ=env).self_test() is False


def test_the_hsm_provider_is_a_declared_interface_not_an_implementation():
    assert Pkcs11KeyProvider.custody == "hsm"
    with pytest.raises(NotImplementedError, match="declared"):
        Pkcs11KeyProvider()


# --- keygen is explicit-only ----------------------------------------------------------------------------

def test_generate_keypair_writes_a_fresh_owner_only_seed_and_returns_its_key_id(tmp_path):
    path = tmp_path / "ledger.key"
    kid = generate_keypair(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_size == 32
    assert FileKeyProvider(path).key_id == kid


def test_generate_keypair_never_overwrites(tmp_path):
    path = tmp_path / "ledger.key"
    generate_keypair(path)
    first = path.read_bytes()
    with pytest.raises(KeyExists):
        generate_keypair(path)
    assert path.read_bytes() == first


def test_generate_keypair_makes_a_different_key_each_time(tmp_path):
    assert generate_keypair(tmp_path / "a") != generate_keypair(tmp_path / "b")


def test_generate_keypair_needs_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_keypair(tmp_path / "no" / "such" / "dir" / "k")


def test_key_generation_exists_in_exactly_one_place_in_the_seal_package():
    """No implicit keygen, enforced structurally: any `.generate(` call in seal/ must sit inside
    keys.generate_keypair. A future `Sealer` that quietly generates a key on first run fails this."""
    def is_generate(n: ast.AST) -> bool:
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"

    offenders = []
    for f in sorted(SEAL.rglob("*.py")):
        tree = ast.parse(f.read_text())
        allowed: set[int] = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "keys.py" \
                    and fn.name == "generate_keypair":
                allowed |= {id(n) for n in ast.walk(fn)}
        offenders += [f"{f.name}:{n.lineno}" for n in ast.walk(tree) if is_generate(n) and id(n) not in allowed]
    assert offenders == []


def test_no_provider_creates_a_file_or_a_key_when_none_is_supplied(tmp_path):
    before = os.listdir(tmp_path)
    for attempt in (lambda: FileKeyProvider(tmp_path / "nope"),
                    lambda: EnvKeyProvider("K", environ={})):
        with pytest.raises(KeyNotConfigured):
            attempt()
    assert os.listdir(tmp_path) == before


# --- trust root ------------------------------------------------------------------------------------------

def tk(seed: bytes, role: str = "ledger") -> dict:
    pub = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {"key_id": key_id(pub), "public_key": pub.hex(), "role": role}


def trust(**over) -> bytes:
    obj = {"v": "cva-seal/1", "deployment_manifest_hash": "a" * 64,
           "keys": [tk(bytes(range(32))), tk(bytes(range(1, 33)), "witness"), tk(bytes(range(2, 34)), "boundary")]}
    obj.update(over)
    return json.dumps(obj, indent=2).encode()


def test_a_valid_trust_root_loads_and_answers_lookups(tmp_path):
    path = tmp_path / "trust_root.json"
    path.write_bytes(trust())
    tr = load_trust_root(path)
    assert isinstance(tr, TrustRoot) and len(tr.keys) == 3
    ledger = tr.by_role("ledger")[0]
    assert tr.public_key(ledger.key_id) == ledger.public_key and tr.public_key("f" * 64) is None
    assert tr.ledger_keys() == {ledger.key_id: ledger.public_key}
    assert [k.role for k in tr.keys] == ["ledger", "witness", "boundary"]


def test_a_trust_root_round_trips_through_its_canonical_bytes():
    tr = parse_trust_root(trust())
    assert parse_trust_root(tr.to_bytes()) == tr
    assert tr.to_bytes() == TrustRoot(tr.deployment_manifest_hash, tr.keys).to_bytes()


def test_a_missing_trust_root_is_an_error(tmp_path):
    with pytest.raises(TrustRootError, match="not found"):
        load_trust_root(tmp_path / "absent.json")


@pytest.mark.parametrize("mutate,needle", [
    (lambda o: o.update(v="cva-seal/2"), "version"),
    (lambda o: o.update(deployment_manifest_hash="A" * 64), "64 lowercase hex"),
    (lambda o: o.update(deployment_manifest_hash="a" * 63), "64 lowercase hex"),
    (lambda o: o.update(keys=[]), "non-empty"),
    (lambda o: o.update(keys="none"), "non-empty"),
    (lambda o: o.update(extra=1), "exactly"),
    (lambda o: o["keys"][0].update(role="admin"), "role"),
    (lambda o: o["keys"][0].update(key_id="1" * 64), "not SHA-256"),
    (lambda o: o["keys"][0].update(public_key=o["keys"][0]["public_key"].upper()), "lowercase hex"),
    (lambda o: o["keys"][0].update(extra="x"), "exactly"),
    (lambda o: o["keys"][1].update(**o["keys"][0]), "duplicate"),
    (lambda o: o.update(keys=[tk(bytes(range(1, 33)), "witness")]), "no ledger key"),
])
def test_a_malformed_trust_root_is_rejected(mutate, needle):
    obj = json.loads(trust())
    mutate(obj)
    with pytest.raises(TrustRootError, match=needle):
        parse_trust_root(json.dumps(obj).encode())


@pytest.mark.parametrize("data", [b"", b"not json", b"[]", b'{"v":"cva-seal/1","v":"cva-seal/1"}', b'{"a":1.5}'])
def test_garbage_trust_roots_are_rejected(data):
    with pytest.raises(TrustRootError):
        parse_trust_root(data)


def test_trust_root_files_may_be_pretty_printed_but_keep_every_other_rule():
    assert parse_trust_root(trust()).keys                       # indent=2 whitespace is fine
    dup_key = trust().replace(b'"role": "ledger"', b'"role": "ledger",\n      "role": "witness"', 1)
    assert dup_key != trust()
    with pytest.raises(TrustRootError, match="duplicate key"):
        parse_trust_root(dup_key)                               # a parser-differential trick still fails


def test_trust_key_is_immutable():
    k = TrustKey("a" * 64, bytes(32), "ledger")
    with pytest.raises(AttributeError):
        k.role = "witness"      # type: ignore[misc]
