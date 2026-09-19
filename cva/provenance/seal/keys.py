"""Signing keys and the trust root (plan §5.9, §5.10, §7.5; decision C-16).

Rules that shape this module:

  * A key is NEVER generated implicitly. `generate_keypair` is the only place a key is created, and it
    is called only by the explicit `cva-seal keygen` command. `Sealer.open()` with no key raises
    `KeyNotConfigured` — there is no fallback and no "development mode".
  * A key file readable by group/other is refused (`KeyPermissionError`) unless the operator passes
    `allow_loose_permissions=True`, and that choice is logged.
  * The demo's custody is a FILE with documented permissions, and the report says so. The real
    deployment answer is an HSM or smartcard: `Pkcs11KeyProvider` is the declared interface, not an
    implementation.
  * Secrets never appear in `repr()`/`str()`. Python cannot zeroise memory, and this module does not
    pretend to: custody hygiene is the deployment's job, stated in the coverage statement.
"""
from __future__ import annotations

import base64
import binascii
import logging
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .canonical import canonical_bytes, parse_strict
from .constants import VERSION
from .errors import (
    InvalidKeyMaterial,
    KeyExists,
    KeyNotConfigured,
    KeyPermissionError,
    NonCanonical,
    TrustRootError,
)
from .records import key_id_of

log = logging.getLogger("cva.provenance.seal")

SEED_LEN = 32
SIGNATURE_LEN = 64
TRUST_ROLES = ("ledger", "witness", "boundary")
_SELF_TEST_MESSAGE = b"cva-seal/1 key self-test"


class KeyProvider(Protocol):
    """What the Sealer needs from a key. `sign` is PURE Ed25519 (PureEdDSA, RFC 8032) over exactly the
    bytes it is given — domain tags are the caller's job (chain.sign_record), never the provider's."""

    custody: str                      # "file" | "env" | "hsm" — printed in every verify report

    @property
    def key_id(self) -> str: ...
    @property
    def public_key(self) -> bytes: ...
    def sign(self, message: bytes) -> bytes: ...


def verify_ed25519(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Pure Ed25519 verification. False (never an exception) for any malformed or invalid input.
    RFC 8032 requires rejecting a non-canonical S (S >= L); the backing library does, and the tests pin it."""
    if len(public_key) != SEED_LEN or len(signature) != SIGNATURE_LEN:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


class _Ed25519Provider:
    custody = "unknown"

    def __init__(self, key: Ed25519PrivateKey, custody: str) -> None:
        self._key = key
        self.custody = custody
        self._public = key.public_key().public_bytes(serialization.Encoding.Raw,
                                                     serialization.PublicFormat.Raw)
        self._key_id = key_id_of(self._public.hex())

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key(self) -> bytes:
        return self._public

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    def self_test(self) -> bool:
        """True iff the key produces a signature that verifies. `SIGNING_KEY` is reported only when this
        holds — a key that loaded but cannot sign must not present itself as a capability (§3.1)."""
        try:
            return verify_ed25519(self._public, _SELF_TEST_MESSAGE, self.sign(_SELF_TEST_MESSAGE))
        except Exception:                                    # noqa: BLE001 - any failure means "cannot sign"
            return False

    def __repr__(self) -> str:                               # never leak key material
        return f"<{type(self).__name__} custody={self.custody} key_id={self._key_id[:16]}…>"

    __str__ = __repr__


def _load_private(material: bytes, source: str) -> Ed25519PrivateKey:
    """Accept exactly two shapes: a raw 32-byte seed, or an unencrypted PKCS#8 PEM Ed25519 key."""
    if material.lstrip().startswith(b"-----BEGIN"):
        try:
            key = serialization.load_pem_private_key(material, password=None)
        except TypeError:
            raise InvalidKeyMaterial(f"{source}: PEM key is encrypted; supply an unencrypted key or use an HSM") from None
        except ValueError as e:
            raise InvalidKeyMaterial(f"{source}: not a valid PEM private key ({e})") from None
        if not isinstance(key, Ed25519PrivateKey):
            raise InvalidKeyMaterial(f"{source}: PEM key is {type(key).__name__}, expected Ed25519")
        return key
    if len(material) == SEED_LEN:
        return Ed25519PrivateKey.from_private_bytes(material)
    raise InvalidKeyMaterial(f"{source}: expected a raw {SEED_LEN}-byte seed or an unencrypted PKCS#8 "
                             f"PEM, got {len(material)} bytes")


class FileKeyProvider(_Ed25519Provider):
    """Key from a file: a raw 32-byte seed or an unencrypted PKCS#8 PEM. Refuses a key file that group or
    other can access (mode & 0o077) unless `allow_loose_permissions=True`, which is logged."""

    def __init__(self, path: str | os.PathLike[str], *, allow_loose_permissions: bool = False) -> None:
        p = os.fspath(path)
        try:
            fd = os.open(p, os.O_RDONLY)
        except FileNotFoundError:
            raise KeyNotConfigured(f"no signing key at {p}; create one explicitly with `cva-seal keygen`") from None
        except OSError as e:
            raise KeyNotConfigured(f"cannot open signing key {p}: {e.strerror}") from None
        try:
            st = os.fstat(fd)                                # same fd for check and read: no TOCTOU
            if not stat.S_ISREG(st.st_mode):
                raise InvalidKeyMaterial(f"{p}: not a regular file")
            loose = stat.S_IMODE(st.st_mode) & 0o077
            if loose:
                msg = f"{p}: key file mode {stat.S_IMODE(st.st_mode):04o} is accessible to group/other"
                if not allow_loose_permissions:
                    raise KeyPermissionError(msg + " (chmod 600, or pass allow_loose_permissions=True)")
                log.warning("%s — proceeding because allow_loose_permissions=True", msg)
            material = os.read(fd, 4096 + 1)
        finally:
            os.close(fd)
        if len(material) > 4096:
            raise InvalidKeyMaterial(f"{p}: file is too large to be a key")
        super().__init__(_load_private(material, p), "file")


class EnvKeyProvider(_Ed25519Provider):
    """Key from an environment variable holding the base64 (standard alphabet, no whitespace) of a
    32-byte seed. An environment variable cannot be cleared from the process's own memory or from
    /proc — the limitation is documented, and a file or HSM is preferred."""

    def __init__(self, var: str, *, environ: Mapping[str, str] | None = None) -> None:
        value = (os.environ if environ is None else environ).get(var)
        if not value:
            raise KeyNotConfigured(f"environment variable {var} is not set; create a key explicitly "
                                   "with `cva-seal keygen`")
        try:
            seed = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise InvalidKeyMaterial(f"{var}: not valid base64") from None
        if len(seed) != SEED_LEN:
            raise InvalidKeyMaterial(f"{var}: expected a {SEED_LEN}-byte seed, got {len(seed)} bytes")
        super().__init__(Ed25519PrivateKey.from_private_bytes(seed), "env")


class Pkcs11KeyProvider:
    """DECLARED, NOT IMPLEMENTED. An HSM or smartcard is the real deployment answer (a key that cannot be
    copied cannot be exfiltrated by whoever compromises the host), but binding PKCS#11 needs a vendor
    module and hardware we do not have. Coverage statement: "custody is a file in the demo; an HSM
    provider is a specified interface, not built"."""

    custody = "hsm"

    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError("Pkcs11KeyProvider is a declared interface only (plan §7.5)")


def generate_keypair(out_path: str | os.PathLike[str]) -> str:
    """Create a NEW raw-seed key file (mode 0600, exclusive create) and return its key_id.

    The ONLY place in the SDK where a key is created. Called by `cva-seal keygen` and by tests —
    never by `Sealer.open()`, never as a fallback. Refuses to overwrite: a silently replaced key is a
    lost trust anchor. The parent directory must already exist.
    """
    p = os.fspath(out_path)
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                             serialization.NoEncryption())
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise KeyExists(f"{p} already exists; refusing to overwrite a signing key") from None
    try:
        os.write(fd, seed)
        os.fsync(fd)
    finally:
        os.close(fd)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return key_id_of(public.hex())


# --- trust root (plan §5.9) -----------------------------------------------------------------------

@dataclass(frozen=True)
class TrustKey:
    key_id: str
    public_key: bytes
    role: str


@dataclass(frozen=True)
class TrustRoot:
    """The set of public keys a verifier trusts, shipped in the bundle. There are no live CAs and no
    network calls, ever. It is NOT self-signed (nothing exists to sign it): its integrity is
    out-of-band and declared in the coverage statement."""

    deployment_manifest_hash: str
    keys: tuple[TrustKey, ...]

    def public_key(self, key_id: str) -> bytes | None:
        return next((k.public_key for k in self.keys if k.key_id == key_id), None)

    def by_role(self, role: str) -> tuple[TrustKey, ...]:
        return tuple(k for k in self.keys if k.role == role)

    def ledger_keys(self) -> dict[str, bytes]:
        return {k.key_id: k.public_key for k in self.by_role("ledger")}

    def to_bytes(self) -> bytes:
        """Canonical bytes, for `cva-seal init` to write and for its digest to be listed in the
        vendoring manifest."""
        return canonical_bytes({
            "v": VERSION, "deployment_manifest_hash": self.deployment_manifest_hash,
            "keys": [{"key_id": k.key_id, "public_key": k.public_key.hex(), "role": k.role} for k in self.keys]})


def parse_trust_root(data: bytes) -> TrustRoot:
    """Strict: no duplicate keys, no floats, exact key sets, lowercase hex, `key_id` must equal
    SHA-256(public_key), no duplicate key ids, at least one ledger key."""
    try:
        obj = parse_strict(data, require_canonical=False)
    except NonCanonical as e:
        raise TrustRootError(f"trust root is not valid: {e.detail}") from None
    if set(obj) != {"v", "deployment_manifest_hash", "keys"}:
        raise TrustRootError(f"trust root keys must be exactly v, deployment_manifest_hash, keys; got {sorted(obj)}")
    if obj["v"] != VERSION:
        raise TrustRootError(f"unsupported trust root version {obj['v']!r}")
    dmh = obj["deployment_manifest_hash"]
    if not _is_hex(dmh, 64):
        raise TrustRootError("deployment_manifest_hash must be 64 lowercase hex characters")
    raw_keys = obj["keys"]
    if not isinstance(raw_keys, list) or not raw_keys:
        raise TrustRootError("keys must be a non-empty array")
    keys: list[TrustKey] = []
    for i, k in enumerate(raw_keys):
        if not isinstance(k, dict) or set(k) != {"key_id", "public_key", "role"}:
            raise TrustRootError(f"keys[{i}] must have exactly key_id, public_key, role")
        if not _is_hex(k["key_id"], 64) or not _is_hex(k["public_key"], 64):
            raise TrustRootError(f"keys[{i}]: key_id and public_key must be 64 lowercase hex characters")
        if k["role"] not in TRUST_ROLES:
            raise TrustRootError(f"keys[{i}]: role must be one of {list(TRUST_ROLES)}")
        if key_id_of(k["public_key"]) != k["key_id"]:
            raise TrustRootError(f"keys[{i}]: key_id is not SHA-256 of public_key")
        keys.append(TrustKey(k["key_id"], bytes.fromhex(k["public_key"]), k["role"]))
    if len({k.key_id for k in keys}) != len(keys):
        raise TrustRootError("duplicate key_id in trust root")
    if not any(k.role == "ledger" for k in keys):
        raise TrustRootError("trust root has no ledger key")
    return TrustRoot(dmh, tuple(keys))


def load_trust_root(path: str | os.PathLike[str]) -> TrustRoot:
    try:
        return parse_trust_root(Path(path).read_bytes())
    except FileNotFoundError:
        raise TrustRootError(f"trust root not found: {path}") from None


def _is_hex(v: object, n: int) -> bool:
    return isinstance(v, str) and len(v) == n and all(c in "0123456789abcdef" for c in v)
