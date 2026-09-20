"""Local accounts: scrypt hashing, roles, lockout (plan D-E9, S5, S8).

No IdP exists on an air gap, so accounts are local and that is a declared limitation (§13),
not an oversight. `admin` manages accounts and **cannot approve**: one person controlling
both identity and decisions is the failure this separation exists to prevent.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .workflow.events import ACTOR_ID_RE

ROLES = ("viewer", "analyst", "approver", "admin")

# OWASP's stated scrypt minimum (N = 2^17, r = 8, p = 1). Argon2id is OWASP's preference;
# scrypt is chosen so no extra dependency enters the bundle (D-E1), and the trade-off is
# recorded here rather than left to be rediscovered.
#
# `maxmem` is the part that bites: OpenSSL defaults to 32 MiB and this parameter set needs
# 128 * N * r = 128 MiB, so the call raises `ValueError: memory limit exceeded` without it.
# S5 requires a test that asserts the call does NOT raise.
SCRYPT_N = 2 ** 17
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 2 ** 28
SCRYPT_DKLEN = 64
SALT_BYTES = 16

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS accounts (
  actor_id TEXT PRIMARY KEY, role TEXT NOT NULL, salt BLOB NOT NULL, hash BLOB NOT NULL,
  created_at REAL NOT NULL, disabled INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0,
  session_epoch INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS ip_failures (
  ip TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0,
  locked_until REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS secrets (k TEXT PRIMARY KEY, v BLOB NOT NULL);
"""


class AccountError(Exception):
    pass


@dataclass(frozen=True)
class Account:
    actor_id: str
    role: str
    disabled: bool = False
    #: Bumped on logout and on any role change, which revokes every live session for this
    #: user server-side (S6). A cookie alone must never outlive a demotion.
    session_epoch: int = 0

    def has_role(self, *roles: str) -> bool:
        return self.role in roles


def derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                          p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN)


def validate_actor_id(actor_id: str) -> str:
    """Enforced HERE, at creation, not at the first analyst event.

    Module C's `_v_analyst` validates `actor_id` as `[A-Za-z0-9._:-]{1,64}`. A user created
    as `a sharma` would authenticate fine and then have their first override refused by the
    ledger — at the worst possible moment, with no obvious cause.
    """
    if not isinstance(actor_id, str) or not ACTOR_ID_RE.fullmatch(actor_id):
        raise AccountError(
            f"{actor_id!r} is not a usable actor id. The ledger records it verbatim and "
            "accepts [A-Za-z0-9._:-], 1 to 64 characters — no spaces and no accents. An "
            "account that cannot be written into a record is an account that cannot make a "
            "decision.")
    return actor_id


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise AccountError(f"role must be one of {list(ROLES)}, got {role!r}")
    return role


class AccountStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- the session-signing secret -------------------------------------------------------
    def secret_key(self) -> bytes:
        """Persisted, so a restart does not silently invalidate every cookie — or, worse,
        so a regenerated key is not mistaken for a logout bug and worked around."""
        row = self._conn.execute("SELECT v FROM secrets WHERE k='session'").fetchone()
        if row is not None:
            return bytes(row["v"])
        key = secrets.token_bytes(32)
        self._conn.execute("INSERT INTO secrets(k,v) VALUES('session',?)", (key,))
        self._conn.commit()
        return key

    # -- accounts ---------------------------------------------------------------------------
    def create(self, actor_id: str, password: str, role: str) -> Account:
        actor_id = validate_actor_id(actor_id)
        role = validate_role(role)
        if len(password) < 12:
            raise AccountError("passwords must be at least 12 characters")
        salt = os.urandom(SALT_BYTES)
        try:
            self._conn.execute(
                "INSERT INTO accounts(actor_id, role, salt, hash, created_at) "
                "VALUES (?,?,?,?,?)",
                (actor_id, role, salt, derive(password, salt), time.time()))
        except sqlite3.IntegrityError:
            raise AccountError(f"account {actor_id!r} already exists") from None
        self._conn.commit()
        return Account(actor_id=actor_id, role=role)

    def get(self, actor_id: str) -> Account | None:
        row = self._conn.execute(
            "SELECT actor_id, role, disabled, session_epoch FROM accounts WHERE actor_id=?",
            (actor_id,)).fetchone()
        if row is None:
            return None
        return Account(row["actor_id"], row["role"], bool(row["disabled"]),
                       int(row["session_epoch"]))

    def list_accounts(self) -> list[Account]:
        rows = self._conn.execute(
            "SELECT actor_id, role, disabled, session_epoch FROM accounts "
            "ORDER BY actor_id").fetchall()
        return [Account(r["actor_id"], r["role"], bool(r["disabled"]),
                        int(r["session_epoch"])) for r in rows]

    def set_role(self, actor_id: str, role: str) -> Account:
        """A role change revokes the user's live sessions (S6): a demoted approver must not
        keep approving on a cookie issued a minute ago."""
        role = validate_role(role)
        cur = self._conn.execute(
            "UPDATE accounts SET role=?, session_epoch=session_epoch+1 WHERE actor_id=?",
            (role, actor_id))
        if cur.rowcount == 0:
            raise AccountError(f"no such account {actor_id!r}")
        self._conn.commit()
        acct = self.get(actor_id)
        assert acct is not None
        return acct

    def set_disabled(self, actor_id: str, disabled: bool) -> None:
        self._conn.execute(
            "UPDATE accounts SET disabled=?, session_epoch=session_epoch+1 WHERE actor_id=?",
            (1 if disabled else 0, actor_id))
        self._conn.commit()

    def set_password(self, actor_id: str, password: str) -> None:
        if len(password) < 12:
            raise AccountError("passwords must be at least 12 characters")
        salt = os.urandom(SALT_BYTES)
        self._conn.execute(
            "UPDATE accounts SET salt=?, hash=?, session_epoch=session_epoch+1 "
            "WHERE actor_id=?", (salt, derive(password, salt), actor_id))
        self._conn.commit()

    def revoke_sessions(self, actor_id: str) -> None:
        self._conn.execute(
            "UPDATE accounts SET session_epoch=session_epoch+1 WHERE actor_id=?", (actor_id,))
        self._conn.commit()

    # -- authentication and lockout (S8) -------------------------------------------------
    def locked_for(self, actor_id: str, ip: str, *, now: float | None = None) -> float:
        """Seconds remaining on the longer of the per-account and per-IP lockouts."""
        now = time.time() if now is None else now
        a = self._conn.execute("SELECT locked_until FROM accounts WHERE actor_id=?",
                               (actor_id,)).fetchone()
        i = self._conn.execute("SELECT locked_until FROM ip_failures WHERE ip=?",
                               (ip,)).fetchone()
        until = max(float(a["locked_until"]) if a else 0.0,
                    float(i["locked_until"]) if i else 0.0)
        return max(0.0, until - now)

    def authenticate(self, actor_id: str, password: str, ip: str = "",
                     *, threshold: int = 5, base_s: int = 2,
                     max_s: int = 3600, now: float | None = None) -> Account:
        now = time.time() if now is None else now
        remaining = self.locked_for(actor_id, ip, now=now)
        if remaining > 0:
            raise AccountError(f"locked for another {int(remaining)} s")

        row = self._conn.execute(
            "SELECT * FROM accounts WHERE actor_id=?", (actor_id,)).fetchone()
        if row is None:
            # Still do the work, so a missing account and a wrong password take the same
            # time. Then record the IP failure: enumerating usernames is reconnaissance.
            derive(password, b"\x00" * SALT_BYTES)
            self._record_ip_failure(ip, now, threshold, base_s, max_s)
            raise AccountError("no such account, or the password is wrong")

        ok = hmac.compare_digest(derive(password, bytes(row["salt"])), bytes(row["hash"]))
        if not ok:
            self._record_failure(actor_id, int(row["failures"]), now, threshold, base_s, max_s)
            self._record_ip_failure(ip, now, threshold, base_s, max_s)
            raise AccountError("no such account, or the password is wrong")
        if row["disabled"]:
            raise AccountError("this account is disabled")

        self._conn.execute(
            "UPDATE accounts SET failures=0, locked_until=0 WHERE actor_id=?", (actor_id,))
        self._conn.execute("DELETE FROM ip_failures WHERE ip=?", (ip,))
        self._conn.commit()
        return Account(row["actor_id"], row["role"], False, int(row["session_epoch"]))

    def _backoff(self, failures: int, threshold: int, base_s: int, max_s: int) -> float:
        if failures < threshold:
            return 0.0
        return float(min(max_s, base_s * (2 ** (failures - threshold))))

    def _record_failure(self, actor_id: str, failures: int, now: float, threshold: int,
                        base_s: int, max_s: int) -> None:
        failures += 1
        self._conn.execute(
            "UPDATE accounts SET failures=?, locked_until=? WHERE actor_id=?",
            (failures, now + self._backoff(failures, threshold, base_s, max_s), actor_id))
        self._conn.commit()

    def _record_ip_failure(self, ip: str, now: float, threshold: int, base_s: int,
                           max_s: int) -> None:
        if not ip:
            return
        row = self._conn.execute("SELECT failures FROM ip_failures WHERE ip=?",
                                 (ip,)).fetchone()
        failures = (int(row["failures"]) if row else 0) + 1
        self._conn.execute(
            "INSERT INTO ip_failures(ip, failures, locked_until) VALUES (?,?,?) "
            "ON CONFLICT(ip) DO UPDATE SET failures=excluded.failures, "
            "locked_until=excluded.locked_until",
            (ip, failures, now + self._backoff(failures, threshold, base_s, max_s)))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


__all__ = ["ROLES", "SCRYPT_MAXMEM", "SCRYPT_N", "SCRYPT_P", "SCRYPT_R", "Account",
           "AccountError", "AccountStore", "derive", "validate_actor_id", "validate_role"]
