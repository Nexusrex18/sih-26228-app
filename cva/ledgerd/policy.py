"""The per-uid record-type allowlist (plan §5.2). A pure function over a policy table.

Testability is the reason it is shaped this way. On a single-uid development box every
process runs as the same user, so a test that drives a real socket proves nothing about the
allowlist — it passes because the uids happen to match. So the decision is a function of
(uid, record type) with the table injected, tested exhaustively with synthetic uids, and
separately exercised once over a real socket to prove the wiring.

Neither uid may write `genesis`, `key_rotation`, `checkpoint` or `anchor_event`: those are
Crypto's own paths, and `SealedLedger.append()` already refuses them. Two independent
refusals for the same act is the intended design, not a duplicate.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

#: What a peer may append, by record type. Empty set = may read status only.
ANALYST_EVENT = "analyst_event"
SCAN_RECORD = "scan_record"

#: Never writable through this daemon, whoever asks.
FORBIDDEN_TYPES = frozenset({"genesis", "key_rotation", "checkpoint", "anchor_event",
                             "degraded_marker", "model_registration", "inference"})

#: `status` answers "is a signing key loaded, and how big is the ledger". Every known peer
#: may ask, because `LedgerdAuditLedger.capabilities()` is an ACTIVE probe and a scanner that
#: cannot ask would have to assume — which is the thing the capability model exists to stop.
_STATUS_OP = "status"
#: `records` is the decision stream. Sealing your own scan record is not a licence to read
#: every override an analyst has ever made, so this is a separate grant.
_RECORDS_OP = "records"
_APPEND_OP_TYPE = {"append_analyst_event": ANALYST_EVENT,
                   "append_scan_record": SCAN_RECORD}


@dataclass(frozen=True)
class Policy:
    """uid -> the record types that uid may append.

    `readers` may call `status` and `records`; an appender is implicitly a reader. An
    empty `writers` and empty `readers` means: this daemon serves nobody, which is a
    misconfiguration worth failing loudly on rather than defaulting open.
    """

    writers: Mapping[int, frozenset[str]] = field(default_factory=dict)
    readers: frozenset[int] = frozenset()
    #: Convenience for single-uid development and the container, where web, scanner and
    #: daemon may legitimately share a uid. It NEVER widens the record-type allowlist —
    #: only who may reach it — and the daemon logs that it is in effect.
    same_uid_ok: bool = False
    owner_uid: int = field(default_factory=os.getuid)

    def known(self, uid: int) -> bool:
        """A peer this daemon serves at all. May ask `status`."""
        return uid in self.readers or uid in self.writers or (
            self.same_uid_ok and uid == self.owner_uid)

    def may_read(self, uid: int) -> bool:
        """May read the decision stream. A writer is NOT implicitly a reader."""
        return uid in self.readers or (self.same_uid_ok and uid == self.owner_uid)

    def may_append(self, uid: int, record_type: str) -> bool:
        if record_type in FORBIDDEN_TYPES:
            return False
        allowed = set(self.writers.get(uid, frozenset()))
        if self.same_uid_ok and uid == self.owner_uid:
            allowed |= {ANALYST_EVENT, SCAN_RECORD}
        return record_type in allowed


def decide(policy: Policy, uid: int, op: str) -> tuple[bool, str]:
    """`(allowed, detail)`. The detail is what the peer is told, and it names the uid."""
    if op == _STATUS_OP:
        if policy.known(uid):
            return True, ""
        return False, (f"uid {uid} is not a peer this daemon serves. cva-ledgerd answers a "
                       "named set of local processes, not whoever can open the socket.")
    if op == _RECORDS_OP:
        if policy.may_read(uid):
            return True, ""
        return False, (f"uid {uid} may not read the decision stream. Appending a record and "
                       "reading every other process's records are separate grants.")
    rtype = _APPEND_OP_TYPE.get(op)
    if rtype is None:
        return False, f"unknown op {op!r}"
    if policy.may_append(uid, rtype):
        return True, ""
    if rtype in FORBIDDEN_TYPES:
        return False, (f"{rtype} is never writable through cva-ledgerd: genesis, "
                       "checkpoints, anchors and rotations are the seal's own paths.")
    return False, (f"uid {uid} may not append {rtype}. The allowlist is per uid AND per "
                   "record type: the web process may record analyst decisions and nothing "
                   "else; the scanner may seal its own scan record and nothing else.")


def from_mapping(data: Mapping[str, object]) -> Policy:
    """Build a policy from config. Unknown record types are a load error."""
    writers: dict[int, frozenset[str]] = {}
    raw_writers = data.get("writers") or {}
    if not isinstance(raw_writers, Mapping):
        raise ValueError("policy.writers must be a mapping of uid -> [record types]")
    for uid, types in raw_writers.items():
        try:
            uid_i = int(uid)
        except (TypeError, ValueError):
            raise ValueError(f"policy.writers: {uid!r} is not a uid") from None
        if isinstance(types, str):
            types = [types]
        if not isinstance(types, (list, tuple, set, frozenset)):
            raise ValueError(f"policy.writers[{uid_i}] must be a list of record types")
        bad = sorted(set(map(str, types)) - {ANALYST_EVENT, SCAN_RECORD})
        if bad:
            raise ValueError(
                f"policy.writers[{uid_i}]: {bad} cannot be granted here. Only "
                f"{[ANALYST_EVENT, SCAN_RECORD]} are writable through cva-ledgerd.")
        writers[uid_i] = frozenset(map(str, types))
    raw_readers = data.get("readers") or []
    if isinstance(raw_readers, (int, str)):
        raw_readers = [raw_readers]
    readers = frozenset(int(u) for u in raw_readers)
    return Policy(writers=writers, readers=readers,
                  same_uid_ok=bool(data.get("same_uid_ok", False)))


def development_policy() -> Policy:
    """Everything runs as one uid: the shape of the real policy, with the owner allowed.

    Used by the container's single-user entrypoint and by tests. It is a real Policy object,
    so the code path under test is the production one.
    """
    return Policy(same_uid_ok=True)


__all__ = ["ANALYST_EVENT", "FORBIDDEN_TYPES", "Policy", "SCAN_RECORD", "decide",
           "development_policy", "from_mapping"]
