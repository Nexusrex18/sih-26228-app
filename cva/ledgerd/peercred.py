"""`SO_PEERCRED`: who is on the other end of the socket, as the KERNEL reports it.

The credentials come from the kernel, not from the connection's own claim, which is the
entire reason this boundary exists (plan D-E3). A web process that has been compromised can
say anything in a JSON frame; it cannot say it is running as a different uid.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

#: Linux: `struct ucred { pid_t pid; uid_t uid; gid_t gid; }`, three native ints.
_UCRED = struct.Struct("3i")
SO_PEERCRED = getattr(socket, "SO_PEERCRED", 17)


class PeerCredUnavailable(Exception):
    """Fail CLOSED. A connection whose peer we cannot identify is refused, never trusted."""


@dataclass(frozen=True)
class PeerCred:
    pid: int
    uid: int
    gid: int


def peer_credentials(sock: socket.socket) -> PeerCred:
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, SO_PEERCRED, _UCRED.size)
    except OSError as e:
        raise PeerCredUnavailable(
            f"SO_PEERCRED is not available on this socket ({e}). The uid of the peer cannot "
            "be established, so the per-uid record-type allowlist cannot be applied and the "
            "connection is refused.") from None
    if len(raw) < _UCRED.size:
        raise PeerCredUnavailable("SO_PEERCRED returned a short structure")
    pid, uid, gid = _UCRED.unpack(raw[:_UCRED.size])
    return PeerCred(pid=pid, uid=uid, gid=gid)


__all__ = ["PeerCred", "PeerCredUnavailable", "peer_credentials"]
