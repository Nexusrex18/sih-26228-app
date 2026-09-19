"""The process-level egress guard — backend_plan.md §5.8, guard one of two.

`pytest-socket` is a pytest plugin whose `--disable-socket` is a command-line flag, and
`cva selftest` is a product entry point, not a pytest run. A guard that exists only as a
flag is armed in CI and absent in the shipped binary, which is the deployment that
matters. So the library is called programmatically, against the API of the pinned 0.8.1
source (read, not recalled):

  * `socket_allow_hosts(allowed, allow_unix_socket=...)` replaces `socket.socket.connect`
    with a guard that raises `SocketConnectBlockedError` for any host outside the list.
  * `disable_socket()` is NOT used. It swaps `socket.socket` for a class whose `__new__`
    raises for every family, loopback included, so "loopback still permitted" would be false.
  * `socket_allow_hosts` does not touch name resolution, and `socket.create_connection`
    resolves BEFORE it connects, so a DNS query for a foreign name would leave the process
    while the connect guard sat idle. The resolver entry points are wrapped here to refuse
    anything that is not a loopback name or an IP literal.

This guard answers WHERE a call was (a traceback at the offending line). It cannot see a
subprocess, a C extension or a vendored binary; only `unshare -rn` proves there was no
route at all. Both ship, and neither is a substitute for the other.
"""
from __future__ import annotations

import ipaddress
import socket
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pytest_socket import SocketBlockedError, SocketConnectBlockedError, socket_allow_hosts

LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.0/8", "::1", "localhost")
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})

#: TEST-NET-1 (RFC 5737): reserved for documentation, never routed, so a canary aimed at
#: it cannot reach anything even if the guard were somehow absent.
CANARY_ADDR = ("192.0.2.1", 9)
CANARY_NAME = "egress-canary.invalid"      # `.invalid` is reserved (RFC 2606)


def _resolvable(host: Any) -> bool:
    """A name may be resolved only if resolving it cannot generate a DNS query."""
    if host is None or host == "":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if str(host).lower() in _LOOPBACK_NAMES:
        return True
    try:
        ipaddress.ip_address(str(host).split("%", 1)[0])
        return True
    except ValueError:
        return False


def _wrap(real: Callable[..., Any], what: str) -> Callable[..., Any]:
    def guarded(host: Any = None, *args: Any, **kwargs: Any) -> Any:
        if not _resolvable(host):
            raise SocketBlockedError(f"A name lookup left the process guard: {what}({host!r}).")
        return real(host, *args, **kwargs)
    return guarded


@contextmanager
def egress_guard() -> Iterator[None]:
    """Arm the process-level guard for the duration of the block, then restore exactly what
    was there before (so nesting works, and a pytest run that armed its own is left alone)."""
    saved_connect = socket.socket.connect
    saved_gai = socket.getaddrinfo
    saved_ghbn = socket.gethostbyname
    saved_ghbn_ex = socket.gethostbyname_ex
    # Allow-list entries are IPs and CIDRs plus "localhost". pytest-socket resolves any
    # non-IP entry when the guard is installed, so arm the connect guard BEFORE wrapping the
    # resolvers — "localhost" resolves through /etc/hosts, no query.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        socket_allow_hosts(list(LOOPBACK_HOSTS), allow_unix_socket=True)
    socket.getaddrinfo = _wrap(saved_gai, "getaddrinfo")
    socket.gethostbyname = _wrap(saved_ghbn, "gethostbyname")
    socket.gethostbyname_ex = _wrap(saved_ghbn_ex, "gethostbyname_ex")
    try:
        yield
    finally:
        socket.socket.connect = saved_connect          # type: ignore[method-assign]
        socket.getaddrinfo = saved_gai
        socket.gethostbyname = saved_ghbn
        socket.gethostbyname_ex = saved_ghbn_ex


@dataclass(frozen=True)
class Canary:
    name: str
    ok: bool
    detail: str


def run_canaries() -> list[Canary]:
    """V13: a guard that is armed while nothing egresses passes trivially, so this makes one
    deliberate outbound attempt of each kind and requires the guard's own error.

    The loopback probe asserts only that the GUARD did not fire. Inside `unshare -rn` the
    loopback interface is down, so the OS refuses the connection; that is the namespace
    doing its job, not the guard refusing a loopback host.
    """
    out: list[Canary] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            socket.create_connection(CANARY_ADDR, timeout=1).close()
            out.append(Canary("outbound connect", False,
                              f"connected to {CANARY_ADDR[0]} — the guard is NOT armed"))
        except SocketConnectBlockedError:
            out.append(Canary("outbound connect", True,
                              f"connect to {CANARY_ADDR[0]}:{CANARY_ADDR[1]} raised "
                              "SocketConnectBlockedError"))
        except Exception as exc:
            out.append(Canary("outbound connect", False,
                              f"raised {type(exc).__name__}, not the guard's own error: {exc}"))

        try:
            socket.getaddrinfo(CANARY_NAME, 80)
            out.append(Canary("name lookup", False, "resolved — the resolver guard is NOT armed"))
        except SocketBlockedError:
            out.append(Canary("name lookup", True,
                              f"getaddrinfo({CANARY_NAME!r}) raised SocketBlockedError"))
        except Exception as exc:
            out.append(Canary("name lookup", False,
                              f"raised {type(exc).__name__}, not the guard's own error: {exc}"))

        try:
            socket.create_connection(("127.0.0.1", 9), timeout=1).close()
            out.append(Canary("loopback permitted", True, "connected to 127.0.0.1"))
        except SocketConnectBlockedError:
            out.append(Canary("loopback permitted", False, "the guard blocked loopback"))
        except OSError as exc:
            out.append(Canary("loopback permitted", True,
                              f"guard did not fire; the OS answered: {exc}"))
        except Exception as exc:
            out.append(Canary("loopback permitted", False,
                              f"raised {type(exc).__name__}: {exc}"))
    return out
