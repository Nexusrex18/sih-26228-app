"""The process-level egress guard (cva/core/egress.py, backend_plan.md §5.8, guard one of two).

Everything here is offline. 192.0.2.1 is TEST-NET-1 (RFC 5737), and inside `egress_guard()` a
connect to it is refused by the guard BEFORE any packet is built. The "guard absent" behaviour
of `run_canaries()` is exercised with the socket functions faked, because the real thing would
send a DNS query for `egress-canary.invalid` and a SYN towards TEST-NET-1: a real network call.
"""
from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError

from cva.core import egress
from cva.core.egress import CANARY_ADDR, CANARY_NAME, egress_guard, run_canaries

# pytest-socket's error constructors call warnings.warn(); the tests assert on the raise.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def snapshot() -> tuple[Any, Any, Any, Any]:
    return (socket.socket.connect, socket.getaddrinfo,
            socket.gethostbyname, socket.gethostbyname_ex)


@pytest.fixture
def loopback_listener() -> Iterator[int]:
    """A real listening socket on 127.0.0.1, opened BEFORE any guard is armed."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    try:
        yield int(srv.getsockname()[1])
    finally:
        srv.close()


# --------------------------------------------------------------------------------------
# the guard refuses what leaves the machine
# --------------------------------------------------------------------------------------
def test_outbound_connect_is_blocked_inside_the_guard() -> None:
    with egress_guard(), pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("192.0.2.1", 9), timeout=1)


def test_outbound_connect_via_a_plain_socket_is_blocked_inside_the_guard() -> None:
    with egress_guard():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            with pytest.raises(SocketConnectBlockedError):
                s.connect(("192.0.2.1", 9))
        finally:
            s.close()


def test_name_lookup_of_a_foreign_name_is_blocked_inside_the_guard() -> None:
    with egress_guard(), pytest.raises(SocketBlockedError):
        socket.getaddrinfo("egress-canary.invalid", 80)


@pytest.mark.parametrize("name", ["example.invalid", "EGRESS-CANARY.INVALID", "a.b.c.d.example"])
def test_every_resolver_entry_point_refuses_a_foreign_name(name: str) -> None:
    with egress_guard():
        with pytest.raises(SocketBlockedError):
            socket.getaddrinfo(name, 80)
        with pytest.raises(SocketBlockedError):
            socket.gethostbyname(name)
        with pytest.raises(SocketBlockedError):
            socket.gethostbyname_ex(name)


def test_a_bytes_hostname_is_still_refused() -> None:
    with egress_guard(), pytest.raises(SocketBlockedError):
        socket.getaddrinfo(b"egress-canary.invalid", 80)


def test_create_connection_to_a_foreign_name_dies_at_resolution() -> None:
    with egress_guard(), pytest.raises(SocketBlockedError):
        socket.create_connection((CANARY_NAME, 80), timeout=1)


# --------------------------------------------------------------------------------------
# ... and lets loopback through
# --------------------------------------------------------------------------------------
def test_loopback_connect_is_not_blocked_by_the_guard(loopback_listener: int) -> None:
    with egress_guard():
        conn = socket.create_connection(("127.0.0.1", loopback_listener), timeout=1)
        conn.close()


def test_loopback_connect_to_a_closed_port_is_an_os_error_not_a_guard_error() -> None:
    with egress_guard():
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=1).close()
        except SocketConnectBlockedError:
            pytest.fail("the guard blocked a loopback connect")
        except OSError:
            pass                                              # the OS answered: not the guard


def test_localhost_name_resolution_is_not_blocked(loopback_listener: int) -> None:
    with egress_guard():
        assert socket.getaddrinfo("localhost", loopback_listener)
        assert socket.gethostbyname("localhost").startswith("127.")


def test_resolving_an_ip_literal_is_not_blocked() -> None:
    with egress_guard():
        assert socket.getaddrinfo("192.0.2.1", 80)            # resolving a literal sends nothing
        assert socket.gethostbyname("192.0.2.1") == "192.0.2.1"


@pytest.mark.parametrize("host,expected", [
    (None, True), ("", True), ("localhost", True), ("LOCALHOST", True),
    ("localhost.localdomain", True), ("127.0.0.1", True), ("::1", True),
    ("192.0.2.1", True), ("fe80::1%eth0", True), (b"127.0.0.1", True),
    ("example.com", False), ("egress-canary.invalid", False), ("localhost.example", False),
    (b"example.com", False),
])
def test_resolvable_allows_only_loopback_names_and_ip_literals(host: Any,
                                                              expected: bool) -> None:
    assert egress._resolvable(host) is expected


# --------------------------------------------------------------------------------------
# run_canaries
# --------------------------------------------------------------------------------------
def test_run_canaries_are_all_ok_inside_the_guard() -> None:
    with egress_guard():
        canaries = run_canaries()
    assert [c.name for c in canaries] == ["outbound connect", "name lookup", "loopback permitted"]
    assert all(c.ok for c in canaries), [(c.name, c.detail) for c in canaries]


def test_run_canaries_details_name_the_guards_own_errors() -> None:
    with egress_guard():
        by_name = {c.name: c for c in run_canaries()}
    assert "SocketConnectBlockedError" in by_name["outbound connect"].detail
    assert "192.0.2.1" in by_name["outbound connect"].detail
    assert "SocketBlockedError" in by_name["name lookup"].detail


def test_run_canaries_reports_failure_when_nothing_is_guarding(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an unarmed process: connect succeeds and the name resolves. No real socket call."""
    class Conn:
        def close(self) -> None:
            pass

    calls: list[Any] = []

    def fake_connect(addr: Any, timeout: Any = None, *a: Any, **k: Any) -> Conn:
        calls.append(addr)
        return Conn()

    def fake_gai(host: Any, *a: Any, **k: Any) -> list[Any]:
        calls.append(host)
        return [(2, 1, 6, "", ("203.0.113.9", 80))]

    monkeypatch.setattr(socket, "create_connection", fake_connect)
    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
    by_name = {c.name: c for c in run_canaries()}
    assert by_name["outbound connect"].ok is False
    assert "NOT armed" in by_name["outbound connect"].detail
    assert by_name["name lookup"].ok is False
    assert "NOT armed" in by_name["name lookup"].detail
    assert by_name["loopback permitted"].ok is True           # it connected: the guard is absent
    assert CANARY_ADDR in calls and CANARY_NAME in calls


def test_run_canaries_reports_failure_when_the_error_is_not_the_guards(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """An OS-level refusal (e.g. no route) is not proof the guard works, so it must not pass."""
    def unreachable(*a: Any, **k: Any) -> Any:
        raise OSError("Network is unreachable")

    def no_dns(*a: Any, **k: Any) -> Any:
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "create_connection", unreachable)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    by_name = {c.name: c for c in run_canaries()}
    assert by_name["outbound connect"].ok is False
    assert "not the guard's own error" in by_name["outbound connect"].detail
    assert by_name["name lookup"].ok is False
    assert "not the guard's own error" in by_name["name lookup"].detail
    assert by_name["loopback permitted"].ok is True           # OSError from the OS is fine here


def test_run_canaries_fails_the_loopback_canary_if_the_guard_blocks_loopback(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*a: Any, **k: Any) -> Any:
        raise SocketConnectBlockedError(["127.0.0.1"], "127.0.0.1")

    def no_dns(*a: Any, **k: Any) -> Any:                     # never let the name canary go out
        raise SocketBlockedError("faked")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    by_name = {c.name: c for c in run_canaries()}
    assert by_name["loopback permitted"].ok is False
    assert by_name["outbound connect"].ok is True             # the guard's own error


# --------------------------------------------------------------------------------------
# restore and nesting
# --------------------------------------------------------------------------------------
def test_guard_arms_by_replacing_the_socket_functions_and_exit_restores_them() -> None:
    before = snapshot()
    with egress_guard():
        inside = snapshot()
        assert inside[0] is not before[0]                     # connect guard installed
        assert inside[1] is not before[1]                     # resolvers wrapped
        assert inside[2] is not before[2]
        assert inside[3] is not before[3]
    after = snapshot()
    assert all(a is b for a, b in zip(after, before, strict=True))


def test_guard_restores_the_socket_functions_when_the_block_raises() -> None:
    before = snapshot()
    with pytest.raises(RuntimeError, match="boom"), egress_guard():
        raise RuntimeError("boom")
    assert all(a is b for a, b in zip(snapshot(), before, strict=True))


def test_after_the_guard_exits_the_resolver_no_longer_refuses_a_name(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the exit really disarmed it, without a real lookup: a fake resolver is the 'real' one."""
    seen: list[Any] = []

    def fake_gai(host: Any, *a: Any, **k: Any) -> list[Any]:
        seen.append(host)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
    with egress_guard(), pytest.raises(SocketBlockedError):
        socket.getaddrinfo("example.invalid", 80)
    # arming resolves "localhost" through the real resolver; the foreign name never got there
    assert "example.invalid" not in seen
    socket.getaddrinfo("example.invalid", 80)                 # disarmed: reaches the fake
    assert seen[-1] == "example.invalid"


def test_nested_guards_inner_exit_leaves_the_outer_guard_armed() -> None:
    before = snapshot()
    with egress_guard():
        outer = snapshot()
        with egress_guard():
            with pytest.raises(SocketConnectBlockedError):
                socket.create_connection(("192.0.2.1", 9), timeout=1)
        # inner has exited: the outer guard's functions are back, and still refuse
        assert all(a is b for a, b in zip(snapshot(), outer, strict=True))
        with pytest.raises(SocketConnectBlockedError):
            socket.create_connection(("192.0.2.1", 9), timeout=1)
        with pytest.raises(SocketBlockedError):
            socket.getaddrinfo("egress-canary.invalid", 80)
    assert all(a is b for a, b in zip(snapshot(), before, strict=True))


def test_nested_guards_still_let_loopback_through(loopback_listener: int) -> None:
    with egress_guard(), egress_guard():
        socket.create_connection(("127.0.0.1", loopback_listener), timeout=1).close()


def test_the_guard_is_reusable() -> None:
    for _ in range(3):
        with egress_guard(), pytest.raises(SocketConnectBlockedError):
            socket.create_connection(("192.0.2.1", 9), timeout=1)


def test_loopback_allow_list_is_the_documented_one() -> None:
    assert egress.LOOPBACK_HOSTS == ("127.0.0.0/8", "::1", "localhost")
    assert CANARY_ADDR == ("192.0.2.1", 9)
    assert CANARY_NAME.endswith(".invalid")
