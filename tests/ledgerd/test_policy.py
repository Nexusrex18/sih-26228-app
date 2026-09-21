"""The uid allowlist as a pure function, with synthetic uids (plan S10).

On a single-uid development box a socket test cannot distinguish an allowlist from a no-op:
it passes because every process happens to share a uid. So the decision is tested here with
uids that do not exist, and the socket wiring is tested separately.
"""
from __future__ import annotations

import pytest

from cva.ledgerd.policy import (
    ANALYST_EVENT,
    FORBIDDEN_TYPES,
    SCAN_RECORD,
    Policy,
    decide,
    development_policy,
    from_mapping,
)

WEB_UID = 4001        # uid A in plan §5.2
SCANNER_UID = 4002    # uid C
STRANGER_UID = 4099

DEPLOYED = Policy(writers={WEB_UID: frozenset({ANALYST_EVENT}),
                           SCANNER_UID: frozenset({SCAN_RECORD})},
                  readers=frozenset({WEB_UID}))


def test_the_web_uid_may_write_analyst_events_and_nothing_else():
    assert decide(DEPLOYED, WEB_UID, "append_analyst_event")[0]
    allowed, detail = decide(DEPLOYED, WEB_UID, "append_scan_record")
    assert not allowed
    assert "per uid AND per record type" in detail


def test_the_scanner_uid_may_seal_its_own_scan_record_and_nothing_else():
    assert decide(DEPLOYED, SCANNER_UID, "append_scan_record")[0]
    assert not decide(DEPLOYED, SCANNER_UID, "append_analyst_event")[0]


def test_an_unknown_uid_is_refused_everything():
    for op in ("status", "records", "append_analyst_event", "append_scan_record"):
        allowed, detail = decide(DEPLOYED, STRANGER_UID, op)
        assert not allowed, op
        assert str(STRANGER_UID) in detail or "unknown op" in detail


def test_the_scanner_uid_cannot_read_the_analyst_record_stream():
    """Least privilege runs both ways: sealing a scan record is not a licence to read
    every decision an analyst recorded."""
    assert not decide(DEPLOYED, SCANNER_UID, "records")[0]


@pytest.mark.parametrize("rtype", sorted(FORBIDDEN_TYPES))
def test_cryptos_own_record_types_are_never_writable_here(rtype):
    wide = Policy(writers={WEB_UID: frozenset({ANALYST_EVENT, SCAN_RECORD})})
    assert not wide.may_append(WEB_UID, rtype), rtype


def test_a_policy_file_cannot_grant_a_forbidden_type():
    with pytest.raises(ValueError, match="cannot be granted here"):
        from_mapping({"writers": {"4001": ["analyst_event", "genesis"]}})


def test_a_policy_file_round_trips():
    p = from_mapping({"writers": {"4001": ["analyst_event"], "4002": "scan_record"},
                      "readers": [4001]})
    assert p.may_append(4001, ANALYST_EVENT)
    assert p.may_append(4002, SCAN_RECORD)
    assert not p.may_append(4001, SCAN_RECORD)
    assert p.may_read(4001)


def test_the_development_policy_is_scoped_to_its_own_uid():
    """`same_uid_ok` widens WHO may connect, never WHAT may be written."""
    p = development_policy()
    assert p.may_append(p.owner_uid, ANALYST_EVENT)
    assert not p.may_append(p.owner_uid + 1, ANALYST_EVENT)
    for rtype in FORBIDDEN_TYPES:
        assert not p.may_append(p.owner_uid, rtype)


def test_an_empty_policy_serves_nobody():
    empty = Policy()
    assert not decide(empty, WEB_UID, "status")[0]
    assert not decide(empty, WEB_UID, "append_analyst_event")[0]
