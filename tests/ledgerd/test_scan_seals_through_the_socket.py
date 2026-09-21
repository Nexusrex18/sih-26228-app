"""`cva scan --audit-ledger-socket` — the O2 handoff, end to end.

`LedgerdAuditLedger` was written and tested against the daemon; what was missing was the CLI
flag that puts it in the scan's audit-ledger slot. The property this file pins down is the
reason the seam exists at all: **the scanner seals a `scan_record` without ever holding the
signing key.** The daemon holds it, allowlists this uid for `scan_record` and nothing else,
and answers `capabilities()` from the real ledger rather than from the fact that a socket
opened.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from cva.cli import _ledgers, code_commit

pytest.importorskip("cva.provenance.seal",
                    reason="the seal SDK needs `cryptography` and `rfc8785`")


def _args(**kw) -> Namespace:
    base = {"inference_ledger": None, "audit_ledger": None, "audit_ledger_socket": None}
    return Namespace(**{**base, **kw})


def test_the_socket_flag_puts_a_ledgerd_handle_in_the_audit_slot(ledgerd: Path):
    from cva.web.workflow.ledgerd_client import LedgerdAuditLedger

    slots = _ledgers(_args(audit_ledger_socket=str(ledgerd)))
    assert isinstance(slots["audit_ledger"], LedgerdAuditLedger)


def test_the_file_flag_still_gives_the_unsigned_chain(tmp_path: Path):
    from cva.core.ledger import JsonlAuditLedger

    slots = _ledgers(_args(audit_ledger=str(tmp_path / "audit.jsonl")))
    assert isinstance(slots["audit_ledger"], JsonlAuditLedger)


def test_the_two_audit_flags_are_alternatives_not_an_addition(tmp_path: Path, ledgerd: Path):
    """Two audit ledgers is not a richer configuration. One slot exists; the record would go
    to whichever won and the report would name the other."""
    with pytest.raises(SystemExit) as e:
        _ledgers(_args(audit_ledger=str(tmp_path / "a.jsonl"),
                       audit_ledger_socket=str(ledgerd)))
    assert "alternatives" in str(e.value)


def test_the_handle_reports_a_signing_key_it_does_not_hold(ledgerd: Path, signing_key: Path):
    from cva.core.capability import Capability

    handle = _ledgers(_args(audit_ledger_socket=str(ledgerd)))["audit_ledger"]
    assert Capability.SIGNING_KEY in handle.capabilities()
    # The key is the daemon's. Nothing in the scanner's slot has a path to it.
    assert not hasattr(handle, "key")
    assert str(signing_key) not in repr(handle)


def test_a_scan_record_appended_through_the_socket_is_accepted_and_sealed(ledgerd: Path,
                                                                         ledger_path: Path):
    handle = _ledgers(_args(audit_ledger_socket=str(ledgerd)))["audit_ledger"]
    record = {
        "type": "scan_record",
        "scan": {
            "scan_id": "s-2026-09-21-0001",
            "report_sha256": "a" * 64,
            "profile_hash": "b" * 64,
            # The full 40-hex commit: `records.py:_v_scan` requires `_hex(..., 40)`, so the
            # short form the CLI used to emit was refused exactly here.
            "code_commit": "c" * 40,
            "finding_counts": {"critical": 0, "info": 1},
        },
    }
    seq = handle.append(record)
    assert int(seq) > 0


def test_a_short_code_commit_is_refused_by_the_daemon(ledgerd: Path):
    """The gap this flag exposed. A 7-hex abbreviation is fine for `JsonlAuditLedger`, which
    validates nothing, and is rejected the moment the same value reaches a real ledger."""
    from cva.web.workflow.ledgerd_client import LedgerRefused

    handle = _ledgers(_args(audit_ledger_socket=str(ledgerd)))["audit_ledger"]
    with pytest.raises(LedgerRefused):
        handle.append({"type": "scan_record",
                       "scan": {"scan_id": "s-2026-09-21-0002",
                                "report_sha256": "a" * 64, "profile_hash": "b" * 64,
                                "code_commit": "abc1234",
                                "finding_counts": {"info": 0}}})


def test_the_handle_refuses_any_record_type_but_scan_record(ledgerd: Path):
    from cva.web.workflow.ledgerd_client import LedgerRefused

    handle = _ledgers(_args(audit_ledger_socket=str(ledgerd)))["audit_ledger"]
    with pytest.raises(LedgerRefused):
        handle.append({"type": "analyst_event", "analyst": {}})


def test_code_commit_is_forty_hex_or_the_word_unknown():
    """Never padded. A zero-padded digest is a lie in the one format a verifier trusts."""
    value = code_commit()
    assert value == "unknown" or (len(value) == 40 and all(
        c in "0123456789abcdef" for c in value)), value


def test_an_image_bakes_the_commit_in_and_only_a_real_one_is_trusted(monkeypatch):
    """No `.git` inside an image, so the build sets CVA_CODE_COMMIT. A malformed value is
    ignored rather than recorded."""
    monkeypatch.setenv("CVA_CODE_COMMIT", "1" * 40)
    assert code_commit() == "1" * 40
    for bad in ("abc1234", "G" * 40, "0" * 39, "1" * 41):
        monkeypatch.setenv("CVA_CODE_COMMIT", bad)
        assert code_commit() != bad


def test_the_daemons_record_is_readable_back_and_names_this_scan(ledgerd: Path,
                                                                 ledger_path: Path):
    from cva.provenance.seal import SealedLedger

    handle = _ledgers(_args(audit_ledger_socket=str(ledgerd)))["audit_ledger"]
    handle.append({"type": "scan_record",
                   "scan": {"scan_id": "s-2026-09-21-0003", "report_sha256": "d" * 64,
                            "profile_hash": "e" * 64, "code_commit": "f" * 40,
                            "finding_counts": {"info": 2}}})
    # Read-only, and with no key: reading back what the daemon sealed must not need the
    # thing the whole seam exists to keep away from the scanner.
    led = SealedLedger.open(ledger_path, read_only=True)
    try:
        payloads = [json.loads(r.payload) if isinstance(getattr(r, "payload", None), str)
                    else getattr(r, "payload", r) for r in led.records()]
    finally:
        led.close()
    assert any("s-2026-09-21-0003" in json.dumps(p, default=str) for p in payloads)
