"""A real `SealedLedger` behind a real `cva-ledgerd` on a real Unix socket.

The daemon is exercised through the socket, not by calling `Daemon.handle` directly, because
`SO_PEERCRED` and the framing are part of what is being tested. The uid allowlist itself is
tested separately as a pure function (`test_policy.py`) — on a single-uid box a socket test
cannot tell an allowlist from a no-op.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

seal = pytest.importorskip("cva.provenance.seal",
                           reason="the seal SDK needs `cryptography` and `rfc8785`")


@pytest.fixture
def ledger_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ledger"
    d.mkdir()
    return d


@pytest.fixture
def signing_key(ledger_dir: Path) -> Path:
    seal.generate_keypair(ledger_dir / "key")
    return ledger_dir / "key"


@pytest.fixture
def ledger_path(ledger_dir: Path, signing_key: Path) -> Path:
    path = ledger_dir / "audit.db"
    led = seal.SealedLedger.init_ledger(
        path, seal.FileKeyProvider(signing_key),
        {"device_id": "test-device", "unit": "test-unit", "profile_hash": "0" * 64,
         "checkpoint_every": 1000})
    led.close()
    return path


@pytest.fixture
def trust_root(ledger_dir: Path, signing_key: Path, ledger_path: Path) -> Path:
    from cva.provenance.seal.records import genesis_prev_hash
    key = seal.FileKeyProvider(signing_key)
    led = seal.SealedLedger.open(ledger_path, key=key)
    tr = seal.TrustRoot(genesis_prev_hash(led.deployment_manifest),
                        (seal.keys.TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    out = ledger_dir / "trust_root.json"
    out.write_bytes(tr.to_bytes() + b"\n")
    return out


@pytest.fixture
def ledgerd(ledger_dir: Path, ledger_path: Path, signing_key: Path):
    """A running daemon. Yields its socket path."""
    from cva.ledgerd.policy import development_policy
    from cva.ledgerd.server import serve

    sock = ledger_dir / "ledgerd.sock"
    server = serve(sock, ledger_path, signing_key, development_policy())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield sock
    finally:
        server.shutdown()
        server.server_close()
        server.daemon.backend.close()
        thread.join(timeout=5)


@pytest.fixture
def client(ledgerd: Path):
    from cva.web.workflow.ledgerd_client import LedgerdClient
    return LedgerdClient(ledgerd)
