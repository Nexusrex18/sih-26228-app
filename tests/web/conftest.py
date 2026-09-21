"""A real app, real fixtures, a real ledgerd behind a real socket.

Nothing here mocks the ledger. The whole design is that `cva-web` cannot reach the key, so a
mocked ledger would test the opposite of what matters.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("flask", reason="Module E's runtime: pip install -e '.[network]'")

from cva.web import fixtures as report_fixtures  # noqa: E402
from cva.web.config import WebConfig  # noqa: E402

seal = pytest.importorskip("cva.provenance.seal",
                           reason="the seal SDK needs `cryptography` and `rfc8785`")


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    out = tmp_path / "reports"
    report_fixtures.write(out)
    return out


@pytest.fixture
def ledger_bits(tmp_path: Path):
    """key + ledger + trust root, the way `cva-seal keygen` / `init` produce them."""
    from cva.provenance.seal.keys import TrustKey
    from cva.provenance.seal.records import genesis_prev_hash

    d = tmp_path / "ledger"
    d.mkdir()
    seal.generate_keypair(d / "key")
    key = seal.FileKeyProvider(d / "key")
    led = seal.SealedLedger.init_ledger(
        d / "audit.db", key,
        {"device_id": "test", "unit": "test", "profile_hash": "0" * 64,
         "checkpoint_every": 1000})
    trust = seal.TrustRoot(genesis_prev_hash(led.deployment_manifest),
                           (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    (d / "trust_root.json").write_bytes(trust.to_bytes() + b"\n")
    return d / "key", d / "audit.db", d / "trust_root.json"


@pytest.fixture
def ledgerd(tmp_path: Path, ledger_bits):
    from cva.ledgerd.policy import development_policy
    from cva.ledgerd.server import serve

    key, ledger, _ = ledger_bits
    sock = tmp_path / "ledgerd.sock"
    server = serve(sock, ledger, key, development_policy())
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
def config(tmp_path: Path, reports_dir: Path, ledger_bits, ledgerd: Path) -> WebConfig:
    _, ledger, trust = ledger_bits
    return WebConfig(reports_dir=reports_dir, index_db=tmp_path / "index.db",
                     accounts_db=tmp_path / "accounts.db", ledgerd_socket=ledgerd,
                     trust_root=trust, ledger_path=ledger,
                     rate_limit_per_minute=100000)


@pytest.fixture
def no_ledger_config(tmp_path: Path, reports_dir: Path) -> WebConfig:
    """The fail-closed case: nothing listening on the socket at all."""
    return WebConfig(reports_dir=reports_dir, index_db=tmp_path / "index.db",
                     accounts_db=tmp_path / "accounts.db",
                     ledgerd_socket=tmp_path / "nothing-here.sock",
                     rate_limit_per_minute=100000)


@pytest.fixture
def app(config: WebConfig):
    from cva.web.app import create_app
    application = create_app(config, TESTING=True)
    store = application.extensions["cva_accounts"]
    store.create("a.sharma", "correct-horse-battery", "analyst")
    store.create("b.rao", "correct-horse-battery", "approver")
    store.create("v.iyer", "correct-horse-battery", "viewer")
    store.create("root.admin", "correct-horse-battery", "admin")
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def scan_id() -> str:
    """The rich fixture: every disposition, a prov.* quarantine, hostile strings."""
    return "s-2026-09-19-0001"


def sign_in(client, actor="a.sharma", password="correct-horse-battery"):
    resp = client.post("/login", data={"actor_id": actor, "password": password,
                                       "csrf_token": _csrf(client)},
                       headers={"Origin": "http://localhost", "Host": "localhost"})
    assert resp.status_code in (302, 303), resp.data[:400]
    return resp


def _csrf(client) -> str:
    from flask import session
    with client.session_transaction() as s:
        from cva.web.security import new_csrf_token
        token = s.get("csrf") or new_csrf_token()
        s["csrf"] = token
    return token


def post(client, url, data=None, **kw):
    data = dict(data or {})
    data.setdefault("csrf_token", _csrf(client))
    kw.setdefault("headers", {"Origin": "http://localhost", "Host": "localhost"})
    return client.post(url, data=data, **kw)


@pytest.fixture
def signed_in(client):
    sign_in(client)
    return client
