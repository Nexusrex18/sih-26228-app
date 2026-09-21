"""S9: the dashboard binds loopback, and leaving loopback is never a quiet default.

Three cases, and the third one is a bug these tests exist to keep fixed:

  1. A non-loopback bind with nothing else set is a load error.
  2. The container case is declared, not inferred: `bind_trusts_container_boundary` lets the
     process bind 0.0.0.0 inside a bridge network, and moves the obligation to the runtime's
     publish spec (`-p 127.0.0.1:…`).
  3. Configuring `tls_*` satisfies the config check — but waitress does not terminate TLS,
     so serving would put plaintext on the LAN while believing it was encrypted. `run()`
     refuses to start instead.
"""
from __future__ import annotations

import pytest

from cva.web.config import ConfigError, WebConfig, from_mapping, load


def test_loopback_is_the_default():
    assert WebConfig().bind_host == "127.0.0.1"
    assert WebConfig().bind_trusts_container_boundary is False


def test_leaving_loopback_without_tls_or_a_declared_boundary_is_a_load_error():
    with pytest.raises(ConfigError, match="S9"):
        WebConfig(bind_host="0.0.0.0")


def test_the_container_boundary_must_be_declared_to_bind_all_interfaces():
    cfg = WebConfig(bind_host="0.0.0.0", bind_trusts_container_boundary=True)
    assert cfg.bind_host == "0.0.0.0"


def test_the_declaration_comes_through_the_environment_as_the_entrypoint_sets_it():
    cfg = load(environ={"CVA_WEB_BIND_HOST": "0.0.0.0",
                        "CVA_WEB_BIND_TRUSTS_CONTAINER_BOUNDARY": "true"})
    assert cfg.bind_host == "0.0.0.0" and cfg.bind_trusts_container_boundary is True


def test_the_declaration_is_a_boolean_not_any_string():
    with pytest.raises(ConfigError):
        from_mapping({"bind_host": "0.0.0.0", "bind_trusts_container_boundary": "sure"})


def test_a_tls_configuration_refuses_to_serve_plaintext(tmp_path):
    from cva.web.app import run

    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    cfg = WebConfig(bind_host="0.0.0.0", tls_certfile=cert, tls_keyfile=key,
                    reports_dir=tmp_path, index_db=tmp_path / "i.db",
                    accounts_db=tmp_path / "a.db")
    with pytest.raises(ConfigError, match="does not terminate TLS"):
        run(cfg)
