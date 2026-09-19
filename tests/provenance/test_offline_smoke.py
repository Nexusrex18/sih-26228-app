"""ADR-006 disconnected-sandbox smoke test for the seal's two dependencies (plan §12, gate C0).

Runs the *specific calls the seal will use* inside a fresh network namespace (`unshare -rn`), so a
library that quietly fetches something on first use fails here, not in the field. "pip install
succeeded" is not a test; this is.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap

import pytest

SMOKE = textwrap.dedent('''
    import socket
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        raise SystemExit("NOT ISOLATED: network reachable")
    except OSError:
        pass
    import hashlib, sqlite3
    import rfc8785
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    assert rfc8785.dumps({"b": 1, "a": [True, "x"]}) == b'{"a":[true,"x"],"b":1}'
    k = Ed25519PrivateKey.from_private_bytes(bytes(32))
    sig = k.sign(b"m"); k.public_key().verify(sig, b"m")
    assert len(sig) == 64 and hashlib.sha256(b"").digest()
    sqlite3.connect(":memory:").execute("PRAGMA journal_mode=WAL")
    print("OK")
''')


def _can_unshare() -> bool:
    if not shutil.which("unshare"):
        return False
    return subprocess.run(["unshare", "-rn", "true"], capture_output=True, check=False).returncode == 0


@pytest.mark.skipif(not _can_unshare(), reason="needs unprivileged user+network namespaces")
def test_seal_dependencies_work_with_the_network_disabled():
    r = subprocess.run(["unshare", "-rn", sys.executable, "-c", SMOKE],
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "OK"
