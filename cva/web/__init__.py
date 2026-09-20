"""`cva-web` — the served analyst dashboard (Module E).

**This module deliberately stays empty.** Two boundaries depend on it:

  * `cva.ledgerd` imports `cva.web.workflow.fold` and `cva.web.workflow.events` so that the
    fold and the enum set exist in exactly one place. If importing `cva.web` pulled in
    `app.py`, the ledger-writing daemon would import Flask — and the process that holds the
    signing key would grow the dependency surface of the process that parses attacker
    strings. `tests/ledgerd/test_import_boundary.py` asserts it does not.
  * `cva.web` must never import `cva.provenance`. The web process has **no key and no
    ledger handle** (plan D-E3, §5.2): it reads the ledger *through* `cva-ledgerd` over a
    Unix socket, and verifies it by shelling out to `cva-seal verify --json`. CI invariant 5
    (`tests/boundaries/test_import_invariants.py`) makes that a build failure rather than a
    convention.
"""
