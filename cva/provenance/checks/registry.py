"""Registration of the scan-side provenance checks (plan §6, §13).

`core.registry` holds `ModelCheck`s, called `check(model, ctx)`. Forcing `LedgerVerify` into `REGISTRY` would
make the orchestrator call it with model arguments, so this module does NOT do that. It exposes the check and
one function Backend calls when the ledger step is wired in:

    from cva.provenance.checks.registry import PROV_CHECKS, assert_taxonomy_ok

**The "no ledger slot yet" half of this note is now stale (2026-09-20, audit item 7).** `RunContext` DOES probe
`INFERENCE_LEDGER` (`core/runcontext.py`), the capability member exists (`core/capability.py`), and `cva scan`
now takes `--inference-ledger` / `--audit-ledger` and constructs the sources. What is still missing is on this
side and is Module C's to close: `PROV_CHECKS` is not registered with `core.registry`, and `LedgerVerify`
exposes `resolve(caps)` but no `check(model, ctx)` convention for the orchestrator to call. Until it lands, a
scan given a ledger resolves the capability and produces no `prov.*` row, so the report carries an explicit
`provenance_summary.not_assessed_reason` rather than letting a satisfied capability with no rows read as
"nothing to report".
"""
from __future__ import annotations

from cva.core.taxonomy import TAXONOMY

from .ledger_verify import LedgerVerify

PROV_CHECKS: dict[str, type[LedgerVerify]] = {LedgerVerify.id: LedgerVerify}


def assert_taxonomy_ok() -> None:
    """A plug-in declaring an `attack_class` that is not in `core/taxonomy.py` is a STARTUP ERROR: its
    coverage row would silently vanish, and a shorter coverage statement looks like a cleaner result."""
    for cid, cls in PROV_CHECKS.items():
        unknown = sorted(set(cls.attack_classes) - set(TAXONOMY))
        if unknown:
            raise RuntimeError(f"{cid} declares attack classes missing from core/taxonomy.py: {unknown}")
