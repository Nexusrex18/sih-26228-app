"""Registration of the scan-side provenance checks (plan §6, §13).

`core.registry` today holds `ModelCheck`s (called `check(model, ctx)`), and the orchestrator has no ledger slot
yet (`RunContext` does not probe `INFERENCE_LEDGER`). Forcing `LedgerVerify` into `REGISTRY` would make the
orchestrator call it with model arguments, so this module does NOT do that. It exposes the check and one
function Backend calls when the ledger step is wired in:

    from cva.provenance.checks.registry import PROV_CHECKS, assert_taxonomy_ok
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
