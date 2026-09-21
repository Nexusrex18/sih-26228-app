"""D-E10: is the `report.json` on disk the one the ledger sealed?

Three states, never two. "Not sealed" and "differs" are different facts, and collapsing
them would let an altered report present as an unsealed one.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

SEALED = "sealed"
DIFFERS = "differs"
NOT_SEALED = "not_sealed"
UNKNOWN = "unknown"

_LABELS = {
    SEALED: "report sealed",
    DIFFERS: "differs from sealed digest",
    NOT_SEALED: "not sealed",
    UNKNOWN: "seal state unknown",
}


@dataclass(frozen=True)
class SealState:
    state: str
    detail: str
    sealed_digest: str | None = None
    on_disk_digest: str | None = None
    ledger_seq: int | None = None

    @property
    def label(self) -> str:
        return _LABELS[self.state]

    @property
    def is_alarm(self) -> bool:
        """`differs` is a red, top-of-page finding: the report was altered after it was
        written. Nothing else on the page can be trusted more than this."""
        return self.state == DIFFERS


def scan_records_for(records: Iterable[Mapping[str, Any]], scan_id: str) -> list[dict[str, Any]]:
    """Every `scan_record` naming this scan, in ledger order.

    Found by `scan_id`, never by `seq` — Backend's `core/ledger.py` says why: the seq is
    assigned after `report.json` is hashed and is deliberately never written back into it.
    """
    out = []
    for r in records:
        if r.get("type") != "scan_record":
            continue
        scan = r.get("scan") or {}
        if scan.get("scan_id") == scan_id:
            out.append(dict(r))
    return sorted(out, key=lambda r: int(r.get("seq", 0)))


def check(on_disk_sha256: str, scan_id: str,
          records: Iterable[Mapping[str, Any]] | None,
          ledger_available: bool = True) -> SealState:
    if not ledger_available or records is None:
        return SealState(UNKNOWN,
                         "The audit ledger is not reachable, so this report's seal could not "
                         "be checked. This is not the same as 'not sealed'.",
                         on_disk_digest=on_disk_sha256)
    matching = scan_records_for(records, scan_id)
    if not matching:
        return SealState(NOT_SEALED,
                         "No scan_record for this scan is in the audit ledger. The scan ran "
                         "without a signing key, or the record was never appended.",
                         on_disk_digest=on_disk_sha256)
    # The LAST record wins: a re-seal after a key rotation appends rather than replaces.
    latest = matching[-1]
    sealed = str((latest.get("scan") or {}).get("report_sha256", ""))
    seq = int(latest.get("seq", 0))
    if sealed == on_disk_sha256:
        return SealState(SEALED,
                         f"sha256 of report.json matches the digest sealed at ledger seq {seq}",
                         sealed_digest=sealed, on_disk_digest=on_disk_sha256, ledger_seq=seq)
    return SealState(
        DIFFERS,
        f"The ledger sealed {sealed[:16]}… at seq {seq}; the file on disk hashes to "
        f"{on_disk_sha256[:16]}…. The report was altered after it was written. Treat every "
        "number on this page as unverified until the original is restored.",
        sealed_digest=sealed, on_disk_digest=on_disk_sha256, ledger_seq=seq)


__all__ = ["DIFFERS", "NOT_SEALED", "SEALED", "UNKNOWN", "SealState", "check",
           "scan_records_for"]
