"""Two ledger protocols, not one — backend_plan.md §5.2.

The orchestrator's last step is *"append scan record"*, and `RunContext.capabilities()`
must probe `INFERENCE_LEDGER` and `SIGNING_KEY`. Crypto's `provenance/` does not exist
yet, so at B3 there would be nothing to call and nothing to probe. A `TODO` at that call
site would leave the orchestrator structurally incomplete through four gates and — worse —
would make `RunContext.capabilities()` unable to answer honestly.

**Why two protocols.** An earlier draft defined a single `Ledger` carrying both
`INFERENCE_LEDGER` and `SIGNING_KEY`. The Module C plan (§3.1, open item O1) found that
conflates two genuinely different objects, and it is right:

  * the **scanner's own audit ledger** — written by the orchestrator's last step, needs a
    signing key of ours;
  * the **field inference ledger under audit** — a read-only input to `prov.*`, needs no
    key of ours at all.

Different files, different keys, different genesis records, a different trust root. One
protocol cannot honestly probe for both — and the single-protocol version would have let
an unsigned test-only JSONL file present itself as a field ledger under audit, which is
false assurance of exactly the kind this contract exists to prevent.

Both protocols live in `core/` and `provenance/` implements them. That does not violate CI
invariant 1: the dependency points from `provenance/` to `core/`, the direction every
module already depends in.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .capability import Capability


@runtime_checkable
class AuditLedger(Protocol):
    """The scanner WRITES this one."""

    def append(self, record: Mapping[str, Any]) -> str: ...
    def capabilities(self) -> set[Capability]: ...   # {SIGNING_KEY} iff it can actually sign


@runtime_checkable
class InferenceLedgerSource(Protocol):
    """The scanner READS this one — the artefact under audit."""

    def records(self) -> Iterator[Mapping[str, Any]]: ...
    def capabilities(self) -> set[Capability]: ...   # {INFERENCE_LEDGER} iff it opened


class NullAuditLedger:
    """Default. The report then honestly says the scan record was not sealed.

    Not a no-op with a comment: it reports NO capability, so `SIGNING_KEY` is absent from
    the scan's CapabilitySet and every consumer sees the truth.
    """

    def __init__(self) -> None:
        self.records_seen: list[Mapping[str, Any]] = []

    def append(self, record: Mapping[str, Any]) -> str:
        self.records_seen.append(dict(record))
        return f"unsealed-{len(self.records_seen)}"

    def capabilities(self) -> set[Capability]:
        return set()


class JsonlAuditLedger:
    """sha256-chained JSONL, **unsigned** — a chain but no key. Local test only.

    It reports NO capability at all, and that is the point. It genuinely has a chain and
    genuinely has no key — and `INFERENCE_LEDGER` is not its capability to report either,
    because that member describes *the artefact being audited*, not the scanner's own log.
    Crypto's `SealedLedger` replaces it at P2 and reports `{SIGNING_KEY}` only once a key
    has loaded AND produced a test signature that verified.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _tail_hash(self) -> str:
        prev = "0" * 64
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    prev = json.loads(line)["record_hash"]
        return prev

    def append(self, record: Mapping[str, Any]) -> str:
        prev = self._tail_hash()
        body = dict(record)
        # Deliberately NOT JCS: this is a test-only chain, and pretending to the canonical
        # encoding would invite someone to verify it with Crypto's verifier and conclude
        # the wrong thing. Module C's rfc8785 path is the real one.
        payload = json.dumps({"prev": prev, "body": body}, sort_keys=True,
                             separators=(",", ":"))
        rh = hashlib.sha256(payload.encode()).hexdigest()
        seq = sum(1 for _ in self.path.read_text().splitlines()
                  if _.strip()) if self.path.exists() else 0
        with self.path.open("a") as fh:
            fh.write(json.dumps({"seq": seq, "prev_record_hash": prev,
                                 "record_hash": rh, "body": body}) + "\n")
        return str(seq)

    def capabilities(self) -> set[Capability]:
        return set()


class NullInferenceLedger:
    """No ledger supplied -> every `prov.*` check resolves UNAVAILABLE, with a reason."""

    def records(self) -> Iterator[Mapping[str, Any]]:
        return iter(())

    def capabilities(self) -> set[Capability]:
        return set()


class JsonlInferenceLedger:
    """Reads a field ledger from disk.

    Probing is ACTIVE per `Plugin-Interfaces.md:98`: it reports `INFERENCE_LEDGER` only if
    the file opened and the first record parsed. A path that exists but holds garbage must
    not present itself as a ledger.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def records(self) -> Iterator[Mapping[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def capabilities(self) -> set[Capability]:
        try:
            next(iter(self.records()))
        except (StopIteration, OSError, json.JSONDecodeError):
            return set()
        return {Capability.INFERENCE_LEDGER}


def scan_record(scan_id: str, report_sha256: str, profile_hash: str, code_commit: str,
                finding_counts: Mapping[str, int]) -> dict[str, Any]:
    """The body Module C §5.7 reserves for `type: "scan_record"`.

    Ordering, which is easy to get wrong in the expensive direction: the report is
    rendered and fsynced BEFORE this is appended, because `report_sha256` binds it — and
    the `seq` that `append()` returns goes to stdout and the log ONLY. Writing `seq` back
    into `report.json` would change the file after it was hashed, and the ledger would
    then hold a digest of a file that no longer exists, reporting *"differs from sealed
    digest"* on every clean scan. The dashboard finds the record by `scan_id`, not by
    `seq`, for exactly this reason.
    """
    return {
        "type": "scan_record",
        "scan": {
            "scan_id": scan_id,
            "report_sha256": report_sha256,
            "profile_hash": profile_hash,
            "code_commit": code_commit,
            "finding_counts": dict(finding_counts),
        },
    }
