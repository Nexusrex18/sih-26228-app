# Module C — decision log

Sign-off for the wire-format decisions in `sih26228-notes/Plan/Module-C-Provenance-Seal-Plan.md` §4.
These change the record format; each is a chain-wide migration if changed after records exist.

| ID | Decision | Status |
|---|---|---|
| D1 | Sign `TAG ‖ canonical_bytes`, tags per record class | **approved** |
| D2 | Typed records with a common header | **approved** (also team-approved 2026-09-19) |
| D3 | Canonical profile = strict subset of RFC 8785 (ASCII, integers only, no floats) | **approved** |
| D4 | One cumulative RFC 6962 tree, signed checkpoint every N records | **approved** |
| D5 | Genesis `prev_record_hash = SHA-256(0x03 ‖ JCS(deployment_manifest))` | **approved** |
| D6 | Two output hashes (fine + coarse `decision_sha256`) and the R0–R3 recompute ladder | **approved** (also team-approved) |
| D7 | Merkle leaf = full signed record bytes | **approved** |
| D12 | Time attestation = signed witness statement, not RFC 3161 | **approved** |
| D15 | Seal raw output; record post-processing separately; TOCTOU-safe two-call API | **approved** |

Approved by the Crypto seat's owner, who delegated the choice to the implementer after a plain-language
review; the implementer re-checked the nine for mutual conflicts and found none.

## Carried into the build
- **D6:** whole-pixel rounding can still flip near a half-pixel boundary. C6's stability study must show
  0 false R1 mismatches; if not, widen the coarse quantisation — never widen a tolerance.
- **D3:** `canonical.ascii_encode()` (added at C1) so callers never hand-roll percent-encoding.

## Decisions taken while building C4 (implementer's calls — flag any you disagree with)

| ID | Decision | Why |
|---|---|---|
| C4-1 | **Payload fsync follows the durability mode.** `per_record`: each payload file is fsynced before the record commits. `group_commit`: payloads are written first and fsynced together at the flush. | Measured on ext4/NVMe: two payload fsyncs cost ~2.3 ms, more than the ledger commit (~2.0 ms). The plan's §9 budget never counted them. Without this, `group_commit` could not meet the 5 ms budget. |
| C4-2 | **`group_commit` uses a background flusher** that fsyncs deferred payloads and the `-wal` file by path, waking every `group_ms` or when `group_n` records are unflushed. Never on the request path. | A synchronous flush every 100th commit produced ~45 ms p99. Also gives a *real* `group_ms` bound (no dependence on a later commit arriving). |
| C4-3 | **A failed background flush is sticky**: further sealing raises `LedgerUnavailable` until an explicit `flush()` succeeds. | If records can no longer be made durable, the loss window is unbounded — fail closed. |
| C4-4 | **Durability is stored in the ledger's `meta` at `init`, not in `SealPolicy`.** | `meta` is insert-only, and the loss window printed in a verify report must not be able to disagree with how the ledger was written. |
| C4-5 | **A model registration is written lazily**, atomically with the first inference that uses a not-yet-registered (model, config) pair. A re-load with different weights writes a new registration. | Keeps `register_*` free of I/O; guarantees registration precedes use. |
| C4-6 | **On recovery the `degraded_marker` is written first, then any registration, then the inference — all in one transaction.** | A declared hole is never followed by an inference that could be seen without it. |
| C4-7 | **`SealedLedger.append()` (the `AuditLedger` protocol) writes only `scan_record` and `analyst_event`.** | Least privilege: genesis, checkpoints, anchors, rotations and markers are Crypto's own paths. |
| C4-8 | **Recovery-on-open and integrity reads use a single read snapshot.** | A multi-process test caught a real race: tip and row-count read as separate statements disagreed while another process was appending. |
| C4-9 | **A key-provider failure at signing time is a `LedgerUnavailable` (`SigningFailed`).** | Otherwise an unplugged HSM would crash a fail-open pipeline instead of producing a declared gap. |
| C4-10 | **`capabilities()` returns capability *names* (strings)**, because `provenance/seal` may not import `core/`. `Capability` is a `str` enum, so consumers' `in` tests work unchanged (asserted in tests against `cva.core.ledger`). | Keeps the seal standalone. |
| C4-11 | **Checkpoints count every record** (genesis, markers, registrations, the checkpoints themselves) toward `checkpoint_every`. | Simplest rule that is derivable from the chain alone. |

### Open / spec ambiguities found
- **`TAG_CHECKPOINT` is defined (D1) but unused**: checkpoint *records* are signed under `TAG_RECORD` like every record. Decide at C7/C8 whether an anchor's checkpoint signature uses it.
- **Alternative to C4-1, not taken:** store payloads in a table inside the ledger DB (one fsync, atomic with the record, `per_record` would then fit the budget). It changes the published payload layout (files → rows), so it needs an explicit decision.
