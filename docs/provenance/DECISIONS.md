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
