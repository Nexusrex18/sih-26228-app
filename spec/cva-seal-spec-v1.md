# cva-seal/1 — wire format and verification procedure

> *Published with the code it describes. The Module C plan, build-decisions log and coverage statement it refers to are kept in the project's planning repository.*

> **Status:** normative, version `cva-seal/1`. Published for gate C8 (the Module C plan §12): a verifier
> written from *this document alone* must agree with the reference implementation on every frozen vector and every
> tamper artefact. If it cannot be written from this text, the text is incomplete — that is a spec bug, not a
> reader error. Frozen vectors: `spec/vectors/cva_seal_v1.json` in the app repository. Decisions behind the design:
> the Module C build-decisions log. How a third party runs a verification: [the verification procedure](../docs/provenance/VERIFICATION-PROCEDURE.md).

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## 1. What is being specified

A **ledger** is an append-only sequence of **records**. Each record is signed, chained to its predecessor, and is a
leaf of a Merkle tree. A **verifier** reads the ledger (as an **export**: §9), a **trust root** (§10) and optionally
**anchors** (§11) and reports **findings**. Nothing here needs a network, a clock, or the software that wrote the
ledger.

Primitives, and only these: SHA-256 (FIPS 180-4), Ed25519 in its *pure* form (RFC 8032 §5.1, no prehash, no context),
and the RFC 6962 / RFC 9162 Merkle tree hash. Hex is always **lowercase**; uppercase hex is *invalid* everywhere
(a second spelling of the same bytes is a malleability channel).

## 2. Encoding

### 2.1 The JSON profile

A record is a JSON object (RFC 8259) restricted as follows. Anything outside the profile makes the record *malformed*.

1. **Text** is ASCII (bytes `0x00–0x7F`); a byte `≥ 0x80` is malformed.
2. **Object keys** match `[a-z0-9_]+`. **Duplicate keys** are malformed (never "last one wins").
3. **String values** contain only printable ASCII `0x20–0x7E`. Caller text outside that range is *percent-encoded by
   the writer* (`%XX` per UTF-8 byte, **uppercase** hex, `%` itself as `%25`); the verifier never decodes it.
4. **Numbers** are integers with `|n| ≤ 2^53 − 1`, written in the shortest decimal form (no `+`, no leading zeros,
   no `-0`, no fraction, no exponent). A float, `NaN` or `Infinity` is malformed.
5. **Literals** `true`, `false`, `null` are allowed as *object values* only. A `null` inside an array is malformed.
6. **Arrays** are homogeneous: all integers, all strings, all booleans, all objects, or all arrays (an empty array
   is homogeneous).
7. **Nesting** of containers is at most **8** deep (the top-level object is depth 1).
8. A stored record is at most **65 536** bytes.
9. The top level is an object.

### 2.2 Canonical bytes

`canon(x)` is the RFC 8785 (JCS) serialisation of `x`. Under this profile it reduces to:

* no whitespace anywhere;
* object members ordered by key, ascending by byte value;
* integers in decimal;
* strings in double quotes, with exactly `"` written `\"` and `\` written `\\`, every other character literal;
* `true`, `false`, `null`.

**A stored record is exactly its canonical bytes** (no trailing newline inside the record). A verifier parses a
stored record, re-serialises the result canonically, and requires byte equality: a record that parses but differs
(key order, whitespace, escapes, trailing bytes) is *non-canonical* — itself a finding (§13, step 1).

## 3. Quantisation

Floats never enter a record. Quantities are converted to integers *before* canonicalising, with one exact sequence
(`floor` is the mathematical floor; arithmetic is IEEE-754 binary64; a float32 is widened exactly first):

| Quantity | Rule | Name |
|---|---|---|
| probability / confidence `p` | `q = floor(p × 1 000 000.0 + 0.5)`, clamped to `[0, 1 000 000]` | `conf_e6` |
| box coordinate (px) | `q = floor(x × 64.0 + 0.5)` | `box_q64` |
| box coordinate, whole pixel | `q = floor(x + 0.5)` | `box_px` |
| preprocessing scalars, filter thresholds | as `conf_e6` | `…_e6` |

The multiply–add–floor sequence MUST NOT be replaced by `round()` (banker's rounding differs at exact halves:
`0.5 → 0` but `1.5 → 2`). Non-finite input is never quantised: the writer seals the marker output `{"nonfinite":true}`
(§5.4) instead. A verifier does not quantise — it only checks hashes — but conformance vectors for the rule are in
`quantise` in the vectors file.

## 4. The record

### 4.1 Header (every type)

| Field | Form |
|---|---|
| `v` | the string `cva-seal/1` |
| `type` | one of `genesis, model_registration, inference, checkpoint, anchor_event, key_rotation, degraded_marker, scan_record, analyst_event` |
| `seq` | integer ≥ 0; `0` exactly for genesis; gapless |
| `prev_record_hash` | 64 hex — a **link** (§6) except in genesis (§4.3) |
| `key_id` | 64 hex = SHA-256 of the raw 32-byte public key of the signer |
| `created_at_utc` | `YYYY-MM-DDTHH:MM:SS.ffffffZ`, exactly 27 characters, a real calendar time; **untrusted**, never used for ordering |
| `nonce` | 32 hex (16 random bytes) |
| *sections* | the type's sections (below), **exactly** those |
| `signature` | 128 hex (the 64 raw bytes) |

A record has **exactly** the header fields, the type's sections and `signature` — no others.

### 4.2 Sections by type

| type | sections |
|---|---|
| `genesis` | `deployment_manifest` |
| `model_registration` | `model`, `config` |
| `inference` | `input`, `model`, `config`, `output` |
| `checkpoint` | `checkpoint` |
| `anchor_event` | `anchor` |
| `key_rotation` | `rotation` |
| `degraded_marker` | `gap` |
| `scan_record` | `scan` |
| `analyst_event` | `analyst` |

Section schemas (every listed key is required unless marked optional; **no other keys**; `H64` = 64 lowercase hex,
etc.; `TEXT(a,b)` = printable ASCII of length `a…b`; `TOKEN(n)` = `[a-z0-9_]{1,n}`; `IDENT(n)` = `[A-Za-z0-9._:-]{1,n}`;
`REF` = `sha256:` + `H64`):

* `deployment_manifest`: `device_id` TEXT(1,128) · `key_id` H64 · `profile_hash` H64 · `unit` TEXT(1,128) ·
  `checkpoint_every` integer ≥ 1 · `spec` = `cva-seal/1`
* `model`: `id` TEXT(1,128) · `weights_sha256` H64 · `format` TOKEN(32) · `arch_hash` H64
* `config`: `preprocess_hash` H64 · `preprocess_ref` REF · `postprocess_hash` H64 · `runtime` TEXT(1,200) ·
  `version_pins_hash` H64 · `code_commit` 40 hex
* `input`: `sha256` H64 · `source_kind` ∈ {`encoded_file`,`model_input_tensor`} · `dims` = [w,h], integers ≥ 1 ·
  `phash` = `null` or `{algo:"phash64-v1", value:16 hex}` · `phash_omitted_reason` ∈ {`not_computed`,`sampled_out`,`unsupported_source`}
  — **required iff `phash` is null, absent otherwise**
* `output`: `jcs_sha256`, `decision_sha256`, `raw_jcs_sha256` (H64) · `payload_ref` REF
* `checkpoint`: `tree_size` integer ≥ 0 · `root_hash` H64
* `anchor`: `checkpoint_seq` integer ≥ 0 · `tree_size` integer ≥ 0 · `root_hash` H64 · `cosigner_key_ids` array of distinct H64 ·
  `medium` ∈ {`write_once`,`cosign`,`logbook`} · `label` TEXT(0,64)
* `rotation`: `new_key_id` H64 · `new_public_key` H64 (64 hex = 32 bytes) · `effective_seq` integer ≥ 1 · `new_key_pop` 128 hex;
  and `new_key_id` = SHA-256(`new_public_key` bytes)
* `gap`: `first_unsealed_utc`, `last_unsealed_utc` (timestamps, first ≤ last) · `reported_count` integer ≥ 0 ·
  `reason` TOKEN(64) · `spill_sha256` H64 **or** the string `unavailable`
* `scan`: `scan_id` IDENT(64) · `report_sha256` H64 · `profile_hash` H64 · `code_commit` 40 hex · `finding_counts` object,
  keys TOKEN(64), values integers ≥ 0
* `analyst`: `actor_id` IDENT(64) · `role` TOKEN(32) · `action` ∈ {assign, acknowledge, override, approve, quarantine, release} ·
  `scan_id` IDENT(64) · `target_type` ∈ {sample, contributor, batch, model, record, dataset} · `target_ref` TEXT(1,200) ·
  `finding_id` IDENT(128) · `justification` TEXT(0,8192) · `request_id` 32 hex · optional: `new_disposition` ∈ {accept, review, quarantine} ·
  `reason_code` ∈ {quality_issue, known_benign, insufficient_evidence, accepted_risk, superseded_by_rescan, other_with_justification,
  known_restore, test_data, superseded_ledger, false_positive_confirmed} · `assignee` IDENT(64) · `refs_seq` integer ≥ 0 · `expected_prev_seq`
  integer ≥ 0. `override` requires `new_disposition`, `reason_code`, `expected_prev_seq` and a `justification` that is not all whitespace;
  `approve` requires `refs_seq`; `assign` requires `assignee`.

### 4.3 Cross-field rules (valid regardless of chain position)

* `genesis`: `seq` = 0; `key_id` = `deployment_manifest.key_id`; `prev_record_hash` = `hex(SHA-256(0x03 ‖ canon(deployment_manifest)))`.
* every other type: `seq ≥ 1`.
* `key_rotation`: `rotation.effective_seq` = `seq + 1`.
* `anchor_event`: `anchor.checkpoint_seq` < `seq`.

A record violating any rule in §2 or §4 is **malformed**.

### 4.4 The inference record

`input.sha256` hashes the *encoded source bytes* (`source_kind = encoded_file`) or the model-input tensor
(`model_input_tensor`); the record states which. `config.preprocess_ref` and `output.payload_ref` are content
addresses of stored payloads (§7). `output.raw_jcs_sha256` hashes the raw, pre-filter output, `output.jcs_sha256`
the filtered decision object at fine precision, `output.decision_sha256` the *decision-only* object (§5).

## 5. Output objects (what the three output hashes cover)

All are canonical bytes of the quantised object, hashed with plain SHA-256.

```
classify   {"task":"classify","top":[{"cls":7,"conf_e6":993118}, …]}
detect     {"task":"detect","detections":[{"cls":3,"conf_e6":871204,"box_q64":[1204,880,3320,2411]}, …],
            "filter":{"conf_thr_e6":250000,"nms_iou_e6":450000}}            # "filter" only on the filtered form
```

* `raw_jcs_sha256` — the **raw** output, fine precision, **no** `filter` block.
* `jcs_sha256` — the **filtered** output, fine precision, with its `filter` block if the pipeline had one.
* `decision_sha256` — the filtered output with **`conf_e6` removed**, boxes as `box_px` (whole pixels), detections
  **sorted by `(cls, box_px)`** ascending (classification keeps its rank order), and the `filter` block kept:
  `{"task":"detect","detections":[{"cls":3,"box_px":[19,14,52,38]}, …],"filter":{…}}`,
  `{"task":"classify","top":[{"cls":7}, …]}`.
* Non-finite output anywhere: all three objects are `{"nonfinite":true}`.

The verifier does not recompute these from a model; it checks only that a stored payload hashes to the committed value (§8).

## 6. Signing, hashing, chaining

Let `canon(r)` be the canonical bytes of record `r` **with `signature` removed**, and `sig(r)` the 64 raw signature bytes.

```
record_hash(r) = SHA-256( canon(r) )
signature(r)   = Ed25519.sign( "cva-seal/1 record\n" ‖ canon(r) )        # pure Ed25519, NOT over record_hash
link(r)        = SHA-256( 0x02 ‖ record_hash(r) ‖ sig(r) )                # the chain covers the signature
r.prev_record_hash = hex( link(previous record) )                         # genesis: §4.3
leaf(r)        = SHA-256( 0x00 ‖ S )  where S = the STORED bytes of r, signature included
```

`"cva-seal/1 record\n"` is the 18 ASCII bytes including the final newline. Hash inputs beginning `0x00–0x03` are
domain-separated; `record_hash` inputs begin with `{` (`0x7B`) and cannot collide with them.

`link` is computed from the record's own *stored bytes* whether or not its signature verifies: a forged signature
must not change what the next record is compared against.

## 7. Payloads

`payload_ref = "sha256:" + h` names a payload whose bytes hash (plain SHA-256) to `h`. Payloads are not part of the
chain; the record's commitment to `h` protects them. A **payload sidecar** (§9.3) supplies them to a verifier that
has only an export.

## 8. The Merkle tree

RFC 6962 §2.1 exactly (also RFC 9162 §2.1, identical for SHA-256), over the leaf hashes of §6:

```
MTH([])   = SHA-256("")          = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
MTH([d])  = d
MTH(D[n]) = SHA-256( 0x01 ‖ MTH(D[0:k]) ‖ MTH(D[k:n]) ),  k = the largest power of two strictly less than n
```

The last node is **never** duplicated at an odd level (CVE-2012-2459). An inclusion proof for leaf `m` of `n` leaves
is the RFC 6962 §2.1.1 audit path, verified as RFC 9162 §2.1.3.2 — with `n` taken from the signed checkpoint or anchor,
never from the presenter of the proof.

A **checkpoint** record at index `i` states `tree_size = i` and `root_hash = MTH(leaf hashes of records 0…i−1)` — it
summarises the records *before* it, so its own index equals its `tree_size`.

## 9. Ledger exports and sidecars

### 9.1 The export

One record per line; each line is a record's stored bytes verbatim followed by `\n` — **including the last line**; no
blank lines; no byte-order mark; nothing else. The position of a line (0-based) is the position of its Merkle leaf.
A last line without `\n` is malformed. (Stricter than "JSON lines" on purpose: every byte of the file then matters, so
no single-bit change is invisible.)

### 9.2 The trust root — §10

### 9.3 The payload sidecar

One JSON object per line: `{"address":"<64 hex>","b64":"<standard base64 of the payload bytes>"}`. The address is the
SHA-256 the record committed to; a wrong payload simply fails to hash to it, so the sidecar needs no trust of its own.

## 10. The trust root

A JSON object `{"v":"cva-seal/1","deployment_manifest_hash":H64,"keys":[{"key_id","public_key","role"}, …]}`, with
`role` ∈ {`ledger`,`witness`,`boundary`}, `key_id` = SHA-256(`public_key`), at least one `ledger` key, no duplicate
`key_id`. It is supplied out of band and is **not self-signed**; its integrity is the operator's responsibility.

* `deployment_manifest_hash` is what genesis must commit to (§4.3): a valid chain from another device fails here.
* `ledger` keys may sign records; the genesis key MUST be one of them. A key introduced by a valid rotation (§12) is
  also a ledger key for as long as the chain says.
* `witness` keys may cosign anchors; `boundary` keys may attest to them. Roles are separate authorities.

## 11. Anchors

An **anchor** fixes what the first `tree_size + 1` records were (the tree *and* the checkpoint record that states it):

```json
{ "v": "cva-seal/1",
  "checkpoint": <the full signed checkpoint record, as an object>,
  "cosignatures": [ {"key_id": H64, "sig": 128 hex} ],
  "attestations": [ {"v","kind":"witness_statement","root_hash","tree_size","observed_at_utc","key_id","sig"} ] }
```

* `cosignature.sig = Ed25519(witness key, "cva-seal/1 cosign\n" ‖ S)` where `S` = the **stored bytes** of the checkpoint
  record (canonical, signature included). `key_id` MUST be a `witness` key of the trust root.
* `attestation.sig = Ed25519(boundary key, "cva-seal/1 witness\n" ‖ canon(statement without "sig"))`; `root_hash` and
  `tree_size` MUST equal the checkpoint's; `key_id` MUST be a `boundary` key; `observed_at_utc` a valid timestamp.
  It is the only thing that bounds absolute time: the first `tree_size` records existed by the earliest such time.
* An anchor is **valid** iff: the checkpoint is a well-formed `checkpoint` record whose `seq` equals its `tree_size`
  and whose signature verifies under a ledger key (§10); **and every** cosignature **and every** attestation present
  verifies. One bad element invalidates the whole anchor.

A **logbook anchor** is just `{"kind":"logbook","tree_size":n,"root_hash":H64}` (no signatures — it was copied from a
physical book). It fixes `n` leaves, not a checkpoint record. Its transcription checksum is the first 8 hex of
`SHA-256("cva-seal/1 anchor-print\n" ‖ decimal(tree_size) ‖ "\n" ‖ root_hash)`.

An anchor is the *only* in-band-independent evidence of tail truncation or of a wholesale rewrite by the key holder.
Records after the newest verified anchor still trust the key holder; that count is the **unwitnessed window** and
MUST be reported.

## 12. Key rotation

The **active key** starts as the genesis key. A record MUST be signed by the active key of its position. A
`key_rotation` record `R` at seq `s`:

1. is signed by the **outgoing** (active) key;
2. carries `rotation.new_public_key` and `rotation.new_key_pop = Ed25519(new key, "cva-seal/1 rotation-pop\n" ‖ canon({"new_key_id":…,"effective_seq":…,"prev_key_id":R.key_id}))`
   — a proof that whoever rotates *holds* the new key, bound to the outgoing key and to this position;
3. makes the new key active from `effective_seq = s + 1`.

If the proof of possession does not verify, the rotation is *not applied* (finding, §13 step 8); records later signed by
the key it named are consequences of that one finding. **Not covered:** a compromised key can validly rotate to a key its
thief controls; emergency revocation needs an out-of-band trust-root update.

## 13. The verification procedure (normative)

Input: the export (§9.1) as a sequence of lines `L[0…n−1]`, the trust root, and optionally anchors, an expected
deployment manifest, an expected inference count, a reference model manifest (model id → allowed weight digests),
payloads, and an input resolver. Output: an ordered list of **findings** `(attack_class, seq, severity, nature)`.

Severities: `critical`, `high`, `medium`, `low`, `info`. The class table:

| class | severity | nature | meaning |
|---|---|---|---|
| `record_edit` | critical | indeterminate | stored bytes differ from what was signed |
| `record_replace`* | high | indeterminate | (reserved; reported as `key_unauthorised` / `record_edit`) |
| `record_replay` | high | indeterminate | byte-identical copy of an earlier record |
| `record_delete` | high | indeterminate | `seq` gap whose missing numbers do not appear later |
| `record_reorder` | high | indeterminate | out-of-order / reused `seq` |
| `input_swap` | critical | indeterminate | resolved input does not hash to `input.sha256` |
| `model_swap` | critical | indeterminate | model digest ≠ its registration / the reference manifest |
| `output_payload_mismatch` | high | indeterminate | payload does not hash to the committed address (or `payload_ref` ≠ `jcs_sha256`) |
| `tail_truncation` | high | indeterminate | an anchor fixes more records than the ledger holds |
| `ledger_fork` | critical | adversarial | two valid histories: anchor root ≠ ledger root, all signatures valid |
| `key_unauthorised` | high | indeterminate; **adversarial** iff the signing key is a *known* key (trust root or rotated-in) and its signature verifies | signed by a non-active key, or a rotation without valid proof |
| `chain_broken` | high | indeterminate | `prev_record_hash` ≠ link of the previous record |
| `nonce_reuse` | high | indeterminate | nonce repeated within an anchoring interval |
| `genesis_mismatch` | high | indeterminate | first record not a genesis, or commits to another deployment |
| `checkpoint_mismatch` | high | indeterminate | checkpoint / anchor event states a root or size the ledger does not have |
| `ledger_incomplete` | high | indeterminate | inference count ≠ the operator's independent count |
| `non_canonical_encoding` | high | indeterminate | valid record, wrong bytes |
| `malformed_record` | high | indeterminate | not a valid record at all (§2, §4) |
| `anchor_invalid` | medium | indeterminate | a supplied anchor that fails its own checks |
| `degraded_gap` | low | quality | a *declared* hole (`degraded_marker`) — evidence, not tampering |
| `clock_regression` | info | quality | `created_at_utc` went backwards — never a tamper signal |

`record_replace` is listed for completeness; a conforming verifier does not emit it.

Findings are reported at the **first** break; its downstream consequences are **folded** into it (a count, not more
findings). The procedure below is exact about what folds. Let `p` be the 0-based position of a line, `leaf(L[p])`
always contributes to the running Merkle accumulator *whatever* the line contains.

**State:** `shift = 0`; `pending = {}` (seq → the reorder finding that explains its absence); `prev_link = none`;
`active_key = none`; `known_keys` = the trust root's keys ∪ keys from valid rotations; `rejected = {}` (key id → the
finding of the rotation that named it); `nonces = ∅`; `verified_cp = {}` (seq → (tree_size, root)); `last_primary =
none` with `last_primary_pos`; `tree_taint = none` (first *tainting* finding); `registrations = {}`.
*Tainting* classes: `record_edit, record_delete, record_reorder, record_replay, ledger_fork, malformed_record,
non_canonical_encoding, key_unauthorised`. A **primary** finding sets `last_primary`/`last_primary_pos = p`.

For each `p`:

**1 PARSE.** If the line lacks its terminating `\n` (last line only) or fails strict parse: emit `malformed_record`
(or `non_canonical_encoding` when it is valid but not canonical); `prev_link = none`; go to the next line. Otherwise
the record `r` is parsed and schema-valid, and `link(r)`, `record_hash(r)` are computed from its stored bytes.

**3 SEQ.** `expected = p + shift`. If `r.seq ≠ expected`:
* if `r.seq ∈ pending`: it is the displaced record a reorder explained — fold into that finding, remove it from `pending`;
  if `pending` is now empty, `shift` returns to its value before that reorder; the link check (step 6) is suppressed;
* else if `r.seq > expected`: `missing = expected … r.seq−1`. If **every** missing number is the `seq` of some **later**
  line: emit `record_reorder` (seq = `r.seq`), `pending[m]` = it for each missing `m`, `shift = r.seq − p`. Otherwise emit
  `record_delete` (seq = `missing[0]`), `shift = r.seq − p`; suppress the link check;
* else (`r.seq < expected`): let `earlier` = positions `< p` holding the same `seq`. If there are some: if `L[p]` is
  byte-identical to the first of them → `record_replay`; else if `active_key` exists and the signature verifies under
  it → `ledger_fork` (adversarial); else `record_reorder`. Then `shift −= 1`, suppress the link check, and remember
  whether it was a byte-identical replay. If there are none: `record_reorder`, suppress the link check.

**genesis.** If `p = 0` and `r.type ≠ genesis`: `genesis_mismatch`.

**4–5 KEY and SIGNATURE.** If `p = 0`: `active_key = r.key_id`, and if that key is not a `ledger` key of the trust
root: `key_unauthorised` (indeterminate). Else if `r.key_id ∈ rejected`: fold into that rotation's finding, the
signature is considered not verified. Else if `r.key_id ≠ active_key`: if the key is known and the signature verifies
under it → `key_unauthorised` (adversarial); known but the signature fails → `record_edit`; unknown →
`key_unauthorised` (indeterminate); and in the two `key_unauthorised` cases, if `r` is a `key_rotation`, register its
`new_key_id` in `rejected`. Else (`r.key_id = active_key`): verify the signature under `active_key`; on failure, if
`last_primary` is a `record_edit` at position `p−1`, fold into it (a *run* of signature failures is one finding, its
finding's `seq` unchanged) and set `last_primary_pos = p`; otherwise emit `record_edit`.

**6 LINK.** `p = 0`: `r.prev_record_hash` must equal `trust_root.deployment_manifest_hash` else `genesis_mismatch`
(if an expected deployment is supplied it must equal too, else a non-primary `genesis_mismatch`). `p > 0`: if
`prev_link` is none or `r.prev_record_hash ≠ prev_link`: if the link check is suppressed, or `last_primary_pos ∈ {p, p−1}`,
fold into `last_primary`; else emit `chain_broken`.

**7 NONCE** (only if the signature verified). If `r.nonce ∈ nonces` and `r` was not a byte-identical replay: `nonce_reuse`.
Add it.

**8 TYPE RULES** (only if the signature verified):
* `checkpoint`: if `tree_size` equals the number of leaves so far **and** `root_hash` equals `MTH` of them: record in
  `verified_cp[r.seq]`. Otherwise fold into `tree_taint` if it exists, else emit `checkpoint_mismatch`.
* `anchor_event`: if `verified_cp[anchor.checkpoint_seq]` is absent or ≠ (`tree_size`,`root_hash`): fold into `tree_taint`
  if it exists, else non-primary `checkpoint_mismatch`. Then `nonces = {r.nonce}` (an anchor closes the nonce window).
* `degraded_marker`: emit `degraded_gap` (non-primary).
* `model_registration`: if a reference manifest is supplied and (`model.id` unlisted or its digest not allowed) →
  `model_swap`; else if the same `model.id` was registered earlier with a different digest → a **non-primary `model_swap`
  at severity `info`** (a legitimate reload and a swap are indistinguishable without a reference). Record the model.
* `inference`: if its `model.id` has no registration → `model_swap`; else if (`weights_sha256`, `arch_hash`, `format`)
  differ from the registration in effect → `model_swap`.
* `key_rotation`: if the proof of possession (§12) fails: `key_unauthorised` (indeterminate), `rejected[new_key_id]` = it,
  the rotation is not applied. Otherwise `active_key = new_key_id`, add the key to `known_keys`.

**9 TIME.** If `created_at_utc` < the previous record's: non-primary `clock_regression` at `info`.
Finally `prev_link = link(r)`.

**After the last line.**
1. **Anchors,** in the order supplied. For each: if it is not parseable or not valid (§11, with the rotated-in ledger
   keys of *this* ledger accepted as ledger keys) → `anchor_invalid`; skip it. Let `t = tree_size` and `covered = t + 1`
   for a checkpoint anchor (`t` for a logbook anchor). If `n < covered` → `tail_truncation`. Else compute `MTH` of the
   first `t` leaves: if it ≠ the anchor's root → fold into `tree_taint` **iff that finding's position is `< t`**,
   otherwise `ledger_fork` (adversarial). Else, for a checkpoint anchor, if `L[t]` is not byte-identical to the anchor's
   checkpoint's canonical bytes → the same fold-or-`ledger_fork` rule (position `< t+1`). Else the anchor is *verified*;
   the strongest verified anchor (largest `covered`) sets the **unwitnessed window** = `n − covered`. (Without a verified
   anchor: the window is the records after the last `anchor_event` in the chain — written by the key holder itself.)
2. **Count.** If an expected inference count is given and the number of `inference` records differs: `ledger_incomplete`.
3. **Per inference record whose signature verified,** in order: if an input resolver is supplied and returns bytes whose
   SHA-256 ≠ `input.sha256` → `input_swap`. Then payloads: if `output.payload_ref` ≠ `"sha256:" + output.jcs_sha256` →
   `output_payload_mismatch`; for each distinct address among {`jcs_sha256`, `raw_jcs_sha256`}: if a payload is supplied
   and its SHA-256 ≠ the address → `output_payload_mismatch`. A payload not supplied is *not a finding* (it is counted
   as missing).
4. An empty export: `genesis_mismatch` (a ledger with no records is not a verified ledger).

### 13.1 Conformance

A verifier conforms if, for every frozen vector, every tamper artefact of the reference lab, and the clean ledgers, it
produces the **same ordered list of `attack_class` values above `info`** and, for the first finding, the same `seq`,
severity and nature as the reference. It MAY report more detail; it MUST NOT report fewer findings, more findings, or a
different class for the same physical change.

**What a verifier MUST say aloud** in every report: the number of records after the newest verified anchor (the
unwitnessed window); that with *no* anchor, tail truncation and a wholesale rewrite by the key holder are not excluded;
and that the trust root was taken on trust.

## 14. Known limits (each has a test that asserts it)

Tail truncation and split-view *after the last anchor*; selective logging (records never submitted); an adversary holding
the signing key from the start; a compromised key rotating to an attacker's key; tampering before the record was sealed;
the host clock; the trust root's integrity.

## Appendix — vectors

> **What the vectors are, and are not.** `cva_seal_v1.json` is generated *from the reference implementation*, so
> regenerating it proves determinism, not conformance to this text: a divergence between the reference and the spec would be
> frozen into the vectors as if it were right. Checking your verifier against them checks it against our implementation's
> output. The independent evidence is a reader written from this text alone — see [the verification procedure](../docs/provenance/VERIFICATION-PROCEDURE.md) §2 — which
> reproduced every check in the file.


`spec/vectors/cva_seal_v1.json`: `keys` (seeds → public keys → key ids) · `trust_root` · `quantise` · `outputs` (raw and
filtered objects → the three canonical objects → hashes) · `records` (a 20-record ledger: genesis, registration, five
classifications, four detections, a degraded marker, a rotation, three more inferences, checkpoints, an anchor event) ·
`record_hashes` · `leaf_hashes` · `roots` (root for every prefix) · `anchor` (cosigned and attested) · `inclusion_proof` ·
`verify` (the expected summary). Plus RFC 8032 §7.1, RFC 8785 and certificate-transparency vectors for the primitives:
`rfc8032.json`, `rfc8785/`, `merkle_ct.json`.
