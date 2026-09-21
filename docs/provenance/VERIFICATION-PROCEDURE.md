# Verification procedure — for a party who does not trust our code

> *Published with the code it describes. The Module C plan, build-decisions log and coverage statement it refers to are kept in the project's planning repository.*

You have been handed a ledger and want to know whether it is intact. You should not have to trust our software, our
word, or our network. This procedure needs only the artefacts below, a computer, and — if you like — a verifier you
wrote yourself from [the specification](../../spec/cva-seal-spec-v1.md).

## 1. What you must be given (and what each is for)

| Artefact | Purpose | Trust |
|---|---|---|
| `ledger.export.jsonl` | the records, one per line, exactly as stored | none needed — it is what is being checked |
| `trust_root.json` | the public keys and the deployment identity | **taken on trust**: obtain it out of band and check its SHA-256 against a value you got separately |
| one or more **anchors** | a signed root fixed at a past moment, ideally cosigned/attested | vouched by the witness / boundary keys in the trust root |
| `ledger.payloads` (optional) | the stored outputs | none — each must hash to what the record committed to |
| an independent inference count (optional) | to detect records never sealed | yours |

Without an anchor you can still detect edits, deletions from the middle, reordering, replay and forgeries — but **not**
removal of the last records, nor a wholesale rewrite by whoever holds the signing key. Say so in your report.

## 2. Pick a verifier (you have three choices)

1. **The reference:** (`pip install -e .` provides the `cva-seal` command; `python -m cva.provenance.seal.cli` is equivalent) `cva-seal verify --records ledger.export.jsonl --trust trust_root.json --anchor A.json`.
2. **The independent implementation:** `python spec/independent_verifier.py ledger.export.jsonl --trust trust_root.json --anchor A.json`
   — under 900 lines, standard library plus `cryptography`, imports nothing from us. *Qualification, stated honestly:* it was
   written by the same author as the reference verifier, from the spec text but after reading the reference, so its agreement
   shows the spec is self-consistent, not that a stranger would reproduce it. That gap was closed for the wire format (§2–§12)
   by a third reader, written by a different person from the spec text alone, which reproduced all 75 frozen-vector checks
   (all 20 prefix roots and signatures, the rotation proof, the checkpoints, the anchor, the inclusion proof, the exact-half
   quantisation cases). The §13 findings procedure has *not* been independently re-derived.
3. **Your own**, written from [the specification](../../spec/cva-seal-spec-v1.md) §13. The spec is normative and complete for this purpose; the frozen
   vectors (`spec/vectors/cva_seal_v1.json`) and the tamper artefacts tell you when yours agrees.

Whichever you use, first confirm it is sound on **known answers**: `cva-seal selftest` (reference), or check your own
against the vectors (RFC 8032 §7.1, the RFC 6962 tree vectors, `cva_seal_v1.json`).

## 3. Run the checks

Run the verifier as above. Read the result in this order.

1. **Exit status / verdict.** `CLEAN` means no finding above *information*. `INTACT, WITH DECLARED GAPS` means the only finding is a `degraded_gap` — a hole the ledger itself declared, evidence rather than tampering (the exit status is still 2, because a declared hole is above *information*). It does **not** mean "genuine input", "the
   model was good", or "nothing was ever removed after the last anchor".
2. **Each finding** has a class and a `seq`. The class names *which arithmetic failed*, not who did it:

   | class | what failed |
   |---|---|
   | `record_edit` | a record's bytes differ from what was signed |
   | `record_delete` / `record_reorder` / `record_replay` | the `seq` numbers show a gap, a shuffle or a copy |
   | `chain_broken` | a record does not chain to its predecessor (usually folded into one of the above) |
   | `key_unauthorised` | signed by a key that was not the active one; or a key rotation that did not prove possession |
   | `nonce_reuse` | one random value used twice within an anchoring interval |
   | `genesis_mismatch` | the history belongs to another device, or does not begin with a genesis record |
   | `checkpoint_mismatch` | a stated Merkle root does not match the records |
   | `ledger_fork` | *(needs an anchor)* two valid histories exist; only the key holder could have signed both |
   | `tail_truncation` | *(needs an anchor)* the anchor fixes more records than the ledger holds |
   | `input_swap` / `model_swap` / `output_payload_mismatch` | the artefact a record points at is not what was sealed |
   | `ledger_incomplete` | *(needs your count)* fewer records than inferences that happened |
   | `anchor_invalid` | a supplied anchor failed its own signatures — it says nothing either way |
   | `degraded_gap` (low) | a **declared** hole: evidence, not tampering |
   | `clock_regression` (info) | the clock stepped back: never a tamper signal |

   One physical change produces **one** finding; its downstream consequences are folded in and counted
   ("`+N folded`"). Fewer, precise findings are a feature.
3. **`nature`.** `adversarial` is printed only when the arithmetic proves a *known* key signed something it should
   not have (or two histories carry that key's signature). Everything else is `indeterminate`: the arithmetic proves
   *that* the ledger differs from what the key holder signed, not *why* — storage corruption looks identical to an edit.
4. **The unwitnessed window.** Note how many records lie after the newest verified anchor, and the time bound
   (`sealed_not_after_utc`) if a witness statement exists. That is the part of the history that still trusts the key holder.

## 4. Look at one record

`cva-seal explain --records X.jsonl --trust T --seq N [--anchor A]` prints, in words: what the record is, whether its
signature verifies, whether it links to its predecessor, whether an anchor covers it, and the findings about it.
To prove **one** record was in the anchored root using only that record, the proof and the anchor:

```
cva-seal proof inclusion --records X.jsonl --seq N --anchor A.json --out p.json      # by the ledger holder
cva-seal proof verify    --proof p.json --anchor A.json --trust trust_root.json       # by you, no ledger needed
```

**Limit — read this before relying on the ledger-free form.** The trust root vouches only for the *genesis* key; a key
introduced by a later `key_rotation` is vouched for by the rotation chain *inside the ledger* ([the specification](../../spec/cva-seal-spec-v1.md) §10, §12).
So `proof verify` works with no ledger **only while the anchor's checkpoint was signed by the genesis key** — that is, an
anchor taken before the first rotation. For an anchor signed by a rotated-in key, `proof verify` correctly reports
`valid: false` ("the checkpoint is signed by key …, which is not a ledger key in the trust root"), because nothing you were
handed establishes that key. In that case verify the anchor **together with the ledger**
(`cva-seal verify --records X.jsonl --trust T --anchor A.json`, which follows the rotation chain), or ask the ledger holder
for an anchor from before the rotation. The tool is not wrong to refuse; the ledger-free promise simply stops at a rotation.

## 5. By hand (if you distrust the tools entirely)

For record *k* with stored bytes *B*: parse *B* and re-serialise canonically ([the specification](../../spec/cva-seal-spec-v1.md) §2.2) — it must equal
*B*. Remove the `signature` field, canonicalise → *C*. Verify the Ed25519 signature over `"cva-seal/1 record\n" ‖ C` with
the active key. Compute `link = SHA-256(0x02 ‖ SHA-256(C) ‖ signature bytes)`; record *k+1*'s `prev_record_hash` must be that
value. Leaf *k* = `SHA-256(0x00 ‖ B)`; the RFC 6962 root of leaves `0…n−1` must equal a checkpoint's `root_hash`. Any tool with
SHA-256 and Ed25519 can do each step.

## 6. What to write in your report

* the artefacts you were given, and how you obtained and checked the trust root;
* which verifier(s) you ran, on which version, and that the known-answer check passed;
* the findings, verbatim, with their classes and seqs;
* the unwitnessed window, and whether an anchor was supplied and by whom it was vouched;
* the standing limits (the coverage statement): tampering *before* sealing, selective logging without a counter, an
  adversary holding the signing key from the start, tail truncation after the last anchor, trust-root integrity.

**Do not write** "the data is authentic". Write what the arithmetic showed and what it cannot show.
