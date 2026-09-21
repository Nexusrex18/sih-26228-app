# Verification procedure: for a party who does not trust our code

You have been handed an audit ledger and want to know whether it is intact. You should not
have to trust our software, our word or our network. This procedure needs the artefacts
below and a computer. Section 4 can be done without any of our code.

This page covers the audit ledger the **dashboard** writes. The full procedure for any
`cva-seal` ledger — anchors, inclusion proofs, `explain`, and the independent verifier in
`spec/independent_verifier.py` — is Module C's:
[`docs/provenance/VERIFICATION-PROCEDURE.md`](provenance/VERIFICATION-PROCEDURE.md).

## 1. What you must be given

| Artefact | Purpose | Trust |
|---|---|---|
| `audit.export.jsonl` | every record, one per line, exactly as stored | none needed: it is what is being checked |
| `trust_root.json` | the public key and the deployment identity | **taken on trust**: get its sha256 from someone other than whoever gave you the file, and compare |
| the report `report.json` (optional) | a scan whose seal you want to check | none: its digest is what a `scan_record` commits to |

The exporter writes the JSONL with `cva-seal export`. The dashboard's **Verification**
page offers the same export to an approver or admin.

## 2. What verification can and cannot tell you (read this first)

Without an external anchor you **can** detect edits, deletions from the middle, reordering,
replay and records signed by a key the trust root does not vouch for. You **cannot** detect:

- removal of the **last** records (tail truncation). Only an anchor fixed at a past moment
  bounds that;
- a wholesale rewrite by whoever holds the signing key. It verifies;
- tampering before a record was sealed. A perfectly sealed record of a doctored input is
  still a perfectly sealed record;
- records that were never written (selective logging), unless you have an independent
  count, which you can pass as `--expected-count`.

The dashboard verifies without an anchor and shows the **unwitnessed window**: how many
records still trust the key holder alone. To bound it, pass an anchor with `--anchor`
(`cva-seal anchor export` makes one); Module C's procedure covers the ceremony.

## 3. Run the reference verifier

```sh
cva-seal verify --records audit.export.jsonl --trust trust_root.json
cva-seal verify --records audit.export.jsonl --trust trust_root.json --json
```

`cva-seal` is installed by `pip install -e .`. `python -m cva.provenance.seal.cli` is
equivalent. The dashboard runs exactly this command, with a fixed argument list, and shows
it on the Verification page so you can run it yourself and compare.

**Exit status:** `0` means no finding above *information*. `2` means there are findings,
listed with their class and `seq`. `1` means the ledger could not be read at all. `1` is
not the same as `2`: one says *we did not check*, the other says *we checked and it is
wrong*.

If you know how many records there should be, from an independent counter:

```sh
cva-seal verify --records audit.export.jsonl --trust trust_root.json --expected-count 412
```

**Each finding's class names which arithmetic failed, not who did it:**

| class | what failed |
|---|---|
| `record_edit` | a record's bytes differ from what was signed |
| `record_delete` / `record_reorder` / `record_replay` | the `seq` numbers show a gap, a shuffle or a copy |
| `key_unauthorised` | signed by a key the trust root does not vouch for |
| `genesis_mismatch` | the history belongs to another deployment, or does not begin with genesis |
| `checkpoint_mismatch` | a stated Merkle root does not match the records |
| `ledger_incomplete` | fewer records than your independent count |
| `degraded_gap` (low) | a hole the ledger **itself declared**: evidence, not tampering |
| `clock_regression` (info) | the clock stepped back: never a tamper signal |

`nature: adversarial` is printed only when the arithmetic proves a known key signed
something it should not have. Everything else is `indeterminate`, because storage
corruption and an edit look identical to arithmetic.

## 4. By hand, with no code of ours

The wire format is specified in Module C's `cva-seal-spec-v1.md`. For record *k* with
stored bytes *B*:

1. Parse *B* and re-serialise it canonically (RFC 8785 JCS). The result must equal *B*.
2. Remove the `signature` field and canonicalise the rest. Call the result *C*.
3. Verify the Ed25519 signature over `"cva-seal/1 record\n" ‖ C` with the key the trust
   root names.
4. Compute `link = SHA-256(0x02 ‖ SHA-256(C) ‖ signature bytes)`. Record *k+1*'s
   `prev_record_hash` must equal it.
5. Leaf *k* = `SHA-256(0x00 ‖ B)`. The RFC 6962 Merkle root of leaves `0…n−1` must equal a
   checkpoint's `root_hash`.

Any tool with SHA-256 and Ed25519 can do each step. Before you trust your own
implementation, check it against the known-answer vectors in `spec/vectors/`: RFC 8032,
the RFC 6962 tree, and RFC 8785 canonicalisation.

## 5. Checking one report's seal

A `scan_record` commits to the sha256 of the `report.json` it seals. Compute it:

```sh
sha256sum report.json
```

Find the `scan_record` for that `scan_id` in the export and compare `report_sha256`. The
dashboard shows the result as a badge with three states: **sealed** (the digests match),
**differs** (they do not, which means the report changed after sealing) and **not sealed**
(no record exists). *Differs* and *not sealed* are different facts, and the badge never
merges them.

## 6. What to write in your report

- the artefacts you were given, and how you obtained and checked the trust root;
- which verifier you ran, and on which version;
- the findings verbatim, with their classes and `seq`s;
- that no anchor was checked, and so the whole history is inside the unwitnessed window;
- the standing limits: tampering before sealing, selective logging without a counter, an
  adversary holding the signing key from the start, tail truncation, trust-root integrity.

**Do not write "the data is authentic."** Write what the arithmetic showed and what it
cannot show.
