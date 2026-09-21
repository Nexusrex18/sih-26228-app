# Demo script: 10 minutes, every step real

This extends the team's five-step `Demo-Script.md` with the two additions the module plans
settled on: **step 0**, the air gap watched rather than asserted, and **the honest limit in
step 4**. It also adds one step only the dashboard can show, **step 5**, a decision that
needs two people.

**Rules** (plan §7.11, D-E13):

- **Never fake a step.** A staged demo of an integrity tool refutes itself.
- The live corpus is sized to what the CPU scans inside the time slot. Anything bigger is
  shown from a prior real run, **and labelled "recorded run" on screen**.
- Every step is rehearsed on data the detectors were not tuned on. `rehearsal-log.md` records
  the timings and every place it went wrong.
- **Wording:** the tool never says "clean", "fixed" or "safe" about data. It says what was
  checked. Presenters say the same. In step 6 in particular, the black-box statement is
  *smaller*, not *cleaner*.

## Before the audience arrives (not timed)

```sh
python -m cva.fixtures                              # demo corpus + model
python -m cva.fixtures --kind mixed_contributors    # step 2's contributors
python -m attacklab --out artifacts/corpus          # step 3's backdoored models
cva-seal keygen --out keys/signing.key
cva-seal init --ledger ledger/audit.db --key keys/signing.key \
  --device-id demo-host --unit demo --trust-out ledger/trust_root.json
python -m cva.web.accounts create a.sharma --role analyst
python -m cva.web.accounts create b.rao --role approver
```

Start `cva-ledgerd` and `cva-web` (SETUP.md §6). Open two browser windows, one signed in as
`a.sharma` and one (private window) as `b.rao`. `attacklab` takes minutes, so it is run
beforehand and its models are the recorded artefacts step 3 scans live.

---

## 0. Unplug the network (0:45)

```sh
docker run --rm --network none --read-only --tmpfs /tmp --cap-drop ALL \
  --security-opt no-new-privileges cva:latest selftest
```

**Say:** "This container has no network interface except loopback. The kernel guarantees
that. The selftest also arms its own egress guard, which raises if anything tries to reach
out. Two independent proofs, and both pass."

**Show:** exit status 0 and the selftest summary.

## 1. Clean pass (1:15)

```sh
cva scan --dataset artifacts/fixtures/demo_coco --model artifacts/fixtures/demo_model.onnx \
  --profile baseline --out reports --audit-ledger-socket run/ledgerd.sock
```

**Show:** the scan list (a new row with a **sealed** badge), then the overview.

**Say:** "Green. And here, the provenance section is present even though nothing is wrong.
A report with no provenance section can't be told apart from one where provenance was never
checked, so it is always shown."

## 2. Inject poison, catch the contributor (1:45)

```sh
cva scan --dataset artifacts/fixtures/mixed_contributors --profile baseline --out reports \
  --audit-ledger-socket run/ledgerd.sock
```

**Show:** **Contributor risk** first, before any finding. `guilty` stands out with its
interval clear of the cohort rate.

**Say:** "Look at `tiny`: 3 flagged out of 5 is a 60% raw rate. `big` is 12 out of 150, 8%.
Ranked by raw rate, `tiny` looks worse. The posterior, with its interval, says five images
are not enough to conclude that. The acceptance officer acts on this table, so it comes
before the individual findings."

## 3. A backdoored model (1:45)

```sh
cva scan artifacts/corpus/models/bd_patch_08.onnx --corpus artifacts/corpus \
  --profile deep --out reports --audit-ledger-socket run/ledgerd.sock
```

**Show:** the Module B finding and its evidence, the reconstructed trigger.

**Contingency, decided at the Module B spike and not on the day:** if trigger
reconstruction is infeasible at the demo's class count, this step shows the STRIP entropy
collapse instead. Whichever is shown is the one the rehearsal log recorded working.

## 4. Hand-edit a ledger record, then point at our own limit (1:30)

```sh
cva-seal export --ledger ledger/audit.db --out /tmp/demo.export.jsonl
```

Open `/tmp/demo.export.jsonl` in an editor and change **one character** inside one
record's payload. Then:

```sh
cva-seal verify --records /tmp/demo.export.jsonl --trust ledger/trust_root.json
```

**Show:** exit status 2 and a `record_edit` finding naming the exact `seq`.

**Then the honest part.** Restore the file, delete the **last** five lines, and verify again:

```sh
cva-seal verify --records /tmp/demo.export.jsonl --trust ledger/trust_root.json
```

**Say:** "Clean. Deleting the tail of a ledger is invisible without an external anchor
fixed at a past moment. This build doesn't take anchors, and the dashboard says so: every
record here is in what it calls the *unwitnessed window*. Pointing at the edge of our own
claim is worth more than a demo with no edges."

## 5. An analyst decision that needs two people (1:30)

In `a.sharma`'s window: **Findings** → a `review` finding → **Open and decide** → **Lower
to accept**, reason code `quality_issue`, justification "Confirmed duplicate cluster from a
scanner re-export, see batch note 14."

**Show:** the finding is now **pending**. The tool's disposition is still shown beside it.

Try to approve it as `a.sharma`: **refused**, you cannot approve your own change. In
`b.rao`'s window: approve it. **Audit trail**: both events are in `seq` order with the
justification in full. **Verify now**: green.

Then, as `a.sharma`, try a lowering whose justification is forty full stops. It clears
the 30-character floor and contains no words, so it is **refused**: the rule counts real
words, because punctuation records nothing. The counter under the field shows this as you
type, and the refusal says `nothing changed`.

## 6. The final report, and a statement that shrinks (1:00)

Scan the same model as a frozen TorchScript archive, which exposes no weights:

```sh
cva scan artifacts/corpus/models/clean_a_frozen.ts --corpus artifacts/corpus \
  --profile deep --out reports --audit-ledger-socket run/ledgerd.sock
```

**Show:** **Coverage** for the ONNX scan, then **Compare** with the TorchScript scan. The
second bar is visibly shorter, and the table lists every attack class it gave up and the
check it would have needed.

**Say:** "Same model, less access, so a smaller statement. That's correct: it lists
exactly what could not be checked, and it never presents a shorter report as a cleaner
result."

## Buffer (1:00)
