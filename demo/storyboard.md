# Video storyboard

Recorded from a **real cold run** on the air-gapped-configured machine (plan §7.11), in the
same order as `script.md`. **Every caption states what was checked and what was not.** A
caption that only celebrates is the video equivalent of a report with no coverage section.

| # | Time | On screen | Caption (checked / not checked) |
|---|---|---|---|
| 0 | 0:00–0:45 | terminal: `docker run --network none … selftest`, exit 0 | *Checked:* no network egress, proven by the kernel and by an in-process guard. *Not checked:* anything about data yet. |
| 1 | 0:45–2:00 | scan list, new **sealed** row → overview → provenance section | *Checked:* the demo corpus under the baseline profile; the report is sealed in the ledger. *Not checked:* the checks listed as UNAVAILABLE on the overview, each with its reason. |
| 2 | 2:00–3:45 | Contributor risk: `guilty` interval clear of the cohort rate; `tiny` (3/5) vs `big` (12/150) | *Checked:* near-duplicate flooding, aggregated per contributor as a posterior with a 95% interval. *Not checked:* poisoning that leaves no regularity in the spaces measured. |
| 3 | 3:45–5:30 | Findings: Module B finding and the reconstructed trigger (or STRIP, per the spike) | *Checked:* the patch-trigger family we train against. *Not checked:* triggers outside our attack families; detection rates do not generalise. |
| 4 | 5:30–7:00 | editor: one character changed → `verify` exit 2, `record_edit` at seq N. Then the tail deleted → `verify` clean | *Checked:* edits, interior deletions, reordering, replay. *Not checked:* **tail truncation without an anchor**, which is shown live. |
| 5 | 7:00–8:30 | two windows: lower → **pending** → self-approve refused → second person approves → timeline → Verify green → punctuation-only justification refused | *Checked:* four-eyes and justification are enforced, and every decision is signed and ordered by seq. *Not checked:* who was at the keyboard; attribution is by the tool, not the analyst's key. |
| 6 | 8:30–9:30 | Coverage → Compare: the TorchScript bar is visibly shorter, with the table of classes it gave up | *Checked:* what each access level allowed. *Not checked:* everything in "covered by nothing", plus the standing limitations printed below it. |
| — | 9:30–10:00 | the standing limitations panel | the invariant, read aloud |

**Recording rules**

- Anything shown from a prior run has **"recorded run"** burned into the corner for the
  whole time it is on screen.
- No cut hides a failure. A retake is a full retake of the step, and the rehearsal log says
  why it was needed.
- The terminal font is large enough to read the `seq` numbers at 1080p. They are the
  evidence.
