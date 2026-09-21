# Operator manual: for analysts and approvers

How to read a report, what each state means, and how to take a decision that will still
hold up when someone asks you about it later. The dashboard is at
<http://127.0.0.1:8713/app/>.

## The banner at the top of every page

It reports whether the audit ledger, the record of every decision, can be trusted right
now. It is above the content on every page because nothing below it is worth more than it.

| Banner | Meaning | What you can do |
|---|---|---|
| **Audit ledger verified** | `cva-seal verify` ran and found nothing above *information* | read and decide |
| **Audit ledger not verified** | verification has not run, or could not | read. Press **Verify now** before deciding |
| **verification FAILED** | the ledger does not verify | **stop.** Decisions shown may not be the ones that were signed. Follow `VERIFICATION-PROCEDURE.md` |
| **Workflow unavailable** | `cva-ledgerd` is unreachable | read. New decisions are refused, never queued |

## Reading a scan, in order

The pages follow the report's fixed read order, so the dashboard and the single-file report
tell the same story in the same sequence.

1. **Overview**: the verdict and the **access assumptions**, meaning what the scanner could
   and could not see. Read the second before the first. A green verdict from a scan that
   could not see the model's weights is a smaller claim than one from a scan that could.
2. **Contributor risk**: before individual findings, because it is usually what you act on.
   Each row shows a **posterior with a 95% interval**, never a raw flag rate: 3 flagged out
   of 5 is 60% and must not outrank 800 out of 10,000. The amber tick is the rate that row
   was compared against. An `exif_cluster` attribution is labelled **hypothesis**.
3. **Findings**: filter by disposition, **nature** and state. *Nature* separates
   **adversarial** (trigger injection) from **quality** (duplicates, out-of-distribution
   samples), and they are not triaged the same way. Severity and confidence are separate
   fields. A high-severity finding with low confidence is a lead, not a conclusion.
4. **Provenance**: the field ledger's integrity. It is always shown, including when clean.
   If this section is absent, provenance was never checked, which is different from being
   fine.
5. **Coverage**: what was assessed, what could not be assessed and why, and what nothing
   covers. **A smaller statement on a black-box scan is correct, not a fault.** The
   calibration chart shows whether the tool's own confidence earns its number.

## The states

- **accept / review / quarantine**: the risk engine's disposition. The dashboard never
  computes one. It shows the tool's, and beside it the effective one after human decisions,
  cited to a ledger `seq`.
- **UNAVAILABLE / DEGRADED**: a check did not run, or ran partly, **with a reason**
  (`capability`: the scan lacked something it needed; `budget`: the tier excluded it).
  These are **hatched, never greyed out.** Absence of evidence is not evidence of absence,
  and a hatched check is a finding about the scan.
- **pending**: a lowering is waiting for a second, different approver.

## Taking a decision properly

Open a finding and press **Open and decide**.

- **Raising** a disposition (towards quarantine) takes effect immediately.
- **Lowering** one takes effect only when a **different** person with the approver role
  approves it. You cannot approve your own change. The four-eyes rule exists so that no
  single account can quietly accept a flagged asset.
- **A justification is mandatory**: at least 30 characters and 5 real words. A string of
  punctuation clears a character count and records nothing, so words are what is counted.
  The counter updates as you type, so a refusal never comes as a surprise.
- **Pick the reason code that is true.** Codes exist so overrides can be counted by cause.
  `other_with_justification` asks for a longer justification because it names no cause.
- **`prov.*` findings** accept only their own codes. If you lower a quarantine on one, the
  dashboard stops and reminds you that the arithmetic did not hold, and that your override
  records a human judgement without making it hold.

**If two analysts act on the same finding at once**, the second is refused with *someone
else acted on this first*. Nothing is overwritten. Re-read the finding and decide again.

**If a decision is refused**, the message says why and says `nothing changed`. Nothing was
recorded. Do not retry in a loop. Read the reason.

## The audit trail

Every scan record and every decision, ordered by **ledger `seq`, not by time**. The host
clock is untrusted, and a clock that steps backwards changes what is displayed, never what
is ordered. Each entry shows who, in which role, what, and the justification in full.

Know what it does not establish: **attribution is by the recording tool, not by your own
key.** Anyone holding your session could record as you. Log out on shared terminals.

## Remediation: what it does and does not claim

`cva remediate` is **not in this build**, and the dashboard says so rather than offering a
button that fails. When it exists, the result of remediating a dataset is a **new artefact
that has been re-scanned**. The dashboard will report *what that re-scan checked*. It will
never say the data is cleaned, fixed or safe. A re-scan that finds nothing is a statement
about the checks, not about the data.
