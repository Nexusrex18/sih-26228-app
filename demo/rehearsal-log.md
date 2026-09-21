# Rehearsal log

One entry per rehearsal: the date, the machine, the timings per step, and **every place it
went wrong**, including the ones that were fixed on the spot. A log with only clean runs is
not a log of rehearsals.

Rehearse on data the detectors were not tuned on (Tasks, "dress rehearsal").

| Field | |
|---|---|
| Date | |
| Machine / image digest | |
| Network state | (must be: cable out, `--network none`) |
| Presenters | |

| Step | Target | Actual | Went wrong | Fixed how |
|---|---|---|---|---|
| 0 air gap | 0:45 | | | |
| 1 clean pass | 1:15 | | | |
| 2 contributor | 1:45 | | | |
| 3 backdoor | 1:45 | | | |
| 4 ledger edit + limit | 1:30 | | | |
| 5 two-person decision | 1:30 | | | |
| 6 coverage shrinks | 1:00 | | | |
| **total** | **9:30 + 0:30 buffer** | | | |

---

## 2026-09-21: pre-rehearsal: what is verified, and what is not yet

**No full rehearsal has been run yet.** This entry records what the test suite already
exercises, so the first rehearsal knows where to look hardest.

**Verified by tests on the development machine:**

- Step 4's commands and exit codes: `tests/docs/test_doc_commands.py` exports a ledger,
  edits one byte, and asserts exit 2. It also asserts exit 1 when the file is unreadable.
- Step 5's rules: the four-eyes flow, self-approval refused, and stale-state refusal, in
  `tests/ledgerd/test_ledgerd_roundtrip.py`. Punctuation-only justifications are refused
  in `test_punctuation_is_not_a_justification`.
- Scans seal through the socket (`--audit-ledger-socket`), in
  `tests/ledgerd/test_scan_seals_through_the_socket.py`.
- Every command in `script.md` names a real subcommand and real flags, in
  `tests/docs/test_doc_commands.py`.

**Not yet verified, so the first rehearsal must cover these:**

- **Step 0.** The image has not been built. The base digest has to be pinned once with
  network access (`make -f docker/image.mk pin`).
- **Every dashboard step in a real browser at phone width (390px).** The pages build, and
  their API routes are tested, but no one has watched them at 390px.
- **Step 3's contingency.** Trigger reconstruction versus STRIP depends on the Module B
  spike's result at the demo class count. Record which one was shown.
- **Timings.** None are measured yet. The live corpus is sized to CPU throughput, which has
  to be measured on the demo machine, not assumed.
