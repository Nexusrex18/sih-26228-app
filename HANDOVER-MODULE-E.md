# Module E — handover

**Branch:** `module-e-governance` (off `origin/modules`, 5 commits, nothing pushed yet)
**Plan:** `../sih26228-notes/Plan/Module-E-Governance-Network-Plan.md` — read §0 and §6 first
**Date:** 2026-09-21

---

## Run it

```bash
cd /home/tanmay0996/sih2/sih-26228-app
npm --prefix frontend install          # once
npm --prefix frontend run build        # -> frontend/out/
.venv/bin/python scripts/dev_dashboard.py --fresh
```

Then <http://127.0.0.1:8713/app/> — sign in `a.sharma` / `demo-password-1234`.
`b.rao` is the **approver** (same password); you need both accounts to exercise four-eyes.
Server-rendered fallback at <http://127.0.0.1:8713/>.

Tests:

```bash
.venv/bin/python -m pytest tests/ledgerd tests/web tests/security tests/boundaries -q
```

---

## Done

| Gate | What exists |
|---|---|
| **N0** | `cva/web/config.py` (unknown key = load error), `cva/web/fixtures.py` — 4 schema-valid fixture reports covering every state: all 3 dispositions, `UNAVAILABLE`/`DEGRADED`, both `exclusion_reason`s, a `prov.*` quarantine, `boundary_flip`/`degraded_gap`/`clock_regression` at their correct non-quarantine dispositions, and hostile contributor/file/category strings |
| **N1** | Jinja views in the report's read order + Next.js pages (scan list, scan detail, findings, contributors) |
| **N2** | `security.py`, `accounts.py` (scrypt N=2¹⁷ with `maxmem`), `auth.py`; 21 tests in `tests/security/` |
| **N3** | `cva/ledgerd/` — SO_PEERCRED, per-uid × per-record-type allowlist, four-eyes, `expected_prev_seq`, idempotency rebuilt from the ledger at startup. `fold.py` property-tested with hypothesis. 32 tests |
| **N4** | `cva-seal verify --json` + `export` added; `verify_bridge.py` (fixed argv, `shell=False`); `seal_check.py` (D-E10); trust banner |
| **N5** | `remediation/jobs.py` + views — reports `cva remediate` **absent from this build** as first-class content rather than offering a button that fails |

**Test count:** 2357 pre-existing + ~80 new, all green.

### Decisions taken beyond the plan (each argued in its commit message)

1. **`cva-web` reads the ledger *through* ledgerd**, never with its own handle. Plan §5.2 gives it a read-only handle but §2 says `provenance.seal` is imported only in ledgerd and the verify bridge — both cannot hold. CI invariant 5 (`tests/boundaries/`) now makes the import a build failure.
2. **ledgerd never reads `report.json`.** It cannot tell a raise from a lowering and does not try; it enforces roles, staleness, and that an `approve` names a real change by someone else. `FoldResult.approvable` exists for this.
3. **ledgerd validates the wire encoding, never repairs it** — re-encoding is not idempotent (`%20` → `%2520`).
4. **Next.js replaces D-E1's server-rendered-only rule.** Runtime still fetches nothing (`output: "export"`); `/app/` needs `'unsafe-inline'` CSP, documented in `security.py:SPA_CSP`. Jinja views kept as the no-script fallback and the XSS-assertion surface.

---

## Left to do

### 1. Finish the Next.js pages *(highest value — this is what judges see)*

Written: `/`, `/scan`, `/findings` (+ `decide-sheet`), `/contributors`.
Missing — copy the pattern from `src/app/contributors/page.tsx`:

- **`/coverage`** — three groups (assessed / not-assessed-with-reason / never-covered), the reliability diagram (use **Recharts**, it is installed; load the `dataviz` skill first), and the two-scan comparison. API: `api.coverage(scanId, compare?)`. Gate N6 wants the black-box statement to look visibly smaller than the white-box one.
- **`/provenance`** — `ProvenanceSummary` already exists in `components/domain.tsx`; add the `prov.*` finding list and the "what this does not establish" panel. Copy the wording from `templates/provenance.html`.
- **`/audit`** — the timeline on the `SeqRail` (component exists). API: `api.audit(scanId?)`. Must state that attribution is tool-asserted, not analyst-signed.
- **`/verification`** — `api.health().verify` has everything; render `VerifyState` including `limitations`.

### 2. Mobile pass *(`apple-design` skill is the reference)*

`decide-sheet.tsx` + `ui/sheet.tsx` already do it properly — drag-to-dismiss with Apple's momentum projection, critically-damped springs, rubber-banding, 44px targets. **Apply the same treatment to the rest** and verify at 390px:

- `/findings` filter rows wrap but are untested on a phone
- `/scan`'s contributor table needs the card/table split `contributors/page.tsx` uses
- Add `prefers-reduced-transparency` and `prefers-contrast` blocks to `globals.css` (only `prefers-reduced-motion` is handled)

### 3. Gates not started

- **N6** — `docs/coverage-standing.yaml`, assembled from each module plan's standing-limitations section. **Backend seam missing:** `cva/report/report_json.py:standing_limitations()` reads a hardcoded list; it needs to merge the yaml. One function.
- **N7** — `docker/Dockerfile`, `entrypoint.sh`, `image.mk`, `hardening.md`. Pin base by digest, non-root, `--read-only`, `--cap-drop ALL`, size < 4 GB, no `nvidia-*`, key in no layer. **`docker` and `podman` are both on this machine.** Needs `make bundle` (Backend B6).
- **N8** — `docs/`: `SETUP.md`, `operator-manual.md`, `THREAT-MODEL.md`, `VERIFICATION-PROCEDURE.md`, generated `LICENSES.md`. Every command in them must be executed (test 10.10). `scripts/vendor_web_assets.py` exists and writes a manifest `LICENSES.md` should be generated from — **it has never been run.**
- **N9** — `demo/script.md`, storyboard, rehearsal log.

### 4. Known gaps

- **`code_commit` is 7 hex, the seal wants 40.** `cva/cli.py:code_commit()` returns `rev-parse --short`; `records.py:_v_scan` requires `_hex(..., 40)`. Harmless today (JsonlAuditLedger does not validate) but `scan_record` through ledgerd **will be rejected**. Fix: full `rev-parse HEAD`. Leave the `"unknown"` fallback — `append_scan_record` catches and reports it honestly. Never pad zeros.
- **`cva remediate` does not exist** (Backend §5.13). UI handles its absence; do not implement the Remediator, plan §0 forbids it.
- **`--audit-ledger-socket` is not wired into `cva scan`.** `LedgerdAuditLedger` is written and tested; `cli.py:_ledgers()` needs the flag. This is handoff O2.
- **Evidence/XSS tests not written.** S1/S3/S4 fixtures exist in `cva/web/fixtures.py:HOSTILE_STRINGS`; the assertions are not.
- `frontend/diagnose.sh` is gitignored scratch — delete it.

### 5. Environment notes

- **Disk was at 97%.** Freed ~1.1 GB; still tight. `next build` needs room.
- **The first `next build` died with `Bus error (core dumped)`** — a corrupt `@next/swc-linux-x64-gnu`. Fix was `rm -rf node_modules/@next/swc-linux-x64-gnu && npm install`. If it recurs, that is the cause, not your code.
- `~/.cache` holds ~2.9 GB of regenerable junk (`ms-playwright`, `puppeteer`, `pnpm`, `pip`).

---

## Do not break

- **`cva/web/` must never import `cva.provenance`** and `cva/ledgerd/` must never import Flask. `tests/boundaries/test_import_invariants.py` enforces both, and `test_invariants_fail_when_violated.py` proves the checks can fail.
- **The dashboard never computes a disposition, confidence or coverage row** (D-E2). It folds ledger events over the risk engine's values and shows both.
- **Fail closed** (D-E4). No optimistic UI, no local queue. A refusal re-renders the state the analyst already had.
- **`UNAVAILABLE` is hatched, never greyed out.** Absence of evidence must not render as evidence of absence — there is a test asserting an all-UNAVAILABLE scan does not look green.
- **The wording rule** (§7.5): the UI never says "cleaned", "fixed" or "safe" about a remediated dataset.
