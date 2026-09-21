# Module E: handover

**Branch:** `module-e-governance` (off `origin/modules`, pushed to origin)
**Plan:** `../sih26228-notes/Plan/Module-E-Governance-Network-Plan.md`. Read §0 and §6 first.
**Date:** 2026-09-21

---

## Run it

```bash
cd /home/tanmay0996/sih2/sih-26228-app
npm --prefix frontend install          # once
npm --prefix frontend run build        # -> frontend/out/
.venv/bin/python scripts/dev_dashboard.py --fresh
```

Then open <http://127.0.0.1:8713/app/> and sign in as `Tan.00` / `12345678`.
`b.rao` is the **approver** (same password). You need both accounts to exercise four-eyes.
Sign-in (`/login`) is the only server-rendered page; every old server-rendered path
redirects into `/app/` (`cva/web/spa_paths.py`). If port 8713 is taken (a stale dev
server), pass `--port 8714 --out .scratch/dev8714`.

Tests. Run them in batches: the whole set in one process was killed for memory (exit 137)
on this machine.

```bash
.venv/bin/python -m pytest tests/ledgerd tests/boundaries tests/report tests/docs -q
.venv/bin/python -m pytest tests/web tests/security -q
.venv/bin/python -m pytest tests/e2e -q        # the whole flow, real processes (~30 s)
```

`tests/e2e/` is the integration suite. It starts `cva-ledgerd` and `cva-web` as separate
processes, set up the way SETUP.md says, and drives them over HTTP with a client that
derives `Origin`/`Referer` from each page's real referrer policy, as a browser does. It
covers sign-in, all 8 pages with their scripts, the two-person decision, the audit trail,
verification, export plus an independent `cva-seal verify`, and a real scan sealed through
the socket while the dashboard is running. It needs `frontend/out` built.

---

## Status by gate

| Gate | State | What exists |
|---|---|---|
| **N0** | done | `config.py` (an unknown key is a load error), `fixtures.py` covering every state, with hostile strings |
| **N1** | done | 9 Next.js pages: `/`, `/scan`, `/contributors`, `/findings` (+ decide sheet), `/provenance`, `/coverage` (reliability diagram in Recharts; two-scan compare with bars to scale), `/audit` (SeqRail), `/verification` (+ ledger export), `/accounts` (admin only, ported from the old Jinja admin page). The server-rendered dashboard is removed; its paths redirect to the SPA. What each page must show is asserted in a real browser (`tests/e2e/test_spa_content.py`) |
| **N2** | done | security baseline. The S1/S3/S4 assertions are written now (`tests/security/test_hostile_report_content.py`) |
| **N3** | done | `cva/ledgerd/` with the per-uid × per-record-type allowlist, four-eyes, `expected_prev_seq` and idempotency |
| **N4** | done | `cva-seal verify --json` / `export`, the verify bridge, the trust banner. `POST /api/audit/export` for the SPA |
| **N5** | done | reports `cva remediate` as **absent from this build**. Do not implement the Remediator (§0) |
| **N6** | done | `docs/coverage-standing.yaml` merged by `cva/report/standing.py` into every report's `standing_limitations`, with reviewers and date printed last. **`reviewed_by` names `PENDING-SECOND-READER`**, and it needs a real second reader |
| **N7** | written, **not built** | `docker/Dockerfile`, `entrypoint.sh`, `image.mk`, `hardening.md`, `ledgerd-policy.json`: three roles, three uids, one group. Building needs `make -f docker/image.mk pin` once with network access, to record the base digest. `build` refuses without it |
| **N8** | done | `docs/SETUP.md`, `operator-manual.md`, `THREAT-MODEL.md`, `VERIFICATION-PROCEDURE.md`, and a **generated** `LICENSES.md` (`scripts/generate_licenses.py`, `--check` in tests). Every doc command is executed or parser-checked (`tests/docs/`) |
| **N9** | written, **not rehearsed** | `demo/script.md`, `storyboard.md`, `rehearsal-log.md` (whose first entry lists what is verified and what is not) |

**Mobile pass:** done in code: 44px targets, a card/table split on `/scan` via a shared
`ContributorCard`, and `prefers-reduced-transparency` / `prefers-contrast` in `globals.css`.
Checked in a real browser at 390px and at desktop width for every page and every fixture
scan by `tests/e2e/test_in_a_real_browser.py` (Playwright): no JS errors, no horizontal
overflow. Screenshots land in `.scratch/e2e-screens/`.

---

## Bugs found and fixed this session (each has a regression test)

1. **The SVG sandbox was being stripped.** The after-request hook overwrote every view's
   CSP, so `/evidence/` SVGs went out without `sandbox`. A view-set CSP is now kept.
2. **Verification failed outside the repo root.** The `python -m` fallback found `cva` only
   when the child's working directory was the repo. It now gets PYTHONPATH from the file's
   own location.
3. **TLS config served plaintext.** Setting `tls_*` satisfied S9's config check, but waitress
   never terminates TLS. `run()` now refuses to start in that state.
4. **`code_commit` was 7 hex.** ledgerd requires 40. It is now the full commit, or
   `CVA_CODE_COMMIT` (baked in at image build, trusted only if exactly 40 hex characters).

## Decisions taken beyond the plan (each argued in its commit message)

1. `cva-web` reads the ledger **through** ledgerd, never with its own handle. CI invariant 5.
2. ledgerd never reads `report.json`.
3. ledgerd validates the wire encoding and never repairs it.
4. Next.js replaces D-E1's server-rendered-only rule. The runtime still fetches nothing.
5. **`bind_trusts_container_boundary`:** inside a bridge network the process must bind
   `0.0.0.0`, so S9 moves to the publish spec. Only the entrypoint sets it, and every
   documented `-p` is `127.0.0.1:` (enforced by a test).
6. **Cleanlab's licence is taken from its wheel, not from the plan.** §7.10 says AGPL. The
   2.9.0 wheel that ships declares Apache and carries the Apache 2.0 text. `LICENSES.md` says
   so and would flip back to the AGPL wording on its own if an older release were pinned.
7. `--audit-ledger-socket` is an **alternative** to `--audit-ledger`. Passing both is refused.

## Still open

- **Build the image**: pin the digest, then run `make -f docker/image.mk image` and `podman`.
  Watch the disk: it was at 93%.
- **Apply the linear.app taste notes** (`.scratch/taste/linear.app.md`) to the dashboard:
  colour only on status, hierarchy by text luminance, hairlines over glows, colour-only
  hover feedback. The sign-in page already follows them.
- **Rehearse the demo** and fill in `rehearsal-log.md`.
- **Second reader** for `docs/coverage-standing.yaml`.
- `scripts/vendor_web_assets.py` has never run (it needs a network) and is now mostly
  moot: the sign-in page bundles IBM Plex from `@fontsource` in `cva/web/static/fonts/`.
- **Two `cva-ledgerd`s.** Module C ships its own daemon (`cva/provenance/seal/ledgerd.py`,
  a different wire protocol) beside Module E's (`cva/ledgerd/`, the one the dashboard and
  the image use). Both kept on the rebase onto `modules`; which one ships is a team call.
- The dashboard's verify bridge follows Module C's `cva-seal verify` contract
  (`--records`, `--trust`, `clean`, `records_in_unwitnessed_window`). It passes no anchor,
  so the Verification page states the unwitnessed window and why it is unbounded.

---

## Do not break

- **`cva/web/` must never import `cva.provenance`**, and `cva/ledgerd/` must never import
  Flask. `tests/boundaries/` enforces both.
- **The dashboard never computes a disposition, confidence or coverage row** (D-E2).
- **Fail closed** (D-E4): no optimistic UI and no local queue.
- **`UNAVAILABLE` is hatched, never greyed out.** Under `prefers-contrast: more` the hatch
  gets denser, not fainter.
- **The wording rule** (§7.5): never "cleaned", "fixed" or "safe" about data.
- **A password is never an argument** (`python -m cva.web.accounts` reads it from the
  terminal or stdin).
- **When staging**, `git add -A` resets `docker/entrypoint.sh`'s executable bit (the file on
  disk is 0644). Re-run `git update-index --chmod=+x docker/entrypoint.sh`. The Dockerfile
  sets `0555` regardless.
