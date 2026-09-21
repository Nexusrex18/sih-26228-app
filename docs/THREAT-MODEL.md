# Threat model

For reviewers. It says who we defend against, what we claim, what we do not claim, and where
the tool itself can be attacked. The standing limitations are maintained in
`docs/coverage-standing.yaml` and printed in every report. This page is the argument behind
them.

## The invariant everything is bounded by

> We detect poisoning that leaves a regularity in pixel, frequency, label-neighbourhood or
> activation space. Poisoning engineered to carry its regularity only in a space we do not
> measure is invisible to us.

A scan that finds nothing is a statement about the spaces that were measured. It is never a
statement that the data is clean.

## Adversaries

| Adversary | Wants | Where they act | What we do |
|---|---|---|---|
| **A dataset contributor** | poisoned samples accepted into training | the data, its labels, its metadata, its file names | Module A detectors, contributor-level aggregation with a posterior and an interval rather than a raw flag rate, and a permutation test for flags spread thinly across many contributors (Sybil) |
| **A model supplier** | a backdoored model accepted | the weights, or the behaviour only | Module B: trigger reconstruction, spectral and activation checks, a fingerprint against a registered reference |
| **Someone with access to the field record** | a history that says something else happened | the inference ledger after sealing | Module C: every edit, deletion, reordering and replay after sealing is detectable. Truncation is detectable relative to an anchor |
| **An insider at the analyst desk** | a quarantine quietly lowered | the dashboard's workflow | four-eyes on every lowering, a mandatory justification, every decision in the signed ledger cited by `seq` |
| **Anyone who controls report content** | code execution in an analyst's browser | contributor names, file names, categories, evidence files | §7.7 S1–S4: autoescaping, a strict CSP, evidence served content-addressed and sandboxed |

**The dataset supplier is an adversary by premise.** Everything derived from their data
reaches the dashboard as a string someone else chose. `cva/web/fixtures.py:HOSTILE_STRINGS`
puts that premise into the fixtures, and `tests/security/test_hostile_report_content.py`
asserts it is inert.

## Out of scope, declared

- **An adversary holding the signing key from the start** can forge a wholly consistent
  history, and it will verify. Mitigation: an HSM or secure element and two-person custody.
  That is a deployment decision.
- **Tampering between the sensor and the SDK.** The SDK signs what it is handed.
- **Tail truncation and split-view within an anchoring interval.** The unwitnessed window is
  printed. It is the bound on insider-grade assurance.
- **Selective logging.** If the SDK is never called there is no gap to find. An independent
  count (`--expected-count`) closes it. That is an integration requirement.
- **Poisoning in a space we do not measure** (the invariant above), and **attacks outside
  our own attack families**. Detection rates are measured against our families and do not
  generalise.
- **A backdoor present since original training with no reference.** Nothing exists to be
  anomalous against. This is the system's largest single blind spot. Registering the
  emitted reference manifest at acceptance closes it for future scans.
- **Compromise of the assurance tool itself.**

## The dashboard's own attack surface (plan §7.7)

`cva-web` is the most exposed process on the host. It is authenticated, it renders
attacker-chosen content, and it can cause records to be signed. It is built so that
compromising it gains as little as possible.

| # | Control | How it holds | Test |
|---|---|---|---|
| S1 | Report content is inert | Jinja autoescape everywhere, no `|safe` on report data; React escapes by default and nothing uses `dangerouslySetInnerHTML` | `test_hostile_report_content.py` parses every page for event-handler attributes and inline script |
| S2 | Strict CSP, `nosniff`, `no-store` on authenticated pages | set by one after-request hook; the SPA's CSP adds `'unsafe-inline'` only because a static export inlines its bootstrap | `test_headers_and_csrf.py` |
| S3 | Evidence served defensively | `/evidence/<sha256>` only; the hash is recomputed; the type comes from magic bytes; SVG goes out under `sandbox` and is rendered through `<img>`; anything unknown is an attachment | `test_hostile_report_content.py` |
| S4 | No user path reaches the filesystem | only a validated `scan_id` and a 64-hex digest ever become a path | traversal and malformed-digest tests |
| S5 | Password storage | scrypt, N=2¹⁷, r=8, p=1, a per-user 16-byte salt | `test_scrypt_runs_at_owasps_parameters_without_raising` |
| S6 | Sessions | signed cookie, `HttpOnly`, `SameSite=Strict`, idle timeout, revoked server-side on logout or role change | cookie-flag and revocation tests |
| S7 | CSRF | a per-session token on every write, **and** an Origin/Referer check against Host | `test_headers_and_csrf.py`, including a real-form regression |
| S8 | Brute force | per-account and per-IP exponential lockout | `test_repeated_failures_lock_the_account` |
| S9 | No plaintext credentials on a shared network | binds loopback. Leaving it needs TLS or a declared container boundary, and `cva-web` refuses to serve when TLS is configured, because its server does not terminate TLS | `test_binding.py` |
| S10 | Least privilege | `cva-web` holds no key and no ledger write path; `cva-ledgerd` allowlists by uid **and** record type; three uids in the image | `tests/ledgerd/test_policy.py`, `tests/boundaries/` |
| S11 | Subprocess hygiene | `cva-seal verify`/`export` run with a fixed argv, `shell=False`, a timeout, and output displayed escaped | `verify_bridge.py`; the cwd-independence regression test |
| S12 | Size and rate limits | 64 KiB bodies; per-route rate limits | `test_an_oversize_body_is_refused` |

**Fail closed (D-E4).** If `cva-ledgerd` is unreachable, the dashboard shows every recorded
decision and refuses new ones. It keeps no local queue and does no optimistic update. A
refusal re-renders the state the analyst already had and says `nothing_changed`.

**The dashboard never computes a disposition (D-E2).** It folds ledger events over the risk
engine's values and shows both side by side. A compromised dashboard can misrepresent what
it shows. It cannot make a lie verify.

## Residual risks, stated plainly

- **Analyst attribution is by the tool, not by the analyst's key.** `actor_id` is asserted
  by `cva-web` after local authentication. Anyone with the `cva-web` uid and a valid session
  can record a decision as that user.
- **A compromised `cva-ledgerd` host account signs whatever it is told.**
- **Login and lockout events are not in the ledger** (v1). They go to the application log.
- **The host clock is untrusted.** Ordering is by `seq`, and every displayed time says so.
