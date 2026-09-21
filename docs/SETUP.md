# SETUP

The install, from a delivered file to a first scan read in the dashboard. This is the PS §2.3
setup deliverable. Every command below is either executed by
`tests/docs/test_doc_commands.py` or parsed there against the real argument parser, so a
flag that stops existing fails CI before it misleads an operator.

Two paths are given. **The image** is the delivered form. **The host install** is the same
software without a container, and it is the path the test suite runs end to end.

---

## 1. Verify what you were given, before anything else

You received `cva-image.tar` and `cva-image.tar.sha256`, and you obtained the expected
hash **separately** (by phone, on paper, in a signed memo). A hash that travels with the
file it describes proves only that the two were copied together.

```sh
sha256sum -c cva-image.tar.sha256
```

Stop if this fails. Do not load a file whose hash you cannot account for.

## 2. Load the image (no network)

```sh
docker load -i cva-image.tar
```

`podman load -i cva-image.tar` works the same way, including rootless.

## 3. Prove the air gap: `selftest`

```sh
HARDEN="--read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges"
docker run --rm --network none $HARDEN cva:latest selftest
```

**Healthy output** ends with the selftest's summary and exit status 0. Two independent
guards are running. `--network none` leaves the container with loopback only, so the
kernel guarantees there is no egress. The selftest also arms its own in-process egress
guard, which raises if anything tries to connect. If either one trips, the tool has a
network dependency it must not have. Report it, and do not work around it.

## 4. Create the signing key: an explicit, witnessed act

The key is created **once**, deliberately, by a named person, and is never created
implicitly by any service. Two people should be present. Keep the file mode `0600`.

```sh
mkdir -p keys
docker run --rm --network none $HARDEN -v "$PWD/keys:/out" --user "$(id -u)" \
  cva:latest seal keygen --out /out/signing.key
```

Host install:

```sh
cva-seal keygen --out keys/signing.key
```

## 5. Create the audit ledger and its trust root

```sh
cva-seal init --ledger ledger/audit.db --key keys/signing.key \
  --device-id analyst-host-01 --unit acceptance-cell --trust-out ledger/trust_root.json
```

`trust_root.json` holds the public key and the deployment identity. A third party
verifies against it. **Record its sha256 somewhere this machine cannot edit**, because a
trust root is only as good as the independent copy it is checked against.

## 6. Start the services

Three roles, three uids. `docker/hardening.md` explains every flag. In short:

```sh
docker run -d --name cva-ledgerd --user 10001:10000 --network none $HARDEN \
  -v cva-ledger:/var/lib/cva/ledger -v cva-run:/run/cva \
  -v "$PWD/keys/signing.key:/var/lib/cva/keys/signing.key:ro" cva:latest ledgerd

docker run -d --name cva-web --user 10002:10000 $HARDEN -p 127.0.0.1:8713:8713 \
  -v cva-reports:/var/lib/cva/reports:ro -v cva-ledger:/var/lib/cva/ledger:ro \
  -v cva-index:/var/lib/cva/index -v cva-run:/run/cva cva:latest web
```

**Publish to `127.0.0.1` only.** `-p 8713:8713` would put analyst passwords on every host
interface in plaintext.

Host install, in two terminals:

```sh
cva-ledgerd --socket run/ledgerd.sock --ledger ledger/audit.db --key keys/signing.key
CVA_WEB_LEDGERD_SOCKET=run/ledgerd.sock CVA_WEB_LEDGER_PATH=ledger/audit.db \
  CVA_WEB_TRUST_ROOT=ledger/trust_root.json CVA_WEB_REPORTS_DIR=reports cva-web
```

Without `--policy`, `cva-ledgerd` runs the single-uid development policy and **says so at
start-up**. A deployment with separate service accounts passes `--policy`, as the image
does.

## 7. Create the accounts

The first admin comes from the shell. The rest can be managed from the dashboard. You
need at least one **analyst** and one **approver**, and they must be different people,
because lowering a disposition needs a second person (four-eyes).

```sh
python -m cva.web.accounts create root.admin --role admin
python -m cva.web.accounts create a.sharma --role analyst
python -m cva.web.accounts create b.rao --role approver
python -m cva.web.accounts list
```

Each `create` asks for the password twice on the terminal. For scripted setup, pipe one
line to `--password-stdin`. **There is no password argument**, because argv is visible to
every user through `ps`. Passwords are at least 8 characters.

In the image, prefix the commands with
`docker run --rm -it --user 10002:10000 -v cva-index:/var/lib/cva/index cva:latest accounts`.

## 8. The first scan

```sh
python -m cva.fixtures
cva scan --dataset artifacts/fixtures/demo_coco --model artifacts/fixtures/demo_model.onnx \
  --profile baseline --out reports --audit-ledger-socket run/ledgerd.sock
```

With `--audit-ledger-socket` the scan seals its `scan_record` through `cva-ledgerd`, so the
scanner never holds the key. Without it, the report says plainly that it was **not
sealed**.

**Profiles and budget tiers.** `--profile` picks which checks are planned, and
`--budget-tier` caps how much compute they may spend. The plan is printed before any work
starts, and a check the tier excludes is reported `UNAVAILABLE` with the reason `budget`,
never silently skipped. A black-box model (`--model-url`, `--model-cmd`) produces a smaller
coverage statement than a white-box ONNX file. That is correct: the smaller statement
reflects what could be checked.

## 9. Open the dashboard

<http://127.0.0.1:8713/app/>. The server-rendered fallback is at <http://127.0.0.1:8713/>.

**Healthy** means:

- the banner at the top of every page says **Audit ledger verified**;
- the scan you just ran is listed with a **sealed** badge;
- its Coverage page lists what was assessed, what was not and why, and the standing
  limitations.

The banner can show three other states:

- **not verified**: `cva-seal verify` has not run, or could not.
- **FAILED**: the ledger does not verify. Stop and follow `VERIFICATION-PROCEDURE.md`.
- **Workflow unavailable**: `cva-ledgerd` is not reachable. Recorded decisions are shown,
  and new ones are refused.

## 10. Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| Banner: *Workflow unavailable* | ledgerd is not running, or the socket path differs | start ledgerd; check `CVA_WEB_LEDGERD_SOCKET` |
| ledgerd logs "starting READ-ONLY" | no key at the mounted path | mount the key read-only; never generate one in place |
| A scan says *not sealed* | no `--audit-ledger-socket`, or ledgerd refused the record | read the WARNING it printed; the refusal names the reason |
| Login rejected with the right password | the account is locked after repeated failures | wait out the lockout: it doubles with each further failure, capped at an hour |
| Login rejected with the right password, not locked | the account is disabled | `python -m cva.web.accounts enable <actor_id>` |
| `cva-web` refuses to start: *does not terminate TLS* | `tls_*` is set | bind loopback and put a TLS-terminating proxy in front |
| `cva-web` refuses to start: *S9* | `bind_host` is not loopback | see `docker/hardening.md` |

## 11. Uninstall and cold restore

**Before removing anything, export the ledger.** It is the one artefact that cannot be
regenerated.

```sh
cva-seal export --ledger ledger/audit.db --out ledger/audit.export.jsonl
cva-seal verify --ledger ledger/audit.export.jsonl --trust-root ledger/trust_root.json
```

Then stop and remove the containers, volumes and image:

```sh
docker rm -f cva-web cva-ledgerd
docker volume rm cva-reports cva-ledger cva-index cva-run
docker image rm cva:latest
```

**Cold restore:** load the image (step 2), put `audit.db`, `trust_root.json` and the key back
in place, and start the services (step 6). The dashboard rebuilds its index from the reports
directory and its view of every decision from the ledger at start-up. There is no other
state to restore.
