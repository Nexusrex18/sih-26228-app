# Running the image: hardening

This page says how each container runs and why each flag is there. `SETUP.md` gives the
install order. Everything here follows plan §7.9 / D-E12. Where the implementation falls
short of the plan, the section **Gaps** says so rather than leaving it out.

## Three roles, three uids, one group

| Role | uid | May append | Holds |
|---|---|---|---|
| `ledgerd` | 10001 | nothing of its own. It signs what the allowlist admits | the signing key, mounted read-only |
| `web` | 10002 | `analyst_event` | nothing secret. It reads the ledger through ledgerd |
| `scan` | 10003 | `scan_record` | nothing secret |

All three share group `cva` (gid 10000), so ledgerd's `0660` socket is reachable by the two
callers and by nobody else. `docker/ledgerd-policy.json` is the per-uid allowlist, with
`same_uid_ok: false`. The entrypoint refuses to start a role under the wrong uid, so a
misconfigured container stops with the reason instead of failing later on a refusal.

## The flags, every time

```sh
HARDEN="--read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges"
```

- **`--read-only`**: the root filesystem cannot be written. The only writable paths are the
  volumes below and `/tmp`.
- **`--tmpfs /tmp`**: matplotlib and torch caches, the selftest's scratch output.
- **`--cap-drop ALL`**: no Linux capabilities. Nothing here binds a port below 1024 or
  changes ownership at runtime.
- **`--security-opt no-new-privileges`**: no setuid escalation, even if a binary allowed it.
- **Never** `--privileged`, never `--network host`.

## Volumes

```sh
docker volume create cva-reports   # scan outputs and the evidence store
docker volume create cva-ledger    # audit.db and trust_root.json
docker volume create cva-index     # the dashboard's index and accounts databases
docker volume create cva-run       # ledgerd's socket, shared by all three roles
```

The **key is not a volume.** It is a single file bind-mounted read-only into the ledgerd
container and no other. It is never baked into a layer, and `make -f docker/image.mk check`
fails the build if a key file is found in the image.

## Step 0: the air-gap proof

```sh
docker run --rm --network none $HARDEN cva:latest selftest
```

With `--network none` the container's only interface is loopback, so the kernel itself
guarantees there is no egress. `selftest` also arms its own in-process egress guard, which
makes two independent proofs. This is demo step 0.

## ledgerd

```sh
docker run -d --name cva-ledgerd --user 10001:10000 --network none $HARDEN \
  -v cva-ledger:/var/lib/cva/ledger \
  -v cva-run:/run/cva \
  -v "$PWD/keys/signing.key:/var/lib/cva/keys/signing.key:ro" \
  cva:latest ledgerd
```

`--network none`: ledgerd only ever talks over the Unix socket. With no key mounted it starts
**read-only**. The dashboard then shows every recorded decision and refuses new ones, and
says why.

## web

```sh
docker run -d --name cva-web --user 10002:10000 $HARDEN \
  -p 127.0.0.1:8713:8713 \
  -v cva-reports:/var/lib/cva/reports:ro \
  -v cva-ledger:/var/lib/cva/ledger:ro \
  -v cva-index:/var/lib/cva/index \
  -v cva-run:/run/cva \
  cva:latest web
```

**`-p 127.0.0.1:8713:8713` is a security control, not a convenience.** Inside a bridge
network the process has to bind `0.0.0.0` to be reachable through `-p` at all, so the
entrypoint sets `bind_trusts_container_boundary`. That setting moves S9 ("no plaintext
credentials on a shared network") onto the publish spec. **`-p 8713:8713` publishes the
dashboard on every host interface over plain HTTP**, which recreates exactly what S9
forbids. To serve beyond the host, put a TLS-terminating reverse proxy in front of it.
`cva-web` refuses to start if `tls_*` is set, because waitress would serve plaintext while
configured as if it were encrypted.

The ledger volume is mounted **read-only** into web. Web needs the file only to run
`cva-seal verify`. Every write goes through ledgerd.

## scan

```sh
docker run --rm --user 10003:10000 --network none $HARDEN \
  -v cva-reports:/var/lib/cva/reports \
  -v cva-run:/run/cva \
  -v "$PWD/data:/data:ro" \
  cva:latest scan --dataset /data/corpus --profile baseline
```

The scanner seals its `scan_record` through ledgerd's socket (`--audit-ledger-socket`), so it
seals without holding a key.

## Accounts

```sh
docker run --rm -i --user 10002:10000 --network none $HARDEN \
  -v cva-index:/var/lib/cva/index \
  cva:latest accounts create root.admin --role admin --password-stdin
```

A password is never a command-line argument, because `ps` shows argv to every user on the
host.

## Podman

The same commands work with `podman`, including rootless. `make -f docker/image.mk podman`
runs the build and checks through it. Rootless podman maps uids 10001–10003 into the
user's subuid range, which is still three distinct uids to `SO_PEERCRED`.

## Gaps

These are stated here so they are not discovered later:

- **The image has not been built on the development machine for this commit.** The base
  digest in `docker/base.digest` has to be recorded once with network access
  (`make -f docker/image.mk pin`). Until it is, `build` refuses to run rather than use a
  tag.
- **One key file, file custody.** The report states custody `file`. A deployment uses an
  HSM (Module C §5.10). A compromised ledgerd host account signs whatever it is told to.
- **Login events are not in the ledger** (v1). They go to the container's stdout log.
