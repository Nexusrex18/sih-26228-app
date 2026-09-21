#!/bin/sh
# The image's single entrypoint. One ROLE per container (plan §5.2, D-E12):
#
#   selftest            the air-gap proof — run it under `--network none` (demo step 0)
#   ledgerd             holds the signing key; the only process that can sign
#   web                 the analyst dashboard; talks to ledgerd over the shared socket
#   scan  [args...]     `cva scan`, sealing its scan_record through ledgerd
#   seal  [args...]     `cva-seal` (keygen, init, verify, export)
#   cva   [args...]     any other `cva` subcommand
#   accounts [args...]  create/disable dashboard accounts (`python -m cva.web.accounts`)
#
# Every role also checks the uid it runs as, because the ledgerd allowlist is per uid: a
# `web` container started as the ledgerd uid would be refused by the daemon anyway, but it
# is clearer to refuse here with the reason.
set -eu

ROLE="${1:-selftest}"
[ "$#" -gt 0 ] && shift

DATA=/var/lib/cva
SOCKET=/run/cva/ledgerd.sock
LEDGER="$DATA/ledger/audit.db"
TRUST="$DATA/ledger/trust_root.json"

need_uid() {
    if [ "$(id -u)" != "$1" ]; then
        echo "cva-entrypoint: role '$ROLE' runs as uid $1 ($2); this container is uid $(id -u)." >&2
        echo "cva-entrypoint: pass --user $1:10000." >&2
        exit 64
    fi
}

case "$ROLE" in
  selftest)
    exec cva selftest --out /tmp/selftest "$@"
    ;;

  ledgerd)
    need_uid 10001 cva-ledgerd
    KEY="$DATA/keys/signing.key"
    if [ ! -f "$KEY" ]; then
        # Never generated here. Creating a key is an explicit, witnessed act (`seal keygen`),
        # and a daemon that quietly made one would hold a key nobody decided to trust.
        echo "cva-entrypoint: no signing key at $KEY. Mount it read-only (see docker/hardening.md)." >&2
        echo "cva-entrypoint: starting READ-ONLY: the dashboard will show decisions and refuse new ones." >&2
        exec cva-ledgerd --socket "$SOCKET" --ledger "$LEDGER" \
            --policy /etc/cva/ledgerd-policy.json "$@"
    fi
    exec cva-ledgerd --socket "$SOCKET" --ledger "$LEDGER" --key "$KEY" \
        --policy /etc/cva/ledgerd-policy.json "$@"
    ;;

  web)
    need_uid 10002 cva-web
    # Inside a bridge network the process must bind 0.0.0.0 to be reachable through `-p`
    # at all. S9 then rests on the publish spec: `-p 127.0.0.1:8713:8713`, never
    # `-p 8713:8713`. The flag below is the declaration that says so; cva-web logs it.
    export CVA_WEB_BIND_HOST=0.0.0.0
    export CVA_WEB_BIND_TRUSTS_CONTAINER_BOUNDARY=true
    export CVA_WEB_REPORTS_DIR="$DATA/reports"
    export CVA_WEB_INDEX_DB="$DATA/index/index.db"
    export CVA_WEB_ACCOUNTS_DB="$DATA/index/accounts.db"
    export CVA_WEB_LEDGERD_SOCKET="$SOCKET"
    export CVA_WEB_LEDGER_PATH="$LEDGER"
    export CVA_WEB_TRUST_ROOT="$TRUST"
    exec cva-web "$@"
    ;;

  scan)
    need_uid 10003 cva-scan
    exec cva scan --out "$DATA/reports" --audit-ledger-socket "$SOCKET" "$@"
    ;;

  seal)
    exec cva-seal "$@"
    ;;

  cva)
    exec cva "$@"
    ;;

  accounts)
    need_uid 10002 cva-web
    export CVA_WEB_ACCOUNTS_DB="$DATA/index/accounts.db"
    exec python -m cva.web.accounts "$@"
    ;;

  *)
    echo "cva-entrypoint: unknown role '$ROLE'." >&2
    echo "roles: selftest | ledgerd | web | scan | seal | cva | accounts" >&2
    exit 64
    ;;
esac
