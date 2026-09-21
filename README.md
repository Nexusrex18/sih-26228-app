# CVA: Computer-Vision Integrity Assurance

**Smart India Hackathon 2026, PS 26228 · Team Zlatan_FC**

CVA audits the three things a computer-vision system is built from: its **training data**, its
**models**, and its **inference records**. It works fully offline, with no cloud and no network
calls, and it produces one report per scan that says what was found and what was checked.

```
   datasets ─┐                                   ┌─ accept
   models   ─┼─► loaders ─► capability ─► checks ─► risk engine ─► report ─┼─ review
   inference ┘   (safe)     negotiation   A·B·D    + dispositions   + coverage └─ quarantine
       │
       └────────► C · provenance seal (signed, hash-chained ledger) ──► verify ──► dashboard (E)
```

## What it does

| Module | Purpose | What it checks |
|---|---|---|
| **A · Data integrity** | Is the training data trustworthy? | Near-duplicates, label errors, duplicate-label conflicts, trigger artifacts, out-of-distribution samples, systematic mislabelling, annotation geometry, negative-space poisoning, metadata anomalies. Findings roll up to the **contributor** that supplied them. |
| **B · Model integrity** | Is this the model you were told it was? | Weight digest, behavioural fingerprint, graph structure, Neural Cleanse trigger reconstruction, STRIP, intrinsic probes, weight and activation statistics, anomalous behaviour. |
| **C · Provenance seal** | Was this inference record altered? | Ed25519-signed, hash-chained records in a Merkle-tree ledger over canonical JSON (RFC 8785, RFC 6962). Offline verification, external anchors, inclusion proofs, an independent verifier, and a native C core. |
| **D · Drift** | Has the incoming data shifted from the reference? | PSI and KS tests over interpretable image axes and embedding projections, with multiplicity control. |
| **E · Governance** | Who decided what, and can it be proven? | Analyst dashboard, two-person approval for lowering a disposition, a signed audit trail, verification, and a generated coverage statement. |

## Key ideas

- **Capability negotiation.** Every check declares what access it needs. Before anything runs, each
  check is resolved to **OK**, **DEGRADED** (runs with a stated reduction) or **UNAVAILABLE** (with
  the reason). The plan is printed before the scan starts.
- **A generated coverage statement.** Each report lists the attack classes that were assessed and
  those that were not, produced from what the checks declare and never typed by hand. A scan with
  less access gives a smaller statement.
- **Dispositions per finding.** Every finding ends as *accept*, *review* or *quarantine*, with the
  rule that decided it.
- **Safe by construction.** Untrusted datasets and models are parsed in a sandbox with size, path and
  pixel limits, and checkpoints are loaded without executing code.
- **Air-gapped.** No module makes a network call, and a built-in egress guard enforces it.
- **Tamper-evident audit trail.** Scan records and analyst decisions are sealed into the ledger by a
  separate process that alone holds the signing key.

## Requirements

- Python 3.11 or newer
- Node.js 20 or newer (only to build the dashboard)
- Linux (the demo's network-isolation step uses `unshare`)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[data,seal,network]"

npm --prefix frontend ci && npm --prefix frontend run build     # the dashboard

make vendor        # optional: fetch the DINOv2 embedding weights (needs a network, once)
```

This installs the console scripts `cva`, `cva-seal`, `cva-ledgerd` and `cva-web`. Every command
below can also be run as `python -m ...`.

## Quick start: the sample demo

Everything below is generated locally. No dataset is downloaded.

### 1. Set up (one time)

```bash
export DEMO=$HOME/cva-demo
mkdir -p $DEMO/{keys,ledger,run,reports,var}

# demo datasets and a small model
python -m cva.fixtures --out $DEMO/fx
python -m cva.fixtures --out $DEMO/fx --kind mixed_contributors

# the signing key and the audit ledger
cva-seal keygen --out $DEMO/keys/signing.key
cva-seal init --ledger $DEMO/ledger/audit.db --key $DEMO/keys/signing.key \
  --device-id demo-host --unit demo --trust-out $DEMO/ledger/trust_root.json

# two accounts: an analyst and a different person who approves
export CVA_WEB_ACCOUNTS_DB=$DEMO/var/accounts.db
echo 'DemoPass-2026' | python -m cva.web.accounts create a.sharma --role analyst  --password-stdin
echo 'DemoPass-2026' | python -m cva.web.accounts create b.rao   --role approver --password-stdin
```

### 2. Start the services (two terminals)

```bash
# terminal 1: the ledger daemon, the only process that holds the signing key
export DEMO=$HOME/cva-demo
cva-ledgerd --socket $DEMO/run/ledgerd.sock --ledger $DEMO/ledger/audit.db --key $DEMO/keys/signing.key

# terminal 2: the dashboard
export DEMO=$HOME/cva-demo
export CVA_WEB_LEDGERD_SOCKET=$DEMO/run/ledgerd.sock CVA_WEB_LEDGER_PATH=$DEMO/ledger/audit.db \
       CVA_WEB_TRUST_ROOT=$DEMO/ledger/trust_root.json CVA_WEB_REPORTS_DIR=$DEMO/reports \
       CVA_WEB_ACCOUNTS_DB=$DEMO/var/accounts.db CVA_WEB_INDEX_DB=$DEMO/var/index.db
cva-web
```

Open <http://127.0.0.1:8713/> and sign in as `a.sharma` (analyst) or `b.rao` (approver), password
`DemoPass-2026`. Use a second browser window for the second account.

### 3. Run scans (a third terminal)

Scans are named by the minute they start, so leave a minute between them.

```bash
export DEMO=$HOME/cva-demo

# a dataset and a model, sealed into the ledger
cva scan --dataset $DEMO/fx/demo_coco --model $DEMO/fx/demo_model.onnx \
  --profile baseline --out $DEMO/reports --audit-ledger-socket $DEMO/run/ledgerd.sock

# a dataset with several contributors
cva scan --dataset $DEMO/fx/mixed_contributors --profile baseline \
  --out $DEMO/reports --audit-ledger-socket $DEMO/run/ledgerd.sock

# the same dataset with no model: a smaller coverage statement
cva scan --dataset $DEMO/fx/demo_coco --profile baseline \
  --out $DEMO/reports --audit-ledger-socket $DEMO/run/ledgerd.sock

# run a scan with the network removed entirely
unshare -rn cva scan --dataset $DEMO/fx/demo_coco --model $DEMO/fx/demo_model.onnx \
  --profile baseline --out $DEMO/reports_airgap
```

Each scan writes `report.json`, `report.html` and `coverage.md` under `<out>/<scan_id>/`, with
evidence stored once by content hash under `<out>/evidence/`. The scans made with
`--audit-ledger-socket` appear in the dashboard.

### 4. Try drift detection

```bash
python -m attacklab.photometric_shift --out $DEMO/drift --brightness 0.2
cva drift --incoming $DEMO/drift/incoming --reference $DEMO/drift/reference --out $DEMO/reports_drift
```

### 5. Try the tamper-evident ledger

This works on a copy of the ledger.

```bash
cp -r $DEMO/ledger $DEMO/ledger_copy
cva-seal anchor export --ledger $DEMO/ledger_copy/audit.db --key $DEMO/keys/signing.key --out $DEMO/anchor.json
cva-seal export --ledger $DEMO/ledger_copy/audit.db --out $DEMO/demo.export.jsonl
cva-seal verify --records $DEMO/demo.export.jsonl --trust $DEMO/ledger/trust_root.json --anchor $DEMO/anchor.json
```

The ledger verifies clean. Now change one character in one line of `$DEMO/demo.export.jsonl` and run
`verify` again: it reports the exact record that was edited. Restore the file, delete its last few
lines and verify with `--anchor`: it reports the truncation.

### 6. Make a two-person decision (in the dashboard)

1. As `a.sharma`, open **Findings**, pick a finding, and lower its disposition with a reason and a
   written justification. It shows as *pending*.
2. Approving your own change is refused.
3. As `b.rao`, approve it.
4. Open **Audit trail** to see both events in ledger order, then **Verify now**.

## Scanning your own data

```bash
# a dataset (COCO, YOLO, VOC or a folder of images) and a model (ONNX, TorchScript or PyTorch)
cva scan --dataset path/to/dataset --model path/to/model.onnx --profile baseline --out reports

# add a clean probe set and reference material for the model checks
cva scan --model path/to/model.pt --corpus path/to/probes --profile deep --out reports

# a black-box model behind a command or a loopback endpoint
cva scan --model-cmd "python serve.py" --input-shape 3,64,64 --num-classes 10 --dataset path/to/dataset
```

Profiles choose which checks run and how much compute they may spend. Run `cva scan --help` for
every option.

## Repository layout

```
cva/
  cli.py                 the `cva` command: scan, drift, selftest, bench
  loaders/               dataset and model loaders with the untrusted-input controls
  core/                  capability model, orchestrator, registries, shared types
  detectors/data/        Module A
  detectors/model/       Module B
  detectors/drift/       Module D
  provenance/            Module C: seal SDK, ledger, verifier, anchors, checks
  risk/                  dispositions, contributor rollup, calibration
  report/                report.json, HTML and the generated coverage statement
  web/  ledgerd/         Module E: dashboard backend and the ledger daemon
frontend/                the dashboard (Next.js, built to a static export)
attacklab/               generators for poisoned data, backdoored models and tampered ledgers
native/                  the C seal core, with C++ and Rust bindings
spec/                    the cva-seal wire-format spec, test vectors, an independent verifier
schemas/  profiles/      the report schema and scan profiles
docs/                    setup, operator manual, threat model, verification procedure
tests/
```

## Tests

```bash
pytest tests -q
```

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md): installation from a delivered image, first scan, first read in the dashboard
- [`docs/operator-manual.md`](docs/operator-manual.md): running the system day to day
- [`docs/THREAT-MODEL.md`](docs/THREAT-MODEL.md): what the system defends against
- [`docs/VERIFICATION-PROCEDURE.md`](docs/VERIFICATION-PROCEDURE.md): how a third party verifies a ledger without our code
- [`spec/cva-seal-spec-v1.md`](spec/cva-seal-spec-v1.md): the published wire format for the provenance seal
- [`docs/LICENSES.md`](docs/LICENSES.md): third-party licences
