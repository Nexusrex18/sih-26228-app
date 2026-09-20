# Vendoring manifest

Every third-party artefact this tool carries or fetches, what it is licensed under, and how
its bytes are pinned. Module A plan §9 assigns this document to Backend. It was owed and not
written (audit item 9), which mattered most for one line: the AGPL decision was ratified on
2026-09-19 and the only places it was recorded were a `pyproject.toml` comment and a module
README.

Two rules this document exists to keep visible:

- **A pin says what it proves.** A digest published by the distributor proves the bytes are
  the ones that party intended to publish. A digest we computed ourselves on first fetch
  proves only that the bytes have not changed *since* — trust on first use. Both are pins;
  they are not the same claim, and the `Digest published by distributor` column keeps them
  apart. `cva/features/vendor.py` carries the same distinction on every artefact and
  `vendor manifest` prints it.
- **A copyleft licence is stated unhedged.** cleanlab is **AGPL-3.0-or-later**. Not
  "AGPL-ish", not "AGPL (but only one call)". The mitigation is recorded below the table,
  not folded into the licence name.

---

## Python dependencies

Resolved and pinned by `make lock` into `requirements.lock`; the exact wheels an air-gapped
host installs are built by `make wheelhouse`. Versions below are the declared floors from
`pyproject.toml` — the lockfile is the authority for what is actually installed.

### Runtime (always installed)

| Package | Licence | Why it is here |
|---|---|---|
| `numpy` | BSD-3-Clause | arrays, throughout |
| `scipy` | BSD-3-Clause | the risk engine's distributions (Jeffreys interval, beta) |
| `torch`, `torchvision` | BSD-3-Clause | model loading, gradients, the embedding backbone. **CPU wheels only** — a CUDA wheel in this environment is a §5.7 violation and `make lock` fails on one |
| `onnx`, `onnxruntime` | Apache-2.0 | the ONNX loader and its inference path |
| `matplotlib` | PSF-based (matplotlib licence) | the reliability diagram and evidence plots |
| `pillow` | MIT-CMU (HPND) | image decoding |
| `jsonschema` | MIT | validates `report.json` and `profile.json` against the published schemas |
| `PyYAML` | MIT | dataset manifests |
| `packaging` | Apache-2.0 / BSD-2-Clause | version comparison — never string-compare a version |
| `pytest-socket` | MIT | a **product** dependency, not a test one: `cva selftest` arms the egress guard through it (§5.8) |

### `data` extra — Module A

| Package | Licence | Why it is here |
|---|---|---|
| **`cleanlab`** | **AGPL-3.0-or-later** | confident learning in `data.label_consistency`. **See the note below.** |
| `imagehash` | BSD-2-Clause | pHash for `data.near_dup`'s cheap tier |
| `adversarial-robustness-toolbox` | MIT | attack-lab reference implementations |
| `scikit-learn` | BSD-3-Clause | clustering and the disposable linear probe. **Load-bearing and, until now, disclosed only in a module README** — it belongs in the plan's stack table (Plan edit, item 8's sibling) |

### `seal` / `provenance` extras — Module C, Mode C

Deliberately short. CI invariant 4 exists to keep it that way: an operator asked to install
the ML stack in order to record one detection declines.

| Package | Licence | Why it is here |
|---|---|---|
| `cryptography` | Apache-2.0 OR BSD-3-Clause | signatures and hashing for the seal |
| `rfc8785` | Apache-2.0 | JCS canonical JSON — the exact bytes that get hashed |

### `network` extra — Module E

| Package | Licence |
|---|---|
| `Flask`, `Werkzeug`, `Jinja2`, `MarkupSafe`, `itsdangerous`, `click`, `blinker` | BSD-3-Clause |
| `waitress` | ZPL-2.1 |

### `dev` extra — Mode A only, never in the bundle tar

| Package | Licence |
|---|---|
| `pytest`, `hypothesis` | MIT / MPL-2.0 (hypothesis) |
| `ruff`, `mypy` | MIT |

---

## The cleanlab / AGPL-3.0-or-later decision

**Ratified 2026-09-19, `backend_plan.md` §5.12. Accepted, not worked around.**

cleanlab is AGPL-3.0-or-later. The AGPL's §13 network clause means that if this tool were
offered to third parties over a network, its corresponding source would have to be offered
with it. That is a real obligation and it is accepted rather than argued down.

What keeps it bounded:

- The dependency is **one call**, `find_label_issues`, inside `data.label_consistency`. It
  is in the `data` extra, not the runtime set, so a deployment that does not run Module A's
  label check does not install it.
- It is therefore **one call to swap** if the obligation ever becomes unacceptable. That is
  a stated exit, not a claim that the licence does not apply.
- Nothing in this repository is licensed more permissively on the assumption that cleanlab
  is absent.

Anyone packaging this tool for distribution reads this section first.

---

## Binary artefacts

Fetched by `make vendor` (`python -m cva.features.vendor fetch`), which verifies each
digest before writing. The **pins are committed; the bytes are not** — an 88 MB checkpoint
does not belong in git history, and `.gitignore` excludes `vendor/weights/*.pth` and
`vendor/dinov2/*`.

| Artefact | Source | Licence | SHA-256 | Digest published by distributor |
|---|---|---|---|---|
| `dinov2_vits14_pretrain.pth` | `dl.fbaipublicfiles.com` (facebookresearch/dinov2) | Apache-2.0 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | **No** — FAIR publishes no digest beside the checkpoint. Computed on first fetch, dev machine, 2026-09-19. Trust on first use |
| `resnet18-f37072fd.pth` | `download.pytorch.org` (pytorch/vision) | BSD-3-Clause | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | **Yes** — torchvision encodes the leading 8 hex of the digest in the filename and checks it on download |
| dinov2 source snapshot | `codeload.github.com/facebookresearch/dinov2` | Apache-2.0 | pinned by commit `e1277af2ba9496fbadf7aec6eba56e8d882d1e35` (a tree, not a file) | Yes — a git commit id is a digest of the tree |

### The offline consequence, stated

`make vendor` needs network, which the offline-first stance rules out for a default
checkout: `vendor/dinov2/` ships with only a `.gitkeep` and `vendor/weights/` does not
exist. On such a checkout `build_extractor` raises `BackboneUnavailable` and three of
Module A's nine detectors cannot produce a finding —

- `data.label_consistency` → `not_performed`
- `data.systematic_mislabel` → `not_performed`
- `data.near_dup` → degrades to pHash-only

— which also means they are never benchmarked, so no calibrator is ever fitted for them.
This is audit item 20 and it is **declared, not fixed**: the report's coverage statement is
the place an operator learns of it. The three options on the table are a vendored artefact
in the repo, a CI job that fetches once and caches, or leaving it declared. Whichever is
chosen, it is chosen out loud.

---

## Keeping this file true

`python -m cva.features.vendor manifest` prints the live state of the binary artefacts —
path, url, licence, pinned digest, whether the file is present and whether it matches. If
that output and the table above ever disagree, the code is right and this file is stale.
