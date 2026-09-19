# CVA — Module B: Model Integrity

Implementation of PS 26228 §2.2.2 (model integrity) plus the model-side half of §2.3's
attack generation, built on the frozen contracts in `sih26228-notes/Architecture/`.

## Why the capability model exists

A binary `white-box | black-box` flag is factually wrong. A bare ONNX inference session
exposes weights and activations but has **no backward pass**, so Neural Cleanse cannot run
on it. Declaring ONNX "white-box, gradients available" declares a check applicable to a
model it cannot execute on.

Every check declares `requires` / `optional` capability sets. The orchestrator resolves all
of them **before running anything**, so coverage is known at minute zero rather than at the
end of a long scan. Three states, plus a separate error channel:

```
requires ⊆ caps?  no  -> UNAVAILABLE + reason + missing[]
                  yes -> optional ⊆ caps?  no  -> DEGRADED + reason + mode
                                           yes -> OK
a check that RAISES   -> ERROR   (a bug in our code, never folded into DEGRADED)
```

`UNAVAILABLE` is not a skip — it is a `Finding` naming exactly what was missing and why.
A shorter report must never look like a cleaner result.

## Checks

| id | needs | catches | runs on ONNX |
|---|---|---|---|
| `model.weight_digest` | file | any byte change — deterministic | yes |
| `model.graph_structure` | architecture | architectural backdoors, custom ops, dead subgraphs | yes |
| `model.behavioural_fingerprint` | predict | substitution, with a **tolerance band** so a benign re-export is not called hostile | yes |
| `model.strip` | predict + clean set | trigger-conditioned backdoors, **black-box** | yes |
| `model.intrinsic_probes` | predict + clean set | reference-free per-class suspicion **ranking** | yes |
| `model.activation_statistics` | activations + clean set | structural damage, dead/saturated units | yes (graph surgery) |
| `model.weight_statistics` | weights + battery | crude weight-level modification | yes |
| `model.anomalous` | predict + clean set | the third PS verdict: strange without a known signature | yes |
| `model.trigger_reconstruction` | predict + clean set; gradients *optional* | patch backdoors, and **reconstructs the trigger image** | DEGRADED (NES) |

Two things worth stating explicitly:

**Neural Cleanse is intrinsically self-referential.** It flags the class whose minimal
trigger is anomalously small *relative to this model's own other classes* — a MAD outlier
test inside the model. It needs gradients; it does not need a reference model. Since
organisers supply no reference battery, it is the strongest reference-free check available.

**STRIP is not a fallback.** ONNX is one of the two formats the PS names and exposes no
gradients, so on a plausible majority of real deliveries STRIP is the *only* backdoor check
that runs at all.

## Running it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .

# build the corpus: clean + backdoored models, with a fitness gate that REFUSES to emit
# a model that did not learn or a "backdoor" that did not take
.venv/bin/python -m cva.attacklab.build

# scan one model -> single-file HTML report
.venv/bin/python -m cva.cli scan artifacts/corpus/models/bd_patch_08.onnx

# score every detector over the whole corpus against ground truth
.venv/bin/python -m cva.cli bench
```

Outputs: `artifacts/reports/<model>.report.html` (per model),
`artifacts/bench/bench.html` (detection matrix + scoreboard),
`artifacts/bench/all_models.report.html` (every scan).

## Evaluating the result

Three surfaces, deliberately separate:

1. **Per-model report** — verdict, access assumptions, the resolved plan, findings with
   evidence images (reconstructed triggers, STRIP entropy histograms, per-class signal
   plots), and a coverage table generated from what the detectors declare.
2. **Benchmark matrix** — every model against every detector, with ground truth from the
   attack lab manifest. Detection rate **and false-alarm rate reported separately**: a
   detector tuned only against attacks has no measured false-alarm rate, and the
   false-alarm column is what decides whether an analyst keeps reading.
3. **Tests** — `pytest` covers the contract invariants, including that ONNX never reports
   gradients, that a raising check surfaces as `ERROR` not `DEGRADED`, that no registered
   check can vanish from a scan, and that a benign re-export is not called a substitution.

## Deliberate limitations

- The corpus is synthetic and offline by design. It exercises the pipeline; it is not
  evidence of field performance. External validation against TrojAI/BackdoorBench is the
  separate, required step.
- Reference models are decorrelated on purpose (different architecture, seed, width, data
  split). A battery of siblings makes every reference comparison unrealistically easy.
- A backdoor present since original training, with no manifest and no reference battery,
  remains the largest blind spot. `model.intrinsic_probes` narrows it; it does not close it.

## Module D — drift detection (first implementation increment)

Compare a declared reference image directory with an incoming image directory, offline:

```bash
python -m cva.drift_cli --reference data/reference --incoming data/incoming --out artifacts/drift
```

This entrypoint requires NumPy, SciPy and Pillow, but does not import Torch or ONNX.
It writes `drift.report.json`, `drift.report.html` and `drift.coverage.md`. Missing reference,
small batches, absent embeddings and unimplemented checks are explicit coverage gaps.
Material drift requests review; no attack intent or field-calibrated confidence is claimed.

Reproducible smoke demo (use a new/empty output directory):

```bash
python -m attacklab.photometric_shift --out /tmp/cva-drift-demo
python -m cva.drift_cli --reference /tmp/cva-drift-demo/reference --incoming /tmp/cva-drift-demo/incoming
python -m pytest tests/detectors/drift tests/boundaries
```

Optional `--reference-embeddings` and `--incoming-embeddings` accept NPZ files containing
`embeddings` (N x D numeric), `sample_ids` (sorted relative image paths), `extractor_id`
and `extractor_version` (scalar strings). Both caches must use the same independent
extractor/version. Pickle loading is disabled. The scanner does not generate or download
embeddings; photometric checks still run without them.

Full sequencing, corrections to the notes, statistical assumptions and remaining MVP
work: [Module D implementation plan](docs/MODULE_D_IMPLEMENTATION_PLAN.md).
