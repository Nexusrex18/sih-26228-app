### 9.4 The detector registry's `requires`/`optional` rows

Backend owns the registry; the rows are the detector authors'. Reproduced from `backend_revised.md` §3.1, whose own header states: *"The capability-enum mapping in the last two columns is this note's work — the source states requirements informally and the registry needs enum values."*

**✅ No inferred rows remain.** All three that were originally marked `[INFERRED]` are settled: `data.label_consistency` on 2026-09-19 (see below), and `model.weight_stats` + `model.neural_cleanse` confirmed by ML-2 in `Plan/Module-B-Model-Integrity-Plan.md` §6.4/§6.5.

**Module A — data integrity (PS §2.2.1)**

| Plug-in | `requires` | `optional` → if missing |
|---|---|---|
| `data.near_dup` | `DATASET_IMAGES` | — |
| `data.label_consistency` | `DATASET_IMAGES`, `DATASET_LABELS` | **none — settled 2026-09-19, no degraded mode.** See below |
| `data.trigger_artifact` | `DATASET_IMAGES` | `MODEL_ACTIVATIONS`, `MODEL_PREDICT` → `DEGRADED`, frequency residue only |
| `data.ood` | `DATASET_IMAGES`, `REFERENCE_CLEAN_SET` | — |
| `data.metadata_anomaly` | `DATASET_IMAGES` (file metadata) | — | *(specced Module A §6.8)* |
| `data.negative_space` | `DATASET_IMAGES`, `DATASET_LABELS`, `MODEL_PREDICT` | `DATASET_CONTRIBUTOR_META` → `DEGRADED`, per-sample mode only |
| `data.annotation_geometry` | `DATASET_LABELS`, `DATASET_CONTRIBUTOR_META` | — |
| `data.systematic_mislabel` | `DATASET_IMAGES`, `DATASET_LABELS`, `DATASET_CONTRIBUTOR_META` | — |
| `data.duplicate_label_conflict` | `DATASET_IMAGES`, `DATASET_LABELS` | — |

**Four Module A rows were missing from this table and are added above (2026-09-20).** §12.1
makes this section the authority for `requires`/`optional`, so four detectors that Module A
had implemented, registered and shipped had **no authoritative row at all** — `registry_rows()`
in `cva/detectors/data/registry.py` was carrying all nine as a stopgap, which inverts §12.1:
the code became the authority because the plan was silent. The rows above are transcribed from
what those detectors actually declare, verified against the registry.

**`data.negative_space`'s contributor capability is `optional` — ruling implemented 2026-09-20.**
It used to hard-require `DATASET_CONTRIBUTOR_META`, so a §3 tier-5 dataset — no contributor
attribution, an explicitly supported case — got **zero** coverage of
`negative_space_poisoning`. The §6.4 signal itself does not need contributor identity; that is
only how the finding is framed, and what the cohort test needs. The row above now matches what
the detector declares.

Without the capability it runs **`DEGRADED`**, per-sample: the confident-detection-where-no-
annotation-exists signal survives, the cohort comparison does not. That mode is capped at `low`
severity and below D5's confidence floor, because one unannotated object is ordinary annotation
noise — only a systematic rate is evidence of poisoning — and every finding says which of the
two modes produced it.

> **This row was wrong in this table for one commit, and that is worth recording.** It was
> transcribed from the registry *before* the ruling was implemented and not revisited after,
> so §9.4 — which §12.1 makes the authority — contradicted shipped code. Exactly the drift
> this section exists to prevent. `tests/conformance/test_plan_rows_match_registry.py` in the
> app repo now reads these rows from the frozen contract and asserts them against
> `registry_rows()`, so the next disagreement is a test failure rather than a discovery.

**`data.label_consistency` is settled and no longer inferred (2026-09-19).** The inferred row gave it `optional: MODEL_PREDICT → DEGRADED`, on the grounds that confident learning needs predicted probabilities and ADR-004 permits using the contributed model's existing frozen outputs. **Withdrawn.** On a label-flip attack the contributed model was trained on the flipped labels and predicts them confidently, so `find_label_issues()` reports no issue on exactly the samples the detector exists to catch — feeding it `MODEL_PREDICT` does not degrade the detector, it inverts it, silently. ML-1's method (`Plan/Module-A-Data-Integrity-Plan.md` §6.2, §12.2) uses a disposable linear probe on the DINOv2 embeddings instead: independent of the artefact under test, always available, so no capability's absence produces a lesser mode. The row carries **no `optional` set**. Two inferred rows remain, both ML-2's.

**Module B — model integrity (PS §2.2.2)**

| Plug-in | `requires` | `optional` → if missing |
|---|---|---|
| `model.weight_digest` | `MODEL_WEIGHTS`, `REFERENCE_MANIFEST` | — |
| `model.fingerprint` | `MODEL_PREDICT`, `REFERENCE_MODEL_BATTERY` | — |
| `model.strip` | `MODEL_PREDICT` **only** | — |
| `model.weight_stats` | `MODEL_WEIGHTS` | `MODEL_ACTIVATIONS` → `DEGRADED`, parameter stats only; `REFERENCE_MODEL_BATTERY` → `DEGRADED`, absolute stats only — **confirmed by ML-2** |
| `model.neural_cleanse` | `MODEL_PREDICT` | `MODEL_GRADIENTS` → `DEGRADED`, gradient-free search — **confirmed by ML-2** |
| `model.intrinsic_probes` | `MODEL_PREDICT` | — |
| `model.graph_structure` | `MODEL_ARCHITECTURE` | — |
| `model.anomalous` | `REFERENCE_MODEL_BATTERY` | — |
| `model.data_consistency` | `DATASET_IMAGES`, `MODEL_PREDICT` | — | **A `Detector`, not a `ModelCheck`** — see below |

**`model.data_consistency` is a `Detector`, not a `ModelCheck` — corrected 2026-09-19.**
`Module-B` §6.9 places it as a `ModelCheck` taking `reference_probes`. But what it actually needs is
*"the **supplied** training data"* — the **contributed dataset under audit** — and `reference_probes`
is the `REFERENCE_CLEAN_SET`, a different `Dataset` that `ModelCheck.check()` has no way to swap for
the contributed one. `Detector.detect(dataset, embeddings, model)` takes exactly this pair, and it
takes `model: ModelHandle | None` *for precisely this kind of check* — the same reason activation
clustering and spectral signatures are `Detector`s despite being model-side methods. It lives in
`detectors/model/` and registers as a `Detector`.

**`file access` was never a `Capability` — corrected 2026-09-19.** `model.weight_digest`'s row read
*"requires: file access, REFERENCE_MANIFEST"*, inherited verbatim from `backend_revised.md` §3.1. The
frozen enum has no such member, so a plug-in declaring it would fail capability resolution at startup
— the row was unbuildable as written. It requires `MODEL_WEIGHTS`, which is what reading a weight
digest actually needs.

**Both previously-inferred rows are now confirmed**, by `Plan/Module-B-Model-Integrity-Plan.md` §6.4
and §6.5 — the detector author ratified the declarations Backend had guessed. **No inferred rows
remain in the registry.**

The reference capabilities sit in `requires`, not `optional`, for the first two rows **on purpose**: with no manifest and no battery there is nothing to be anomalous *against*, so the honest state is `UNAVAILABLE` with "no reference supplied" — not a `DEGRADED` comparison against nothing. Per §5.5 #9 we generate all three references ourselves, so in practice they resolve `OK` — *but only because we built them, which the report must say.*

**Module D — distribution shift (PS §2.2.4)**

| Plug-in | `requires` |
|---|---|
| `drift.distribution` (PSI with the corrected `(1/N + 1/M)·χ²(B−1)` null, KS, MMD, energy) | `DATASET_IMAGES`, `REFERENCE_CLEAN_SET` |
| `drift.interpretable_axes` | `DATASET_IMAGES` |
| `drift.semantic_axis` | `DATASET_IMAGES`, `REFERENCE_CLEAN_SET` — it clusters the reference set |
| `drift.vs_manipulation` | output of the above |

**Module C**

| Plug-in | `requires` | `optional` → if missing |
|---|---|---|
| `prov.ledger_verify` | `INFERENCE_LEDGER` | `REFERENCE_MANIFEST` → `DEGRADED`, in-chain `model_registration` records only |
| `prov.recompute` | `INFERENCE_LEDGER`, `MODEL_PREDICT` | **none** — degrades at runtime on `payload_missing`, see below |

**`prov.recompute` declares no `optional` set — corrected 2026-09-19.** An earlier row gave it
`optional: DATASET_IMAGES → DEGRADED`. Wrong capability: `DATASET_IMAGES` is probed from the
**contributed dataset under audit**, whereas the images `prov.recompute` re-runs come from the
**inference ledger's payload store** — a different artefact, with no enum member describing it, and
deliberately so (the payload store is not part of the chain; §7.7). The honest shape is Module C
§7.7's: the check resolves `OK` on capability and reports `payload_missing` → `DEGRADED` at runtime
when a referenced payload is absent, **never a false pass**. A capability it cannot actually probe
would have resolved `DEGRADED` on every scan of a dataset that happened to be image-less, for a
reason unrelated to what it needs.

From the Module C plan §3.3, otherwise adopted verbatim. **`prov.ledger_verify` must not declare `SIGNING_KEY`** — verification uses the public trust root only, and requiring a private key to *verify* would make an air-gapped third-party audit impossible, which is the opposite of the module's purpose.

**PS-audit and threat-sweep rows, now placed.** `model.anomalous` (G7), `model.graph_structure`
(13 B2), `model.intrinsic_probes` and `model.data_consistency` are registered above;
`data.negative_space` and `data.metadata_anomaly` sit in Module A's table; `drift.semantic_axis` in
Module D's.

**G3 is closed inside `model.weight_stats`, not by a separate detector — ML-2's call, adopted.** An
earlier draft of this plan listed `model.activation_statistics` as a new row. `Module-B` §6.4 points
out that `model.weight_stats` **already declares `optional: MODEL_ACTIVATIONS` for exactly this
purpose**, so a separate detector would be a second plug-in competing for one capability and one PS
clause. The activation-statistics path inside `weight_stats` is extended to satisfy G3 outright
(dead units, saturated units, dynamic range, divergence vs the reference battery). One clause, one
detector.

