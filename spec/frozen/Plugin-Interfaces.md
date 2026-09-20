---
tags: [architecture, interfaces]
status: decided
---

# Plug-in Interfaces

Concrete contracts for the extension points named in [[../Decisions/ADR-008-Plugin-Architecture]]. This is the first real blocker on [[../Team/Tasks|the task board]] — nothing in Module A/B/D can be written against an interface that doesn't exist yet.

**Amended 2026-09-18** after a teammate-led review (`Consolidated/01-Review-of-Existing-Plan.md`) found the original `access_level` design was factually wrong, not just simplified — see the capability model below. The `Finding` and `Sample` shapes were also widened for the same reason: the originals couldn't represent things the system is required to say. Full argument: `Consolidated/03-Architecture.md`, `06-Risk-Reporting-and-Governance.md`.

**Amended a third time 2026-09-19** after a contract-level review (`Post-Merge-Review-2026-09-19.md`) found six shapes in this file that could not express things the system is required to do — all of them inside the P0 freeze, so all of them cheaper now than after the loaders exist. In order: the capability set was attached to `ModelHandle` although two thirds of the enum describes the dataset and the run context; `Detector` received no model, so activation clustering and spectral signatures were unbuildable; `ModelCheck` typed a battery of reference *models* as a `Dataset`; `DriftTest` declared no capabilities and received no images; and `Finding` carried neither `scan_id` nor `produced_by`. Each is marked inline below.

**Amended again 2026-09-18** to fix a conflict this file itself introduced in the first amendment (`Sample.labels` and `Dataset.annotations` were both stored representations of the same information, which can drift apart), and to add the `PyTorchLoader` decided in [[../Decisions/ADR-001-Scope-Freeze-Formats]]. Source: `Tanmay/backend_revised.md` §1.1, §5.5 #5/#10.

**Amended a fourth time 2026-09-19** by Backend, landing with the `core/types.py` commit that puts
the freeze in force (`Plan/backend_plan.md` §5.11). Eight ADDITIVE changes — no field is removed or
retyped, so nothing already written against this contract breaks: `Finding.nature` (§5.9),
`Finding.exclusion_reason` (§5.5), `Finding.availability` (Backend), `Evidence.data` plus the
content-addressed `Evidence.path` rule (§5.3), `Sample.contributor_source` (§5.10),
`Capability.SUSPECT_INPUTS` (Backend), the two ledger protocols (§5.2, resolving Module C's O1) and
`Remediator` (§5.13). Each is marked `[added]` inline with the argument for it.

## Shared types

**`Finding`** — the one shape every detector-type plug-in returns, and the only thing [[Module-E-Dashboard-Governance|Module E]] needs to understand to aggregate anything:
```
Finding {
  finding_id: str
  scan_id: str                   # which scan produced this — needed to group per-scan generated
                                 #   coverage, and required for PS §2.3's reproducible audit log
  detector_id: str
  detector_version: str          # ties a finding to the exact code that produced it
  target_type: sample | contributor | batch | model | record | dataset
  target_ref: str                # id within that type — "the affected asset"
  severity: info | low | medium | high | critical    # impact
  confidence: float              # [0,1], calibrated belief — kept separate from severity on purpose
  score_raw: float                # the detector's own raw score, before calibration
  threshold: float                 # the threshold applied, for recalibration/audit later
  reason: str                     # human-readable, per PS §2.2.5 — should name the evidence, not just a score
  attack_class: str               # declared taxonomy id — this is what makes the coverage statement generatable, not hand-written
  evidence: [Evidence]            # {kind, caption, path, data}; kind ∈ image_crop | contact_sheet | heatmap | plot | table | hash | json
  access_assumptions: [str]       # required PER ASSESSMENT by PS §2.2.2, not just once per module
  limitations: [str]
  disposition: accept | review | quarantine
  disposition_rule: str           # which rule in the disposition profile fired — "why this disposition?" always has a specific answer
  produced_by: str                # run context: code commit + profile hash, so a finding is reproducible
  nature: adversarial | quality | indeterminate       # [added] §5.9
  exclusion_reason: capability | budget | None        # [added] §5.5
  availability: OK | DEGRADED | UNAVAILABLE | ERROR   # [added] Backend
}
```
**`nature` [added]** (§5.9). PS §2.1 says data may be mislabelled *"deliberately or inadvertently"*,
and the dispositions express severity of RESPONSE, not nature of CAUSE. Without this field a
contributor supplying duplicates and OOD imagery — a collection-quality problem — and a contributor
injecting triggers collapse into the same `quarantine` bucket, losing exactly the distinction an
acceptance officer most needs, because the two have entirely different remedies. `indeterminate` is
the honest default and most detectors will emit it; the field earns its place on the two that can
tell the difference.

**`exclusion_reason` [added]** (§5.5). The temptation was a fifth `Availability` state for budget
exclusion. Resisted: `Availability` is frozen at four and every consumer — renderer, coverage
generator, the dashboard [[Module-E-Dashboard-Governance|Module E]] is already mocking — switches on
exactly those four, and a fifth breaks all of them for information that is not a *state* at all but
an *attribute of one*. `UNAVAILABLE` continues to mean "this did not run, and here is why"; this
says which kind of why. A check that COULD NOT run and one that WAS NOT ASKED TO are different facts
about the scan, and the report must be able to say which.

**`availability` [added]** (Backend). The resolution state this finding was produced under. The
state machine below requires all four states to print at equal prominence, yet the frozen shape
carried no per-finding availability — and joining findings back to the coverage rows by
`detector_id` fails the moment one detector emits both an `OK` finding and a `DEGRADED` note.
`exclusion_reason` does not cover it: that one only qualifies `UNAVAILABLE`.
**Why severity and confidence are separate fields, not one.** The PS says "confidence or severity" — that's a floor, not a design. Severity is impact; confidence is belief. A high-confidence low-severity finding and a low-confidence high-severity one call for opposite actions, and one float can't distinguish them.

**Why `target_type` is polymorphic.** It's what lets one record serve sample-, contributor-, model- and record-level findings without four parallel schemas — [[Module-E-Dashboard-Governance|Module E]] groups by it.

**`Evidence`** — `{kind, caption, path, data}`, the shape `Finding.evidence` holds.

**`path` is the BARE CONTENT HASH** (§5.3), not a scan-relative path. Content addressing exists so
an identical crop is stored once; a per-scan `evidence/` folder stores it once *per scan that
references it*, and worse, forces `path` to be scan-relative — re-coupling it to `scan_id`, the
exact volatile field content addressing exists to remove. Renderer and dashboard both resolve it
against the shared `<out_dir>/evidence/`.

**`data` [added]** is the in-memory payload for `table`/`json`/`hash` evidence, and it is TRANSPORT,
not the record: `EvidenceStore.materialise` writes it, hashes it, and fills `path` at write time.
Without it a detector would have to know the output directory in order to emit a two-line table,
which puts filesystem layout inside every plug-in. It is also the channel for per-detector
diagnostics that are not `Finding` fields. There is deliberately **no extension dict on `Finding`**:
one would become an untyped side-channel that Module E cannot render and the schema cannot validate.

A plug-in that can't run under the available capabilities still returns a `Finding` — see the `UNAVAILABLE`/`DEGRADED` states below. Evidence explicitly names what's missing and why. No silent skips, ever (PS §2.2.6 fallback requirement).

**`Sample`** — canonical per-image shape every `DatasetLoader` produces:
```
Sample {
  sample_id: str
  content_sha256: str           # also half the feature-cache key
  path: Path
  width: int
  height: int                   # width/height are MANDATORY — YOLO's normalised boxes are
                                #   unrecoverable in absolute pixels without them
  labels: [Label]
  contributor: str | None       # None = genuinely unknown. NEVER a default.
  batch: str | None
  source_meta: dict
  contributor_source: sidecar | directory | format_field | exif_cluster | none | None  # [added] §5.10
}
```
**`contributor_source` [added]** (§5.10) records which of the five precedence tiers below actually
resolved `contributor`. The precedence list already requires that *"the report should say which
level was used"*, and no field carried it, so the report could not. A contributor flagged on EXIF
clustering and one flagged from a signed sidecar are not the same claim — the first is a derived
proxy the report must label a hypothesis, the second is auditable evidence — and the party being
flagged is likely another unit. Same reason `disposition_rule` exists: the claim carries its own
provenance.

**`Dataset`**
```
Dataset {
  samples: [Sample]
  categories: [Category]
  annotations() -> [Annotation]  # DERIVED VIEW over every sample's labels — not stored
  capabilities() -> set[Capability]   # DATASET_* only; probed, not assumed — images actually
                                      #   decodable, labels actually present, contributor actually
                                      #   resolved on at least one Sample
}
```
**`Sample.labels` is the only stored representation; `Dataset.annotations()` is a derived view, not a second field.** The first version of this amendment stored both, which is exactly the two-sources-of-truth bug the rest of this document argues against elsewhere: a store and a projection can't disagree, two stores can. Per-sample detectors iterate `Sample.labels` directly (inline wins where the work happens); `annotations()` computes a flat, COCO-shaped view on demand for bbox-heavy work and export.

**`contributor` lives on `Sample`, not on `Dataset`.** The original design put it in a dataset-level metadata dict, which can't answer "which contributor sent sample 4471?" — and a submission has *many* contributors, which is the premise of PS §2.2.1's source-level aggregation. `None` rather than a default string is load-bearing: a default would let the [[Module-A-Data-Integrity|contributor rollup]] produce a confident-looking single-contributor report from a dataset carrying no contributor data at all.

Resolution precedence when populating `contributor` (each level below the first is weaker evidence, and the report should say which level was used):
1. an explicit `contributors.yaml` sidecar — preferred, auditable
2. directory structure (`data/contrib_B/…`)
3. COCO `images[].source` / a YOLO `data.yaml` extension field
4. EXIF camera-serial clustering — a **derived proxy**, must be labelled as a hypothesis in the report, not a fact
5. none of the above → `contributor: None`, and the rollup reports `UNAVAILABLE` for that sample rather than guessing

**`CapabilitySet`** — the set plug-ins are resolved against. **It is assembled per scan by the orchestrator, not returned by any one object** (amended 2026-09-19): the enum spans three different sources, and no single component can answer for all of them — a `ModelHandle` cannot know whether a reference clean set was supplied or a signing key exists.
```
Capability =
  # --- probed from the Dataset, by DatasetLoader
  DATASET_IMAGES | DATASET_LABELS | DATASET_CONTRIBUTOR_META
  # --- probed from the ModelHandle, by ModelLoader
  | MODEL_PREDICT | MODEL_LOGITS | MODEL_ACTIVATIONS | MODEL_WEIGHTS
  | MODEL_GRADIENTS | MODEL_ARCHITECTURE
  # --- resolved from the scan's run context / profile, by the orchestrator
  | REFERENCE_CLEAN_SET | SUSPECT_INPUTS | REFERENCE_MODEL_BATTERY | REFERENCE_MANIFEST
  | INFERENCE_LEDGER | SIGNING_KEY

CapabilitySet = ModelHandle.capabilities()      # MODEL_* only
              ∪ Dataset.capabilities()          # DATASET_* only
              ∪ RunContext.capabilities()       # REFERENCE_*, INFERENCE_LEDGER, SIGNING_KEY
```
**`SUSPECT_INPUTS` [added]** (Backend) is a run-context capability: inputs supplied for
assessment *that are not asserted clean*. It exists because `model.strip` cannot be honest without
it. STRIP is an INPUT-LEVEL trigger detector — it asks whether THIS INPUT carries a trigger — so a
probe set that is clean by construction carries none, and running STRIP as a model-level check from
clean probes alone is not a use the method supports: it ends up measuring a training artifact (the
backdoored model is simply more confident) rather than a backdoor signature. With this member
`model.strip` `requires` a suspect set and resolves `UNAVAILABLE` without one, which is the honest
state. It is distinct from `REFERENCE_CLEAN_SET`, whose whole content is the clean assertion.

Each source probes only what it can observe; the orchestrator unions the three into the single `CapabilitySet` that the `OK`/`DEGRADED`/`UNAVAILABLE` state machine below resolves against, and that the report header prints. Splitting it this way is what keeps probing *active* — a `RunContext` that reports `REFERENCE_CLEAN_SET` present must have actually opened it, exactly as a `ModelHandle` reporting `MODEL_GRADIENTS` must have attempted a backward pass.

**`ModelHandle`** — canonical wrapper every `ModelLoader` produces, built around a **probed capability set**, not a declared format-based label:
```
ModelHandle {
  predict(x) -> output
  capabilities() -> set[Capability]     # MODEL_* only; PROBED against the actual loaded model,
                                        #   not assumed from file format
  get_weights() -> dict | None
  get_graph() -> Graph | None
  weight_digest() -> str
}
```
**Why not `access_level: "white-box" | "black-box"`.** That was the original design, and it's factually wrong: a bare ONNX inference session has weights and activations but genuinely **no backward pass** — the `Gradient` operator isn't registered at inference time. Declaring ONNX "white-box, gradients available" (as the original Module B note did) means declaring Neural Cleanse applicable to a model it cannot run on. A binary enum structurally cannot express "weights yes, gradients no."

What each adapter actually probes to:
| Adapter | Capabilities |
|---|---|
| **ONNX** (inference session) | predict, logits, weights, architecture. **Activations only after graph surgery** (extending `graph.output` with the wanted intermediate tensors before the session is created — no ONNX Runtime API returns them otherwise). **Gradients: permanently absent on the deployed scanner.** They exist only via a separate `onnxruntime-training` build (`generate_artifacts()`), and per [[../Decisions/ADR-001-Scope-Freeze-Formats]] that package is **not bundled** — a second ONNX Runtime variant in one air-gapped image invites version skew nobody can debug in the field. `MODEL_GRADIENTS` therefore probes `False` for every ONNX model, always; this is a declared, deliberate narrowing, not an unresolved gap. |
| **TorchScript** | predict, logits, activations, weights; gradients if not frozen (`torch.jit.freeze` inlines parameters as constants and sets `requires_grad = False`, which a probe must actually attempt, not infer from the file extension) |
| **PyTorch `nn.Module`** (via `PyTorchLoader`, see [[../Decisions/ADR-001-Scope-Freeze-Formats]]) | predict, logits, activations, weights, architecture, **and gradients that always work** — the one adapter where Neural Cleanse's full-confidence path is reliable rather than conditional |
| **Query-only endpoint** | predict; logits if returned |

**Probing is still active, not declarative, for everything except the ONNX-gradients row above** — an adapter attempts the minimal operation (extract one activation, check `requires_grad`) and records what actually happened, because only an attempt discovers a frozen TorchScript model. ONNX gradients are the one capability decided in advance rather than probed, precisely because we chose not to vendor what probing them would require. Results cache per model digest.

**Why build a third loader at all.** PS §2.2.6 names *"PyTorch/TorchScript"* as one format category, not two — a `.pt`/`.pth` checkpoint loaded via `torch.load` into an `nn.Module` is not a TorchScript archive, and `TorchScriptLoader` can't open one. Without `PyTorchLoader`, the richest capability row (`nn.Module`: "everything, including gradients that always work") was unreachable, and the two model formats we *could* actually scan both had a conditional or absent gradient story. `PyTorchLoader` must call `torch.load(path, weights_only=True)` — mandatory, not advisory, since the model supplier is untrusted by premise and unrestricted `torch.load` executes pickle — and the dependency pin is `torch>=2.6.0`, because below that version `weights_only=True` didn't fully close the RCE path (CVE-2025-32434; the flag existed since 2.4 but the fix landed in 2.6.0).

Every `Detector`/`ModelCheck`/`DriftTest` plug-in declares `requires` and `optional` capability sets. Before anything runs, the orchestrator resolves each plug-in against the assembled scan-level `CapabilitySet`:
```
requires ⊆ CapabilitySet?   no ──► UNAVAILABLE + reason + missing[]
                            yes ─► optional ⊆ CapabilitySet?  no ──► DEGRADED + reason + mode used
                                                              yes ─► OK
```
All three states print in the report at equal prominence — `UNAVAILABLE` is not a skip, it's a `Finding` with `evidence` naming exactly what capability was missing and why. A plug-in that *raises* is a separate `ERROR` state (a bug in our code), never conflated with a legitimate degradation — collapsing the two would let defects hide inside what looks like honest coverage reporting.

Neural Cleanse is the concrete case this exists for: it declares `MODEL_GRADIENTS` as `optional`, not `requires`. On a `PyTorchLoader` or an unfrozen `TorchScriptLoader` model, gradients resolve `True` and it runs at full confidence (`OK`). On ONNX, or a frozen TorchScript model, gradients resolve `False` and it falls back to a gradient-free search at lower confidence and higher cost (`DEGRADED`) — and the report says which happened and why, never a silent skip, never a crash. For ONNX specifically this `DEGRADED` path is now the permanent, declared outcome (see the adapter table above), not a per-model coin flip.

## Interfaces

**`DatasetLoader`**
```
supports(path) -> bool
load(path) -> Dataset
```
Implementations for this submission: `COCOLoader`, `YOLOLoader` (see [[../Decisions/ADR-001-Scope-Freeze-Formats]]).

**`ModelLoader`**
```
supports(path) -> bool
load(path) -> ModelHandle
```
Implementations: `ONNXLoader`, `TorchScriptLoader`, `PyTorchLoader` (see [[../Decisions/ADR-001-Scope-Freeze-Formats]]). All three probe capabilities on load per the table above — no format gets a hardcoded capability set, including the one (ONNX gradients) that's decided in advance rather than probed.

**`Detector`** (Module A)
```
requires: set[Capability]
optional: set[Capability]
detect(dataset: Dataset,
       embeddings: EmbeddingIndex | None,
       model: ModelHandle | None) -> [Finding]
```
Implementations: near-duplicate (pHash), label-flip (Cleanlab), trigger/OOD (activation clustering + spectral signatures via ART, k-NN in embedding space). See [[Module-A-Data-Integrity]].

**Why `Detector` takes a model at all** (added 2026-09-19). Most Module A detectors run purely on the dataset and the independent backbone's embeddings, per [[../Decisions/ADR-009-Embedding-Backbone]] — and they must, since judging a dataset with the model trained on it is circular. But **activation clustering and spectral signatures are *defined on* the contributed model's activations** and cannot exist otherwise: they are model-integrity methods applied to a data question. They declare `MODEL_ACTIVATIONS` in `requires` and resolve `UNAVAILABLE` when it is absent — which is also why the original signature, passing no model, could not have expressed two of the four trigger-artifact methods at all. `None` is the normal case; only those two detectors ask for it, and the report labels their findings as depending on the contributed model.

**`ModelCheck`** (Module B)
```
requires: set[Capability]
optional: set[Capability]
check(model: ModelHandle,
      reference_probes: Dataset | None,        # REFERENCE_CLEAN_SET — images to probe with
      reference_battery: ModelBattery | None   # REFERENCE_MODEL_BATTERY — reference MODELS
      ) -> [Finding]

ModelBattery {
  models: [ModelHandle]          # known-good reference models
  manifest: Manifest | None      # REFERENCE_MANIFEST — declared weight digests / fingerprints
}
```
**Two different references, two different types** (amended 2026-09-19). The original signature passed a single `Dataset` as `reference_battery`, but weight statistics compares per-layer moments **against other models**, and the substitution check compares against a **declared digest or fingerprint** — neither is expressible as a `Dataset`. STRIP and the behavioural fingerprint want probe *images*; weight statistics and the digest check want reference *models* and a manifest. Splitting the parameter is what lets each plug-in declare the reference capability it actually needs (`REFERENCE_CLEAN_SET` vs `REFERENCE_MODEL_BATTERY` vs `REFERENCE_MANIFEST`) and resolve `UNAVAILABLE` independently, rather than as one undifferentiated missing reference.
Implementations: Neural Cleanse, STRIP. Each declares its capability needs; the orchestrator resolves `OK`/`DEGRADED`/`UNAVAILABLE` before calling `check()` at all, per the state machine above. See [[Module-B-Model-Integrity]].

**`DriftTest`** (Module D)
```
requires: set[Capability]
optional: set[Capability]
assess(reference: Dataset, incoming: Dataset,
       reference_dist: EmbeddingDistribution | None,
       incoming_dist: EmbeddingDistribution | None) -> [Finding]
```
Implementations: embedding-distance, PSI, KS, interpretable axes. See [[Module-D-Drift-Detection]].

**Two corrections, both 2026-09-19.** `DriftTest` was the only plug-in interface with **no `requires`/`optional` declaration**, yet Module D is precisely the module that depends on `REFERENCE_CLEAN_SET` — with no reference the honest state is `UNAVAILABLE`, and it previously had no way to say so, nor any way to appear in the generated coverage statement. And the signature took **embedding distributions only**, which cannot express the interpretable-axes test at all: brightness, RMS contrast, Laplacian sharpness, JPEG quality and EXIF sensor/ISO are computed from images and file metadata, not from embeddings. Since interpretable axes are what make a drift finding actionable (*"illumination shifted 2.3σ"* rather than *"dimension 341 moved"*), the interface has to carry the `Dataset`; the embedding distributions stay as the cached, optional fast path for the purely statistical tests.

**`AuditLedger` / `InferenceLedgerSource`** — **two** protocols, not one **[added]** (§5.2;
adopted from [[Module-C-Provenance-Seal]] §3.1's open item **O1**, resolved 2026-09-19):
```
AuditLedger                          # the scanner WRITES this one
  append(record: Mapping) -> str     # returns record id
  capabilities() -> set[Capability]  # {SIGNING_KEY} iff it can actually sign

InferenceLedgerSource                # the scanner READS this one — the artefact under audit
  records() -> Iterator[Mapping]
  capabilities() -> set[Capability]  # {INFERENCE_LEDGER} iff it opened
```
An earlier draft had a single `Ledger` carrying both capabilities. The Module C plan found that this
conflates two genuinely different objects, and it is right: the **scanner's own audit ledger**
(written by the orchestrator's last step, needs a signing key) and the **field inference ledger under
audit** (a read-only input to `prov.*`, needing no key of ours) are different files, with different
keys, different genesis records and a different trust root. One protocol cannot honestly probe for
both — and the single-protocol version would have let an unsigned test-only JSONL file present
itself as a field ledger under audit, which is false assurance. Backend ships `NullAuditLedger`,
`JsonlAuditLedger` (both reporting `{}` — a chain with no key is not a `SIGNING_KEY`) and
`NullInferenceLedger`; Crypto's `SealedLedger` implements **both** and replaces all three at P2.
Probing is active here as everywhere: `capabilities()` opens the ledger and attempts the operation,
never returns a constant.

**`Remediator`** **[added]** (§5.13) — the interface [[../Decisions/ADR-004-Scope-Boundaries]]'s
amendment needed and the frozen contract did not have, so the amendment had nowhere to land:
```
requires: set[Capability]
optional: set[Capability]
remediate(dataset: Dataset, findings: [Finding],
          model: ModelHandle | None) -> RemediationResult

RemediationResult {
  kind: "dataset" | "model"
  artefact_path: Path
  artefact_digest: str        # a NEW artefact with its OWN digest
  manifest: [RemovalRecord]   # what was removed/pruned, and which Finding justified it
  source_scan_id: str
}
```
The ADR states the boundary and this shape is what makes it structural rather than a convention:
*"a `Detector` or `ModelCheck` plug-in may never retrain. A `Remediator` is a separate, explicitly
invoked component that can, and its output is re-assessed rather than trusted."* It is therefore a
**separate interface, never a `Detector`/`ModelCheck` subtype** — so no baseline plug-in can be
registered as one, or vice versa — and the orchestrator's baseline path never constructs one; it is
reached only through `cva remediate`, a distinct entry point. A CI import invariant enforces both.
The baseline no-retraining constraint (PS §2.2.6) is unchanged and absolute.

## What's deliberately not an interface
Per [[../Decisions/ADR-003-Crypto-Design]], the seal/audit log (Module C) and the report-aggregation logic (Module E) stay concrete core modules — only one implementation of each is planned, so wrapping them in an interface now would be speculative. **The two ledger protocols above are not a counter-example** (noted 2026-09-19): they are not an extension point for alternative seal implementations but the *shape the orchestrator calls through before Crypto's `provenance/` exists*, and the thing `RunContext.capabilities()` probes `INFERENCE_LEDGER` and `SIGNING_KEY` against. Without them the orchestrator would carry a `TODO` at its last step through four gates and could not answer either capability honestly. Capability negotiation itself is also core, not a plug-in point — it's the thing plug-ins are resolved against.

## The three schemas frozen alongside these types
`Finding`, `Evidence`, `Dataset`/`Sample`, `ModelHandle`, `ModelBattery`, the `Capability` enum and the `CapabilitySet` assembly rule, and `Availability` (`OK`/`DEGRADED`/`UNAVAILABLE`/`ERROR`) are frozen together **from the `core/types.py` commit**, not at the end of P0 — moved forward 2026-09-19 (`Plan/backend_plan.md` §5.11). Four seats are told to build against these types today; if the types can still move until the end of P0, every seat building today is building on sand and the freeze protects nobody during the only window in which it matters. Changes after that commit need whole-team agreement, per [[../Team/Tasks]]. Three JSON Schemas are versioned artifacts alongside them, each with a versioned `$id` and a `schema_version` field on every instance (a report that can't be re-read next month isn't the "reproducible audit log" PS §2.3 asks for):
- `report.schema.json` — the assurance report. This is the named PS §2.3 submission deliverable.
- `profile.schema.json` — config profiles (thresholds, disposition rules; unknown keys are a load error, not a warning).
- `scenario.schema.json` — attack-lab scenario manifests (crosses into [[../Attacks/Attack-Simulation-Plan]]).

The sealed ledger record is deliberately **not** a fourth schema here — its shape is fixed by what gets hashed (see [[Module-C-Provenance-Seal]]), and a separate schema describing the same structure would be a second definition that could silently drift from the one thing that's actually hashed.

## Related
- [[../Decisions/ADR-008-Plugin-Architecture]]
- [[System-Overview]]
- [[../Team/Tasks]]
- `Consolidated/03-Architecture` · `Consolidated/06-Risk-Reporting-and-Governance` · `Consolidated/01-Review-of-Existing-Plan`
