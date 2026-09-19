# Module A — Data Integrity

Implements `sih26228-notes/Plan/Module-A-Data-Integrity-Plan.md` (PS §2.2.1): nine `Detector`
plug-ins, registered in core's `DETECTOR_REGISTRY`, plus ground-truth attack scripts in `attacklab/`.
Built against the real core types (`cva/core/types.py`) and loaders; the one thing still faked is the
embedding backbone (see below).

## Layout and conventions

| | |
|---|---|
| `data.near_dup` · `data.duplicate_label_conflict` | `near_duplicate.py`, `duplicate_label_conflict.py` |
| `data.label_consistency` · `data.systematic_mislabel` | `label_flip.py`, `systematic_mislabel.py` |
| `data.trigger_artifact` · `data.ood` | `trigger_ood.py` (one file, two ids) |
| `data.metadata_anomaly` · `data.annotation_geometry` · `data.negative_space` | one file each |
| shared plumbing | `base.py` (Params, evidence store, `make_finding`, ctx helpers) |
| real core types, re-exported in one place; plus the ONLY remaining stand-in (the fake embedding backbone) | `_stub_types.py` |
| attack classes | names in `taxonomy.py`; **definitions live in `cva/core/types.py::ATTACK_CLASSES`** |

* **Registration** is `@register_detector` (`cva.core.registry.DETECTOR_REGISTRY`), signature
  `detect(dataset, embeddings, model, ctx)` — the same contract `model.data_consistency` uses.
* **Zero-argument construction.** Everything a detector needs arrives through `ctx`
  (`CheckContext`): thresholds in `ctx.profile["<detector_id>"]` (unknown keys *inside that
  namespace* are an error; other plug-ins' keys are ignored), evidence dir = `ctx.out_dir`,
  `ctx.scan_id`, `ctx.rng_seed`. No constructor injection.
* **Datasets are used through the frozen contract only** — `samples`, `categories`, `annotations()`,
  `capabilities()`. The real `InMemoryDataset` has no `sample(id)`, `category_name(id)` or `len()`; detectors
  use `base.sample_by_id` / `category_name` / `n_samples`, which derive them. Enforced by
  `tests/detectors/data/test_real_types.py` (a static AST guard on detector source, plus every test
  running on the real type and a round trip through the real `COCOLoader`). An earlier stand-in `Dataset`
  with those conveniences let five of six detectors crash on a real dataset.
* One `Finding` type (`cva.core.types.Finding`). `finding_id` = `sha256(detector_id‖version‖
  target_type‖target_ref‖attack_class)[:16]` with `\x1f` separators; never hashes `scan_id`.
  `disposition` is the placeholder `review` / `pending_risk_engine`; `confidence` is uncalibrated.
* **Never a silent skip.** Input a detector cannot test (too few files, no cohort, no peers, no
  embeddings/reference) yields a `not_performed` finding naming what was missing — never `[]`, which
  would read as "checked, nothing wrong". `[]` means the check ran and found nothing.
* **Other modules' faults are reported, never raised.** A model adapter that misreports its capabilities or
  cannot be queried, or a bad setting, does not crash Module A and is not swallowed: it is recorded and
  surfaced as a `DEGRADED` finding while the rest of the detector still runs. Unreadable files are counted
  and disclosed on findings (and, if most are unreadable, reported as `not_performed`).
* Findings that name a contributor state the resolution tier; `exif_cluster` reads as `HYPOTHESIS`.

Run: `pip install -e '.[data]'` then `python -m pytest tests/detectors/data`.

## Statistical policy (multiplicity)

Three detectors run one hypothesis test per (contributor × class …) cell. Uncorrected, that is
~K² tests per contributor — invisible at K=4, ~6,000 at COCO's K=80.

* `systematic_mislabel`: Bonferroni over every (contributor, declared, read-as) cell that has a
  cohort; two-sample Fisher test against pooled peers; a cell must also beat the **worst single
  peer** by `gap_min`.
* `annotation_geometry`: KS p-value (the weakest across peers) Bonferroni-corrected over every
  (contributor, category, feature); must differ from **every** peer, same direction.
* `negative_space`: Fisher vs pooled peers, Bonferroni over contributors, rate ≥ `ratio_min` × worst peer.

**Known cost of the worst-peer gate:** two contributors applying the *same* wrong mapping each become
the other's worst peer, so **neither is flagged** (a lone poisoned contributor is). This is pinned by
`test_KNOWN_LIMITATION_two_contributors_sharing_a_mislabelling_mask_each_other` and stamped on the
finding. A median-of-peers gate would recover collusion detection with three or more peers at the price
of weaker protection with few — a design choice, not made here.

A one-sample binomial against a cohort rate estimated from the same data is anti-conservative, hence
Fisher. **Validated on synthetic K=4 and K=6 only** (0 false positives on clean data over 4–8 seeds
at K=6); **not tested at COCO scale** — `min_n` / `p_max` will need re-tuning there, and the
finding's `limitations` say so.

## What the numbers are — and are not

Every test runs on a **synthetic** corpus (`attacklab/synth_dataset.py`) with a **stub** embedding
backbone that was **tuned against the detectors the tests then exercise**, so a green suite
validates the harness and detector mechanics, **not detection quality**. Specifically:

* A plain random-projection stub gave 73% probe accuracy and 55/240 false flags on clean data, so a
  class-separable block was added; `semantic_weight` (0.8) was chosen by sweeping detector
  pass/fail over the test corpora. The suite is one fitted hyperparameter away from green.
* An earlier global-statistics block was **derived from the OOD attack's own images** and made
  `data.ood` recall go 0 → 1.0. It was **removed**. `test_ood_insertion_found` is now `xfail` for
  exactly that reason; `data.ood` is covered by `test_ood_mechanism_with_explicit_vectors`
  (hand-built vectors, nothing fitted).
* No AUROC / detection rate is claimed. The plan's P3 gate (≥0.90 AUROC @ 1% FPR) is **not
  measured**; it needs the real DINOv2 backbone, a real corpus, and an evaluation harness
  (`cva/bench/` now provides the harness: `stats.py` — `wilson`, `bootstrap_rate`, `required_n`,
  `separation`; `protocol.py` — `group_split`, `pick_threshold`, `nested_evaluate`; plus
  `sensitivity.py`, `ablation.py`, `validate.py`. **Module A does not use it yet** — that, with real
  embeddings, is the work that would produce a number.)

Honest negatives found while building:

* **ART's spectral signatures and activation clustering did not separate poisoned from clean
  activations** on the toy CNN (mean spectral z ≈ 0; 6–10% of poisoned samples in the small
  cluster). ART's `SpectralSignatureDefense` removes a fixed fraction of every class and its
  `relative-size` analyser crashes when no cluster qualifies, so its building blocks are used behind
  gates that stay silent on clean data, capped at `low`, and unit-tested on **planted** clusters
  only. Whether ART earns its dependency tree for two methods that didn't work is an open question.
* `data.ood`'s threshold is seed-sensitive when estimated from a 200-image reference.
* `data.negative_space` recall was 0.31 with a coverage rule and ~1.0 with IoU matching.
* `data.annotation_geometry` mis-flagged untampered contributors when a tampered peer sat in the
  cohort (fixed: differ from every peer).
* `data.metadata_anomaly`: `single_camera` fired on a group where one file named a camera among many
  stripped ones (fixed: needs `camera_min_share` of files naming a camera); `pooled_cameras` was
  documented but never implemented (now implemented).

## Open items — owners other than Module A

1. **[Backend issue — to be filed]** **No orchestrator path runs any `Detector` yet.** `cva/core/orchestrator.scan(model, ctx)` takes no
   dataset and iterates only `REGISTRY` (ModelChecks); `DETECTOR_REGISTRY` is never read, so
   `model.data_consistency` is equally unreachable. `report_json.coverage_of` builds coverage from
   `result.plan`, so **Module A appears in the coverage statement only once the orchestrator adds a
   detector pass.** `test_module_a_reaches_the_real_coverage_generator` shows the generator accepts
   Module A's rows once they are in a plan; `test_orchestrator_style_run_…` shows zero-arg + `ctx` works.
2. **V9 threshold jitter (`ctx.jitter` / `ctx.opt` routing) is unreconciled.** The review of this
   branch cites `main@66852b5` (V9) and `main@9c2db62`; neither commit exists in this repository (its
   `main` is `d5f63c7`), and `CheckContext.opt` here is a plain lookup with no jitter, so it could not
   be reproduced or adapted. Module A reads its tunables from `ctx.profile[<detector_id>]` through ONE
   function, `Params.from_ctx` — so routing them through `ctx.opt` (or a namespaced form of it) is a
   one-function change once the owner of `context.py` decides the convention.
3. **`data.ood` needs reference *embeddings*; `ctx.probes_x` holds raw images.** It reads
   `ctx.profile["reference_embeddings"]` (an `(M, d)` array in the same space as `embeddings`) — the
   same channel Module B uses for `trigger_candidate`. Comparing `probes_x` with embeddings would
   need an extractor Module A must not own. `CheckContext` needs a reference-embeddings field.
4. `backend_plan.md` §9.4 has no registry rows for `negative_space`, `annotation_geometry`,
   `systematic_mislabel`, `duplicate_label_conflict` (§9.5 lists their classes).
   `registry.registry_rows()` has them.
5. **`data.trigger_artifact` `requires` conflict:** the plan (§3/§6.3) puts `MODEL_ACTIVATIONS` in
   `requires`; the registry makes it `optional` (→ `DEGRADED` to frequency residue). One id can't do
   both without killing the model-free method; followed the registry.
6. **`negative_space` hard-requires `DATASET_CONTRIBUTOR_META`** (decision pending): a dataset with no
   contributor attribution gets no coverage of `negative_space_poisoning`. The alternative is
   `optional` with a degraded, clearly-labelled dataset-level mode; the registry row Backend adds
   must match whichever is chosen.
7. Detection vs classification: Module A assumes boxes / multi-label images; Module B's models are
   classifiers. Label checks use each image's *dominant* category (stated in `limitations`);
   `negative_space` assumes a `(N, M, 6)` detector output.
8. `finding_id`: plan §5.3 says `[:16]`; core's `Finding` default is `uuid[:12]` (random, not a hash).
   Module A follows the plan. `finding_id` also excludes dataset identity, so the same `sample_id` in
   two datasets collides in a shared store — a §5.3 question.
9. `exifread` is listed in the plan; Pillow's `ExifTags` was used. `scikit-learn` is not listed.
10. `attacklab/` now ships in the installed package (it must, to be importable); the air-gapped
   scanner image should exclude it — the layout rule is "not under `cva/`", not "not in the repo".
11. **Licences.** cleanlab is AGPL-3.0-or-later. `find_label_issues` is one call to swap. "Accepted,
   backend_plan.md §5.12" is a plan clause, not a distribution decision for a network-delivered tool —
   worth re-ratifying with whoever owns distribution.

## Not done

* ADR-006's disconnected-sandbox smoke test for imagehash / cleanlab / ART / the backbone: **not
  written and never executed** for Module A. (Module C's `tests/provenance/test_offline_smoke.py` is
  written but needs Linux `unshare -rn` and there is no CI, so it is skipped on every machine the
  team owns — "written" is not "executed" for either module.)
* No CI workflow; the import-boundary rules are pytest tests (`tests/boundaries/`) with nothing to run them.
* P3 quality measurement with confidence intervals. The harness exists in `cva/bench/`; Module A is
  not wired to it and has no real backbone or corpus to run it on.
* Held-out evaluation (Crypto's, per the plan).
