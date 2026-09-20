# Module D implementation plan and progress

Scope: PS 2.2.4, the Module D notes, Consolidated/17 full-product direction and
Consolidated/19 MVP sequencing. This PR targets `modules` and incorporates Backend's
B3–B7/reporting follow-ups. It is a first working increment, not completion of Module D.

## Current implementation

- Plug-ins implement the **existing** `core/interfaces.py:DriftTest` signature:
  `assess(reference, incoming, reference_dist, incoming_dist)`. Both dataset parameters
  are canonical `Dataset`s when present; `reference` now explicitly permits `None`
  for the plan-required reference-free image-axis description. Every plug-in declares `requires`, `optional`, `attack_classes`;
  the orchestrator resolves these declarations before execution and validates taxonomy.
- No `DriftBatch` or `BatchDriftTest` contract, and no added core registry. Concrete checks
  are composed explicitly at the entrypoint and passed as `Sequence[DriftTest]`. Configuration
  belongs to the detector package. The sole protocol amendment is the optional reference annotation; shared finding types are unchanged.
- The image-directory adapter returns an `InMemoryDataset` implementation with cached
  measurements. Existing COCO/YOLO Dataset instances work through `measure_dataset`;
  the CLI currently accepts image directories, without inventing annotations or provenance.
- Image decoding runs in shared S3, with S6 path containment and a 16-million-pixel S7
  ceiling checked **before** full decode. Header bombs, broken images, escaping symlinks
  and files modified during loading fail explicitly. S3's Python-level confinement limits
  remain disclosed. Whole-dataset decode is bounded by the shared sandbox timeout.
- PSI uses `(1/N + 1/M)` scaling and survival-function tails. Shared pooled quantile edges
  are label-invariant and include both infinite tails; duplicate edges collapse; a half-count
  prevents infinities. Sparse bins use a seeded conditional permutation test.
- KS plus BH correction across both statistics and axes; a minimum KS effect gate prevents
  significance alone being called material drift. Sparse permutation budgets expand to at
  least `ceil(2 * family_size / alpha)` so one PSI signal can survive BH. The effective budget
  and minimum attainable p-value are recorded, not hidden behind the requested default.
- Brightness, RMS contrast, sharpness, residual noise, RGB histogram proportions; JPEG
  quantization and EXIF ISO where complete. Missing metadata is stated. Correlated colour
  bins are not independent pieces of evidence; repeated-monitoring FDR is not claimed.
- Optional aligned `EmbeddingRows` is an adapter for DriftTest's existing `Any` distribution
  arguments, not a core contract. KS needs observations; a moments-only
  `EmbeddingDistribution` reports unavailable. Extractor identity/version/dimensions and
  sample order must match. Fixed projections and descriptive centroid distances are provided.
- Reference/incoming content-hash overlap prevents independent two-sample assessment,
  including renamed copies. Small comparison batches are unavailable. Without a reference,
  image axes still produce descriptive summaries, explicitly without PSI/KS or a drift claim.
- Shared risk engine evaluates an explicit review-capped drift profile. Detectors do not
  set dispositions. Confidence remains an explicitly uncalibrated zero placeholder until a
  held-out calibration artifact exists; `1-p` is never sold as an attack probability.
- `python -m cva.cli drift` is a lazy subcommand; no Torch/ONNX import is needed for it.
  The unreleased `cva.drift_cli` compatibility shim has been removed. Shared report builders and generated coverage
  are used unchanged in shape; dataset-only capability scoping is opt-in on the result.
- Normal scan IDs use the shared grammar with atomic, per-output-root/day allocation.
  Existing scan directories are never overwritten. `--selftest` uses the shared pinned clock
  and `selftest_scan_id(seed)`; finding IDs stay scan-invariant. The deployment should use a
  common output root for a scan namespace (the grammar permits 10,000 scans/day per namespace).
- Deferred semantic and manipulation checks declare `semantic_shift` and
  `suspicious_manipulation`, respectively. Unavailable rows remain in generated coverage.

## Review resolutions

1. Safety: shared S3/S6/S7; malformed-image errors reach argparse without a traceback.
2. Contracts: deleted parallel core types/registry; frozen Dataset/DriftTest now used.
3. Renderer: explicit asset kind and per-result relevant capabilities; current model behavior
   is preserved. The newer Backend renderer intentionally includes every capability for
   model scans; this PR does not add or remove those rows.
4. Coverage: shared `coverage_of`, shared Markdown writer, no hand-written replacement.
   Full JSON is validated against the published report schema. Backend's recent update also
   fixed the formerly missing common envelope fields; that code is reused, not duplicated.
5. Taxonomy: deferred checks corrected.
6. Independence: same content and partial overlap are unavailable.
7. Sparse PSI: family-aware permutation budget plus significance and null-calibration tests.
8. Risk: shared risk engine/profile owns policy; gaps keep their explicit reasons and do not override the verdict; no fabricated
   calibration. Batch-level drift does not imply contributor-risk assessment.
9. Identity: new scan IDs, preserved outputs, deterministic IDs only in explicit selftest.
10. Access: capabilities probed from the Dataset and reference, scoped by plug-in declarations;
    no hard-coded `DATASET_IMAGES=True` and no irrelevant model rows in the drift header.

## Remaining product work

1. Expose canonical COCO/YOLO routing and shared embedding cache through the drift CLI;
   preserve sample hashes, extractor version, labels and contributor provenance.
2. Add camera/sensor categories and feature-missingness assessment, timestamp/label/contributor
   sidecars, and stronger interpretation. Do not treat pixels as independent images.
3. Manipulation classification is deliberately deferred in full per Plan §7 and the final
   review ruling. Neither the three-feature nor seven-feature classifier is required by this PR.
   Any future implementation needs held-out calibration and a review-only profile ceiling.
4. Add budgeted MMD and semantic cluster occupancy/nearest-example evidence. Reference-only
   k-means cannot create an absent reference cluster: use outlier distance for novelty and
   occupancy for mixture changes. Do not invent terrain/season labels from unlabeled clusters.
5. Measure field false positives, power, uncertainty and throughput. Publish deployment
   dependency hashes/licences and prove offline behavior under OS-level egress controls.

## Statistics and validation

The notes' unscaled-PSI explanation had its direction reversed: for ordinary N,M the
unscaled chi-square threshold is too large and **misses shifts**. Fixed 0.25 heuristics are
a different issue and can overflag small batches. Tests pin the corrected numeric threshold.

Sources:
- https://scholarworks.wmich.edu/dissertations/3208/
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- https://docs.scipy.org/doc/scipy/reference/stats.html

Regression coverage includes dense/sparse null sampling, constant/extreme inputs, family-wise
permutation resolution, multiple clean fixture seeds, injected brightness drift, canonical
Dataset acceptance, declared capabilities, taxonomy, schema validation, model-renderer
compatibility, overlap, scan identity/selftest, corrupt images and decompression-bomb headers.
Synthetic tests are not field validation. Runtime/confidence limitations stay in reports.

## September 20 review resolutions

- Clean assessed findings can produce ACCEPT; declared UNAVAILABLE gaps stay in coverage.
  ERROR forces REVIEW in the drift wrapper until Backend's equivalent shared fix merges.
  No finding disposition is overridden after the shared risk engine runs.
- `profiles/drift.json` holds the ordered D1–D7/C1–C3 policy, with D6 ungated for the
  uncalibrated drift evidence and capped at review. No runtime shadow rule is inserted.
  Backend still owns the general calibration-aware routing fix across modules; this explicit
  drift profile follows its severity-only ruling without claiming to fix other modules.
- `--profile` reads schema-validated PSI and per-axis effect settings. Explicit CLI flags
  override the profile. Independent reference/incoming floors are honoured; `--min-samples`
  sets both. The effective policy and settings are hashed by the shared hash helper and
  saved to `effective.profile.json`. Reproduction commands run from that scan directory.
- Genuine capability exclusions retain CAPABILITY, policy exclusions use BUDGET, and data
  validity/size, cache mismatch and unfinished implementation gaps use None with a reason.
- MMD/energy-distance cuts appear in distribution finding limitations even when the check
  cannot run, and in generated Markdown coverage. Raw observations remain a private adapter
  with validated row count (`n`), not interchangeable with moments-only EmbeddingDistribution.
- The image-folder adapter counts unsupported files. Failed scan/report runs remove only
  their reserved directory; shared immutable evidence is retained for concurrent readers.
- `attacklab.photometric_shift.inject` splits an existing declared-clean corpus into disjoint
  source batches, transforms a seeded fraction of incoming images, and records parameters,
  source/output hashes and measured brightness truth. Brightness, contrast, sensor noise and
  JPEG recompression are supported. The original synthetic generator remains a smoke helper.
- Tests pin clean ACCEPT, error REVIEW, profile/schema/hash behavior, numeric shift magnitude,
  per-axis effects, metadata and manual pixel statistics, seeded reproducibility and cleanup.
  The unscaled-PSI regression asserts missed shifts, not the notes' reversed false-alarm claim.

## Shared-surface decisions requiring Backend review before merge

The four-argument DriftTest protocol remains; its reference annotation is now Dataset | None
because §5 explicitly requires image-axis reporting without a reference. DriftScanResult still
adds `asset_kind`, `relevant_capabilities`, and `drift_summary`; shared writers consume them,
so Module E must preserve these dataset/report fields. The profile schema adds consumed drift
knobs. The shared coverage writer adds finding limitations for dataset results only. The CLI's
lazy imports stay in place, and build_model's docstring is restored.

The final review assigns Plan corrections to its maintainer: reconcile §3's reference
requirement with §5, correct §6.2's unscaled-PSI explanation, and align Consolidated/19's
manipulation scope with Plan §7. This PR documents the decisions without editing another
owner's plan. The actual helper/test decomposition is recorded here; separate psi.py/ks_test.py
wrappers would add no behavior, so statistics remain together with dedicated regression tests.

Validation results for this revision are recorded in the PR body. Synthetic regressions do
not establish field accuracy. At the default 20-image comparison floor, small shifts can be
missed; this is a minimum for attempting a test, not a recommended sample size or power claim.
