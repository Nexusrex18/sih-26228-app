# Module D implementation plan and progress

Scope: PS 2.2.4, using the notes repository's Module D architecture and build plan,
Consolidated/17 full-product direction and Consolidated/19 MVP sequencing. The initial implementation was built against main. This PR is based on modules,
which supplies canonical Dataset contracts and loaders. The standalone DriftBatch path
is an explicit image-folder/cache adapter; canonical Dataset integration remains below.
It does not replace or modify the frozen DriftTest protocol in core/interfaces.py.

## Delivered first increment

- Dataset batch/config contracts, separate drift registry and dataset orchestrator.
- Offline image-folder adapter, strict image decoding, stable content IDs, optional
  aligned NPZ embeddings with extractor identity/version. No downloading a backbone.
- PSI with `(1/N + 1/M)` scaling; survival function avoids catastrophic tail subtraction.
  Label-invariant pooled quantile bins include both infinite tails, collapse duplicate
  edges and use a half-count pseudocount. Sparse expected counts use a seeded
  conditional permutation test. Counts, method and sample sizes appear in evidence.
- KS and BH correction across both tests and all axes within each detector. A minimum
  KS effect gate prevents statistical significance alone being called material drift.
- Embedding projection tests and descriptive centroid Euclidean/diagonal Mahalanobis
  distance; incompatible extractor versions are explicitly unavailable.
- Brightness, contrast, Laplacian sharpness, residual noise, per-image RGB histogram
  proportions; JPEG quantization and EXIF ISO only with complete metadata.
- Existing HTML rendering reused; dataset JSON, generated coverage, seeded photometric
  fixtures, math/negative/integration/error/reproducibility tests.

## Resolutions of contradictory notes

1. Without a reference, descriptive image metrics are possible but drift is not. Both
   registered drift comparisons require a reference; unavailable does not mean clean.
2. The notes' claim that dropping the PSI scale raises false alarms has its direction
   reversed for N,M > 2: the unscaled chi-square threshold is larger and misses shifts.
   The regression asserts the correct numeric threshold and sample-size dependence;
   it does not encode the incorrect proposed test.
3. KS requires observations. The proposed mean/covariance-only EmbeddingDistribution
   cannot support it. DriftBatch retains aligned rows; the future cache adapter must do so.
4. A p-value is not calibrated confidence in drift, and especially not malicious intent.
   P/q values are evidence; Finding.confidence remains explicitly uncalibrated (0),
   until held-out calibration artifacts exist. No arbitrary `1-p` conversion.
5. Statistical drift has quality nature. This initial path routes material drift or
   incomplete coverage to REVIEW, never quarantine. This is a local deterministic
   policy until the shared calibrated risk engine exists.
6. New four-role ownership supersedes ML-1 wording: Detection owns drift; Core owns
   eventual shared dataset/cache integration; Adversary owns held-out evaluation.

## Next increments (required product work, not removed from scope)

1. Integrate canonical COCO/YOLO Dataset and shared independent-backbone cache when Core
   delivers them. Preserve sample alignment, extractor version and batch identity.
   Current folder ingestion reads images only, not annotations.
2. Add categorical camera/sensor comparison, per-feature missingness assessment and
   metadata sidecars for time, labels and contributor provenance. Preserve sample unit;
   do not inflate N by treating image pixels as independent images.
3. Implement the MVP three-feature manipulation triage (breadth, class-conditionality,
   contributor concentration) once labels/contributors and held-out attack/benign corpora
   exist. Train on development data; calibrate separately; freeze before held-out scoring.
   Report feature contributions, missing evidence and review-only policy. Expand to seven
   features only with temporal/spatial/manifold inputs and independent evaluation.
4. Add bounded MMD permutation testing (memory/sample budgets recorded), semantic cluster
   occupancy and nearest-example evidence. Reference-only k-means cannot create an
   "absent reference cluster"; use outlier distance for novelty and occupancy for mixture
   change. Do not invent terrain/season labels from unlabeled clusters.
5. Validate on independent benign drift and attack datasets with matched nuisance factors;
   report false-positive rate, power, intervals and performance. Export versioned
   calibration and deployment bundles with dependency licences/hashes and egress tests.

## Acceptance gates

First increment: numerical PSI regression, sparse/constant/extreme samples, clean null
simulation, known brightness shift, no-reference/small-batch/embedding-version rejection,
corrupt-image fail-fast, report serialization, deterministic fixture hashes, import boundaries.

MVP completion additionally requires the three-feature triage above, held-out calibration,
shared loader/cache integration and target-machine throughput measurement. This increment
is not a declaration that Module D or the complete product is finished.

## Statistical sources

- Yurdakul, Statistical Properties of Population Stability Index:
  https://scholarworks.wmich.edu/dissertations/3208/
- SciPy KS documentation (continuous independent two-sample assumptions):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- SciPy statistical functions / FDR adjustment:
  https://docs.scipy.org/doc/scipy/reference/stats.html

BH is applied within each detector; it is not global control across scans and assumes
appropriate dependence. Sequential monitoring needs its own policy. Sparse PSI uses
permutations; the dense scaled chi-square result remains an approximation with smoothing
and estimated bins. No field performance claims follow from synthetic tests.

## Validation performed

Python 3.14, NumPy 2.5.3, SciPy 1.18.1, Pillow 12.3.0. Targeted tests and existing
import-boundary tests: 28 passed against the modules branch. CLI fixture generation
and HTML/JSON/coverage report smoke passed. Network connect is blocked in the end-to-end
test. Full Module B tests were not run: its Torch/ONNX test environment is not installed
in this temporary environment. Statistical methods and approximation/fallback status
are recorded in each axis's evidence.

PR validation on `modules`: changed-file Ruff passes; 28 targeted drift/import-boundary
checks pass. `mypy cva/core` reports one pre-existing `quantise.py:97` no-any-return
error in this dependency environment, reproduced on an unmodified checkout of the base.
No additional core type errors were introduced.
