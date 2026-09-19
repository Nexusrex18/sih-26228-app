# Module B — status, and the unresolved problem

## What works, verified

- **Capability probing is active and correct.** Verified across four adapters:
  - ONNX: predict, logits, weights, architecture, activations (via graph surgery).
    `MODEL_GRADIENTS` absent **always** — a declared narrowing, with the reason printed.
  - Frozen TorchScript: keeps gradients, **loses** weights and activations. Discovered by
    attempting a backward pass and reading `state_dict`, not inferred from the extension.
  - PyTorch `nn.Module`: everything, and the only reliable full-confidence gradient path.
  - Callable/black-box: predict only, every absence carrying a reason.
- **The three-state machine.** `UNAVAILABLE` emits a `Finding`, never a silent skip; a
  raising check surfaces as `ERROR`, never folded into `DEGRADED`.
- **Neural Cleanse falls back cleanly.** On ONNX it runs the gradient-free NES path,
  reports `DEGRADED`, names the tier reached and applies a stated confidence penalty.
- **The report and benchmark render**, with per-detector detection **and false-alarm**
  rates reported separately.
- **The corpus fitness gate works** — it refused to emit a corpus three times: a model that
  never learned (0.52), a "backdoor" that never took (ASR 0.58), and a task so easy models
  memorised it (acc 1.000).

## Superseded — see "Measured on the hardened corpus" below

## What did not work on the FIRST corpus

**The three backdoor detectors do not separate clean from backdoored on this corpus.**
Measured on a matched pair (clean acc 0.985 / ASR 0.000, backdoored acc 0.998 / ASR 1.000):

| signal | clean | backdoored | expected |
|---|---|---|---|
| STRIP normalised low-tail | 0.133 | 0.049 | lower for backdoored — **direction correct**, but the absolute 0.35 threshold flags both |
| Neural Cleanse max anomaly index | 7.78 | 0.69 | higher for backdoored — **inverted** |
| Intrinsic suspicion | 7.99 | 0.97 | higher for backdoored — **inverted** |

### Diagnosis

**Cause 1 — the adaptive lambda was missing.** Fixed. A fixed L1 weight never converges on
the minimal mask, so every class returns a similar norm and the MAD test has nothing to
find. After the fix masks sparsify properly (L1 34–103 → 15–22). The verdict stayed
inverted, which ruled this out as the whole story.

**Cause 2 — the task is too easy, and this inverts the detectors rather than merely
weakening them.** Clean models reached **acc 1.000**. Decision boundaries are then
degenerate, a minimal universal trigger is small for *every* class, and Neural Cleanse's
premise — that one class is unusually reachable — collapses. Worse, on the *clean* model
two shape classes (triangle, cross) are intrinsically easier to reach than the rest, which
the anomaly index reads as a backdoor. Hence a false positive on clean and a false
negative on backdoored.

Mitigation in progress: 8 classes instead of 5, lower shape contrast, heavier background
nuisance, and an **upper** bound in the fitness gate (`MAX_CLEAN_ACC = 0.99`) — a model at
1.000 is as unfit a test subject as one at chance.

**Cause 3 — STRIP is framed wrongly, and this is a design error, not a tuning problem.**
STRIP is an *input-level* trigger detector: it asks whether *this input* carries a trigger.
Our probe set is clean by construction, so nothing in it carries a trigger and a backdoored
model behaves normally on all of it. Using it as a *model-level* check from clean probes
alone is not a use the method supports.

The honest fix, not yet made:
- `model.strip` should `require` a **suspect** input set and report `UNAVAILABLE` when only
  clean probes exist. Its current threshold is measuring a training artifact (the
  backdoored model is simply more confident) rather than a backdoor signature.
- The real reference-free, black-box, model-level test is the **universal-perturbation
  margin** — currently buried as one signal inside `model.intrinsic_probes`. It deserves to
  be its own check, and it is the one that should carry the black-box path.

## What this means for the plan

The architecture held: every one of these was found *and localised* without changing a
contract, and each fix is inside one detector. But two planning claims need correcting:

1. **"STRIP is the only backdoor check that runs on ONNX"** — repeated through the
   consolidated docs. It is not usable in the form assumed. The ONNX black-box path needs
   the universal-perturbation check instead, and the coverage statement must say so.
2. **The attack lab needs a difficulty gate, not just a fitness gate.** An easy corpus does
   not under-test a margin-based detector, it *inverts* it. That belongs in
   `07-Attack-Lab-and-Benchmark` as a corpus requirement.

## Next

1. Rebuild on the hardened corpus and re-measure all three signals.
2. Split the universal-perturbation margin into `model.universal_margin`.
3. Re-scope `model.strip` to require suspect inputs.
4. Only then quote any detection number.


---

# Measured on the hardened corpus (2026-09-19)

Corpus: 8 classes, clean accuracy 0.78–0.91, all backdoors ASR ≥ 0.997. Anomaly-index
thresholds as shipped (`nc_anomaly_threshold = 2.0`, `um_anomaly_threshold = 2.0`,
`intrinsic_threshold = 2.5`).

| model | truth | universal_margin | neural_cleanse | intrinsic |
|---|---|---|---|---|
| clean_a | clean | 0.77 | **2.13** | 1.50 |
| clean_b | clean | 1.48 | 0.91 | 2.45 |
| clean_wide | clean | **2.89** | 1.21 | **3.22** |
| perturbed | clean | 0.67 | **3.14** | **2.62** |
| bd_patch_08 | BACKDOORED | 0.67 | **4.92** | 1.77 |
| bd_patch_05 | BACKDOORED | 0.77 | 1.60 | **3.37** |
| bd_blended_20 | BACKDOORED | 1.39 | **4.44** | **2.96** |
| bd_sig_10 | BACKDOORED | 1.57 | 1.20 | 2.02 |

(bold = fires at the shipped threshold)

## Neural Cleanse — no longer inverted, and the ranking is real

On the first corpus the clean model outscored the backdoored one (7.78 vs 0.69). That is
gone. **The two highest scores in the whole corpus are now backdoored models** (4.92, 4.44)
and the highest clean score is 3.14.

At the shipped threshold of 2.0 it scores **2/4 detection at 2/4 false alarm** — unusable.
The separation exists but the threshold does not reflect it: 2.0 came from the published
heuristic for a different setting and was never calibrated to this corpus.

**A threshold of 3.5 would give 2/4 detection at 0/4 false alarm on this corpus.** That
number is FITTED ON THE DATA IT IS REPORTED AGAINST and must not be quoted as a result. It
is stated here only as evidence that the separation is real. A credible figure requires the
held-out-family discipline: calibrate on patch and blended, evaluate on SIG and clean-label.

## The two misses are different, and only one is a defect

- **`bd_sig_10` (1.20) is an EXPECTED miss.** SIG is a global sinusoidal overlay. Neural
  Cleanse optimises a small localised mask, so a trigger spanning the whole image is
  outside its premise by construction. This belongs in the coverage statement as a
  *measured* limitation, not a bug — and it is exactly the kind of declared blind spot the
  PS asks for three times.
- **`bd_patch_05` (1.60) is a poison-rate floor.** Same trigger family as `bd_patch_08`
  (4.92), at 5% instead of 8%. This is the sweep that turns "may miss adaptive attacks"
  into "detects patch triggers at poison rate ≥ X%". Run it.

## `model.universal_margin` does not work — an honest negative

Detection **0/4**, false alarm **1/4**. Every backdoored model (0.67–1.57) scores below the
worst clean model (2.89). The check is not merely mis-thresholded, it has no separation at
all on this corpus.

This matters because it was introduced specifically to carry the ONNX and black-box path
after STRIP was found mis-specified. **So that path currently has no working detector.**
Candidate causes, untested: the hill-climb is too weak to find a genuinely minimal
perturbation in 50 steps; an additive universal shift may be the wrong parameterisation
where the trigger is multiplicative/localised (a mask, as Neural Cleanse uses); and cost
normalisation by success rate may dominate the statistic.

Until it is fixed or withdrawn, **the honest position is that we have no black-box
model-level backdoor detector** and the coverage statement must say so.

## `model.intrinsic_probes` is at chance

2/4 detection, 2/4 false alarm. It was designed as a *ranking* to target expensive checks,
not a verdict, and its disposition is capped at `review` — so this is closer to its
intended role than the numbers suggest. But it is not currently earning its cost.

## Status

| check | state |
|---|---|
| capability probing, 4 adapters | working, verified |
| three-state machine + ERROR channel | working, tested |
| `weight_digest`, `graph_structure`, `fingerprint`, `anomalous`, `activation_statistics` | working |
| `trigger_reconstruction` (Neural Cleanse) | **separation real, threshold uncalibrated** |
| `universal_margin` | **no separation — negative result** |
| `intrinsic_probes` | at chance |
| `strip` | correctly reports UNAVAILABLE without suspect inputs |

## Next, in order

1. Poison-rate sweep for the patch family — produces the first real coverage number.
2. Calibrate the Neural Cleanse threshold on patch + blended, **evaluate on SIG**.
3. Fix or withdraw `universal_margin`. Try a mask parameterisation rather than an additive
   shift; if it still fails, withdraw it and declare the black-box path uncovered.
4. Only then quote a detection rate.
