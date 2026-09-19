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

## What does not work yet

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
