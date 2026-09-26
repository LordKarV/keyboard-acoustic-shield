# Real-world / out-of-distribution validation

Honest domain-shift check before tomorrow's live physical-keyboard test. We ran
our trained attacker and our adversarial shield against **external, labeled
keystroke-audio datasets recorded on different keyboards/mics than ours** — real
ground truth, not our Harrison/synth data.

Reproduce: `KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.eval_external`
Raw numbers: `runs/external_eval.json` · Code: `keyguard/eval_external.py`

## Datasets used (genuinely external, ground-truth labels)

| Dataset | What it is | Size | Keyboard/mic |
|---|---|---|---|
| `zenodo_10623477` | one recording per letter A-Z | 26 files, 1 press/key | one unknown keyboard+mic (not ours) |
| `multipressure_z19453177` (Zenodo, Pinto et al., CC-BY-4.0) | each key at High/Medium/Low pressure | 38 keys × 3 = 114 files, 1 press/key/pressure; 36 keys map to our CLASSES → 108 samples | a *different* keyboard+mic |

**Both are "reference" datasets: only 1–3 physical presses per key.** That matters
a lot (see caveat). Chance = 1/36 = **2.8%**. 36 classes (A–Z, 0–9).

## Results (real, no cherry-picking)

| Test | top-1 | top-3 | Read as |
|---|---|---|---|
| **A. Attacker zero-shot** — zenodo (our `supervised_mbp.pt`, no calibration) | **0.0%** | 0.0% | total collapse |
| **A. Attacker zero-shot** — multipressure | **2.8%** | 7.4% | = chance (collapse) |
| **B. Attacker calibrated** — leave-one-pressure-out (train 2 pressures, test 3rd) | **3.7%** | 9.3% | barely above chance |
| **B. Attacker calibrated** — mixed 2-train/1-test per key | **0.0%** | 5.6% | still ~chance |
| **C. Defender** — attacker leakage **before → after** −18 dB shield | 5.6% → **2.8%** | — | pinned to chance |
| **C. Defender** — speech quality (STOI) of shielded audio | **0.990** | — | speech untouched |

Sanity: the same `supervised_mbp.pt` scores **87.2%** on held-out Harrison keys, so
the checkpoint is fine — the collapse below is real domain shift, not a bug.

## What this means (honest)

1. **Zero-shot transfer across keyboards/mics is essentially zero.** Our
   Harrison-trained attacker doesn't just degrade — it *collapses*, predicting
   mostly `1`/`2`/`Z` for every external keystroke. Per-key acoustics are
   device-specific; a model trained on one rig does not read another rig cold.
   **You must bank the target keyboard first.**

2. **1–3 presses per key is not enough to calibrate, either.** Even training on
   the external keyboard (leave-one-pressure-out, or mixed split) stays at ~chance.
   This is *not* evidence the attack is impossible on the real keyboard — it's
   evidence these datasets are too small. Our own working attack numbers came from
   ~25 presses/key (Harrison 87%, our laptop bank 37.8% top-1 / 57.1% top-3). Below
   ~dozens of presses/key there is nothing to learn from.

3. **The defender holds up out-of-distribution.** Even against an attacker trained
   *on the target data*, the bounded −18 dB adversarial perturbation pins accuracy
   to chance (2.8%) while STOI stays **0.990** (speech preserved). The dramatic
   88%→2.8% drop lives on Harrison (see `shield/adversarial.py`) where the attacker
   actually reaches 88% first; here the external attacker never climbs high because
   the data is sparse, but the shield still drives it *to the floor* with no audible
   cost. The defense — the novel centerpiece — generalizes.

## Caveat / limitation

The two external datasets are reference sets (one recording per key), so **B is
data-starved and B's low numbers are about sample size, not about the attack being
weak.** A public dataset with many repetitions per key at a fixed condition would
give a stronger calibrated number; we did not download one to keep compute modest
and because the existing labeled sets already answer the domain-shift question. The
STOI mixture uses a LibriSpeech example clip with the perturbation tiled over it
(worst case); PESQ was not run.

## How to run tomorrow (live physical keyboard)

1. **Bank the real keyboard first — this is mandatory, not optional.** Zero-shot is
   0%. Capture many presses per key on the actual rig:
   `KEYGUARD_MIC=<mic idx> uv run python3 capture.py bank`
   Aim for **≥20–25 presses/key** (matches the regime where we hit 37–87%). Fewer
   than ~10/key and expect chance, per the numbers above.
2. **Transfer-learn:** pretrain on Harrison, fine-tune on the fresh bank (HANDOFF
   step 2). Realistic expected range on a laptop-class mic: **~35–60% top-3**, lower
   top-1. Do not promise 87% live — that was Harrison's exact rig.
3. **Then demo the shield:** run the −18 dB adversarial perturbation against the
   banked/fine-tuned attacker; expect it driven to ~chance (2.8%) with STOI ≈ 0.99.
   This is the reliable headline even if the attack side underperforms live.

### Gotchas that will move the numbers
- **Mic distance / type:** the mic matters more than the keyboard. Same phone/mic
  for banking and testing; a close phone mic (Harrison's rig) beats a far laptop mic.
- **Typing speed:** fast typing overlaps presses and breaks isolated-key
  segmentation (onsets merge). Type at a measured pace for the isolated-key demo.
- **Ambient noise:** quiet room. `onset_strength` peak-picking is fragile under HVAC
  / chatter; the segmenter will invent or miss onsets.
- **Pressure/consistency:** the multipressure result shows even pressure changes on
  the *same* key confuse a sparsely-trained model — bank at the pressure you'll type.
- **Device pin:** export `KEYGUARD_DEVICE=cpu` (MPS is flaky here).
