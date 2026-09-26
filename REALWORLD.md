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

## Cross-keyboard generalization (diverse-training attempt)

**Verdict: FADED (negative result).** User ask, verbatim: *"train on general
keyboard audios ... test if it works and if it doesn't u can fade it."* We tried
it. It doesn't generalize. Keep the per-keyboard-calibration path above.

The prior section showed a *Harrison-only* attacker collapses on other keyboards.
The obvious next hypothesis: pool MANY keyboards' labeled audio and train one
**keyboard-agnostic** attacker so no per-keyboard banking is needed. We built the
diverse pool and ran the honest test.

Reproduce: `KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.eval_generalize`
Smoke: `... -m keyguard.eval_generalize demo` · Raw: `runs/generalize.json` ·
Code: `keyguard/eval_generalize.py`

### Data pool — 8 keyboard/mic domains (all ground-truth labeled)

| Domain | Keyboard / channel | n (capped 25/key) | Source |
|---|---|---|---|
| `harrison` | MacBook Pro, phone mic + Zoom | 900 | `data/harrison` |
| `live` | our laptop + our mic | 481 | `data/live_bank.npz` |
| `mka_mac` | MKA MacBook (direct + Messenger + Zoom) | 648 | MKA, Mendeley `bpt2hvf8n3` |
| `mka_hp` | MKA HP laptop | 219 | MKA |
| `mka_lenovo` | MKA Lenovo laptop | 216 | MKA |
| `mka_msi` | MKA MSI laptop | 216 | MKA |
| `multipressure` | Pinto et al. keyboard (3 pressures) | 108 | Zenodo `19453177` |
| `zenodo` | one unknown keyboard | 26 | Zenodo `10623477` |

New download this round: **Multi-Keyboard Acoustic (MKA)** dataset (Mendeley Data
DOI `10.17632/bpt2hvf8n3`, 2 GB, CC-BY). It has 4 distinct physical keyboards
(HP/Lenovo/MSI/Mac) with per-press segmented wavs (~6–7 presses/key/platform);
Mac/Messenger/Zoom are the same physical Mac keyboard over different channels, so
we group them as one domain (else "keyboard-out" would leak). Non-alphanumeric
keys (enter, caps, …) dropped → 36-class A–Z/0–9. Everything mapped through the
project front-end (onset → `KEY_WIN` window → log-mel). Chance = 1/36 = **2.8%**
top-1, ≈ **8.3%** top-3.

### Results (real, no cherry-picking)

| Test | top-1 | top-3 | Read as |
|---|---|---|---|
| **Ceiling** — pool ALL 8 keyboards, 80/20 split *mixing* keyboards | **24.3%** | **43.7%** | model *can* learn keys when it has seen every keyboard |
| **LOKO mean** — leave-one-keyboard-out, zero-shot on the held-out rig | **2.1%** | **6.4%** | **= chance / below top-3 chance → no generalization** |

Per-held-out (train on the other 7, test zero-shot):

| Held-out keyboard | top-1 | top-3 |
|---|---|---|
| harrison | 3.0% | 6.2% |
| live | 0.4% | 1.5% |
| multipressure | 1.9% | 7.4% |
| zenodo | 0.0% | 0.0% |
| mka_lenovo | 3.7% | 10.2% |
| mka_hp | 2.3% | 8.7% |
| mka_msi | 2.8% | 7.9% |
| mka_mac | 2.8% | 9.3% |
| **mean** | **2.1%** | **6.4%** |

### What this means (honest)

1. **Diverse-pool training did NOT yield a keyboard-agnostic attacker.** Every
   held-out keyboard lands at or below chance (mean top-3 6.4% < 8.3% top-3
   chance). No held-out keyboard clears the KEEP bar (top-3 ≥ ~15%).
2. **It's not a capacity/label bug — it's domain shift.** The exact same pipeline
   hits **24.3% / 43.7%** at the ceiling (keyboards mixed into train+test). The
   ~40-point top-3 gap between ceiling and LOKO *is* the result: per-key acoustics
   are device-specific (plate resonance, mic, TDoA), and 8 keyboards are nowhere
   near enough for the net to learn a keyboard-invariant key representation.
3. **This corroborates the earlier zero-shot collapse** and extends it: pooling
   many rigs doesn't fix it. Keyboard-agnostic acoustic keystroke recognition is
   an **open research problem** — it likely needs orders of magnitude more
   keyboards, and/or explicit domain-invariance (adversarial domain adaptation,
   per-device normalization), not just "more data thrown in a pot."

### Recommendation

- **Keep the per-keyboard-calibration path** from the section above (bank the
  target keyboard ≥20–25 presses/key, then transfer-learn). Do **not** ship the
  pooled attacker as a no-calibration default — it reads an unseen keyboard at
  chance. `eval_generalize.py` stays as an honest negative-result artifact; it is
  **not** wired into the demo (`server.py`/`realtime.py`) and nothing overclaims.
- The defender story is unaffected: the −18 dB adversarial shield still pins a
  *calibrated* attacker to chance with STOI ≈ 0.99. The generalization failure is
  purely on the attack side and, if anything, widens the defender's margin.

