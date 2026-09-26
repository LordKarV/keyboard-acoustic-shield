# Arena: Continuous CTC attacker vs the DSP shield

**Question:** Does the deterministic DSP shield (`keyguard/shield/shield.py`) stop the
continuous overlap keystroke transcriber (`MtlCRNN`, `runs/ctc_skaid_crnn.pt`) from
reading real typing — and does it *keep* stopping it once the attacker adapts?

**Setup.** Real SKAID continuous typing (`data/continuous/skaid/labels.jsonl`, 51 sessions).
Held-out = every 5th session (unseen typist/recording), matching `load_skaid`. Metric =
MEAN sequence CER over held-out 5 s chunks (n = 586). Chance ~ 95–100 %. The shield is
applied to each held-out session's **raw 16 kHz audio** using its **true OS onset
timestamps** (victim-side timing), then re-chunked and decoded — exactly the deployment
assumption. Speech quality = STOI of shielded vs. original recording (PESQ lib not
installed in this env; STOI is the honest perceptual-preservation proxy available).

Harness: `keyguard/arena/ctc_arena.py` (clean-vs-shielded eval + chunk dump) and
`keyguard/arena/adaptive_train.py` (retrain `MtlCRNN` on shielded audio, reusing the exact
`train_overlap` recipe: class-weighted dense frame-CE + onset-BCE).

## Results

| Shield | Attacker | MEAN CER | vs clean | STOI |
|---|---|---|---|---|
| — (clean) | original (`ctc_skaid_crnn.pt`) | **40.7 %** | — | 1.00 |
| default (`str1.0 rand1.0 key14 ctx12 dec0`) | original (non-adaptive) | **55.5 %** | +14.8 pp | 0.914 |
| default | **adaptive** (retrained on shielded) | **39.4 %** | −1.3 pp | 0.914 |
| strong (`str1.0 rand2.0 key24 ctx16 dec400`) | original (non-adaptive) | **58.9 %** | +18.3 pp | 0.880 |
| strong | **adaptive** (retrained on shielded) | **41.6 %** | +0.9 pp | 0.880 |

Adaptive checkpoints: `runs/ctc_shielded_adaptive.pt`, `runs/ctc_strong_adaptive.pt`
(warm-started from the clean attacker, 2500 steps each, eval on shielded held-out).

## What this means

- **Non-adaptive: the shield helps, modestly.** Against the fixed attacker it lifts CER by
  ~15 pp (default) to ~18 pp (strong) while keeping speech intelligible (STOI 0.88–0.91).
  But the attacker is *not blocked* — 55–59 % CER is still far below chance, i.e. it keeps
  reading a large fraction of keys through the shield.

- **Adaptive: the shield does NOT hold. Attacker win.** Retraining `MtlCRNN` on shielded
  audio claws CER all the way back to the clean baseline — 39.4 % (default, actually
  *below* clean) and 41.6 % (strong). The ~15–18 pp gain evaporates entirely. Pushing the
  shield harder (wider key window, 400 decoys, 2× residue randomization) bought only
  +2.7 pp of non-adaptive CER and still collapsed under adaptation, at a real STOI cost.

- **Why.** The shield's transform is *deterministic given the onset timestamps*: it removes
  the keystroke's above-neighbour "excess" energy and scrambles the residue with a random
  per-stroke spectral gain. That defeats the attacker it was designed against — the
  self-supervised *clustering* attacker in `adversarial.py`, whose clusters stop lining up
  with keys. But a **supervised CTC attacker with ground-truth labels** simply relearns the
  new (shielded) feature distribution. Because STOI stays 0.88–0.91, most of the signal —
  including the keystroke-correlated spectral structure that survives the "keep min(current,
  neighbour)" step — is still present, and per-frame frame-CE recovers the key identity from
  it. Decoys add insertions the non-adaptive decoder trips on, but the adaptive model learns
  to tell white-noise decoy transients from real keystroke spectra and ignores them.

## Verdict (min-max)

Against the **continuous CTC attacker**, the current deterministic DSP shield is a
**non-robust defense**: a meaningful gain against a frozen attacker, ~0 gain against an
adaptive one. Higher shield strength does not change the outcome and costs speech quality.
This is the expected failure mode of a fixed, hand-designed perturbation in a min-max game
— it is exactly why `adversarial.py` frames the shield as a *learned* perturbation co-trained
against the attacker. To beat this attacker the shield must either (a) be co-trained
against the CTC attacker directly (learned, adaptive perturbation under the STOI budget,
not a fixed DSP recipe), or (b) destroy the per-key discriminative information that survives
at STOI 0.88–0.91 — which almost certainly requires spending more of the perceptual budget
than STOI ~0.9 allows. A shield validated only against a non-adaptive attacker overstates
its protection by ~15–18 pp CER.

## Reproduce

```bash
# clean vs shielded (+ dump shielded chunks for the adaptive attacker)
KEYGUARD_DEVICE=cuda PYTHONPATH=$(pwd) python -m keyguard.arena.ctc_arena \
  --ckpt runs/ctc_skaid_crnn.pt --dump-train scratch/shielded_train.pt \
  --dump-shielded-test scratch/shielded_test.pt

# adaptive attacker (min-max step)
KEYGUARD_DEVICE=cuda PYTHONPATH=$(pwd) python -m keyguard.arena.adaptive_train \
  --train scratch/shielded_train.pt --test scratch/shielded_test.pt \
  --out runs/ctc_shielded_adaptive.pt --steps 2500 --warm runs/ctc_skaid_crnn.pt
```

Strong shield: add `--strength 1.0 --randomize 2.0 --decoys 400 --key-frames 24 --ctx-frames 16`.
