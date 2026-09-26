# Keyguard — research concept

Two contributions: (1) an overlap-robust, free-typing acoustic keystroke attack, and
(2) a perceptually-constrained *adversarial* defense against an adaptive eavesdropper.

---

## Part 1 — The attack: free-typing, overlap-robust keystroke recognition

### The gap
Every published acoustic keystroke attack we found (Zhuang 2005; Berger 2006;
Harrison 2023; Okada 2026) assumes **isolated keystrokes** — one clean click per
window, keys ~0.8 s apart, or per-key training recordings. The 2023-era survey states
it plainly: current systems are "constrained by the assumption that keystrokes don't
significantly overlap during high-speed typing," and list overlap as **future work**.
Slater et al. (ACSAC 2019) is the closest prior — an RNN/CTC transcriber robust to
overlap (CER 36%→7.4% known, 41%→15% unknown) — but it predates the
self-supervised-audio-encoder era and used bespoke hardware data.

**Nobody has a working, modern, free-typing, overlap-robust acoustic keylogger** from
a commodity laptop mic. That's the target.

### Why the attack works — two pillars
- **Pillar 1 — per-key acoustics.** Plate resonance varies by key location, so per-key
  timing + spectral (plate-resonance) features make keys distinguishable. A CNN learns
  this: **87.2% top-1** on Harrison MacBook data (chance 2.8%).
- **Pillar 2 — language redundancy.** English carries ~1 bit/char, so a *weak*
  acoustic guess + a language model recovers the text. We feed the attacker's per-key
  **top-3 candidate lattice** to **Gemini** (`gemini-flash-latest`, `attack_lm.py`,
  wired into `/api/demo`); it is verified live — e.g. it recovers `MEETING` from a
  corrupted top-1 `NAAG1MH` and `PASSWORD` from `OSAAEPTS`. This is the lever that
  turns a weak acoustic stream into readable English. **Random passwords are the
  honest boundary** — no language prior, so Pillar 2 can't help (with the shield on,
  the lattice is garbage and even Gemini can't reconstruct — the LLM-augmented
  attacker is still defeated).

### Why isolated-keystroke methods fail on free typing
A keystroke is a broadband transient (~120 ms push+release). At 300–500 CPM inter-key
intervals fall to 60–150 ms, so a fixed window holds 2–3 clicks and neighbouring
windows are near-identical; a per-window classifier emits runs of the same character
(we reproduced exactly this). The problem is the *formulation*, not the features.

### Method: keystroke ASR
Treat it as speech recognition for keys — transcribe the whole stream, no hard
segmentation.
1. **Data engine.** Overlap-add per-key samples at inter-key intervals from a
   lognormal typing-dynamics distribution → unlimited `(audio, string)` pairs with a
   controllable overlap knob. Sound adds linearly, so this is a faithful physical
   model.
2. **Acoustic model.** log-mel (4 ms hop) → CNN → BiGRU → **CTC** over 37 symbols
   (blank + 36 keys). CTC learns the alignment, so overlap needs no segmentation.
3. **LM fusion.** CTC beam search + char n-gram shallow fusion, optional LLM n-best
   rescoring — the 36%→7% lever.

### Honest status of the attack
- Isolated-key classifier: **87.2% top-1** on Harrison; **37.8% top-1 / 57.1% top-3** on
  our own laptop-mic bank (real but weak signal — laptop mic, quiet keys).
- The CTC transcriber pipeline is **verified** (overfits synthetic overlap to ~4% CER)
  but **generalization collapsed** on the weak laptop-mic signal (same common letters
  regardless of input). Bottleneck is signal strength, not code. Path forward: a
  stronger/fixed sensor (phone mic, or contact/accelerometer via Arduino),
  transfer-learning from Harrison, and a training-collapse fix (multi-task onset head).

---

## Part 2 — The defense: a perceptually-constrained adversarial shield

### The contribution
A shield that outputs audio **simultaneously** (1) perceptually transparent — a human,
PESQ/STOI, *and* a speech recognizer (Whisper) hear the original speech unchanged — and
(2) privacy-robust against a **retraining** attacker who trains on the *defended*
audio and still can't read keys above chance. Formally:

```
min_D  max_A  Leakage( A( D(x) ) )      subject to   Perceptual( D(x), x ) <= eps
```

### Why this is genuinely hard (and why a deterministic transform doesn't cut it)
1. The inner `max_A` makes the optimal defense a function of the *current* attacker,
   which retrains — **no fixed transform is optimal** against a moving adversary.
2. The minimal, speech-preserving perturbation is defined *through the attacker's
   neural network*: adversarial-example optimization against A plus a
   psychoacoustic/perceptual loss. There is **no closed form**.
3. "Just garble the keystroke band" is the trivial **max-perturbation** solution that
   **violates the perceptual constraint** — it breaks Whisper and destroys overlapping
   speech. The interesting region is the *constrained minimum*, reachable only by
   optimization.

### Architecture
D (a small causal defender net) and A (our CTC/CNN keystroke reader) **co-train
GAN-style**: A is the discriminator (can it read keys?), D is the generator (make keys
unreadable) under a hard perceptual regularizer — a STOI/PESQ floor plus a
Whisper-transcription-intact check. The training loop *is* the arena. Implemented in
`keyguard/shield/adversarial.py` as a universal/learned bounded adversarial
perturbation vs a differentiable attacker, with adversarial co-training rounds.

### Lineage
- **Adversarial examples / adversarial audio** — Szegedy et al. 2013; Carlini & Wagner
  2018 (targeted attacks on speech-to-text). We invert the objective: perturb *away*
  from key-readability, not toward a target transcript.
- **Psychoacoustic hiding** — Schönherr et al. 2019 (adversarial audio under a masking
  threshold). We reuse the perceptual-masking idea as the *constraint* that keeps the
  perturbation inaudible and speech-preserving.
- **Adversarial-audio privacy (the paradigm we adapt, and closest prior art).**
  EveGuard (Chang et al., IEEE S&P 2025, arXiv 2411.10034) is the closest published work
  — a learned, speech-preserving adversarial perturbation against a side-channel
  eavesdropper — but for the **vibration** channel (mmWave/accelerometer/laser)
  protecting **spoken speech**. AntiFake (Yu et al., CCS 2023) applies the same toolbox
  to blocking voice cloning. We did **not** invent adversarial-audio privacy.
- **Defensible novelty:** to our knowledge, the **first perceptually-constrained
  adversarial shield for the acoustic *keystroke* side-channel** (secret = typed text) —
  adapting the adversarial-audio-privacy paradigm (EveGuard S&P'25, AntiFake CCS'23) to a
  channel and a secret it hasn't been applied to, framed as a min-max arms race against a
  *retraining* attacker. We do not claim to have "solved" the vulnerability.

### Baselines already measured
- DSP shield (STFT inpaint + residue randomization + decoys): deterministic baseline.
- Neural separator (`keyguard/shield/separator.py`): preserves speech (STOI 0.99,
  PESQ 2.90) while cutting an adaptive attacker **86% → 63%**.
- RNNoise-style suppression: ~no effect (**86% → 84%**).

### Adversarial shield D — arena numbers
The co-training loop is served at the dashboard's `/api/arena` from
`runs/arena/adv-*/arena.json`. At an inaudible **-18 dB** perturbation budget
(chance = 2.8%, 36 classes):

- Clean (unprotected) attacker on Harrison isolated keys: **87.2%**.
- Shielded, **every round: 2.8% (= chance)** — the deployed shield always drives the
  attacker to chance.
- A **retraining** attacker fighting a *frozen* shield claws back across rounds —
  **20.6% → 47.8% → 55.6% → 47.2% → 18.3%** (up to ~56% mid-game) — but D re-solves
  against it every round and re-blinds it to **2.8%**. Final shielded = **2.8%**.

This is a real bouncing arms race, not a flat monotonic curve: a retraining attacker
recovers against a static D, but D wins each round it re-optimizes and drives the
attacker back to chance.

---

## Evaluation (the money figures)
- Attack: CER vs overlap rate (typing-speed sweep); known vs unknown typist; with/without LM; password guess rank.
- Both: bits leaked/keystroke (info-theoretic lower bound).
- Defense: attacker accuracy **after retraining** on defended audio, vs PESQ/STOI and
  Whisper WER on the same audio — the two axes of the min-max.

## References
- Slater et al., *Robust keystroke transcription from the acoustic side-channel*,
  ACSAC 2019 — overlap-robust CTC transcriber (CER 36→7.4%).
- *A Survey on Acoustic Side Channel Attacks on Keyboards*, arXiv 2309.11012 — states
  overlap/high-speed as open.
- Harrison, Toreini, Mehrnezhad 2023 (arXiv 2308.01074) — CoAtNet, isolated keys.
- Okada et al. 2026 (arXiv 2607.22094) — self-supervised, isolated keys.
- Szegedy et al. 2013; Carlini & Wagner 2018 — adversarial examples / adversarial audio.
- Schönherr et al. 2019 — psychoacoustic adversarial hiding.
- Chang et al., *EveGuard: Defeating Vibration-based Side-Channel Eavesdropping with
  Audio Adversarial Perturbations*, IEEE S&P 2025 (arXiv 2411.10034) — closest prior art.
- Yu et al., *AntiFake: Using Adversarial Audio to Prevent Unauthorized Speech
  Synthesis*, CCS 2023.
- Asonov & Agrawal, *Keyboard Acoustic Emanations*, IEEE S&P 2004.
