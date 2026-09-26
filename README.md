# Keyguard

**Reading keystrokes through a video call — then defeating the attack with an adversarially-trained AI shield.**

Keyguard is a HackGT (ML/AI track) project with two halves:

1. A **red team** of acoustic keystroke eavesdroppers that reconstruct what you
   type from the *sound* of your keyboard leaking through a Zoom/Meet call.
2. A **blue team** shield whose job is a constrained min-max game: emit audio that
   a human, PESQ/STOI, and a speech recognizer all hear as unchanged speech, while
   a **retraining** attacker still can't read the keys above chance.

The headline goal is deliberately hard: not "we broke a static attacker," but
"the attacker retrained on our *shielded* audio and still can't read the keys —
and the speech is untouched."

> Ethics: only our own devices, consenting teammates/judges, and fake passwords.
> The shield runs on the *victim's* side and uses the victim's own OS key-event
> timestamps (timing only, never which key) — the asymmetry that lets it be surgical.
> Nothing here targets third parties.

---

## What works today

| Component | Status | File |
|-----------|--------|------|
| Keystroke onset segmentation | Done | `keyguard/segment.py` |
| Mel-spectrogram features | Done | `keyguard/features.py` |
| **Supervised attacker** (CoAtNet-flavored CNN + SE blocks) | Done — **87.2% top-1** on Harrison data (chance 2.8%); **37.8% top-1 / 57.1% top-3** on our own laptop-mic bank | `keyguard/attackers/supervised.py` |
| Timing-only attacker (leak catcher) | Done | `keyguard/attackers/timing.py` |
| Overlap/free-typing CTC transcriber | Pipeline verified (~4% CER on synthetic overlap); generalization collapses on weak laptop-mic signal | `keyguard/ctc/` |
| LM-fused CTC decoder (beam search + char-LM) | Done | `keyguard/ctc/decode.py` |
| **DSP shield** (STFT inpaint + residue randomization + decoys) | Done (deterministic baseline) | `keyguard/shield/shield.py` |
| **Neural separator** (speech-preserving) | Done — STOI 0.99, PESQ 2.90; adaptive attacker 86% → 63% | `keyguard/shield/separator.py` |
| **Adversarial shield D** (min-max, AI-essential centerpiece) | Done — arena at −18 dB budget (chance 2.8%): attacker **87.2% clean → 2.8% shielded (= chance) every round**; a retraining attacker claws back to ~56% mid-game vs a *frozen* shield, but D re-solves and re-blinds it to **2.8%** each round | `keyguard/shield/adversarial.py` |
| Synthetic speech+keystroke mixtures | Done (LibriSpeech + keystrokes) | `keyguard/synth.py` |
| Evaluation harness (adaptive attacker + PESQ/STOI) | Done | `keyguard/eval/` |
| Dashboard | Done | `keyguard/server.py`, `keyguard/web/` |
| Real-time BlackHole virtual mic | Offline core ready | — |

---

## The attack (red team) — two pillars

A keystroke is a sharp broadband transient. Given call audio we **segment** onsets,
cut a fixed window around each, turn it into a **log-mel spectrogram**, and classify
it. But *why* the attack works — and why it's genuinely dangerous — rests on two
pillars:

**Pillar 1 — per-key acoustics.** Each key has a distinguishable sound: the plate
resonates differently by location, so **per-key timing + spectral (plate-resonance)
features** distinguish keys. A CNN learns this. On the public
[Harrison dataset](https://github.com/JBFH-Dev/Keystroke-Datasets) (MacBook Pro,
phone + Zoom recordings) our CoAtNet-style classifier hits **87.2% top-1** (chance
2.8%), reproducing Harrison et al. 2023 (arXiv 2308.01074). On our own laptop-mic
captured bank the signal is real but weaker — **37.8% top-1 / 57.1% top-3** — laptop
mic + quiet keys, not a code problem.

**Pillar 2 — language redundancy (load-bearing AI).** English carries only ~1
bit/character, so a *weak* acoustic guess plus a language model recovers the text.
This is where AI does real work on the attack side: the attacker's **per-key top-3
candidate lattice** is handed to **Gemini** (`gemini-flash-latest`,
`keyguard/attack_lm.py`, wired into `/api/demo`), which multiplies those weak guesses
against English redundancy. This is verified live — from a corrupted top-1 stream
`NAAG1MH` Gemini recovers `MEETING`, and from `OSAAEPTS` it recovers `PASSWORD`, using
only the top-3 lattice. **Random passwords are the honest hard case** — no language
prior, so Pillar 2 can't rescue them (in the demo, `HUNTER2` reads at 100% with no
shield but 28.6% with the shield on, and Gemini cannot reconstruct the garbled
lattice) — which is exactly why an LLM-augmented attacker is still defeated by the
shield.

The overlap/free-typing **CTC transcriber** (`keyguard/ctc/`) treats typing as speech
recognition — no hard segmentation, the CTC blank is the detector. The pipeline is
verified (it overfits synthetic overlapping typing to ~4% CER), but generalization on
the weak laptop-mic signal collapsed (the model emitted the same common letters
regardless of input). The bottleneck is **signal strength, not the code**. Path
forward: a stronger/fixed sensor (phone mic Harrison-style, or a contact/accelerometer
sensor via Arduino), transfer-learning from Harrison, and a training-collapse fix
(multi-task onset head).

## The shield (blue team) — a constrained adversarial min-max

The reframe that makes AI **essential** to the defense:

```
min_D  max_A  Leakage( A( D(x) ) )      subject to   Perceptual( D(x), x ) <= eps
```

- **D** — a small causal defender network — emits a *bounded* perturbation added to
  the audio.
- **A** — the attacker network (our CTC/CNN keystroke reader) — **retrains** to beat
  whatever D is doing.
- **Perceptual constraint** — a human, PESQ/STOI, *and* Whisper must hear the
  original speech unchanged.

**Why this is provably not a deterministic transform, and why AI is essential:**

1. The inner `max_A` means the optimal defense depends on the *current* attacker,
   which retrains. No fixed transform is optimal against a moving adversary.
2. The minimal, speech-preserving perturbation is defined *through the attacker's
   neural network* — adversarial-example optimization against A, plus a
   psychoacoustic/perceptual loss. There is no closed form.
3. "Just garble the keystroke band" is the trivial **max-perturbation** solution
   that **violates the perceptual constraint** — it breaks Whisper and destroys
   overlapping speech. That's the whole point: the hard, interesting region is the
   *constrained* minimum, and only optimization finds it.

**Architecture — the arena is the training loop, not an LLM tuning knobs.** D and A
co-train GAN-style: **A is the discriminator** (can it read the keys?), **D is the
generator** (make the keys unreadable) under a hard perceptual regularizer (STOI/PESQ
floor + Whisper transcription must stay intact). Implemented in
`keyguard/shield/adversarial.py` as a universal/learned bounded adversarial
perturbation against a differentiable attacker, with adversarial co-training rounds.

**Where the earlier work now sits:** the deterministic DSP shield
(`keyguard/shield/shield.py` — timestamp-conditioned STFT inpainting + per-stroke
residue randomization + decoys) and the neural separator
(`keyguard/shield/separator.py` — preserves speech at STOI 0.99 / PESQ 2.90 while
cutting an adaptive attacker 86% → 63%) are the **baselines**. RNNoise-style
suppression does essentially nothing (86% → 84%). The old "LLM tunes shield config"
idea is demoted to an optional *proposer*, not the AI story.

**Adversarial shield results (dashboard arena).** The co-training loop is served at
`/api/arena` from `runs/arena/adv-*/arena.json`. At an inaudible **−18 dB**
perturbation budget (chance = 2.8%, 36 classes):

- Clean (unprotected) attacker on Harrison isolated keys: **87.2%**.
- Shielded, **every round: 2.8% (= chance)** — the deployed shield always drives the
  attacker to chance.
- A **retraining** attacker fighting a *frozen* shield partially claws back across
  rounds — **20.6% → 47.8% → 55.6% → 47.2% → 18.3%** (up to ~56% mid-game) — but D
  re-solves against it every round and re-blinds it to **2.8%**. Final shielded =
  **2.8%**.

This is a real bouncing arms race, not a flat monotonic curve: a retraining attacker
recovers against a static D, but D wins each round it re-optimizes.

## Metrics (`eval/metrics.py`)

- Text accuracy (edit-distance similarity) & top-1 / top-3
- **Mutual information in bits/keystroke** — a data-processing lower bound on leakage
- Cluster purity vs true keys
- Speech quality: **PESQ + STOI**, plus a Whisper transcription-intact check for the
  adversarial shield's perceptual constraint

---

## Setup

```bash
uv sync                                   # install deps
git clone --depth 1 https://github.com/JBFH-Dev/Keystroke-Datasets data/harrison

# 1. train the baseline attacker (~2 min on M-series MPS)
uv run python3 train_attacker.py

# 2. capture your own labeled typing bank (SPACE included)
uv run python3 capture.py bank

# 3. train the overlap/free-typing CTC transcriber
uv run python3 train_ctc.py

# 4. launch the dashboard
uv run uvicorn keyguard.server:app --port 8000   # open http://localhost:8000
```

Attack-side LM (Pillar 2): set `GEMINI_API_KEY` (in `.env`, loaded by
`keyguard/config.py`); the demo uses `gemini-flash-latest`. Without a key the LM pass
falls back to the acoustic top-1 and everything still runs end to end.

## Layout

```
keyguard/
  segment.py features.py audio.py config.py   # signal front-end
  attackers/  supervised.py timing.py          # red team
  ctc/        data.py model.py decode.py       # overlap/free-typing transcriber
  shield/     shield.py separator.py adversarial.py   # blue team (baseline -> neural -> min-max)
  eval/       harness.py metrics.py baselines.py
  synth.py                                      # speech+keystroke mixtures
  server.py web/index.html                      # dashboard
data/harrison/   # Harrison et al. dataset (git-cloned, not committed)
runs/            # trained models, results
```

## Honest limits
- Laptop-mic signal is weak (quiet keys) — the attack works but is far below the
  87.2% Harrison number, and the free-typing CTC model doesn't yet generalize off it.
- Random passwords have no language prior, so Pillar 2 can't rescue them — the honest
  hard case, stated as such.
- The adversarial shield D is the centerpiece: at a −18 dB budget the deployed shield
  drives the attacker to chance (2.8%) every round; a retraining attacker claws back to
  ~56% mid-game against a frozen shield, but D re-solves and re-blinds it to 2.8%.

## Credit / reproductions
- **Attack — a faithful reproduction/extension, not a novel attack.** Asonov & Agrawal,
  IEEE S&P 2004 (*Keyboard Acoustic Emanations* — each key has a distinct sound);
  Harrison, Toreini, Mehrnezhad 2023 (arXiv 2308.01074, *A Practical Deep Learning-Based
  Acoustic Side Channel Attack on Keyboards* — up to **95% lab / 93% Zoom**; dataset +
  supervised attacker); Slater et al., ACSAC 2019 (*Robust keystroke transcription from
  the acoustic side-channel* — overlap-robust CTC prior for the free-typing transcriber).
- **Defense — closest prior art is EveGuard.** Chang et al., *EveGuard*, IEEE S&P 2025
  (arXiv 2411.10034) — adversarial-audio side-channel *defense*, but for the **vibration**
  channel protecting **spoken speech**; and AntiFake (Yu et al., CCS 2023). Keyguard adapts
  this adversarial-audio-privacy paradigm to the acoustic **keystroke** side-channel
  (secret = typed text) — a channel and secret it hasn't been applied to. Carlini & Wagner
  (adversarial audio) and Schönherr et al. (psychoacoustic hiding) are the underlying
  lineage. We did not invent adversarial-audio privacy and do not claim to have "solved"
  the vulnerability.
