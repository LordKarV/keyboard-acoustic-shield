# Keyguard

**We can hear your password. Then we make it impossible.**

## Inspiration
Your keyboard leaks. Every key has its own plate resonance, and recent work (Harrison et al. 2023) read keystrokes over Zoom at **93%** accuracy from sound alone. So anyone on a call with you is effectively running a keylogger. Standard defenses like white noise or muting are either audible or easy to filter out. We wanted a shield that a person can't hear and that still works when the attacker retrains to beat it.

## What it does
Keyguard is an **attack and a defense**, and we show both live.

**The attack has two pillars.**
- **Acoustics:** a CoAtNet-style CNN on log-mel features reads single keystrokes at **87.2% top-1** (random guessing is 2.8% over 36 keys). For fast, overlapping free typing, a CRNN transcriber (BiGRU with an onset head) reads **real continuous typing from people it has never heard** at **~41% CER** (random is ~95%). Only one published system has read overlapping typing (Slater, ACSAC '19).
- **Language:** English carries about 1 bit per character, so a weak acoustic signal is still enough. We give **Gemini** the attacker's per-key **top-3 candidate lattice**, and it reconstructs the text: `QBTTVPSE` becomes **`PASSWORD`**, and `NAAG1MH` becomes **`MEETING`**.

**The defense is the centerpiece.** The shield **D** is a learned **universal adversarial perturbation**: a waveform about 18 dB quieter than a keystroke. It's added onto every keystroke at the exact moment your OS reports a key press, and it's trained against attackers **A** that keep retraining:

$$
\min_{D}\,\max_{A}\;\mathcal{L}_{\text{leak}}\big(A(D(x))\big)\quad\text{s.t.}\quad \mathcal{P}\big(D(x),\,x\big)\le\varepsilon
$$

- The attacker drops from **87.2% to 2.8% (random guessing) in every round**.
- Arms race: a retraining attacker climbs back to 20.6% → 47.8% → 55.6%, and D re-solves and blinds it again each time.
- **One perturbation defeats four different architectures at once** (SE-CNN, wide CNN, ResNet, frame-GRU), including a ResNet that had climbed back to 91% partway through the race.
- A **transfer attacker it never trained against** also drops to 2.8%, so D isn't overfit to its sparring partners.
- **Speech stays clear:** STOI 0.96. On the leakage vs. speech-quality Pareto chart, it sits alone in the safe corner, ahead of every noise-based baseline.

## Why the AI has to be there
Our first shield was plain signal processing: it used the OS key timestamps to inpaint each keystroke's STFT magnitude, randomize the leftover sound, and inject decoy clicks. It hurt a fixed attacker (CER 40.7% → 55.5%). Then we let the attacker **retrain on shielded audio**, and it went straight back to **39.4%**. **Rule-based defenses lose to an adaptive attacker. We measured that.** The fix of "just garble the keystroke band" wrecks speech and Whisper. What's left is a real min-max problem where the smallest perturbation that stays inaudible is defined *through the attacker's own network*. There's no closed-form answer, so the defense has to be learned.

## How we built it
- **Stack:** PyTorch (CUDA / MPS / CPU), librosa, FastAPI, and a vanilla JS/SVG dashboard.
- **Front-end:** onset segmentation and log-mel features. We found and fixed a mismatch between our differentiable torch log-mel and librosa (STFT pad mode plus a missing 80 dB floor; correlation went from 0.956 to 1.0). That fix is what made the adversarial gradients line up with the real attacker.
- **Overlap transcriber:** trained on **SKAID** (26 typists, 67k keystrokes). CTC collapsed to the English letter prior, and we traced three root causes: SpecAugment was shifting audio frames but not their labels, the curriculum was feeding mostly empty chunks, and logits were being shipped off the GPU. We replaced CTC with **class-weighted dense per-frame CE plus onset-BCE**. The attacker went from random-level output to reading real typing.
- **Adversarial arena:** population co-training, streamed live to the dashboard. **Backboard** gives it memory across sessions: it records every attacker the shield has faced, so the defense is always evaluated against the full history.
- **Live calibration:** a guided per-key recorder banks the target keyboard (including space, for word boundaries) and collects labeled real typing to fine-tune on.
- **Demo resilience:** a cache-backed demo mode, so a noisy hall can't break the live moment.

## Challenges
- **Overlapping keystrokes:** clicks 24 ms apart, each ringing for 120 ms. Hand-tuned onset detectors capped recall at 56–78%, so we moved to learned per-frame onsets.
- **Cross-keyboard generalization (honest negative result):** a model pooled across 8 keyboards scores **6.4% top-3** zero-shot on a keyboard it has never seen. Per-key acoustics are specific to each device, so a real attacker has to calibrate on your keyboard. That's why our live pipeline banks the target keyboard.
- **Compute:** a loaded laptop and unreliable MPS, so we split training across a Mac and a CUDA GPU box, handing data off through git.

## What we learned
Rule-based privacy breaks as soon as the adversary can learn. Defenses have to be trained *against* adaptive attackers and constrained by human perception. An LLM turns a mediocre side-channel into a dangerous one.

## What's next
- Run the learned shield against the adaptive overlap transcriber (right now it has been proven against single-key attackers).
- Ship it as a real-time virtual mic (BlackHole) on the victim's side.
- Extend it to other acoustic side channels on calls.

## Standing on the shoulders of
Asonov & Agrawal '04 · Zhuang, Zhou & Tygar '05 · Cecconello et al., *Skype & Type* '19 · Slater et al., ACSAC '19 · Harrison, Toreini & Mehrnezhad '23 (2308.01074) · LLM-assisted keystroke correction (2504.11622, 2502.09782) · **EveGuard, IEEE S&P '25 (2411.10034)**, the closest prior defense (vibration channel, speech) · AntiFake, CCS '23 · Szegedy '13, Carlini & Wagner '18 · ASCA survey (2309.11012) and SoK (2308.03806).

**Novelty, stated carefully:** the attack reproduces and extends prior work. To our knowledge, Keyguard is the **first perceptually-constrained adversarial shield for the acoustic keystroke channel**, bringing EveGuard-style adversarial audio privacy to a new channel and a new secret.

## Ethics
Only our own devices, consenting participants, and fake passwords. The shield runs on the victim's side and uses only the OS **timing** of key presses, never which key was pressed.

## Built with
Python · PyTorch · librosa · FastAPI · Gemini · Backboard · JavaScript · SVG
