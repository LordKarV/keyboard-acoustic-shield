# Catalog: Existing Open-Source Acoustic Keystroke Side-Channel Attacker Implementations

> **Purpose (KeyGuard / HackGT):** Survey of public, openly-licensed code and pretrained
> models that infer typed keys/text from keyboard audio, so we can learn from proven
> implementations while building the *attacker* half of an authorized defensive research
> project. Everything below is public research code, tested on our own devices only.
>
> **Compiled:** 2026-09-26. Metadata (stars, last push, license) fetched from the GitHub API
> on that date and will drift over time.
>
> **Key finding up front:** There is **no public code release** for a true *continuous /
> overlapping* CTC/seq2seq keystroke transcriber. The seminal continuous work (Slater et al.,
> ACSAC 2019) never released code. Every runnable repo below is an **isolated-key classifier**
> (segment each keypress, then classify a mel-spectrogram). For a continuous CTC transcriber we
> must **compose** pieces: segmentation + feature extraction from these repos, n-gram/LM
> decoding from keytap2/3, and a CTC training loop adapted from a general ASR (Conformer/
> DeepSpeech) recipe.

---

## Quick comparison table

| Repo | Task | Architecture | Framework | Pretrained weights | Dataset | License | Last push | Stars |
|------|------|--------------|-----------|--------------------|---------|---------|-----------|-------|
| [soheil/DeepKeyAttack](https://github.com/soheil/DeepKeyAttack) | Isolated | CoAtNet (CNN+self-attn) | PyTorch | Not provided | Harrison phone/Zoom | None stated | 2023-08 | 52 |
| [MomirMilutinovic/…-coatnet](https://github.com/MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet) | Isolated | CoAtNet | PyTorch | Not provided | Harrison datasets | None stated | 2024-07 | 11 |
| [JBFH-Dev/Keystroke-Datasets](https://github.com/JBFH-Dev/Keystroke-Datasets) | Data only | — (official paper data) | — | — | MBP phone WAVs + Zoom | **MIT** | 2023-08 | 34 |
| [ggerganov/kbd-audio](https://github.com/ggerganov/kbd-audio) | Isolated + **unlabeled continuous** | classic DSP + n-gram LM | C++ | keytap needs training clip | user-recorded | **MIT** | 2023-01 | 9032 |
| [shoyo/acoustic-keylogger](https://github.com/shoyo/acoustic-keylogger) | Isolated | unsupervised cluster + CNN | PyTorch | Not provided | self-recorded | None (archived) | 2022-11 | 159 |
| [cnoujaim/Keyboard-Acoustic-Emanations-Python](https://github.com/cnoujaim/Keyboard-Acoustic-Emanations-Python) | Isolated | features + classifier + LM | Python | Not provided | self-recorded | None stated | 2021-11 | 5 |
| [aworse/KDAA](https://github.com/aworse/KDAA) | Isolated | log-mel CNN (jamo) | PyTorch | Not provided | Korean self-recorded | None stated | 2026-09 | 0 |
| [yossigor/AcousticKeylogger](https://github.com/yossigor/AcousticKeylogger) | Isolated + dict attack | classic ML | Python | Not provided | self-recorded | **GPL-3.0** | 2019-03 | 23 |
| [rodrigoalvesvieira/SoundKeylogger](https://github.com/rodrigoalvesvieira/SoundKeylogger) | Isolated (demo) | pattern analysis | C | No | self-recorded | None stated | 2016-07 | 69 |
| [KeystrokeTranscriptionWithCWT/KeystrokeRecordings](https://github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings) | Data only (continuous) | — (CWT thesis data) | — | — | continuous-typing WAVs | None stated | — | low |

---

## Detailed entries

### 1. soheil/DeepKeyAttack  — *most runnable CoAtNet reproduction*
- **URL:** https://github.com/soheil/DeepKeyAttack
- **What it does:** Clean re-implementation of Harrison et al. 2023 ("A Practical Deep
  Learning-Based Acoustic Side Channel Attack on Keyboards", arXiv 2308.01074). Classifies
  **isolated** laptop keystrokes (36 classes: a–z, 0–9) recorded via phone mic or Zoom.
- **Architecture:** CoAtNet (convolution + self-attention hybrid) over mel-spectrogram images.
- **Framework:** PyTorch. Ships `CoAtNet.py` (model), `train.py`, `infer.py` — a tidy,
  self-contained train/infer split that is easy to read and adapt.
- **Pretrained weights:** Not provided (train yourself; the Harrison datasets are public).
- **Dataset:** Harrison phone/Zoom keystroke recordings.
- **License:** None declared (⚠️ legally "all rights reserved" — usable as *reference*, but do
  not vendor code without contacting the author).
- **Status:** 52★, last push 2023-08, not archived.
- **Take:** Best-organized modern PyTorch CoAtNet attacker; ideal template for our
  isolated-key baseline classifier.

### 2. MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet — *reproduction + segmentation script*
- **URL:** https://github.com/MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet
- **What it does:** Student reproduction of the same Harrison 2023 paper. **Isolated** keys.
- **Architecture:** CoAtNet over mel-spectrograms (PyTorch, in a Jupyter notebook
  `keyboard-side-channel.ipynb`).
- **Key asset:** `isolate_key_presses.py` — standalone **energy/threshold-based keystroke
  segmentation** that turns a raw recording into per-key clips. Directly reusable and the
  single most valuable file here for our segmentation stage.
- **Pretrained weights:** Not provided. **License:** None declared. **Status:** 11★, 2024-07.
- **Take:** Borrow the segmentation script; the notebook is a readable end-to-end walkthrough.

### 3. JBFH-Dev/Keystroke-Datasets — *official paper data (MIT)*
- **URL:** https://github.com/JBFH-Dev/Keystroke-Datasets
- **What it does:** Official data artifact from Harrison et al. 2023 (author Josh Harrison).
  Contains `MBPWavs` (MacBook Pro keystrokes recorded by nearby phone) and `Zoom` recordings.
  **Datasets only — no model code** in this repo (the paper's code was never released as a
  separate repo; the reproductions above fill that gap).
- **License:** **MIT** — the one clearly-licensed, freely usable asset. Good for benchmarking
  our classifier against the published 95%/93% numbers.
- **Status:** 34★, last push 2023-08.

### 4. ggerganov/kbd-audio — *the classic; only public "continuous" attacker*
- **URL:** https://github.com/ggerganov/kbd-audio
- **What it does:** C++ toolkit (`keytap`, `keytap2`, `keytap3`). `keytap` = supervised isolated
  detection from a short training clip. **`keytap2`/`keytap3` recover text from an *unlabeled*
  continuous recording with no per-key training**, using clustering of keystroke waveforms plus
  **English letter/n-gram frequency statistics** to solve the substitution cipher. keytap3 fully
  automates this.
- **Architecture:** Classic DSP (adaptive-threshold onset detection, cross-correlation
  similarity) + statistical **n-gram language decoding** — no deep learning.
- **License:** **MIT**. **Status:** 9032★, 608 forks, last push 2023-01. Most mature project here.
- **Take:** Even though it's not a neural CTC model, keytap2/3 is the closest thing to a
  continuous free-typing attacker with public code. **Borrow: (a) the adaptive-threshold
  onset/keystroke detector, (b) the n-gram/language-model decoding that maps acoustic clusters
  to text without labels** — that decoding layer is exactly what a CTC beam-search + LM rescorer
  needs on our side.

### 5. shoyo/acoustic-keylogger — *unsupervised pipeline, well-documented*
- **URL:** https://github.com/shoyo/acoustic-keylogger
- **What it does:** End-to-end keylogging pipeline (Zhuang et al. 2005 "Keyboard Acoustic
  Emanations Revisited" lineage). Peak-detection segmentation → MFCC/mel features →
  unsupervised clustering → CNN classification. **Isolated** keys.
- **Framework:** PyTorch; Dockerized with a Postgres data store for keystroke samples.
- **Pretrained weights:** Not provided. **License:** None declared; repo **archived (read-only)**.
- **Status:** 159★ (most-starred pure-Python attacker), last push 2022-11.
- **Take:** Good reference for the full data-prep pipeline (segmentation + feature extraction +
  unsupervised label discovery) and for the classic-vs-DL comparison.

### 6. cnoujaim/Keyboard-Acoustic-Emanations-Python — *isolated + language-model correction*
- **URL:** https://github.com/cnoujaim/Keyboard-Acoustic-Emanations-Python
- **What it does:** Detects key presses, extracts features, classifies, then applies a
  **language-model correction pass** — reports 88.8% char-level / 46.1% word-level accuracy.
  Isolated keys.
- **License:** None declared. **Status:** 5★, 2021-11.
- **Take:** Small but shows a spellcheck/LM post-correction stage worth mirroring after CTC decode.

### 7. aworse/KDAA — *freshest (2026) log-mel CNN*
- **URL:** https://github.com/aworse/KDAA
- **What it does:** Korean Dubeolsik keyboard acoustic attack. Pipeline: record → segment →
  **log-mel spectrograms → CNN jamo classifier** (CPU, Python 3.11). Isolated keys.
- **License:** None declared. **Status:** 0★ but **most recently updated (2026-09)** — modern,
  minimal, easy-to-follow PyTorch code.
- **Take:** Useful clean modern segmentation→log-mel→CNN reference; non-English target shows the
  pipeline generalizes.

### 8. yossigor/AcousticKeylogger — *GPL, password dictionary attack*
- **URL:** https://github.com/yossigor/AcousticKeylogger
- **What it does:** Classic-ML keystroke recovery aimed at building smart dictionary attacks on
  passwords. Isolated. **License: GPL-3.0** (only copyleft repo — reusable, but GPL obligations
  apply if we redistribute). **Status:** 23★, 2019.

### 9. rodrigoalvesvieira/SoundKeylogger — *old C demo*
- **URL:** https://github.com/rodrigoalvesvieira/SoundKeylogger
- Early (2016) pattern-analysis proof-of-concept in C. 69★. Historical interest only.

### 10. KeystrokeTranscriptionWithCWT/KeystrokeRecordings — *continuous-typing dataset*
- **URL:** https://github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings
- Recordings for the thesis "Keystroke Transcription From Acoustic Emanations Using Continuous
  Wavelet Transform" (~80.3% accuracy, Springer 2024). **Data only, no model code**, but it is
  one of the few *continuous free-typing* datasets available — valuable for training/evaluating
  our CTC transcriber.

---

## Papers with strong methods but NO public code (for reference / to re-implement)

- **Slater et al., "Robust Keystroke Transcription from the Acoustic Side-Channel," ACSAC 2019**
  (Two Six Labs). The canonical **continuous / overlapping** attacker: end-to-end audio→keystroke
  transcription with a **recurrent (RNN) speech-transcription architecture (CTC-style)** that
  handles fast typing, overlapping waveforms, and local noise adaptation. **No code released** —
  this is the paper our CTC transcriber should conceptually reproduce.
  PDF: https://www.acsac.org/2019/program/final/1/137.pdf
- **Harrison, Toreini, Mehrnezhad 2023, arXiv 2308.01074** — CoAtNet isolated classifier
  (95% phone / 93% Zoom). Code not released by authors; reproduced by repos #1 and #2 above;
  official data in #3.
- **"Improving Acoustic Side-Channel Attacks... Transformers and LLMs," arXiv 2502.09782 (2025)**
  — Vision Transformers (ViT/Swin/DeiT/BEiT) + LLM (Llama-3, GPT-4o) typo correction on the
  Harrison data. **No code/weights released.**
- **"Impact Analysis of Speech Representation Learning Models for ASCA," arXiv 2606.21210**
  — introduces **KEYAC** dataset and evaluates self-supervised speech encoders (wav2vec2/HuBERT/
  WavLM-family) as keystroke feature extractors. Code/data availability not confirmed public.
- **"Improved CoAtNet for robust ASCA on keyboards," Int. J. Info. Security 2025** — CoAtNet
  variant, +5–6% over Harrison baseline. Paper only.

## General ASR building blocks to adapt for a CTC keystroke transcriber (openly licensed)
Since no keystroke-specific CTC repo exists, adapt a mature Conformer-CTC / DeepSpeech ASR recipe
and swap the vocabulary for keys:
- [LuluW8071/Conformer](https://github.com/LuluW8071/Conformer) — Conformer-CTC training with
  PyTorch Lightning (clean, small).
- [sooftware/conformer](https://github.com/sooftware/conformer) / [jaketae/conformer](https://github.com/jaketae/conformer)
  — reference Conformer implementations.
- [espnet/espnet](https://github.com/espnet/espnet) — full CTC/attention/RNN-T recipes if we want
  a heavier toolkit.

---

## Recommendations for KeyGuard's attacker (what to borrow)

1. **Isolated baseline (fast win):** Start from **soheil/DeepKeyAttack** (PyTorch CoAtNet,
   clean train/infer split) and train on **JBFH-Dev/Keystroke-Datasets** (MIT). Reproduce the
   ~95% isolated-key number as our attacker baseline and defense target.
2. **Segmentation:** Reuse **MomirMilutinovic's `isolate_key_presses.py`** and/or **kbd-audio's
   adaptive-threshold onset detector** for the keypress detection stage.
3. **Continuous CTC transcriber (the real goal):** No drop-in repo exists. Compose:
   - Front end: log-mel / MFCC features (from shoyo or aworse pipelines).
   - Acoustic model: a **Conformer/CRNN encoder trained with CTC loss** (adapt LuluW8071/
     Conformer or espnet), vocabulary = keys + blank, trained on continuously-typed audio.
   - Decoder: **beam search + n-gram/character language model** — mirror **keytap2/keytap3's
     n-gram frequency decoding** (MIT) and add an LLM/spellcheck rescoring pass as in
     cnoujaim's LM-correction step.
   - Data: use the **CWT continuous recordings** repo plus our own recorded free-typing sets.
4. **Licensing caution:** Only **kbd-audio** and **JBFH-Dev data** (MIT) and **yossigor**
   (GPL-3.0) are clearly licensed. The CoAtNet reproductions (#1, #2) and shoyo declare no
   license — treat as read-only reference, re-implement rather than copy.
