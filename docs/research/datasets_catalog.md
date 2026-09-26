# Public Labeled Keystroke-Audio Datasets — Catalog

Project: KeyGuard (HackGT) — authorized defensive acoustic keystroke side-channel research.
Scope: ONLY publicly available, openly-licensed academic/research datasets. Audio of
keyboards with per-key or per-text labels. Priority = **continuous / overlapping free
typing** (real-world), which the existing `data/pool/*.npz` (isolated-key windows) lacks.

Last updated: 2026-09-26

---

## TL;DR

- **Best NEW find (downloaded): CWT KeystrokeRecordings — "Pangrams" & "ThreeStates".**
  Real **continuous / overlapping free typing** of full sentences (pangrams) with known
  text labels. This is exactly the overlapping-typing data the pool is missing.
- Downloaded into `data/external/`:
  - `cwt_keystroke_transcription/PangramsandThreeStates/` — 40 WAVs, 81 MB (**continuous typing**)
  - `zenodo_keyboard_sound_16564409/` — 415 WAVs, ~9.9 MB (isolated keys, keyboard-type labels)
- Most real-world continuous datasets from top papers (Slater ACSAC'19, Skype&Type) are
  **not publicly downloadable** — recorded commands / status noted below.
- HuggingFace and kbd-audio yielded no NEW usable keystroke-audio beyond what is already pooled.

---

## DOWNLOADED datasets

### 1. CWT KeystrokeRecordings (Pangrams + ThreeStates) — CONTINUOUS TYPING ⭐
- **Source:** https://github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings
  ("Keystroke Transcription From Acoustic Emanations Using Continuous Wavelet Transform" thesis)
- **Downloaded to:** `data/external/cwt_keystroke_transcription/PangramsandThreeStates/`
- **Contents downloaded:** 20 × `Panagram_XX.wav` + 20 × `ThreeStates_XX.wav` = 40 files, **81 MB**
- **Format:** WAV, mono, **48 kHz**, 16-bit PCM
- **Duration:** Pangrams ≈ 29.6 s each; ThreeStates ≈ 14.6 s each
- **Labels:** per-text (each Pangram file = a full pangram sentence, e.g. "the quick brown
  fox…", typed naturally; ThreeStates = fixed text states). No per-keystroke timestamps
  shipped, but the typed text is known → usable for CTC / transcription-style training.
- **Typing type:** **CONTINUOUS / OVERLAPPING free typing** (natural rhythm, full sentences). ✅
- **Keyboards/typists:** single keyboard, single typist (thesis author).
- **License:** GPL-3.0.
- **Download command (what was run):**
  ```bash
  RAW="https://raw.githubusercontent.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings/main/KeystrokeRecordings/PangramsandThreeStates"
  for i in $(seq -w 1 20); do
    curl -sL -o "Panagram_$i.wav"   "$RAW/Panagram_$i.wav"
    curl -sL -o "ThreeStates_$i.wav" "$RAW/ThreeStates_$i.wav"
  done
  ```
- **Full repo (adds isolated-key sets, ~320 MB total):**
  ```bash
  git clone --depth 1 https://github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings
  ```
  Other folders (NOT downloaded — isolated repeated single keys, similar to existing pool):
  - `Hundreds/` — 152 WAVs, 182 MB (each file = ~100 presses of ONE key; per-key label)
  - `TwentyFives/` — 38 WAVs, 53 MB (each file = ~25 presses of ONE key; per-key label)

### 2. Zenodo 16564409 — "Keyboard sound dataset" (isolated, keyboard-type labels)
- **Source:** https://zenodo.org/records/16564409
- **Downloaded to:** `data/external/zenodo_keyboard_sound_16564409/` (415 WAVs, ~9.9 MB; zip removed)
- **Format:** WAV, mono, **44.1 kHz**, 16-bit, **0.25 s** clips
- **Structure:** 4 folders by switch type — `blue/`(100), `red/`(136), `membrane/`(99),
  `pantograph/`(80). `wav_keyboard_sound_train.csv` / `wav_keyboard_sound_test.csv` give
  `x,y` (path, class 0–3).
- **Labels:** keyboard **switch TYPE** (0..3), NOT per-character. Isolated single keystrokes.
- **Typing type:** ISOLATED single keys.
- **License:** MIT (Sony Corporation 2020).
- **Value:** keyboard-hardware diversity for the shield / augmentation, not per-key attack labels.
- **Download command:**
  ```bash
  curl -sL -o keyboard_sound_dataset.zip \
    "https://zenodo.org/api/records/16564409/files/keyboard_sound_dataset.zip/content"
  unzip keyboard_sound_dataset.zip
  ```

---

## NOT downloaded — catalogued (already pooled, gated, or not audio)

### A. JBFH-Dev / Keystroke-Datasets (Harrison et al. 2023, arXiv:2308.01074)
- https://github.com/JBFH-Dev/Keystroke-Datasets — MIT.
- Two sets: `MBPWavs/` (iPhone 13 mini next to 16" M1 Pro MacBook) and `Zoom/` (Zoom
  recording). Keys 0–9, q–m; each WAV = ~25 presses of ONE key. Per-key labels, **isolated**.
- **Already in pool as `harrison.npz`.** Clone: `git clone https://github.com/JBFH-Dev/Keystroke-Datasets`

### B. Slater et al., "Robust Keystroke Transcription from the Acoustic Side-Channel" (ACSAC'19) ⭐ (gated)
- Paper: https://www.acsac.org/2019/program/final/1/137.pdf | DOI 10.1145/3359789.3359816
- **17 users, ~86k keystrokes, real-world CONTINUOUS typing tasks** — the single most
  valuable continuous dataset conceptually. Two Six Labs. **No public download found**
  (not on Zenodo/OSF/GitHub/figshare). Action: would need to email authors for research access.

### C. Skype & Type (Cecconello et al., Univ. Padua)
- Paper: https://spritz.math.unipd.it/datasets/locard_pubs/Skype_and_TypeKeyboardEavesdroppingInVoice-over-IP.pdf
- 8 users typed a pangram; also 260-sample per-user sets (10 × 26 letters), WAV 44.1 kHz.
  Mostly isolated per-letter samples. Project page `spritz.math.unipd.it/projects/dontskypetype/`
  returned 404 — **no working public download link found**. Contact authors if needed.

### D. Multi-Pressure Keyboard Key Sounds (MDPI Data 2026, Zenodo 19453177)
- https://zenodo.org/records/19453177 — alphanumeric + space at low/med/high pressure.
- **Already in pool as `multipressure.npz`.** Isolated keys, per-key labels.

### E. Multi-Keyboard Acoustic (MKA) — Mendeley Data
- https://data.mendeley.com/datasets/bpt2hvf8n3/4 — HP, Lenovo, MSI, Mac, Messenger, Zoom.
- **Already in pool as `mka_hp/lenovo/msi/mac.npz`.** Isolated/segmented keys, per-key labels.

### F. Sakuzas/key_stroke_sound (HuggingFace)
- https://huggingface.co/datasets/Sakuzas/key_stroke_sound
- **Already in pool as `hf_sakuzas_keyboard.npz`.** Isolated keys.

### G. Zenodo 15148876 / 17608769 — "Audio dataset of keyboard keystrokes" (CC-BY-4.0)
- Small (~1.5 MB `Keystrokes_Dataset.zip`, plus spectrograms/CSV in 17608769). Isolated keys.
- Overlaps the existing pool `zenodo.npz`. Low incremental value; skipped.
  DL: `curl -sL -o k.zip "https://zenodo.org/api/records/17608769/files/Keystrokes_Dataset.zip/content"`

### H. OSF 5xwng — "A Dataset of Noisy Typing on QWERTY Keyboards"
- https://osf.io/5xwng/ — **NOT AUDIO.** Keystroke-timing / text-entry logs from 9 studies. Excluded.

### I. ggerganov/kbd-audio (Keytap) & shoyo/acoustic-keylogger
- Tools/pipelines, not datasets. kbd-audio ships only English n-gram frequency text files
  (`data/english_*grams.txt`), no labeled audio. shoyo has code, no packaged public audio corpus.

---

## Notes on overlapping/continuous vs isolated

| Dataset | Continuous/overlapping? | Labels | Status |
|---|---|---|---|
| **CWT Pangrams/ThreeStates** | **YES (free sentences)** | per-text | **Downloaded** ✅ |
| CWT Hundreds/TwentyFives | No (repeated single key) | per-key | command recorded |
| Zenodo 16564409 | No | keyboard-type | Downloaded |
| Slater ACSAC'19 | **YES (real-world)** | per-key + timing | not public (gated) |
| Skype&Type | partial (pangram) | per-key | no working DL link |
| Harrison/MKA/multipressure/Sakuzas/zenodo | No | per-key | already in pool |

**Only the CWT Pangram/ThreeStates set provides real continuous/overlapping typing audio
that is both public and downloadable.** The pool's other sources are all isolated-key.
