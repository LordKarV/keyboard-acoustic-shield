# Keyboard / Keystroke AUDIO Datasets — Research Log

Goal ranking: **BEST** = MacBook built-in keyboard, continuous free-typing/sentences, keylog+timestamps → **GOOD** = any laptop, continuous, labeled → **OK** = isolated per-key MacBook presses (acoustic bank).

Our front-end for "how to ingest": resample to **16 kHz**, onset detect → **0.3 s window** → **log-mel**; emit SKAID-style `labels.jsonl` rows `{wav, keys, onset_samples}`.

Date of survey: 2026-09-26.

---

## Ranked table (best MacBook-continuous first)

| Rank | Dataset | URL | Host | MacBook? | Recording | Mic | Scale | Labels | License | Tier |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Harrison "Zoom" companion** (same repo as MBPWavs) | https://github.com/JBFH-Dev/Keystroke-Datasets | GitHub | ✅ MacBook Pro 16" M1 Pro | Isolated per-key | Zoom built-in recording | 36 keys × 25 | per-key `.wav` files (filename = label) | MIT | OK (real MacBook, new-to-us medium) |
| 2 | **CWT "KeystrokeRecordings"** (Pangrams + banks) | https://github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings | GitHub | ❓ unknown keyboard (thesis) | Pangrams = **continuous sentences**; also isolated banks | single mic (unstated) | Pangram_01–15+ continuous; TwentyFives (25/key); Hundreds (100/key) | text-known (pangram string); banks = filename per key. **No onset timestamps** | GPL-3.0 | GOOD-ish (continuous, weak labels) |
| 3 | **KAD** (mechanical) | https://zenodo.org/records/14809537 · https://github.com/spatam/kad | Zenodo/GitHub | ❌ mechanical (phone @17 cm) | Isolated per-key | Smartphone | 36 keys × 75 (→2,700 augmented) | per-key `.wav`, `augmented_audio_list.csv` | MIT | OK (extra isolated bank) |
| 4 | Zenodo 17608769 "Audio dataset of keyboard keystrokes" v5 | https://zenodo.org/records/17608769 | Zenodo | ❓ unstated | likely isolated | unstated | tiny (~5 MB: CSV + spectrograms + audio) | CSV | CC-BY-4.0 | OK-minor |
| 5 | Zenodo 15148876 (Viana do Castelo) | https://zenodo.org/records/15148876 | Zenodo | ❓ unstated | likely isolated | unstated | tiny (1.5 MB zip) | unstated | CC-BY-4.0 | OK-minor |
| 6 | Kaggle "keystroke-noiseless-final" | https://www.kaggle.com/datasets/nguyncaoduy/keystroke-noiseless-final | Kaggle | ❓ (JS page, unverified) | unknown | unknown | unknown | unknown | Kaggle ToS | unverified |
| 7 | Kaggle "Keyboard Stroke Sound" | https://www.kaggle.com/datasets/long4b/keyboard-stroke-sound | Kaggle | ❓ (JS page, unverified) | unknown | unknown | unknown | unknown | Kaggle ToS | unverified |

### Best-fit target that is NOT downloadable (be honest)
| Would-be #1 | Slater et al. ACSAC'19 "Robust Keystroke Transcription from the Acoustic Side-Channel" | https://dl.acm.org/doi/10.1145/3359789.3359816 · paper: https://www.acsac.org/2019/program/final/1/137.pdf | — | laptop typing | continuous | 17 users, **86k keystrokes**, real-world typing tasks, with keylog | **NOT publicly released** — no repo/Zenodo/OSF found. This is exactly our BEST profile; only the paper is public. |

---

## Per-dataset notes

### 1. Harrison et al. — "Zoom" companion dataset  ⭐ recommended download
- Repo `JBFH-Dev/Keystroke-Datasets` is the official data drop for the 2023 paper *"A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards"* (Harrison, Toreini, Mehrnezhad; arXiv 2308.01074). **This is the same paper as our "MBPWavs".**
- The repo contains **two** datasets: **(a) Phone** (iPhone 13 mini next to keyboard on microfibre cloth) = the MBPWavs we already have; and **(b) Zoom** — the *same* MacBook Pro 16" M1 Pro, 36 keys (1-0, q-m), 25 presses each, but captured through **Zoom's built-in meeting recording** (the victim's own mic + VoIP codec).
- **Why it matters:** the Zoom set is a distinct acoustic medium (VoIP-compressed, exactly the 93% "new best for the medium" result in the paper) on a real MacBook. We likely have only the Phone half. This is the closest new MacBook data to our own laptop-mic setup after the phone recordings.
- License MIT. Download: `git clone` the repo (files are committed directly, no LFS noted).
- **Ingest:** each `.wav` = 25 presses of one key. Run onset detection per file; every detected onset inherits the key from the filename; window 0.3 s @16 kHz → log-mel; write `labels.jsonl` `{wav, keys:[<key>], onset_samples:[...]}`. Treat Phone and Zoom as two separate acoustic domains (do NOT mix into one train fold — use for cross-medium eval, which is our shield's threat model).

### 2. CWT "KeystrokeRecordings"  ⭐ recommended download (only new *continuous* audio)
- Backs the thesis *"Keystroke Transcription From Acoustic Emanations Using Continuous Wavelet Transform"* (also Springer 2024 chapter, ISBN 978-981-97-2458-1_1). GPL-3.0.
- Three folders (verified via GitHub API):
  - **`PangramsandThreeStates/`** — `Panagram_01..15+` `.wav`, ~2.84 MB each = **continuous free-typing of full pangram sentences**. This is real continuous typing; the "label" is the known pangram text (character sequence), so you get ground-truth *content* but **no per-keystroke onset timestamps** — you must derive onsets yourself and align to the known string.
  - **`TwentyFives/`** — one `.wav` per key, 25 presses each (a-z, digits, backspace, caps, enter, comma, dot…). Direct MBPWavs-style isolated bank on a different keyboard.
  - **`Hundreds/`** — one `.wav` per key with **100 presses each** — a larger isolated bank (more presses/key than MBPWavs' 25).
- Keyboard/device is **not stated** in the repo (thesis rig; probably not a MacBook). Single unnamed mic. Sample rate unstated (we resample anyway).
- **Ingest (continuous pangrams):** onset-detect each `Panagram_*.wav`; you'll get M onsets. Align to the pangram's character order (drop non-printing keys). If onset count ≈ char count, zip them into `labels.jsonl`; otherwise use DTW / manual trim. Weak supervision but the only new continuous source.
- **Ingest (banks):** same recipe as MBPWavs; `Hundreds/` gives 4× the presses/key.

### 3. KAD (spatam / Zenodo 14809537)
- Mechanical keyboard, smartphone mic at 17 cm, **isolated** keystrokes. 36 keys (a-z, 0-9) × 75 `.wav` files = 900 files, → 2,700 after augmentation. `augmented_audio_list.csv` indexes them. 123.9 MB, MIT, published Feb 2025. Source code by email request; data is public on Zenodo.
- Not a MacBook, but a clean labeled isolated bank for a cross-device negative/robustness set.
- **Ingest:** filename→key; onset per file; 0.3 s @16 kHz log-mel. Good for "unseen keyboard" generalization tests of the shield.

### 4–5. Small Zenodo isolated sets (17608769, 15148876)
- Both CC-BY-4.0, very small (5 MB and 1.5 MB). Landing pages omit device/typing-mode; 17608769 ships a CSV + spectrogram archive + audio zip (structured labels implied), 15148876 is a single `Keystrokes_Dataset.zip` from Polytechnic Institute of Viana do Castelo. Likely isolated single-keyboard sets. Low priority; download only if we want more label diversity. Verify contents after unzip.

### 6–7. Kaggle sets (unverified)
- `nguyncaoduy/keystroke-noiseless-final` and `long4b/keyboard-stroke-sound` — Kaggle pages are JS-rendered and did not expose metadata to fetch. Need a logged-in look or the Kaggle API (`kaggle datasets download`). Device/MacBook status unknown. Flagged for manual check, not relied upon.

---

## Dead ends / already-have (checked, nothing new)
- **Slater et al. ACSAC'19** (17 users / 86k keystrokes, continuous, with keylog) — the ideal profile, but **no public data release** found (paper only). Worth emailing authors.
- **ggerganov/kbd-audio** (keytap/2/3) — tools only; training data is user-recorded via its `record`/`record-full` tools. No shipped MacBook dataset. Useful as a *recorder* to make our own MacBook continuous data + keylog.
- **shoyo/acoustic-keylogger** — pipeline code only, no dataset shipped (repo archived Jan 2023).
- **EchoCrypt / WOOT'25** (Ayati et al., "Noisy Keyboards…", https://github.com/Botacin-s-Lab/EchoCrypt) — reuses Harrison's Phone/Zoom 36×25 data; no new recordings.
- **DECKER** (arXiv 2605.03384, cross-keyboard) — data behind an anonymous 4open.science repo we couldn't read; revisit post-deanonymization.
- **KeyRecs, EmoSurv, Aalto, TypeNet, IEEE-DataPort "keystrokes"** — **keystroke *dynamics*** (press/release timings only), **no audio**. Not usable for acoustic training.
- Already in hand and still current: **SKAID** (Zenodo 17282184 — 26 people continuous + keylogs, our best), **MKA** (Mendeley bpt2hvf8n3, incl. a Mac subset), **multipressure** (Zenodo 19453177 / MDPI data11070168), **keyboard sound dataset** (Zenodo 16564409), **one-per-letter** (Zenodo 10623477), **MBPWavs** (= Phone half of #1).

---

## Recommended next downloads (top 3)

1. **Harrison "Zoom" companion** — https://github.com/JBFH-Dev/Keystroke-Datasets
   Real **MacBook Pro** audio in a second medium (VoIP/Zoom) we probably don't have; drops straight into the MBPWavs pipeline and gives us a cross-medium eval that matches the shield's threat model.

2. **CWT KeystrokeRecordings** — https://github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings
   The **only new continuous free-typing audio** we found that is actually downloadable (`Panagram_*` sentences), plus a 100-press/key isolated bank. Labels are known-text (no onset timestamps), so use for continuous-typing robustness, not as gold timing truth.

3. **KAD** — https://zenodo.org/records/14809537
   Easy Zenodo download, 900→2,700 labeled isolated presses on a *different* (mechanical) keyboard — ideal cross-device generalization / negative set for stress-testing the shield.

> Net finding: **no publicly downloadable MacBook + continuous free-typing + keylog dataset exists** as of this survey. SKAID (already held) remains our best continuous+keylog source; the best *additional* MacBook data is isolated (Harrison Phone+Zoom). For true continuous MacBook data with timestamps, the practical path is to **record our own** using ggerganov/kbd-audio's `record-full` (audio + synced keylog) on the user's own MacBook.
