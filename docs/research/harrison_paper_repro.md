# Harrison et al. (2023) — Reproduction Notes & Release Status

**Paper:** J. Harrison, E. Toreini, M. Mehrnezhad, *"A Practical Deep Learning-Based
Acoustic Side Channel Attack on Keyboards"*, IEEE EuroS&PW / SiLM Workshop 2023.
arXiv: **2308.01074**.

- Abstract / PDF: https://arxiv.org/abs/2308.01074 · https://arxiv.org/pdf/2308.01074
- Workshop PDF: https://silm-workshop.github.io/2023-papers/silm2023-acoustic.pdf
- HTML (readable): https://ar5iv.labs.arxiv.org/abs/2308.01074

---

## 1. RELEASED vs NOT — the short answer

| Item | Released? | Where |
|---|---|---|
| **Per-key audio datasets** (phone + Zoom, MacBook Pro) | **YES — public, MIT** | github.com/JBFH-Dev/Keystroke-Datasets |
| **Keystroke isolation/segmentation function** (`isolator`) | **YES** (posted in the dataset README) | same repo README |
| **Attack model + training/inference code** (CoAtNet) | **NO — not public.** Paper: *"The attack code is available upon request to researchers."* | — (email authors) |
| **Pretrained weights** | **NO** | — |

**Bottom line for the user's exact question:** The authors did **NOT** release the attack
algorithm/model code. They released **only the datasets** (and a short keystroke-isolation
snippet in the README). The CoAtNet training/inference code is "available upon request to
researchers" — it is not on GitHub. The paper describes the architecture and recipe in enough
detail to reproduce, and a **third-party reimplementation exists** (see §5).

---

## 2. The authors' repo (dataset only)

**URL:** https://github.com/JBFH-Dev/Keystroke-Datasets — **License: MIT** (~34 stars).
Owner JBFH-Dev = Josh Harrison (paper's first author).

Contents:
- `MBPWavs/` — **phone-recorded** set (this is the "MBPWavs" we already use).
- `Zoom/` — **Zoom-recorded** set. So **YES, both a phone set and a Zoom set of the same
  MacBook Pro exist.**
- `README.md`, `LICENSE`. **No model/training code.**

Data format:
- **36 keys**: `0-9` and `a-z` (README lists them as "1-0, q-m").
- **One `.wav` file per key**, each file containing **25 keystrokes** in a row at slightly
  varying intervals/strengths.
- Same person, same environment for every key.
- **Phone set:** iPhone 13 mini next to a **16-inch M1 Pro MacBook Pro (2021)**, on a
  microfibre cloth to damp vibration.
- **Zoom set:** captured via Zoom's built-in *record meeting* function.

**Download:** `git clone https://github.com/JBFH-Dev/Keystroke-Datasets.git`
(or download the ZIP from the repo's Code button). The `.wav` files are committed directly.

### Their segmentation code (verbatim from the dataset README)

```python
def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)   # summed-FFT energy envelope
    threshed = energy > threshold                        # energy threshold
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*scan) + size//2
        if timestamp > prev_end + (0.1*sample_rate):     # >100ms since last stroke
            keystroke = signal[timestamp-before:timestamp+after]
            strokes.append(torch.tensor(keystroke)[None, :])
            prev_end = timestamp+after
    return strokes
```

Zoom usage (from README): adaptive threshold search until exactly 25 strokes are found,
window `before=2400, after=12000` → **14400 samples per stroke (~0.33 s)**, `size=48, scan=24`,
starting `threshold≈0.06`, `step=0.005`.

---

## 3. Exact architecture & training recipe (from the paper, for reproduction)

- **Model:** CoAtNet — hybrid described as *"two depth-wise convolutional layers followed by
  two global relative attention layers."* Paper does **not** print exact channel/block counts;
  it is a small CoAtNet-style config. (Third-party repo and follow-up "Improved CoAtNet"
  papers treat it as CoAtNet-0-scale.)
- **Input representation:** mel-spectrogram, **64 mel bands**, STFT **window/n_fft = 1024
  samples**, **hop 500** (**255 for the MacBook/phone keystrokes**), rendered to **64×64
  images** fed to the classifier.
- **Sample rate:** **44.1 kHz** (iPhone recordings).
- **Isolated keystroke window:** **fixed length 14400 samples (≈0.33 s)** per key (phone).
  For Zoom, adaptive threshold until the correct keystroke count is recovered.
- **Segmentation:** summed-FFT **energy envelope + threshold**, with a **100 ms** refractory
  gap between accepted peaks (see `isolator` above).
- **Data augmentation:** random **time-shift up to ±40%** of the window, plus **SpecAugment**-
  style masking of a **random 10% of both the time and frequency axes**.
- **Optimizer:** **Adam**, **lr = 5e-4**, **batch size 16**, **1100 epochs**; validation
  checked every 5 epochs.
- **Split:** stratified/random train–test split (exact ratio not stated in the paper).
- **Classes:** 36-way (0-9, a-z).

**Headline results:** **95%** top-1 with the phone mic; **93%** over Zoom.

**Threat-model caveats to flag for our shield work:**
- The mic was a **phone ~17 cm** from the leftmost side of the laptop — **not the MacBook's
  built-in mic.** Distance/mic matters for our defensive scenario.
- The attack is **isolated single-keystroke 36-way classification**, one clean key at a time,
  25 identical presses per key — **not continuous free-typing transcription.**

---

## 4. What transfers to OUR (harder) continuous-typing task

- **Reusable directly:** the phone + Zoom datasets, the `isolator` segmentation idea
  (energy-envelope peak picking), the mel-spectrogram → 64×64 → CoAtNet pipeline, and the
  augmentation recipe (time-shift + SpecAugment).
- **Does NOT transfer cleanly:** their released approach assumes **cleanly separable, isolated
  keystrokes** with a known count. Real free-typing has **overlapping key sounds, digraph
  timing, variable rates, and no per-key ground-truth boundaries** — their segmentation and
  36-way classifier are not a transcription system. For continuous typing you need a
  sequence/CTC model (see §6) or a robust onset detector + language model on top.
- For a **defensive shield**, note the attack's dependence on: an **external mic close to the
  keyboard**, a **quiet single-typist environment**, and **clean segmentation** — all of which
  are levers a defense (masking noise, mic-distance, typing-style perturbation) can attack.

## 5. Third-party reimplementation (fills the missing code)

**MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet**
https://github.com/MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet

- `keyboard-side-channel.ipynb` — **CoAtNet definition + training** (adapts public CoAtNet
  implementations: chinhsuanwu, xmu-xiaoma666, anastasiialobanova).
- `isolate_key_presses.py` — CLI keystroke isolation:
  `python isolate_key_presses.py <recording> <keystroke_count> --init_treshold <t> --step <s> --prefix <outdir>`
- No pretrained weights, no license file stated, no reproduced-accuracy numbers published in
  the README. Faithfulness is student-project level — verify before trusting.

Related follow-ups (not the original authors): "Improved CoAtNet for robust ASCA"
(link.springer.com/article/10.1007/s10207-025-01194-x) and EchoCrypt / WOOT'25 LLM-typo work
(github.com/Botacin-s-Lab/EchoCrypt) both build on the JBFH-Dev dataset.

## 6. Slater / Compagno et al., ACSAC 2019 (continuous free-typing, CTC) — release status

*"Robust Keystroke Transcription from the Acoustic Side-Channel"* (Slater, Novotney, Moore,
Morgan, Tenaglia — Two Six Labs). PDF: https://www.acsac.org/2019/program/final/1/137.pdf ·
DOI: https://dl.acm.org/doi/10.1145/3359789.3359816

- Dataset: 17 users, ~86k keystrokes, real-world typing tasks.
- **Neither the dataset nor the code appears to be publicly released.** No GitHub repo found
  under the authors/Two Six Labs; confirms prior finding (data unreleased) and adds that
  **code is also unreleased.** Would require contacting the authors.
- (A *different* CWT-based group released recordings at
  github.com/KeystrokeTranscriptionWithCWT/KeystrokeRecordings — not Slater's data.)
