# Maximizing Accuracy of an Acoustic Keystroke Attacker on Isolated Keys

**Scope:** Isolated / well-separated single keystrokes, ~36 classes (A–Z, 0–9). Baseline: a CNN at ~87% top-1 on the Harrison dataset. Goal: (a) push per-key accuracy higher, (b) generalize across keyboards, (c) adapt to a new keyboard with few labeled presses.

**Context:** Authorized defensive research (HackGT "KeyGuard"). All techniques below come from published academic work; we build the attacker only to build the shield. No third-party targeting.

> Note on citations: this memo synthesizes the classic literature (Asonov & Agrawal 2004; Zhuang/Zhou/Tygar 2005/2009; Berger/Wool/Yeredor 2006; Cecconello "Skype & Type" 2019; Harrison/Toreini/Mehrnezhad 2023) and recent 2024–2026 deep-learning work (DECKER cross-keyboard; LLM "typo" correction; SSL speech-representation analysis; improved-CoAtNet variants). Some very recent arXiv IDs (e.g., DECKER 2605.03384, SSL analysis 2606.21210, LLM-typo 2504.11622) are cutting-edge and should be re-verified before publication, but the *techniques* they describe are well-grounded and standard.

---

## TOP ACTIONABLE LEVERS (ranked)

Ranked by expected accuracy gain per unit of engineering effort for *our* setting (isolated keys, already at 87% on one keyboard).

1. **Fix the segmentation window and tighten it around the push transient.** Harrison's exact recipe: fixed 0.33 s window (14,400 samples @ 44.1 kHz) around the energy onset, then a 64-mel × 64-frame spectrogram (n_fft=1024, hop=500). The single biggest source of per-key error in isolated-key setups is misaligned/over-long windows that dilute the "push peak" cue. Align windows to onset, keep them short, and be consistent between train and test. **(Feature representation — biggest low-effort win.)**

2. **Add SpecAugment + time-shift + light noise augmentation.** Harrison used time-shift up to 40% + SpecAugment (10% time/freq masking). This is exactly what took a strong CNN from good to 95%. It regularizes without destroying per-key spectral cues (masking is sparse). Add mild Gaussian noise (calibrated so clean-set accuracy barely drops) for robustness. **(Augmentation — cheap, large gain, already partially in Harrison's pipeline.)**

3. **Use a conv+attention hybrid (CoAtNet) or ViT head instead of a plain CNN.** Harrison's 95%/93% (phone/Zoom) came from CoAtNet on 64×64 mel images vs. Asonov's 79% top-1 with an FFT+NN. The convolution stages capture local spectral texture (the transient), the relative-attention stages capture global time–frequency structure. This is the most likely single-model upgrade from an 87% CNN. **(Architecture.)**

4. **Language-model / dictionary fusion on top of per-key posteriors — when the target is text.** Even a perfect isolated-key model benefits enormously when the victim types words. Options in increasing power: HMM + English bigram/trigram re-scoring (Zhuang: recovers up to 96% of chars with *no* labels), dictionary correlation (Berger: 90%+ in top-50 for long words), and modern LLM "typo correction" (GPT-4o/fine-tuned Llama lifts BLEU from ~0.08 to 0.92 at medium noise). For *random passwords* the LM helps less (Zhuang still cracked 90% of 5-char alpha in <20 guesses via candidate ranking). **(Post-processing — huge when text is linguistic; keep raw top-k for password/random targets.)**

5. **For cross-keyboard generalization: acoustic style randomization in training + a domain-invariant embedding.** DECKER's recipe closes most of the cross-device gap: random IIR filters + spectral-envelope (mel) warp ±12% + exponential-decay perturbation during training, plus a domain-adversarial gradient-reversal layer and supervised cross-keyboard contrastive alignment. Took unseen-keyboard key accuracy from ~58% (plain encoder) to ~81%. The augmentation piece alone (ASR) is the highest-leverage part and is drop-in. **(Cross-device generalization + augmentation.)**

Secondary levers (high value, more effort): self-supervised speech embeddings (WavLM) as features for few-shot/cross-device (#11); few-shot fine-tuning recipes (#12); test-time / per-key calibration and ensembling (#13).

---

## 1. INPUT FEATURE REPRESENTATION

The single most consequential set of choices for isolated keys. A keystroke has two acoustic events — the **push (touch) peak** and the **release peak** ~100 ms later. The push transient carries most of the discriminative energy; the release is noisier. Isolate around the push.

### 1.1 Segmentation / onset detection
- **Energy-threshold onset detection (FFT energy / signal power).** Asonov & Agrawal and most successors detect the press by a spike in windowed energy. Harrison used FFT-energy thresholding with a *fixed* window length after onset.
- **Harrison's exact window:** fixed length **14,400 samples = 0.33 s @ 44.1 kHz**, starting at detected onset. For Zoom recordings they used an **adaptive prominence threshold** that iteratively lowered the peak-prominence until exactly the known number of keystrokes (25/recording) was detected — a practical trick for noisy/compressed channels.
- **Practical guidance for us:** onset-align every clip. Prefer a *short* window that captures the push transient (~40–100 ms) plus enough tail for the release harmonics. Over-long windows (or center-of-clip cropping) are a common cause of accuracy loss. Keep the exact same segmentation at train and test time.
- **Fine-grained alternative (Cecconello "Skype & Type"):** take an **8–10 ms window around the press peak** and FFT that. Very tight windows work with classical features; deep-learning mel pipelines prefer the longer 0.2–0.33 s window so the 2-D model can see time structure.

### 1.2 Feature families (evolution + when to use)
| Representation | Papers | Params | Notes for us |
|---|---|---|---|
| **Raw FFT of press window** | Asonov 2004; Cecconello 2019 | 8–10 ms window around peak; full FFT | Simple, works with LR/NN; 79% top-1 (Asonov). Baseline only. |
| **Cepstrum / MFCC** | Zhuang 2005/09; Berger 2006 | MFCC on press segment | Better than raw FFT for classical classifiers (Zhuang found cepstrum > FFT). Compact. |
| **Mel-spectrogram (2-D image)** | Harrison 2023 | **64 mel bands, n_fft=1024, hop=500 (→255 for Zoom to keep frame count), window→64×64 px** | **Current SOTA input for CNN/CoAtNet. Recommend as default.** |
| **Log-mel spectrogram** | DECKER 2024–26 | log-mel → ECAPA-TDNN encoder | Log compression stabilizes dynamic range; better for cross-device. Prefer log-mel over linear-mel. |
| **Self-supervised speech embeddings** | SSL-ASCA analysis 2026 | WavLM/wav2vec2/HuBERT (768-d), XLS-R (1024-d); **resample audio to 16 kHz** | Frozen backbone + small head. Best cross-keyboard/codec robustness (see §5.3). |

### 1.3 Concrete recommended feature config (starting point)
- Sample rate 44.1 kHz for a near-field phone; resample to 16 kHz if feeding an SSL model.
- Window: onset-aligned, 0.25–0.33 s (match Harrison at 14,400 samples to reproduce their result exactly).
- Mel-spectrogram: **64 mel bins, n_fft=1024, hop=500, Hann window, log-magnitude (dB)**, resized/cropped to 64×64. Tune hop so frame count ≈ 64 for whatever sample rate/window you pick (Harrison dropped hop to 255 for MacBook-Zoom audio precisely to preserve frame count).
- Per-image normalization (mean/std or min-max to [0,1]).
- **Ablations worth running:** mel bins {40, 64, 128}; include vs. exclude release peak in window; log-mel vs. PCEN (per-channel energy normalization is known to help transient/impulsive sounds and cross-device level differences — worth a try even though the keystroke papers haven't reported it).

---

## 2. MODEL ARCHITECTURE

### 2.1 Progression (what beat what)
- **FFT + shallow neural net** (Asonov 2004): 79% top-1 / 88% top-3.
- **Logistic regression on FFT** (Cecconello): top-5 ~91.7% with known typing style.
- **CNN on mel-spectrogram**: strong; our current 87%.
- **CoAtNet (conv + relative self-attention hybrid)** (Harrison 2023): **95% (phone) / 93% (Zoom)** on 36 classes. Depthwise-conv stages then global relative-attention stages. This is the reference architecture to beat.
- **Vision Transformer (ViT)**: used alongside CoAtNet in the LLM-typo-correction work as the spectrogram classifier; competitive but data-hungry — use pretrained ImageNet weights.
- **ECAPA-TDNN** (DECKER): 1-D time-conv encoder with channel attention + attentive statistics pooling on log-mel frames; strong embedding backbone for cross-keyboard.
- **SSL backbone + KAN head** (SSL-ASCA 2026): Kolmogorov-Arnold Network head (learnable spline activations) beat FCN/CNN heads on out-of-domain and VoIP-codec conditions (e.g., WavLM+KAN 61.7% vs WavLM+CNN 49.5% on unseen keyboards).

### 2.2 Harrison CoAtNet training recipe (reproduce this)
- Optimizer Adam, initial max LR **5e-4**, batch size **16**, **cross-entropy** loss, **~1,100 epochs**, random stratified split.
- Only **25 presses per key** per keyboard (36 keys) — notably small; augmentation (§3) is doing heavy lifting. This is important for our few-shot goals: high accuracy is achievable with tens of presses per key *if* augmentation and a good backbone are used.

### 2.3 Recommendations for us
1. Swap plain CNN → **CoAtNet** (or an ImageNet-pretrained EfficientNet/ConvNeXt-tiny as a cheaper stand-in) on 64×64 log-mel. Expect the biggest single-model jump.
2. Keep an **embedding model** (ECAPA-TDNN or SSL backbone) in parallel for the cross-keyboard track — image CNNs overfit device coloration; embedding + metric learning generalizes better.
3. **Ensemble**: average posteriors of (CoAtNet on mel) + (SSL+head on raw audio). Cheap, reliably adds 1–3 points, and the two make uncorrelated errors (image texture vs. learned speech features).

---

## 3. DATA AUGMENTATION (the accuracy multiplier)

Two objectives, sometimes in tension: (a) regularize/expand a small per-key set, (b) simulate other keyboards/channels for cross-device generalization — **without smearing the per-key spectral fingerprint** that distinguishes keys.

### 3.1 Augmentations that preserve per-key cues (safe, use liberally)
- **SpecAugment** (time + frequency masking): Harrison masked ~10% on each axis. Masks are *sparse*, so they regularize without erasing the transient. **Use this — it is part of the 95% recipe.** Time-warping (the third SpecAugment op) is optional and lower-value for short clips.
- **Time-shift / random crop within the window** (up to 40% in Harrison): makes the model robust to onset-detection jitter. Directly relevant since our segmentation won't be pixel-perfect.
- **Additive Gaussian noise on the waveform** (LLM-typo paper): `S_noisy = S + η·N(0,1)`, with η calibrated to target ~95/85/70% clean accuracy tiers. Train across tiers for graceful degradation. Also add **real ambient noise** (office/room tone) at varying SNR.
- **Mixup / SpecMix** (audio literature, cross-domain): interpolate two clips + labels. Helps calibration and margins. Use *low* mixing (α small) so per-key identity isn't destroyed; treat as a mild regularizer, not a core cue-preserving augment.

### 3.2 Augmentations for CROSS-DEVICE generalization (DECKER "Acoustic Style Randomization")
This is the key set for keyboard transfer. It perturbs the *channel/keyboard transfer function* while leaving the *relative* per-key structure intact:
- **Random IIR filters** (order 6–12): simulate different keyboard/mic frequency responses (spectral coloration).
- **Spectral-envelope / mel-frequency warp ±12%**: simulates a different keyboard's resonance without shifting the fine per-key contrasts uniformly.
- **Exponential-decay perturbation** (γ ∈ [0.85, 1.15]): varies the transient decay envelope (different key travel / dampening).
- **Codec / VoIP simulation**: encode-decode through Opus/AAC/Speex at Zoom/Teams-like bitrates to build robustness to compression (the SSL-ASCA "codec generalization" protocol shows this is where naive models collapse).
- **Reverberation / room impulse responses (RIR)**: convolve with varied RIRs to simulate rooms/distances.

**Why it doesn't destroy per-key cues:** these transforms model *device/channel* variation (a smooth transfer function applied to *all* keys the same way within a sample), so the *between-key* discriminative differences survive; the model learns to be invariant to the channel, not to the key. Contrast with heavy pitch-shift or aggressive spectral masking, which *can* wipe key-specific resonances — use those sparingly.

### 3.3 Recommended augmentation stack
Training: onset time-shift (±40%) → ASR (random IIR + mel-warp ±12% + decay perturb) with prob ~0.5 → SpecAugment (2 freq + 2 time masks, ≤10%) → additive noise tier (rotate clean/med/high) → optional codec pass for VoIP robustness. Keep a *clean* validation set to detect cue destruction (if clean-val accuracy drops materially, an augment is too aggressive).

---

## 4. LANGUAGE-MODEL / DICTIONARY FUSION (when target is text, not random keys)

Even a perfect per-key classifier gains from linguistic priors; a mediocre one gains enormously. Keep this modular so it can be *disabled* for random-password targets (where it hurts) and *enabled* for prose.

- **Unsupervised HMM + n-gram language model** (Zhuang/Zhou/Tygar 2005/2009): bootstrap labels from unlabeled typing using English letter statistics + spelling/grammar, then feed back to retrain the classifier. Recovers **up to 96% of characters with no labeled training data**. Even in our supervised setting, an HMM/n-gram rescoring layer over per-key posteriors corrects many single-key errors.
- **Dictionary attack via cross-correlation** (Berger/Wool/Yeredor 2006): no training; ranks dictionary words by acoustic correlation of the key sequence. **90%+ correct in top-50** for 7–13-char words. Good for word-level targets.
- **LLM "typo correction"** (2025): treat the classifier's raw string as noisy text and have an LLM fix it. Few-shot prompt ("You are an expert at correcting typos", 2–3 example pairs) with GPT-4o, or **QLoRA-fine-tuned Llama-3.2-3B** (AdamW, LR 2e-4, 3 epochs over increasing noise) that reaches 90–99% of GPT-4o quality at 67× smaller. Lifts BLEU from ~0.08→0.92 (medium noise), ~0.02→0.64 (high noise). Notably they found **no beam search needed** — a single deterministic generation sufficed.
- **Beam search with LM priors** (DECKER's text track): constrained beam search over per-key posteriors + LLM prior. Character accuracy 81.3% (raw) → 86.9% (GPT-2) → 91.6% (LLaMA-2 7B); sentence exact-match 42.6% → 72.8% (GPT-4).
- **For passwords / random strings:** don't apply a natural-language LM. Instead rank candidate strings by combined per-key posterior probability and let the attacker enumerate top-k (Zhuang: 90% of 5-char alpha passwords in <20 guesses; 80% of 10-char in <75). Report top-1/top-3/top-5/top-k for our defensive evaluation.

**Recommendation:** implement per-key softmax → optional n-gram/LLM rescoring as a *separable* stage. Always log raw top-k so the defensive analysis can measure the attacker with and without linguistic help.

---

## 5. CROSS-KEYBOARD GENERALIZATION & FEW-SHOT ADAPTATION

This is where naive spectrogram CNNs fail hardest — they memorize one keyboard's/mic's coloration. Three complementary tools.

### 5.1 Domain-invariant embedding (DECKER, the current cross-keyboard reference)
Four-stage recipe; unseen-keyboard key accuracy **~58% (plain ECAPA) → 81.3%**:
1. **Keyboard Signature Normalization (KSN):** a learnable inverse-filter front-end (four parallel 1-D CNNs, 64 channels each, varying kernel sizes + dilations) in the *waveform* domain that suppresses device-specific spectral coloration before feature extraction.
2. **Domain-adversarial disentanglement (gradient-reversal layer):** a keyboard-ID classifier trained through a GRL so the embedding becomes *un*predictable of keyboard identity: minimize `L_key − λ·L_domain`. Removes device bias. (DANN/GRL alone got unseen accuracy to 71.6%.)
3. **Supervised cross-keyboard contrastive alignment:** supervised contrastive loss pulls same-key embeddings *across devices* together, pushes different keys apart. This explicitly aligns key clusters across keyboards.
4. **Acoustic Style Randomization** (the augmentation from §3.2) to simulate unseen keyboards during training.

### 5.2 Few-shot adaptation to a NEW target keyboard (limited labeled presses)
Goal: given only a handful of labeled presses on the target keyboard, adapt. Recipes, cheapest first:
1. **Frozen-backbone + retrain small head.** Train a strong device-invariant encoder (§5.1 or an SSL backbone §5.3) on many keyboards; on the target, freeze it and fit only a lightweight classifier (FCN/KAN/logistic) on the few labeled presses. This is exactly the SSL-ASCA setup (frozen WavLM + small head) and works with tens of presses.
2. **Prototypical / metric-based few-shot.** With the contrastive-aligned embedding (§5.1-3), compute a class prototype = mean embedding of the few labeled presses per key on the target keyboard; classify new presses by nearest prototype. No gradient steps needed; scales down to a few shots per key.
3. **Fine-tune with heavy augmentation.** Harrison showed 25 presses/key + augmentation → 95% *on that keyboard*. So a viable recipe: collect ~10–25 presses/key on the target, apply the full augmentation stack (§3.3) to inflate, fine-tune (low LR, few epochs, maybe LoRA/adapter layers to avoid catastrophic forgetting). Expect to recover most of the in-domain accuracy.
4. **Self-training / feedback loop** (Zhuang-style): use the LM to auto-label the target-keyboard typing, then fine-tune the acoustic model on those pseudo-labels — closes the gap with *zero* manual labels if the victim types prose.
5. **Test-time adaptation:** update BatchNorm statistics (or use domain-specific BN) on unlabeled target audio; cheap, label-free, corrects for channel shift.

### 5.3 Self-supervised speech representations (best cross-device features)
- Backbones: **WavLM (best), wav2vec2, HuBERT, XLS-R**; resample audio to **16 kHz**; keep **frozen**, train only a downstream head.
- Heads tested: FCN (90 units), small CNN (2 conv blocks, k=3, maxpool 2), and **KAN (1 hidden layer, 30 units, grid=5, spline order=3)** — KAN was best OOD.
- Training: AdamW, 20 epochs, batch 16, LR 2.2e-4, dropout + early stopping.
- Results (KEYAC, 37 keyboards, unseen-keyboard OOD): WavLM+KAN **61.7%** vs WavLM+CNN 49.5%; under VoIP codec, KAN held **50.9%** vs CNN 35.9%. Absolute numbers are lower than DECKER because this is a harder 37-keyboard benchmark with frozen features — the *lesson* is that SSL features + a nonlinear head generalize far better than mel-CNN across devices/codecs.
- **Takeaway:** for the cross-keyboard/few-shot track, a frozen WavLM + KAN/prototype head is the most label-efficient starting point; for single-keyboard peak accuracy, CoAtNet-on-mel still wins.

---

## 6. ROBUSTNESS / NOISE / CODEC HANDLING (from improved-CoAtNet + LLM-typo work)
- **Denoising / spectral subtraction / bandpass** before mel extraction to attenuate steady room noise (keystrokes are broadband transients; a high-pass around a few hundred Hz removes rumble without losing the click).
- **Train across noise tiers** (§3.1) rather than clean-only; naive CoAtNet loses >50% accuracy under high noise, but LLM correction restores it — combine both.
- **Adaptive peak-prominence onset detection** for compressed/VoIP audio (Harrison's Zoom trick).
- **Codec-aware training** (encode through Opus/Speex) for Zoom/Teams targets; the SSL analysis shows this is the failure mode that separates robust from brittle models.

---

## 7. CONCRETE EXPERIMENT PLAN FOR "KEYGUARD" (attacker side, to then defend)
1. **Reproduce Harrison** exactly (64-mel, n_fft=1024, hop=500, 0.33 s window, CoAtNet, Adam 5e-4, bs16) → confirm ~95% single-keyboard; this validates our pipeline and beats our 87% CNN.
2. **Augmentation ablation:** add SpecAugment + time-shift + noise tiers; measure clean vs. augmented; add DECKER ASR (IIR + mel-warp ±12% + decay perturb) and codec pass.
3. **Cross-keyboard test:** train on N−1 keyboards, test on held-out keyboard; compare (a) mel-CoAtNet, (b) ECAPA + GRL + contrastive (DECKER-lite), (c) frozen WavLM + KAN. Report the seen→unseen gap (DECKER's was −17.6 points; that gap is what the *defense* must widen).
4. **Few-shot curve:** on the held-out keyboard, sweep {1, 5, 10, 25} labeled presses/key with prototype and fine-tune recipes.
5. **LM fusion:** measure raw top-1/3/5 and post-LM char/word accuracy on prose; keep raw top-k for random-string targets.
6. **Then flip to defense** (out of scope for this memo, but the levers tell us what to disrupt): what breaks the attacker is (i) destroying onset detectability, (ii) homogenizing per-key acoustic signatures, (iii) injecting adversarial/masking sounds that survive the attacker's augmentation, (iv) channel/codec perturbations the attacker didn't train on. The most robust attackers (augmented + LM-fused + device-invariant) are hardest to defeat, so the shield must attack the *feature* (onset + per-key spectral distinctiveness), not just add broadband noise the attacker already trains against.

---

## KEY SOURCES
- Asonov & Agrawal, "Keyboard Acoustic Emanations," IEEE S&P 2004 — FFT + NN, 79% top-1 / 88% top-3.
- Zhuang, Zhou, Tygar, "Keyboard Acoustic Emanations Revisited," CCS 2005 / TISSEC 2009 — cepstrum + HMM + n-gram LM + feedback; 96% chars, no labels. https://people.eecs.berkeley.edu/~tygar/papers/Keyboard_Acoustic_Emanations_Revisited/TISSEC.pdf
- Berger, Wool, Yeredor, "Dictionary Attacks Using Keyboard Acoustic Emanations," CCS 2006 — no training, correlation + dictionary, 90%+ top-50. https://dl.acm.org/doi/10.1145/1180405.1180436
- Cecconello et al., "Skype & Type," ACM TOPS 2019 — 8–10 ms FFT window, logistic regression, top-5 91.7%. https://spritz.math.unipd.it/datasets/locard_pubs/Skype_and_TypeKeyboardEavesdroppingInVoice-over-IP.pdf
- Harrison, Toreini, Mehrnezhad, "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards," EuroS&PW 2023, arXiv:2308.01074 — 64-mel/n_fft=1024/hop=500, CoAtNet, SpecAugment+time-shift, 95%/93%. https://arxiv.org/abs/2308.01074
- "Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' Typo Correction," arXiv:2504.11622 — CoAtNet/ViT + LLM correction, QLoRA Llama-3.2-3B, noise-tier training. https://arxiv.org/html/2504.11622
- DECKER, "Domain-invariant Embedding for Cross-Keyboard Extraction and Recognition," arXiv:2605.03384 — KSN + GRL + contrastive + Acoustic Style Randomization; 58%→81% unseen keyboard. https://arxiv.org/abs/2605.03384 *(very recent; re-verify)*
- "Impact Analysis of Speech Representation Learning Models for Acoustic Side-Channel Attack," arXiv:2606.21210 — frozen WavLM/wav2vec2/HuBERT + FCN/CNN/KAN heads, KEYAC 37 keyboards, codec generalization. https://arxiv.org/html/2606.21210v2 *(very recent; re-verify)*
- "A Survey on Acoustic Side Channel Attacks on Keyboards," arXiv:2309.11012 — overview of segmentation/features/models. https://arxiv.org/pdf/2309.11012
- "Improved CoAtNet for robust acoustic side-channel attack classification on keyboards," Int. J. Information Security 2025 — robustness/denoising improvements over baseline CoAtNet.
