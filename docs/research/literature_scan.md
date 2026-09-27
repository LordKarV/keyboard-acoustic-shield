# Literature Scan — Acoustic Keystroke Side-Channel (Attack) + Adversarial Audio Shield (Defense)

Scope: authorized defensive-security project. Attacker = continuous free-typing transcription from audio on the user's own MacBook; Defender = adversarial audio "shield." Scan date 2026-09-26.

Conventions: each item gives **exact citation + URL**, a one-line *what-it-is*, and a one-line *what we can lift* (for attacker CER or defense). Code/data status is called out explicitly. arXiv IDs of the form `26xx.xxxxx` are 2026 preprints.

Already covered elsewhere in our notes (NOT re-covered here except via their citation graph): Harrison et al. 2023 (arXiv 2308.01074, CoAtNet, isolated keys, data released/code not); Slater & Compagno et al. ACSAC'19 "Robust keystroke transcription…" (continuous CTC, nothing released, https://dl.acm.org/doi/10.1145/3359789.3359816); Zhuang/Tygar; Asonov & Agrawal "Keyboard Acoustic Emanations (Revisited)"; ggerganov/kbd-audio (keytap).

---

## (A) Continuous-typing attack methods

- **DECKER: Domain-invariant Embedding for Cross-Keyboard Extraction and Recognition** — Zhang et al., AsiaCCS'26. https://arxiv.org/abs/2605.03384 (HTML: https://arxiv.org/html/2605.03384).
  - *What-it-is:* End-to-end ASCA pipeline that recognizes keystrokes across **unseen keyboards/mics** then reconstructs sentences with constrained LLM decoding; introduces the HEAR dataset (53 participants, 37 laptop keyboards, external-mic / device-mic / VoIP settings).
  - *What we can lift:* The whole cross-keyboard recipe (see §B) plus **constrained LLM sequence reconstruction** on top of noisy per-key predictions — directly applicable as our decode-time CER reducer. HEAR is the closest thing to a continuous, multi-keyboard, VoIP-realistic benchmark (access on request, academic only).

- **Transforming Keystroke Noise to Text: Self-Supervised Acoustic Eavesdropping Attacks on Keyboards** — Okada, Ito, Ueno, Hayashi, Homma, 2026. https://arxiv.org/abs/2607.22094 (HTML: https://arxiv.org/html/2607.22094v2).
  - *What-it-is:* **No-labels** attack: keystroke segmentation → 902-D acoustic features (MFCC + spectral + derivatives) → UMAP(5-D) → clustering (K=25) → character-BERT (6 layers, 256 hid) masked-LM decode with iterative feedback (50 iters) + Gemini-2.0-Flash cleanup + label spreading. Assumes non-overlapping keystrokes (~240 cpm). Isolated-ish but decodes free English text.
  - *What-it-is / lift:* Levenshtein "accuracy" 90–99% at 150–250 keystrokes across laptops, through-wall, and Zoom/Teams/Meet. **Lift:** the unsupervised cluster→char + iterative LM-feedback loop is a strong fallback when we have no labels for a target keyboard; the ambiguity-aware (probabilistic cluster→char) embedding is a clean way to feed acoustic uncertainty into an LM. **No code/data released.**

- **Hear to reveal: Stealing keystroke content from keyboard acoustic side-channel (HERO)** — Knowledge-Based Systems, 2026. https://www.sciencedirect.com/science/article/abs/pii/S0950705126014152 (paywalled; abstract only, WebFetch 403).
  - *What-it-is:* Self-supervised pretraining on unlabeled keystroke audio to learn reusable acoustic representations that improve generalization to new users/keyboards.
  - *What we can lift:* Confirms **SSL pretraining on unlabeled keystroke audio** is the current generalization lever; motivates our own contrastive/masked pretrain before few-shot calibration. No code seen; paywalled.

- **Practical Acoustic Eavesdropping On Typed Passphrases** — 2025. https://arxiv.org/abs/2503.16719 (HTML: https://arxiv.org/html/2503.16719v1).
  - *What-it-is:* Unsupervised clustering + dictionary decoding of space-delimited passphrases (EFF wordlist); cross-correlation on raw audio best (median 93%). Word-level, not continuous free text.
  - *What we can lift:* Limited for CTC (assumes word segmentation). Useful bits: **raw-audio cross-correlation** as a cheap per-key similarity feature and their MFCC hyperparameter sweep. **No code/data.**

- **A Practical Deep Learning-Based ASCA on Keyboards (extended)** and related isolated-key SOTA:
  - **Improved CoAtNet for robust acoustic side-channel attack classification** — Int. J. Information Security, 2025. https://link.springer.com/article/10.1007/s10207-025-01194-x — *lift:* architecture tweaks + noise-augmentation regime that harden CoAtNet (paywalled).
  - **ASCA-Net: DL-based ASCA for keystroke recognition across multiple platforms** — J. Computer Virology & Hacking Techniques, 2026. https://link.springer.com/article/10.1007/s11416-026-00648-7 — *lift:* multi-platform (phone/Zoom) classifier design (paywalled).
  - **Acoustic Side Channel Attack on Keyboards Based on Typing Patterns** — 2024. https://arxiv.org/abs/2403.08740 — *lift:* uses inter-key **timing/typing-pattern** priors to disambiguate keys; a timing prior we can fuse into the LM/decoder.

---

## (B) Domain adaptation / few-shot / sim2real

- **DECKER (as above, arXiv 2605.03384)** — the primary reference for this section. Four concrete stages we can copy:
  1. **Keyboard Signature Normalization** — remove device "coloration" (per-keyboard spectral response) so features are device-neutral.
  2. **Domain-Adversarial Disentanglement** — gradient-reversal head that suppresses keyboard-identity info in the embedding.
  3. **Supervised Cross-Keyboard Contrastive Alignment** — pull same-key/different-keyboard samples together (this is the contrastive lever we want).
  4. **Acoustic Style Randomization** — synthesize responses of *unseen* keyboards from seen ones = **sim-to-real / augmentation** for keyboard timbre.
  - *What we can lift:* All four, directly. #4 is the cheapest CER win for us: augment our MacBook recordings with randomized keyboard "styles" to pre-adapt before few-shot calibration on the true target. No public code found yet (recent).

- **Impact Analysis of Speech Representation Learning Models for ASCA** — 2026. https://arxiv.org/abs/2606.21210 (HTML: https://arxiv.org/html/2606.21210v2).
  - *What-it-is:* Benchmarks **wav2vec 2.0, HuBERT, WavLM** as frozen/fine-tuned front-ends for keystroke inference on HEAR, across keyboards and VoIP; WavLM most robust; continuous >> harder than isolated; LM fusion helps but can't fully offset VoIP compression.
  - *What we can lift:* Use a **pretrained speech SSL encoder (WavLM/HuBERT) as our front-end** instead of raw log-mel → much better cross-keyboard/cross-mic generalization and few-shot data efficiency. Reports CER from single digits (clean) to 40%+ (VoIP). No code URL confirmed (states intent to release).

- **Few-shot calibration observation (multiple papers, incl. 2607.22094 / press coverage):** accuracy climbs from ~80% at ~150 keystrokes to ~100% at ~200–250 on a given device.
  - *What we can lift:* Budget a **~200-keystroke in-session calibration set** per new keyboard; that is our few-shot target and a realistic demo constraint.

---

## (C) Decoders & LM fusion

- **Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction (EchoCrypt)** — Ayati et al., 2025. https://arxiv.org/abs/2504.11622 (HTML: https://arxiv.org/html/2504.11622v1). **Builds directly on Harrison 2023.**
  - *What-it-is:* Vision-Transformer (Swin/ViT) keystroke classifier from mel-spectrograms + **LLM post-hoc "typo" correction**. LLMs: GPT-4o, Llama-3.2-1B/3B/8B, and **LoRA-fine-tuned Llama-3.2-3B**. On noisy Phone data, CoAtNet baseline BLEU 0.078 → GPT-4o 0.916; LoRA-Llama-3B 0.908 ("67× smaller than GPT-4o"). Clean: Swin 100% on Phone. Isolated keys (a-z,0-9; excludes space/backspace/enter); MacBook Pro; synthetic Gaussian noise.
  - *What we can lift:* **This is our LM-fusion blueprint.** A small **LoRA-fine-tuned local LLM** as the sentence-level corrector on top of noisy per-key logits gives the biggest practical CER/BLEU jump under noise, cheaply and offline. **CODE RELEASED (see §D).**

- **DECKER's constrained LLM decoding** and **2607.22094's iterative character-BERT + label-spreading feedback** (both above): two alternative decode strategies —
  - *lift:* constrained/beam LLM decode (DECKER) for coherence; iterative feedback (Okada) when confidence is low. Combine with a CTC front-end: CTC posteriors → LM rescoring/correction.

- **General CTC/seq2seq decode references** (not keystroke-specific, but transferable): CTC decoding-strategy comparisons and Bayes-risk CTC (controllable alignment, https://arxiv.org/abs/2210.07499) — *lift:* beam-search + shallow-fusion LM over our 37-key CTC, and alignment control to handle key-onset jitter.

---

## (D) Released code / data we could pull

- **EchoCrypt (code + weights)** — https://github.com/seyyedaliayati/EchoCrypt — License **CC0-1.0**. Training code for CoAtNet/ViT/Swin classifiers, LLM-correction scripts (GPT-4o + Llama), **LoRA fine-tuning**, preprocessing for Phone/Zoom audio. Pretrained checkpoints on HuggingFace: `huggingface.co/seyyedaliayati/` (Phone + Zoom models). 8GB VRAM for inference. *Isolated keys, not continuous.*
  - *Lift:* Grab the classifier + LoRA-correction code and adapt to our 37-key CTC; the correction stage transfers as-is.
- **HEAR dataset** (from DECKER) — the only realistic **continuous, 37-keyboard, VoIP** set. Access **on request, academic only**; sample data via awesomepapers.io/cybersecurity/datasets/hear. *Lift:* our best evaluation target for cross-keyboard/VoIP CER; request early.
- **Isolated-key audio datasets on Zenodo/Mendeley (public, ready to pull for pretraining/augmentation):**
  - Multi-Pressure Keyboard Key Sounds — https://zenodo.org/records/16564409 (also MDPI Data 2026, https://www.mdpi.com/2306-5729/11/7/168): alphanumerics + space, 3 pressure levels, spectrograms + metadata.
  - Multi-Keyboard Acoustic (MKA) — https://data.mendeley.com/datasets/bpt2hvf8n3 — 6 sources (HP, Lenovo, MSI, Mac, Messenger, Zoom); good for **cross-keyboard augmentation**.
  - Audio dataset of keyboard keystrokes — https://zenodo.org/records/11486430 and https://zenodo.org/records/15148876 (membrane/pantograph/mechanical switch variety).
  - *Lift:* All are isolated keypresses — use for **SSL/contrastive pretraining and Acoustic-Style-Randomization augmentation**, not directly for continuous CTC labels (they lack SKAID-style continuous labels.jsonl; we'd synthesize sequences by concatenation).
- **SSL speech encoders (off-the-shelf weights)** — WavLM/HuBERT/wav2vec2 on HuggingFace — *lift:* drop-in front-end per arXiv 2606.21210.

Honest gaps: DECKER, HERO, the SSL-impact paper, and all the continuous/free-typing attacks have **no released code**; HEAR is gated. The only pullable attack code is EchoCrypt (isolated).

---

## (E) Defense / countermeasure techniques for the shield

Two families: (1) classic acoustic masking/obfuscation; (2) **adversarial-perturbation shields** (our target — a learned, near-imperceptible additive signal that breaks the attacker's classifier while preserving call intelligibility).

**Classic masking / obfuscation**
- **Anand & Saxena, "Mitigating Acoustic Side-Channel Attacks on Password Keystrokes"** — FC'16. https://fc16.ifca.ai/preproceedings/21_Anand.pdf — *techniques:* white-noise masking, **fake-keystroke injection** (decoy sounds), keystroke-timing obfuscation. Trade-off: constant noise fatigues users; fake keystrokes need calibration. *Lift:* fake-keystroke injection is a strong, simple baseline to beat/compare against; also a data-augmentation for a robust attacker.
- **Anand & Saxena, "Keyboard Emanations in Remote Voice Calls: Password Leakage and Noise(less) Masking Defenses"** — 2018. https://www.researchgate.net/publication/323786190 — *technique:* "noiseless" masking (inject sounds shaped to be unobtrusive in a call). *Lift:* directly on-topic for a VoIP-call shield; motivates perceptually-shaped masking over white noise.
- **SoK: Acoustic Side Channels** — 2023. https://arxiv.org/abs/2308.03806 — *what-it-is:* survey cataloguing masking, obfuscation, sound-dampening, key-layout randomization. *Lift:* taxonomy + related-work coverage for our writeup (full-text needs the PDF; abstract confirms "countermeasures" section).
- Key-layout randomization / touchscreen remapping — recurring cheap defense for virtual keyboards (from survey arXiv 2309.11012) — not applicable to a physical MacBook but worth a sentence.

**Adversarial-perturbation shields (the core of what we build)**
- **Qin, Carlini, Cottrell, Goodfellow, Raffel — "Imperceptible, Robust, and Targeted Adversarial Examples for ASR"** — ICML 2019. https://proceedings.mlr.press/v97/qin19a/qin19a.pdf — *what-it-is:* uses the **psychoacoustic masking threshold** (not L_p norm) to bound perturbations below human hearing, plus robustness over acoustic transforms. *Lift:* **the exact objective for our shield** — optimize a perturbation that maximizes attacker CER subject to staying under the masking threshold; gives robustness-over-room-response too.
- **Szurley & Kolter — "Perceptual Based Adversarial Audio Attacks"** — 2019. https://arxiv.org/abs/1906.06355 — *what-it-is:* psychoacoustic attack done purely in the time domain via DFT symmetry. *Lift:* time-domain formulation is efficient for a **real-time streaming shield**.
- **"Are Deep Speech Denoising Models Robust to Adversarial Noise?"** — 2025. https://arxiv.org/abs/2503.11627 — *what-it-is:* psychoacoustic-constrained PGD forces speech models to unintelligible output while staying imperceptible. *Lift:* recipe/hyperparameters for **psychoacoustic-PGD**; also a warning that our shield must survive the attacker's denoiser.
- **Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Examples** — IJCAI 2023. https://www.ijcai.org/proceedings/2023/0535.pdf — *what-it-is:* imperceptible adversarial protection deployed as a *privacy shield* (same posture as ours). *Lift:* deployment/threat framing and perceptual-loss design for a defender-side shield.
- **WaveGuard: Understanding and Mitigating Audio Adversarial Examples** — USENIX Sec'21. https://www.usenix.org/system/files/sec21fall-hussain.pdf — *what-it-is:* input-transformation *defenses that detect/remove* audio adversarial examples. *Lift:* this is the **adversary's counter-move** — our shield must be robust to these transforms (EOT over WaveGuard-style filters) to survive the "arms race" demo.
- Perceptual metrics to gate the shield: **STOI** (intelligibility, preferred) and **PESQ** (quality, 0.5–4.5) — report both; STOI correlates better with human understandability.

---

## TOP 5 actionable ideas for us (ranked)

1. **LoRA-fine-tuned local LLM as the sentence-level corrector on top of CTC/per-key logits.** Biggest, cheapest CER/BLEU win under noise, offline. Pull EchoCrypt code to bootstrap. — https://github.com/seyyedaliayati/EchoCrypt , https://arxiv.org/abs/2504.11622
2. **Adopt DECKER's cross-keyboard recipe for our attacker's robustness**, especially Acoustic-Style-Randomization augmentation + supervised contrastive cross-keyboard alignment + domain-adversarial head; then ~200-keystroke few-shot calibration on the target MacBook. — https://arxiv.org/abs/2605.03384
3. **Swap raw log-mel for a pretrained SSL speech front-end (WavLM/HuBERT).** Best cross-mic/cross-keyboard generalization and few-shot data efficiency; keep our onset→0.3s→CTC head on top. — https://arxiv.org/abs/2606.21210
4. **Build the shield as a psychoacoustic-masking-bounded adversarial perturbation (Qin-2019 objective), optimized with EOT over room/mic/denoiser transforms**, gated by STOI/PESQ so the call stays intelligible. — https://proceedings.mlr.press/v97/qin19a/qin19a.pdf , https://arxiv.org/abs/2503.11627
5. **Harden the shield against the attacker's counter-defenses (WaveGuard-style input transforms) and benchmark against the fake-keystroke-injection baseline** — the honest "arms race" evaluation. — https://www.usenix.org/system/files/sec21fall-hussain.pdf , https://fc16.ifca.ai/preproceedings/21_Anand.pdf
