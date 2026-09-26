# Transcribing Fast / Overlapping / Continuous Typing from Audio — Techniques Memo

**Project:** KeyGuard (HackGT) — authorized defensive acoustic keystroke research on own devices with fake passwords.
**Scope:** How to transcribe *continuous, overlapping, un-segmented* free typing from a mic, and how to fix our CTC ConvNet transcriber that collapses to a near-constant output at ~84% CER.
**Status of our system:** CTC-based ConvNet (CTC blank = keystroke detector), trained on synthetic overlap-added audio. Symptom: **class-prior collapse** (predicts one dominant symbol / all-blank everywhere), CER ~84%.

---

## 0. TL;DR — Ranked Actionable Fixes for Collapse + Accuracy

Ordered by expected impact-per-effort for our specific failure (collapse on synthetic overlap-added data).

1. **Kill the all-blank attractor with an entropy / label-prior term on the CTC posterior.** Add either (a) EnCTC maximum-entropy regularization `L = L_CTC − β·H(feasible paths)` (β≈0.1–0.2, anneal down), or (b) label-prior CTC: divide each path's per-frame prob by `P(k)^α` with **α≈0.3** and `P(k)` = running average of posteriors (recomputed each epoch). This is the single highest-leverage fix: it directly attacks the "blank dominates every feasible path" dynamic that produces our constant output. Label-prior CTC drops blank posterior mass from ~0.80 → ~0.32 in the reference work.
2. **Add a Sound-Event-Detection (onset) multitask head with real frame-level targets.** Because synthetic overlap-added audio *has exact ground-truth key press times*, you can supervise a per-frame onset/BCE head jointly with CTC. This gives dense gradients from frame 1 (CTC alone gives almost none early), anchors the alignment, and prevents the "output blank everywhere" cold-start. Weight ~0.3–0.5 of total loss. This alone often un-sticks a collapsed CTC.
3. **Fix the sim-to-real / class-balance bug first — collapse at 84% CER usually means the model has learned the trivial prior.** With ~30 keys uniformly you'd expect ~30x better than random; 84% ≈ "always emit the most common char + blanks." Check: (a) is the blank frame:label frame ratio enormous (e.g. 50:1)? Downsample blank-dominant frames or shorten the frame rate; (b) is your synthetic key distribution uniform while the LM prior is English? Match training char distribution to target text; (c) is there a constant synthetic artifact (same click sample, same SNR, aligned grid) the net exploits? Randomize aggressively (see §5).
4. **Switch CNN→CRNN (CNN + BiLSTM/BiGRU), i.e. the Slater/Deep-Speech-2 recipe.** A pure ConvNet has a fixed receptive field and no sequence memory; overlapping keystrokes need temporal context to disambiguate press-vs-release and adjacent keys. Slater et al. (ACSAC 2019), the only published overlapping-typing transcriber, use exactly CNN→BiRNN→dense→CTC and reach **~7–15% CER on real continuous typing**. Add a Conformer only if you have the data/compute; CRNN is the proven baseline here.
5. **Curriculum + beam-search LM fusion at decode.** Train slow→fast (start with non-overlapping isolated keys, ramp inter-key interval down and overlap up over epochs) so CTC learns clean alignments before facing overlap. At inference, use prefix-beam-search CTC decoding with a character/word n-gram (or small neural) LM via shallow fusion: `score = log p_CTC + λ·log p_LM + β·|tokens|` (λ≈0.3–1.0, word-insertion bonus β). Slater used an LM head; recent 2024–2025 work uses LLM "typo correction" as a post-pass. Expect large CER drops on English free text.

If you do only three things: **#1 (entropy/prior term), #2 (onset multitask head), #3 (audit the synthetic data / blank balance).** Those three directly cause and cure collapse.

---

## 1. The Reference Method — Slater et al., ACSAC 2019

*"Robust Keystroke Transcription from the Acoustic Side-Channel."* Slater, Novotney, Moore, Morgan, Tenaglia (Two Six Labs). This is **the** paper for overlapping/continuous typing — everyone else does pre-segmented isolated-key classification. Extracted from the paper + author slides:

**Why it matters for us:** They explicitly break the two assumptions every prior attack made — (a) that keystroke detection/segmentation is trivial (TPR/TNR = 100%), and (b) that key acoustics are consistent across users. Their whole point is that *segmentation is the hard part*, and they solve it by **not segmenting at all** — end-to-end audio→text with CTC, exactly the regime we're in.

### Data
- Hardware: Razer BlackWidow mechanical keyboard (Cherry MX Blue), Blue Yeti Pro mic placed **<1 m to the left** of the keyboard.
- 17 users, 76 sessions, **~1100 keystrokes/session, 86k keystrokes total, 8.2 hours of audio.**
- Fixed, variable, and **free-text** formats; English prose and code tasks.
- Keystrokes captured via software keylogger; audio–keylog sync via an **audio chirp**. Crucially: paired audio + transcript **with no detailed per-key timing** used for the RNN — CTC handles alignment.

### Features / preprocessing
- Filter noise, **downsample**, generate **10 ms spectrogram frames** (fine time resolution — important for overlap; press and release transients are milliseconds apart).
- (Their segment-based baseline used 40 ms spectrograms; the end-to-end RNN used the 10 ms spectrogram sequence.)

### Model
- **"Adapted from Deep Speech 2":** Input spectrogram sequence → **Convolutional layers → Recurrent (BiRNN) layers → Dense layer → per-frame probability distribution over keys** (+ a blank/"no keystroke" symbol `_`).
- Trained with **CTC loss**. Decode = argmax per frame then collapse blanks + duplicates; "[Insert Language Model Here]" — they add an LM for English.
- Label set = key classes + blank.

### Training
- Split each session into **20-second cuts**.
- **4-fold cross-validation**; stratify **per session** (= "known typist") or **per user** (= "unknown typist", the harder generalization test).
- Paired audio/keylog transcripts only; **no frame-level timing labels** — CTC learns the alignment.
- Baseline for comparison: "Skype & Type" (Compagno 2017) power-threshold detection + logistic-regression classification.

### Reported results (CER = (ins+del+sub)/ref chars)
| Setting | CER |
|---|---|
| RNN + true key timings (upper bound) | **7.41%** |
| End-to-end RNN, audio only (English, session strat) | **14.3%** |
| Oracle segment baseline (Skype & Type) | 36.0% |
| Random | 86.7% |

RNN median CER across conditions: **6.8% (best: session-strat) → 12.4% (worst: user-strat / unknown typist).** Task/stratification breakdown medians: 12.4% / 10.5% / 9.7% / 6.8%. An LM improved English transcription further.

**Key takeaways for us:**
- They trained on **real continuous typing**, not synthetic overlap-added audio. Our sim-to-real gap is a prime suspect for collapse.
- **10 ms frames** (not 40 ms) for the end-to-end path — high temporal resolution matters when clicks overlap.
- CRNN, not pure CNN.
- **Random baseline is 86.7% CER — our 84% means our model is barely above random / has collapsed to the class prior.** That is the diagnostic smoking gun.

---

## 2. Diagnosing "class-prior collapse" (predicts one symbol everywhere)

Collapse in CTC-for-sparse-events has a well-understood mechanism and several compounding causes:

### Mechanism (why CTC collapses to blank)
Blank is the most "versatile" token — it appears in the overwhelming majority of feasible alignment paths. Early in training the randomly-initialized net finds that emitting blank everywhere gives a locally decent CTC loss; gradient then concentrates on that dominant (blank-heavy) path and **reinforces itself**, ignoring alternative alignments. This is the same dynamic that produces "peaky" CTC in ASR, but for **sparse events** (keystrokes are ~2–5% of frames) it goes all the way to full collapse instead of just peakiness. Reference works: EnCTC (Liu et al., NeurIPS 2018); "Why does CTC result in peaky behavior?" (Zeyer et al. 2021); label-prior CTC (Huang et al. 2024).

### Compounding causes specific to our setup
1. **Extreme blank:label frame imbalance.** If frame rate is high and typing is sparse, 95%+ of frames are blank. The all-blank solution is a strong attractor. → reduce frame rate, or use entropy/prior term, or class-weight.
2. **Synthetic data too clean / too regular.** Overlap-added synthetic audio often reuses the same click sample, same amplitude, same SNR, and lays keys on a regular grid. The net can either (a) overfit a shortcut that doesn't transfer (real-typing collapse), or (b) find the data so "easy" that blank-everywhere already minimizes loss on the majority. → aggressive randomization (§5).
3. **Training distribution ≠ inference distribution.** Uniform random keys in training but English at test → the model's learned prior fights the LM; if training text is a fixed template, it memorizes the template and emits its mode.
4. **Learning rate / warmup too aggressive** → net dives into the blank basin before learning anything. → lower LR, warmup, gradient clipping.
5. **No dense supervision at cold start.** CTC gives near-zero useful gradient in the first epochs on sparse events. → onset multitask head (§3) provides it.

### Fast diagnostics to run
- Print the **per-frame posterior for blank**: if it's ~0.9–1.0 everywhere → classic collapse.
- Print the **argmax token histogram** over a val batch: one or two tokens → confirmed.
- Compare CER to the **majority-class baseline** and to **random (86.7% in Slater)**: at 84% you're at random → the model has learned nothing but the prior.
- Turn off overlap entirely (isolated keys, wide spacing): if it *still* collapses, the problem is the loss/optimization, not the overlap. If it learns, the problem is overlap difficulty/curriculum.

---

## 3. Loss formulations to prevent collapse (concrete)

Let `y_k^t` = softmax posterior for token k at frame t; blank = `∅`.

### (a) EnCTC — maximum-entropy regularization (NeurIPS 2018) — *recommended first*
$$\mathcal{L} = \mathcal{L}_{CTC} \;-\; \beta \, H(\text{feasible paths})$$
where `H` is the entropy of the distribution over feasible alignment paths given the target label sequence. The entropy term **prevents error signal from collapsing onto the single dominant (blank) path** and encourages exploration of neighboring alignments, yielding **smoother, wider non-blank activations**. The paper also adds an **entropy-based pruning** that removes unreasonable alignments to keep it tractable.
- Practical proxy if full-path entropy is awkward to implement: add a **per-frame entropy bonus** `+β Σ_t H(y^t)` (maximize output entropy) early in training, annealed to 0. Simple, and directly discourages the spiky all-blank output.
- **β ≈ 0.1–0.2**, anneal downward over training.

### (b) Label-prior CTC (Huang et al., 2024) — *strong, simple, well-tuned*
Reweight each alignment path by dividing per-frame probs by the token prior:
$$P_{\text{prior}}(\pi|X) = \prod_{t=1}^{T} \frac{y_{\pi_t}^t}{P(\pi_t)^{\alpha}}, \qquad \mathcal{L}=-\log\!\!\sum_{\pi\in\mathcal{B}^{-1}(W)}\!\! P_{\text{prior}}(\pi|X)$$
- This **boosts alignment paths that contain fewer blanks**, directly countering blank dominance.
- Estimate `P(k)` by **marginalizing the posteriorgram over all frames each epoch** (init uniform, update at epoch end); or, simpler, count token frequency in Viterbi alignments.
- **α = 0.3** is optimal in the reference (α > 0.3 prevented convergence). Reduced blank prior from **0.80 → 0.32**; improved boundary accuracy 12–40%. For our sparse-event case this both de-peaks *and* de-collapses.

### (c) CR-CTC — consistency regularization (2024) — *for accuracy + robustness once un-collapsed*
Run **two independently SpecAugment-ed views** of the same spectrogram through the shared encoder; add symmetric KL between their per-frame distributions:
$$\mathcal{L} = \tfrac12\big(\mathcal{L}_{CTC}(x^a)+\mathcal{L}_{CTC}(x^b)\big) + \alpha\,\mathcal{L}_{CR},\quad \mathcal{L}_{CR}=\tfrac12\big[D_{KL}(\text{sg}(z^b)\|z^a)+D_{KL}(\text{sg}(z^a)\|z^b)\big]$$
- **α = 0.2**; use **stop-gradient** on the target branch. Use **~2.5× stronger time-masking** than default (masked frames must be predicted from context — self-distillation).
- Explicitly **suppresses peaky/overconfident distributions** and improves generalization (LibriSpeech WER 2.52→2.10 for Zipformer-M). Doubles as sim-to-real regularizer since the two augmentations force invariance.

### (d) Cheap complementary tricks
- **Blank class weighting / focal-style down-weighting** of blank frames in an auxiliary frame-CE loss.
- **Label smoothing** on any auxiliary CE head.
- **Softmax temperature > 1** early to keep the distribution soft, anneal to 1.

---

## 4. Multitask onset / Sound-Event-Detection head (stabilizer)

Add an **auxiliary per-frame head** trained jointly with CTC:

- **Onset head:** per-frame binary "is a key pressed onset within this frame?" with BCE loss. Because **synthetic overlap-added audio has exact ground-truth onset times**, this is free dense supervision — the thing CTC most lacks at cold start.
- Optional **key-activity head** (sound-activity detection, SAD) and even a **frame-level key-class head** (multi-label, since keys overlap) — polyphonic SED framing: keystrokes are overlapping sound events with onsets.
- Reference: SED multitask work shows **onset + activity auxiliary heads give up to ~3× event-based F-score** over a bare SED model (URBAN-SED); the temporal anchors condition the primary head.

**How it fixes collapse:** dense gradients from epoch 0 pull the shared encoder toward "there is/ isn't an event here," so by the time CTC needs to place labels the features already localize keystrokes. Set total loss:
```
L = L_CTC + λ_onset · BCE_onset (+ λ_act · BCE_activity)
```
with **λ_onset ≈ 0.3–0.5**, optionally decayed once CTC is healthy. This is arguably the most reliable single un-sticking trick for a collapsed sparse-event CTC.

**Alternative architecture (Onsets-and-Frames style, from music transcription):** two heads — an onset detector and a frame/label detector where the frame head is *conditioned on* the onset head's output. Very well matched to keystrokes (press onset = the discriminative transient; the release/tail is the "frame").

---

## 5. Sim-to-real: generating synthetic overlap that transfers

Our collapse is likely partly a **synthetic-data artifact**. Slater trained on real typing; we must make synthetic look real enough that blank-everywhere is *not* a valid shortcut and features transfer.

**Generate overlap-added audio like this:**
- **Sample from a bank of many real recorded click samples per key** (record dozens per key on the actual demo keyboard). Never reuse one canonical click. Include both **press and release transients** as separate events (release is quieter, ~30–120 ms after press — this is where "overlap" really bites at speed).
- **Realistic inter-key timing:** draw inter-key intervals from a real typing distribution (log-normal, mean ~120–250 ms for fast typists, heavy left tail for common digraphs). Model **key press–release durations** and let neighboring keys overlap in time (release of key N overlaps press of key N+1). Slater's appendix specifically histograms "time from key release to subsequent key press" — replicate that distribution.
- **Amplitude / velocity jitter** per keystroke; different keys have different loudness.
- **Room + channel randomization (domain randomization):** convolve with a bank of room impulse responses (RIRs), random mic distance/position, add real ambient noise at varied SNR (e.g. 5–30 dB), random EQ, and simulate the real capture chain (codec, AGC). This is what makes it transfer and simultaneously kills the "one constant artifact" shortcut.
- **Match the character distribution to the target domain** (English text corpus for the demo), not uniform keys — otherwise the learned prior fights reality.
- **De-grid:** never align onsets to spectrogram frame boundaries; jitter onsets by sub-frame amounts.

**Pitfalls:**
- Overlap-add with a single click sample + fixed SNR + uniform keys ⇒ trivial for the net ⇒ it minimizes loss by emitting the prior ⇒ **collapse** and zero real-world transfer. (This matches our symptom.)
- Test on a *held-out real recording* every epoch; synthetic val CER can look fine while real collapses.
- **Mix in real data** as soon as you have any (even 10–20 min of real continuous typing with keylog) — a few real batches per step dramatically closes the gap; or fine-tune on real after synthetic pretraining.

---

## 6. Curriculum design (slow → fast)

CTC learns clean alignments far more easily on separated events, then transfers to overlap:
1. **Stage 0 (warmup):** isolated single keystrokes, wide spacing (>400 ms), high SNR. Get the onset head + CTC to reliably fire on non-blank. Verify no collapse here first.
2. **Stage 1:** normal typing speed, minimal overlap, moderate noise.
3. **Stage 2:** fast typing, press/release overlap, digraph timing, full domain randomization (RIRs, noise, EQ).
4. **Stage 3:** real recordings / fine-tune.
Ramp overlap and speed **continuously** (anneal the inter-key-interval distribution downward) rather than in hard jumps. Keep the entropy/prior term strongest in Stage 0–1, anneal it as alignments stabilize.

---

## 7. Model architecture recommendations

| Option | Verdict for us |
|---|---|
| **Pure CNN (current)** | Insufficient alone — fixed receptive field, no sequence memory to resolve overlap; prone to prior collapse. |
| **CRNN = CNN + BiLSTM/BiGRU** (Slater / Deep Speech 2) | **Recommended default.** Proven on this exact task (7–15% CER real). CNN front-end (2–3 conv blocks over spectrogram) → 3–5 BiLSTM/BiGRU layers (256–512 units) → dense → CTC. Robust with modest data. |
| **Conformer / transformer-CTC** | Best ceiling if you have data + compute; convolution-augmented self-attention captures long context. Consider only after CRNN works; more data-hungry and more prone to instability on tiny datasets. CR-CTC (§3c) was designed on Zipformer/Conformer. |
| **CoAtNet / ResNet classifiers** (2023–2025 ASCA works, 99%+ acc) | These are **isolated-key classifiers on pre-segmented audio**, not continuous transcribers — not directly usable for overlap, but good feature-extractor backbones. |

**Front-end features:** log-mel or linear spectrogram, **10 ms hop** (per Slater) — do not use 40 ms for the end-to-end path; overlap disambiguation needs the resolution. 64–128 mel bins. Consider per-frequency normalization.

---

## 8. Decoding strategy + language-model fusion

- **CTC prefix beam search** (not greedy argmax) with beam width 16–64.
- **Shallow fusion** with a character- or word-level LM trained on the target text domain:
  `score(y) = log p_CTC(y|x) + λ · log p_LM(y) + β · word_count(y)`
  Tune **λ ≈ 0.3–1.0**, word-insertion bonus **β** to counter deletion bias. Slater explicitly added an LM for English and it helped.
- **Neural LM rescoring / n-best rescoring:** decode n-best with the n-gram, rescore with a small RNN/transformer LM or a character-level model.
- **LLM post-correction (2024–2025):** recent ASCA work ("LLM-assisted spectrogram typo correction," 2504.11622; ASCA-Net semantic post-correction) feeds the noisy transcript to an LLM to fix "typos." Cheap to bolt on for the demo: pipe raw CTC output through a spell/LM corrector. Big CER win on English free text.
- **Keyboard-adjacency prior:** substitution errors cluster among physically/acoustically adjacent keys; a confusion-aware LM or edit model exploits this.

---

## 9. Concrete recommended configuration (starting point)

- **Features:** log-mel spectrogram, 16 kHz, 25 ms window, **10 ms hop**, 80 mel bins, per-frame CMVN.
- **Model:** CNN (2× conv blocks, stride only in freq) → 4× BiGRU(320) → dense → softmax over {keys + blank}.
- **Loss:** `L = L_CTC(label-prior, α=0.3) + 0.4·BCE_onset + 0.2·CR_consistency`, plus per-frame entropy bonus β=0.1 annealed to 0 over first 10 epochs.
- **Optim:** Adam, LR 3e-4 with 1–2k step warmup, grad clip 5.0, batch by ~20 s cuts.
- **Data:** domain-randomized synthetic (multi-sample click bank, RIRs, noise 5–30 dB, real timing distribution, English char distribution) with curriculum Stage 0→3; fine-tune on any real continuous recording.
- **Decode:** CTC prefix beam search (beam 32) + char/word n-gram LM shallow fusion (λ≈0.5) + optional LLM post-correction.
- **Track:** blank posterior mean, argmax token histogram, and **real-recording CER** every epoch. Target: get below the 86.7% random floor immediately, then chase Slater's 10–15% real-typing range.

---

## Sources
- [Slater et al., ACSAC 2019 — Robust Keystroke Transcription from the Acoustic Side-Channel (paper/slides PDF)](https://www.acsac.org/2019/program/final/1/137.pdf) · [ACM DL](https://dl.acm.org/doi/10.1145/3359789.3359816)
- [Liu et al., NeurIPS 2018 — CTC with Maximum Entropy Regularization (EnCTC)](https://papers.nips.cc/paper/7363-connectionist-temporal-classification-with-maximum-entropy-regularization.pdf) · [proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e44fea3bec53bcea3b7513ccef5857ac-Abstract.html)
- [Huang et al., 2024 — Less Peaky and More Accurate CTC Forced Alignment by Label Priors](https://arxiv.org/abs/2406.02560)
- [CR-CTC: Consistency Regularization on CTC for Improved Speech Recognition, 2024](https://arxiv.org/html/2410.05101v1)
- [Zeyer et al., 2021 — Why does CTC result in peaky behavior?](https://arxiv.org/pdf/2105.14849)
- [Onsets, Activity, and Events: A Multi-task Approach for Polyphonic Sound Event Modelling](https://lacuna.tiptreesystems.com/work/onsets-activity-and-events-a-multi-task-approach-for-polyphonic-sound-event/)
- [A Survey on Acoustic Side Channel Attacks on Keyboards, 2023](https://arxiv.org/pdf/2309.11012)
- [Practical Acoustic Eavesdropping on Typed Passphrases, 2025](https://arxiv.org/pdf/2503.16719)
- [Making Acoustic Side-Channel Attacks Viable with LLM-Assisted Spectrogram "Typo" Correction, 2025](https://arxiv.org/pdf/2504.11622)
- [A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards, 2023 (CoAtNet)](https://arxiv.org/pdf/2308.01074)
- [ASCA-Net (keystroke recognition + semantic post-correction), 2026](https://link.springer.com/article/10.1007/s11416-026-00648-7)
