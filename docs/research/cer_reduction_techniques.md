# Lowering CER on the continuous keystroke transcriber

Defensive research memo (HackGT "KeyGuard"). Public techniques only; own-device
acoustic side-channel, attacker-to-build-a-shield. Ranked, implementation-ready.

## Where we are (the exact setup this memo targets)

- **Encoder**: `MtlCRNN` (`keyguard/ctc/train_overlap.py`) = 3× Conv2d blocks that
  pool **frequency only** (`MaxPool2d((2,1))`, so time/frame alignment is preserved)
  → 2-layer BiGRU, hidden 192, dropout 0.2 → `head: Linear(384, 37)` CTC logits +
  `onset_head: Linear(384, 1)` per-frame onset logit.
- **Frames**: 16 kHz, `N_FFT=512`, `HOP=128` → **8 ms/frame**, `N_MELS=64`, log-mel
  normalized per-clip (`model.logmel`).
- **Symbols**: 37 = blank(0) + A–Z,0–9 (`VOCAB`, `data.py`).
- **Loss (current)**: `LAMBDA_FRAME=1.0` dense class-weighted per-frame CE
  (`fce_w[BLANK]=BLANK_W=0.05`, `ignore_index=-100`) + `LAMBDA_ONSET=0.5` onset BCE +
  `LAMBDA_CTC=0.0` (off) + `BETA_ENT=0.0`. Frame targets = key id at onset frame
  ±`FRAME_WIDTH=1`, blank elsewhere (`frame_key_target`).
- **Decoding (current)**: CTC **greedy** (`greedy_decode`: argmax → collapse repeats →
  drop blank). `lm_decode.LMDecoder` (pure-python prefix-beam + order-3 `CharLM`,
  optional pyctcdecode/KenLM) and `lm_correct.correct()` (SymSpell word-splitting)
  exist but are **not** in the reported ~40.7% greedy path.
- **Data**: SKAID `data/continuous/skaid/labels.jsonl` — 51 sessions, 67,747
  keystrokes, held-out = every 5th whole session (26 typists, unseen). Plus
  `data/pool/*.npz` isolated-key banks across **10 keyboards** (harrison,
  hf_sakuzas, kad_spatam_mechanical, mka_hp/lenovo/mac/msi, multipressure, zenodo, live).
- **Reported**: ~40.7% CER mean / 37.5% median greedy on real held-out typists.
- **Dominant error = SUBSTITUTIONS** (acoustically adjacent keys: H→G, U→P), plus
  insertions/doublings. This is the failure mode every recommendation below is
  scored against.

Key implication of the frame-CE regime: the CTC head is effectively a **per-frame
classifier**, so per-frame substitutions dominate. Two lever families attack this
directly: (A) *decode-time* corrections that exploit language structure and the
onset head; (B) *train-time* changes that make each per-frame decision less
confusable. Augmentation and data strategy compound both.

---

## TOP 6 RANKED BY CER-REDUCTION-PER-EFFORT

> Effort = engineer-days; Impact = absolute CER points on the real held-out set.
> Ranges are literature-anchored (Slater ACSAC'19 saw 36%→7% from LM decoding on a
> comparable acoustic front-end) and de-rated for our unseen-typist, no-space setup.

| # | Lever | Effort | Est. abs. CER Δ | Attacks |
|---|-------|--------|-----------------|---------|
| 1 | **Onset-gated decoding** (emit exactly one key per detected onset) | 0.5–1 d | **−4 to −10 pts** | insertions/doublings, blank-collapse tails |
| 2 | **Label smoothing + focal frame-CE + adjacency-aware smoothing** | 0.5 d | **−3 to −7 pts** | substitutions (direct) |
| 3 | **KenLM char n-gram shallow fusion in prefix-beam** (wire up existing decoder) | 1 d | **−4 to −8 pts** | substitutions, insertions |
| 4 | **Freq-only SpecAugment + RIR/noise/VoIP** (alignment-safe aug) | 1–1.5 d | **−3 to −6 pts** | substitutions (generalization to unseen typist/mic) |
| 5 | **Isolated-key pretrain (`data/pool/*.npz`) → fine-tune on SKAID** | 1–2 d | **−3 to −6 pts** | substitutions (better per-key acoustic model) |
| 6 | **Ensemble / TTA + confidence rescoring + LLM post-correction** | 1–2 d | **−3 to −6 pts** (mostly on *word* readability) | substitutions, run-together text |

These are **largely additive up to a point** (decode-side #1+#3+#6 overlap; train-side
#2+#4+#5 overlap). A realistic stacked target: **40.7% → high-20s% char CER**, with
*word*-level readability improving much more once #3/#6 re-insert boundaries.

---

## 1. DECODING IMPROVEMENTS

### 1.1 Onset-gated decoding — RANK #1 (best ROI)

**What.** We already train a per-frame onset head (`onset_head`, BCE against
`onset_target`). Use it to *count and locate* keystrokes, then emit **exactly one key
per detected onset** by reading the CTC/frame classifier posterior at that onset
frame — instead of letting greedy collapse decide how many symbols come out.

**Why it fixes our failure mode.** Insertions and doublings (H→GG, spurious extra
letters) come from greedy collapse emitting a symbol whenever the argmax changes,
even inside one physical click or during ambiguous release transients. Onset gating
fixes the *length* of the output to the number of true events, eliminating that whole
error class and preventing the doubled-char artifacts `lm_correct._destutter` is
currently patching after the fact. Substitutions at true onsets remain, but the
sequence length is now correct, which also makes downstream LM fusion far more
effective (the LM no longer has to fight length errors).

**Implementation (concrete).**
1. Peak-pick the onset probability `σ(onset_logit)` over frames: threshold `τ≈0.5`,
   enforce a **minimum inter-onset distance** of ~5 frames (40 ms; a human cannot
   press two keys <~40 ms apart, and CLIP≈one key width). Use
   `scipy.signal.find_peaks(prob, height=τ, distance=5)`.
2. For each detected onset frame `f`, choose the key as
   `argmax_{s≠blank} mean(logits[f-1:f+2, s])` (average over ±1 frame = `FRAME_WIDTH`
   window; excludes blank). This is a ~15-line function next to `greedy_decode`.
3. Emit that key. Output length = number of peaks.

**Calibrate τ on held-out** to match predicted vs true keystroke counts (minimize
count error, not CER, to avoid overfitting the threshold). Expect the onset head to
be well-trained already (it carries `LAMBDA_ONSET=0.5`).

**Refinement (joint head).** Add a small **duration/velocity feature** later, but v1
is just peak-pick + argmax-at-peak. Because the frame-CE head is already supervised
to place the key id at the onset frame ±1, reading it at the onset peak is
maximally aligned with training.

**Est. impact.** −4 to −10 pts; removes most insertions/doublings outright.
**Effort.** 0.5–1 day. **Do this first.**

### 1.2 CTC beam search + char/word n-gram LM (KenLM) shallow fusion — RANK #3

**What.** Replace greedy with prefix-beam search + a **char-level KenLM** via shallow
fusion: `score = logP_acoustic + α·logP_LM(char|context) + β·|emitted|`. The scaffold
already exists (`lm_decode.LMDecoder`, `build_kenlm`, `decode.beam_search`); it's just
not on the reported path and KenLM/`lmplz` isn't installed here.

**Why it fixes our failure mode.** Substitutions like U→P and H→G are acoustically
close but produce **low-probability character sequences** ("...PENED" vs "...HENED").
A char LM re-ranks toward legal English letter transitions, correcting exactly the
adjacent-key confusions that dominate. This is the single biggest documented lever in
the literature for this task (Slater et al. 36%→7% region).

**Implementation (concrete).**
- **Build the LM on real English**, not the synthetic corpus currently defaulted in
  `_corpus_lines`. Use a large char-tokenized English corpus (Gutenberg / OpenWebText
  subset / `wordfreq` top-N expanded to running text). **Order 5–6** char n-gram
  (`KENLM_ORDER=5` is set; try 6). Install `kenlm` + `lmplz` (the code auto-detects and
  lights up the fast path).
- **Because we have no space symbol**, either (a) train the char LM on
  space-*stripped* text so it matches our stream, or (b) add a space symbol to the
  vocab (37→38) and let the LM insert boundaries — (b) is higher-value long-term (see
  §4 note) but (a) ships today with zero retraining.
- **Fusion weights**: start `α=0.5`, `β=1.0` (defaults in `LMDecoder`), then grid
  `α∈{0.3,0.5,0.8,1.2}`, `β∈{0,0.5,1,2}` on held-out. Beam width **64–100**
  (`beam_width=64` set; widen to 100 for the eval run, it's offline).
- **Prefix-beam details for CTC**: keep separate blank/non-blank prefix probabilities
  (`decode.beam_search` already does `(pb, pnb)` per prefix and handles the
  repeated-char merge correctly — the `ch == prefix[-1]` branch). Apply the LM score
  **only when extending with a new non-blank char** (already the case: `add = lp[s] +
  α·lm.logprob(prefix, ch)`), never on blank/repeat, or you double-count.
- **Combine with onset gating (#1)**: best combo is **onset-gated beam** — constrain
  the beam to emit one non-blank per onset window and let the LM pick *which* key.
  This bounds length (from #1) *and* corrects substitutions (from LM). Implement as a
  beam over onset slots rather than over all T frames: at slot `i`, candidates =
  top-k keys by acoustic posterior in that onset window, scored with the LM given the
  prefix. Much smaller search, and it directly targets the two live error classes.

**Est. impact.** −4 to −8 pts on top of greedy; more when combined with #1.
**Effort.** 1 day (mostly corpus + KenLM install + weight sweep; decoder exists).

### 1.3 Confidence-based rescoring

**What.** For each emitted key, compute a confidence (softmax max at the onset frame,
or margin between top-2 logits). Keep an **n-best list** from the beam and rescore by
combining acoustic confidence, LM score, and an **adjacency penalty** (penalize
candidates that would require a physically implausible key given neighbors — see §2.4).

**Why.** Low-confidence, low-margin frames are exactly where substitutions happen; a
targeted rescore lets the LM/adjacency prior override only where the acoustic model is
unsure, without disturbing confident keys. This is the safe, precision-preserving way
to apply priors.

**Implementation.** Take the beam's top-10 hypotheses (`beam_search` already tracks a
sorted beam dict — expose top-k instead of top-1). Final score =
`logP_ac + α·logP_charLM + γ·logP_adjacency`. Tune `γ` small (0.1–0.3) on held-out.
**Est. impact.** −1 to −3 pts incremental. **Effort.** 0.5 day (given #1.2).

### 1.4 LLM-based post-correction for run-together text — part of RANK #6

**What.** Feed the beam n-best (or the single decoded string) to an LLM with a prompt
to (a) insert word boundaries and (b) fix implausible letters, constrained to
minimal edits. Hooks exist: `decode.llm_rescore` (Grok/xAI) and
`lm_correct.correct()` (offline SymSpell Viterbi word-splitting, no API needed).

**Why.** Our stream has **no spaces** ("HOPEYOURWEEKISOFFTOAGO"); much of the *usable*
error is that words run together and a few letters are wrong. An LLM/dictionary pass
recovers readability and fixes residual substitutions the char LM missed, using
long-range word/semantic context the n-gram cannot. This is a *readability/word-CER*
win more than a raw-char-CER win.

**Implementation.**
- **Offline first**: `lm_correct.correct()` already does Damerau-Levenshtein SymSpell
  + wordfreq Viterbi segmentation and de-stuttering. Run it as the final stage; it's
  CPU-only, deterministic, and reproducible for the paper.
- **LLM on top** (optional, for demo): give it the top-8 beam candidates + the
  onset-count as a length hint ("the string has exactly N letters"), ask for minimal-
  edit English with spaces. The length hint from #1 sharply constrains hallucination.
- **Report char-CER on space-stripped text** (as `lm_correct._cer` does) so word
  insertion doesn't distort the metric — but *also* report word-level accuracy, which
  is what actually matters for a defensive threat demonstration.

**Est. impact.** small on space-stripped char-CER (−1 to −3), large on word
readability. **Effort.** offline path is ~0 (exists); LLM path 0.5 day.

---

## 2. REDUCING SUBSTITUTION ERRORS (train-time)

### 2.1 Label smoothing + focal loss on the frame-CE — RANK #2

**What.** Change the per-frame CE (`fce = CrossEntropyLoss(weight=fce_w, ignore_index=-100)`)
to (a) **label smoothing** `ε≈0.1` and (b) **focal** reweighting `(1−p_t)^γ`, `γ≈1–2`.

**Why it fixes our failure mode.** Substitutions between acoustically similar keys are
the *hard*, low-margin examples. Focal loss concentrates gradient on exactly those
frames (the easy blank/clear-key frames already dominate the loss otherwise). Label
smoothing prevents the head from becoming overconfident on the training keyboards,
which improves calibration on **unseen typists** (our eval) and makes the onset-frame
margins the beam/rescorer (#1.3) relies on more meaningful. Both are ~1-line changes.

**Implementation (concrete).**
- Label smoothing: `nn.CrossEntropyLoss(weight=fce_w, ignore_index=-100,
  label_smoothing=0.1)`. Try `ε∈{0.05,0.1,0.15}`.
- Focal: implement a class-weighted focal CE (there's no built-in with both weight and
  focal). `loss = Σ w_c · (1−p_t)^γ · (−log p_t)` over non-ignored frames, `γ=1.5`.
  Keep the `BLANK_W=0.05` down-weight. Combine LS + focal or A/B them.
- Keep `LAMBDA_ONSET=0.5`; these change only the frame-CE term.
**Est. impact.** −3 to −7 pts. **Effort.** 0.5 day. **Second-do-first.**

### 2.2 Adjacency-aware label smoothing (keyboard prior) — high value, cheap

**What.** Instead of uniform label smoothing, distribute the smoothed mass over
**acoustically/spatially neighboring keys**. Build a 36×36 confusion-neighbor matrix
(from the model's own held-out confusion matrix, or a physical QWERTY-adjacency +
key-position prior) and use it as the soft target.

**Why.** Our substitutions are *structured* (H↔G, U↔P — physical neighbors whose
strikes sound alike). Uniform smoothing wastes mass on impossible confusions
(H↔0). Distributing smoothing mass along the *actual* confusion structure tells the
model "if not H, it's most likely G/J/N", which both regularizes and lets the
downstream adjacency rescorer (§1.3) be consistent with training.

**Implementation.** Compute the confusion matrix from a first `MtlCRNN` run's held-out
frame predictions; row-normalize, floor, use as soft targets: `target = (1−ε)·onehot +
ε·neighbor_dist`. `ε≈0.1`. This is a data-driven upgrade of §2.1 — do §2.1 first, then
swap in the neighbor distribution.
**Est. impact.** −1 to −3 pts on top of §2.1. **Effort.** 0.5 day.

### 2.3 More temporal context — Conformer / dilated attention

**What.** Give the encoder a larger receptive field so it disambiguates a key from its
*coarticulation* with neighbors. Options, in ascending effort:
- (a) **Widen the BiGRU** hidden 192→320 and/or add a 3rd layer.
- (b) Insert **1–2 self-attention layers** on top of the BiGRU (a mini-Conformer
  block: conv + MHSA + FFN), heads=4, d=256.
- (c) Full **Conformer** encoder replacing the CRNN.

**Why.** Overlapping/fast typing makes a keystroke's spectrum bleed into neighbors;
substitutions rise with overlap (the `evaluate` overlap sweep shows CER climbing with
overlap%). Longer context lets the model use the *sequence* of clicks, not just the
local frame, to resolve the ambiguous ones.

**Caution.** More capacity + only 67k keystrokes risks overfitting the training
typists → *worse* unseen-typist CER. Pair any capacity increase with §3 augmentation
and §4 pretraining, and watch the held-out gap. **Prefer (a) then (b).** Keep
frequency-only pooling; do **not** add time downsampling (breaks frame alignment).
**Est. impact.** −2 to −5 pts *if* regularized; risk of regression otherwise.
**Effort.** (a) 0.5 d, (b) 1–2 d, (c) 3+ d.

### 2.4 Spatial/keyboard-adjacency prior at decode — see §1.3

Encode a physical key-position cost: penalize hypotheses requiring implausible
same-hand rapid jumps, or up-weight neighbors of the acoustic argmax. Cheapest as a
**decode-time** rescoring term (`γ·logP_adjacency` in §1.3) rather than a train-time
change. **Est. impact.** −1 to −2 pts. **Effort.** 0.5 day.

### 2.5 Mixup / manifold-mixup that preserves onset timing

**What.** Mix two clips' *log-mels* with `λ~Beta(0.2,0.2)` and mix their **frame
targets the same way** (soft-label mixup on the frame-CE). Crucially, **do NOT
time-shift** — mixup is per-frame elementwise, so onset frames stay put; the frame
label at frame `f` becomes `λ·onehot(key_A[f]) + (1−λ)·onehot(key_B[f])`.

**Why.** Regularizes the per-frame classifier and expands the effective typist/keyboard
distribution (mixing across the 10 pool keyboards). Manifold-mixup (mix at the BiGRU
feature layer) is even safer for timing since it mixes representations, not raw frames.

**Caution/notes.** The onset-BCE target should be `max` (union) of the two onset maps,
not the mix, so onsets aren't blurred below threshold. Mixing two continuous typing
streams superimposes clicks (physically like two-finger noise) — acceptable as
regularization but keep mixup probability moderate (`p≈0.3`).
**Est. impact.** −1 to −3 pts. **Effort.** 0.5–1 day.

### 2.6 Ensembling + test-time augmentation — part of RANK #6

- **Ensemble**: train 3–5 `MtlCRNN` with different seeds / the conv vs crnn encoder /
  ±attention; average the per-frame log-posteriors before decoding. Different models
  make different substitutions; averaging cancels them.
- **TTA**: run each clip through a few **timing-safe** perturbations (gain, mild EQ,
  additive noise at high SNR — never time-shift) and average posteriors.
- **Weight averaging (SWA/EMA)**: cheapest "ensemble" — keep an EMA of weights during
  training (decay 0.999), evaluate the EMA. Almost free, reliably −0.5 to −1.5 pts.

**Est. impact.** ensemble −2 to −4 pts (at N× inference cost); EMA −0.5 to −1.5 pts
(free); TTA −1 to −2 pts. **Effort.** EMA 0.25 d; ensemble 1–2 d (mostly training time).

---

## 3. ALIGNMENT-PRESERVING AUGMENTATION — RANK #4

The hard constraint: **nothing may move keystroke timing**, because frame-CE and
onset-BCE labels are per-frame. `spec_augment` currently time-shifts via `np.roll`
(`SPEC_TIME_SHIFT=0.40`) which desynced labels → SpecAugment was disabled
(`KEYGUARD_SPEC_AUG=0`). The fix is to keep only timing-invariant ops.

### 3.1 Frequency-masking-only SpecAugment (safe)

- **Remove the time-roll and time masks** from `augment.spec_augment`; keep **only
  frequency masks**. Frequency masking does not touch the time axis → labels stay
  aligned.
- **Params**: 1–2 frequency masks, each width `F ~ Uniform(0, 8)` mel bins (of 64;
  ~≤12%), fill = per-clip mean (as now). Total masked freq area ≤ 20% (`SPEC_MAX_AREA`).
  Concretely set `SPEC_TIME_SHIFT=0.0`, drop the `axis==0` (time) mask loop, keep the
  `axis==1` (freq) loop. Re-enable `KEYGUARD_SPEC_AUG=1`.
- **Why it helps substitutions**: forces the model not to rely on a single mel band,
  improving robustness to the spectral differences between the training keyboards and
  an unseen typist's keyboard/mic — the exact distribution shift causing held-out subs.

### 3.2 Additive noise (already present — tune)

`add_noise` exists (Gaussian + optional `data/noise/*.wav`). Length-preserving, safe.
- **SNR range**: current `(5, 20) dB`. Widen slightly to `(0, 25) dB`; keep a fraction
  clean. Add real ambient/room clips to `data/noise/`.
- Add a **MUSAN** noise bank (OpenSLR SLR17) — the TODO at `augment.py:148` — for
  realistic babble/HVAC. Safe (additive, same length).

### 3.3 Room impulse response / reverb (add — the biggest missing safe aug)

- Convolve with **RIRs** (OpenSLR SLR28, the TODO at `augment.py:145`) via
  `scipy.signal.fftconvolve(y, rir)[:len(y)]` — **truncate to original length** and
  **align on the direct-path peak** so onset frames don't shift (RIR pre-delay would
  move timing; roll the RIR so its max is at t=0 before convolving).
- **Why**: reverb is the dominant real-world channel effect and a top cause of
  unseen-room substitution errors. Apply `p≈0.5`, mix a handful of small/medium room
  RIRs.
- **Timing caveat**: reverb *smears* energy forward in time (physically real, and the
  labels are the onsets, which are unchanged) — acceptable. Just don't add bulk delay.

### 3.4 Codec / VoIP simulation (already present — keep & extend)

`_voip` in `train_overlap.py` band-limits 300–3800 Hz + coarse quantize (`p=0.5`).
Length-preserving, safe. Extend with an actual codec round-trip (μ-law, or an Opus
encode/decode at 8–16 kbps) for realism if a call-audio threat model matters. Keep
band-limit conservative — keystroke transients have energy up to ~8 kHz, so aggressive
low-pass removes discriminative bands (could *increase* subs); keep cutoff ≥3.8 kHz.

### 3.5 Gain / EQ (already present — keep)

`random_gain` (`±12 dB`) and `style_randomize` (2–4 peaking biquads, `±9 dB`, envelope
γ 0.85–1.15) are the DECKER-style device randomizer — all length-preserving, all safe.
These are doing the heavy lifting for cross-device generalization; keep them ON
(they're at `p=0.8` for style/noise). Consider raising EQ diversity (more filters, or
a random spectral-tilt term) to better cover the 10-keyboard gap.

**Net for §3.** Re-enabling freq-only SpecAugment + adding RIR is the concrete win.
**Est. impact.** −3 to −6 pts on unseen typists. **Effort.** 1–1.5 days.

---

## 4. DATA STRATEGY — RANK #5 (pretraining) + notes

### Is 67k keystrokes / 51 sessions enough?

**Borderline.** For a 36-class per-frame classifier it's workable (that's why we're at
40% not chance), but **26 typists is the real bottleneck** for generalization — the
held-out CER gap is a *typist/keyboard* gap, not a sample-count gap. More *typists* and
*keyboards* would help more than more keystrokes from the same people. Two cheap
multipliers: the 10 isolated-key keyboards (§4.1) and augmentation (§3).

### 4.1 Pretrain on isolated-key data → fine-tune on SKAID — RANK #5

**What.** `data/pool/*.npz` holds isolated single-key recordings across **10
keyboards**. Pretrain the encoder to classify isolated keys (clean per-key acoustic
model), then fine-tune the whole `MtlCRNN` on SKAID continuous typing.

**Why it fixes substitutions.** The core substitution problem is a *weak per-key
acoustic model*. Isolated keys give clean, unambiguous per-key supervision across 10
keyboards — far more keyboard diversity than SKAID alone — teaching the CNN/BiGRU
features that separate H from G *before* the harder continuous/overlap task muddies
them. This is the standard "pretrain the acoustic front-end, then adapt" recipe.

**Implementation (concrete).**
1. **Pretrain**: use `synth_line` (already synthesizes continuous typing *from* the
   per-key banks via `KEYGUARD_BANK=data/pool/<kbd>.npz`) OR build isolated-key frame
   targets directly. Train `MtlCRNN` on a mix of all 10 pool banks with full §3
   augmentation. This is essentially `train_overlap.build_pool` with `root` cycling
   over the 10 npz banks.
2. **Fine-tune**: load those weights, train on SKAID (`KEYGUARD_SKAID=...`) with a
   **lower LR** (`3e-4 → 5e-5`), optionally freeze the CNN for the first ~500 steps.
3. Keep held-out = every 5th SKAID session (unchanged) so the comparison is honest.

**Est. impact.** −3 to −6 pts. **Effort.** 1–2 days (data plumbing + two-phase train).

### 4.2 Self-training / pseudo-labeling

**What.** Run the current model on **unlabeled** continuous-typing audio (or the
non-held-out SKAID with confidence filtering), keep only **high-confidence,
onset-count-consistent** transcriptions as pseudo-labels, retrain.

**Why.** Expands effective data toward the real distribution cheaply. The onset head
(#1) gives a natural confidence/consistency filter: keep chunks where predicted onset
count matches a stable estimate and mean per-frame confidence is high.

**Notes.** Risk of amplifying systematic substitutions (the model reinforces its own
H→G bias). Mitigate: only pseudo-label with **LM-fused + adjacency-checked** decodes
(#1.2/#1.3), and mix ≤50% pseudo-labeled with real. **Est. impact.** −1 to −3 pts.
**Effort.** 1–2 days. Lower priority than #4.1.

### 4.3 Speaker/typist normalization

**What.** Reduce typist/keyboard variability the model must fight:
- **Per-clip feature normalization** — already done (`logmel` subtracts mean/divides
  std). Consider **per-session/CMVN** normalization if session boundaries are known.
- **Typist embedding / FiLM conditioning**: at *train* time only (we don't know the
  typist at test), better to make features *invariant* than conditioned. The repo
  already has domain-invariance machinery (`keyguard/gen_variants/dann.py`,
  `invariant.py`) — a **DANN/gradient-reversal** on typist/keyboard id would push the
  encoder toward typist-invariant features, directly targeting the held-out gap.

**Why.** The held-out failure is distribution shift across typists/keyboards; invariance
training attacks it at the representation level, complementing augmentation (§3).
**Est. impact.** −1 to −3 pts. **Effort.** 1–2 days (reuse existing DANN variant).

### 4.4 Add a space symbol (structural note, medium-term)

Our stream has no space, so words run together and the char LM can't model word
boundaries natively. SKAID likely logs the space key; adding a 38th symbol (space) and
labeling its onsets would let the beam+LM insert boundaries during decoding instead of
post-hoc segmentation (§1.4), a cleaner and higher-accuracy path. Non-trivial (retrain
+ relabel) but strategically the right move if word-level accuracy is the headline
metric. **Effort.** 1–2 days incl. retrain.

---

## Recommended execution order (stacking for max CER drop)

1. **Onset-gated decoding** (#1.1) — 0.5 d, biggest ROI, no retrain.
2. **Label smoothing + focal frame-CE** (#2.1) — 0.5 d, retrain.
3. **KenLM char n-gram beam fusion**, combined with onset gating (#1.2) — 1 d.
4. **Freq-only SpecAugment + RIR/noise/VoIP** (#3) — 1–1.5 d, folds into #2 retrain.
5. **Isolated-key pretrain → SKAID fine-tune** (#4.1) — 1–2 d.
6. **EMA weights now; ensemble + TTA + LLM/dictionary post-correction** (#2.6, #1.4) — 1–2 d.

Then, if word-level readability is the headline: **add a space symbol** (#4.4) and
**adjacency-aware smoothing / DANN invariance** (#2.2 / #4.3).

Measure each lever against the same 586 held-out chunks, report **both** space-stripped
char-CER (`lm_correct._cer`) and word accuracy, and keep the onset-count sanity check as
a guardrail.
