# Reading the target user's REAL fast typing — a ranked adaptation recipe

Authorized defensive research (HackGT "KeyGuard"): build the strongest acoustic
keystroke transcriber for the *target user's own keyboard* so the adversarial shield
has a real adversary to defeat. This doc gives a concrete, ranked, implement-now
recipe with hyperparameters, grounded in what we already measured.

Scope: our CRNN (CNN -> BiGRU -> CTC + onset head, 8 ms frames = `HOP=128`,
`N_MELS=64`, 36 classes + blank) in `keyguard/ctc/model.py` /
`keyguard/ctc/train_overlap.py`. Two existing checkpoints:

- **SKAID base** (`runs/ctc_skaid_crnn.pt`): trained on REAL continuous typing from
  26 people. ~37-41% CER on SKAID held-out. **~88-93% CER (chance) zero-shot on the
  target keyboard** — cross-keyboard acoustic gap.
- **synth-from-bank**: trained on overlap-added isolated clips of the target's own
  keys. ~5% CER on the synthetic distribution, **~88-101% CER on the target's REAL
  typing** — sim-to-real gap.

---

## TL;DR — the single highest-leverage action

**Fine-tune the SKAID base (not the synth model) onto the target keyboard with
AdaBN + a small adapted head, using the ~200 real keys heavily augmented, and keep a
SKAID replay co-batch to prevent forgetting.**

Rationale in one line: **SKAID already solved the hard part** — it learned the
temporal structure of *real* continuous typing (coarticulation, press/release,
rollover, digraph timing), which is exactly what the synth-from-bank model never
saw. The synth model has the opposite deficit. So the residual gap on SKAID is a
*domain shift in the acoustic front-end* (plate resonance + mic + room), and domain
shift over an already-correct sequence model is precisely what a few-hundred-sample
BN-recalibration + low-rank head adaptation can close — whereas no amount of
synth-overlap fixes a model that never learned real dynamics. Start from real,
adapt the acoustics. Do **not** keep polishing the synth-from-bank path as the base.

Second-highest-leverage action (do it in parallel, it is cheap and it is the real
bottleneck long-term): **collect ~15-20 more minutes of labeled real typing** (see
§5). 200 keys is enough to *adapt* but not enough to *validate*; the CER number you
get from 200 keys is high-variance.

---

## Why each failure happens (diagnosis drives the fixes)

### Why overlap-add of isolated clips fails on real continuous typing (Q1)

Overlap-add of isolated clips is acoustically linear-correct (sound adds linearly)
but *dynamically wrong*. The distribution mismatch has five concrete sources:

1. **Press vs release are two events, not one.** A real keystroke is a *push*
   transient AND a *release* transient ~80-120 ms later, with different spectra and
   levels. Isolated clips recorded at a measured pace typically capture a clean
   push+release pair with a long silent tail. In fast typing the *release of key N*
   overlaps the *push of key N+1*, and sometimes the release is clipped by the next
   key entirely (key rollover — the user is already pushing the next key before
   releasing the last). Overlap-add of whole clips reproduces neither the truncated
   releases nor the push/release *interleaving*.

2. **Coarticulation / mechanical coupling.** On a real board, holding one key
   changes the plate/chassis resonance for the neighbouring press (the plate is
   pre-loaded, finger damping differs). Independent clips are recorded from rest, so
   they lack this state-dependent spectral coloration.

3. **Digraph timing distribution is wrong.** Our synth used a broad WPM/lognormal
   gap model (`data.synth_line`, `wpm=(150,560)`). Real inter-key intervals are
   strongly **digraph-dependent** (e.g. `th`, `er`, `in` are fast; same-finger and
   cross-hand-return digraphs are slow) and bimodal. A model trained on the wrong gap
   distribution keys off the wrong evidence spacing.

4. **Mic/room channel.** Isolated clips and the real fast session may share the mic,
   but level (AGC pumping under continuous typing), room reverb tail overlapping the
   next click, and mic proximity drift all differ. Our synth had no reverb/RIR and
   idealized levels.

5. **Level/energy normalization.** Continuous typing has near-constant broadband
   energy; the network's BatchNorm/instance stats and any AGC settle to a different
   operating point than the sparse isolated-clip stream.

**Concrete techniques that close the sim-to-real gap** (in rough order of payoff):

- **Split each isolated clip into a push segment and a release segment** and
  overlap-add them *independently* on the timeline, so releases can be truncated /
  interleaved / dropped under rollover. This is the biggest single realism fix and is
  cheap given we already have 36x~34 clips. (Detect the two transients per clip with
  the same onset detector in `keyguard/ctc/segment.py`; take push = onset -> onset+35 ms,
  release = the second transient.)
- **Drive the synth with REAL digraph timing distributions extracted from SKAID
  keylogs**, not a generic lognormal. Fit inter-key-interval distributions per digraph
  (or per key-class-pair) from `data/continuous/skaid/labels.jsonl` and sample gaps
  from those. Add per-session rollover probability (fraction of gaps < release length).
- **Convolve with a small bank of room impulse responses** (a few real RIRs or a
  synthetic image-method set, RT60 ~100-400 ms) so the reverb tail of one click
  realistically bleeds into the next.
- **Inject real noise/babble** — talking, coughing, HVAC — at SNR 0-20 dB. We have
  the noisy target session; also pull babble from the SKAID audio itself and any
  ambient recording. This is *required*: one of the two target sessions has
  talking+coughing, and the model must be robust to exactly that.
- **Codec/VoIP band-limit** (already in `_voip`, 300-3800 Hz + coarse quantize),
  gain and EQ jitter — timing-safe channel augmentations.
- **Mixup of real + synth** at the waveform or feature level (see §4). This is the
  single most effective "glue" between the two distributions.

Even with all of this, expect synth-from-bank to remain a *support* signal, not the
base. Its job is to add target-keyboard acoustic coverage of rare keys/digraphs, not
to be the primary training distribution.

### Why zero-shot SKAID collapses on the target keyboard

Per `REALWORLD.md`: per-key acoustics are device-specific (plate resonance
geometry, mic, TDoA) and 26-keyboard diversity is nowhere near enough to learn a
keyboard-invariant representation (LOKO mean top-1 = 2.1% = chance). The sequence
model is fine; the *front-end feature-to-key mapping* is mis-calibrated for the new
rig. That is a classic covariate shift — and BN stats + a light head re-fit are the
textbook fix.

---

## The ranked recipe (implement top to bottom)

### Step 0 — Fix the labels first: robust keylog->audio alignment (Q3)

Everything downstream needs the 200 keys correctly placed on the audio timeline.
Reuse and harden the existing aligner (`keyguard/continuous_convert.py`:
`_best_offset_ms` brute-force offset search + nearest-onset snap, `MIN_HIT_RATIO`
gate).

- **Estimate ONE global mic-latency offset per session** by brute-force scoring
  candidate offsets against detected onsets (already implemented). Do the coarse
  sweep (±full range) then a fine sweep, maximizing onset hits within `HIT_TOL_MS`.
- **Under talking/coughing noise, do NOT trust every detected onset.** Restrict the
  onset detector to the keystroke band and use *spectral-flux with a high-frequency
  emphasis* (key clicks are broadband/HF-rich; speech energy is <4 kHz and coughs are
  low-mid). Practically: band-pass 2-8 kHz before `onset_strength` so speech/cough
  onsets don't dominate peak-picking. This is the key robustness fix for the noisy
  session.
- **Snap each keylog press to the nearest detected onset within ~40-60 ms**
  (`SNAP_TOL_S`); if none is within tolerance, keep the offset-mapped time but mark
  the frame as *low-confidence* (see below). Do not drop the key — 200 keys is too
  precious.
- **Quality-gate per session, not per key** (`MIN_HIT_RATIO`, currently 0.55).
  For the clean session you should see a high hit ratio; for the noisy session
  expect lower — keep it but weight its uncertain frames down.

**Recommended loss: dense class-weighted frame-CE (keep it), with CTC as a small
auxiliary — NOT CTC-primary.**

- Frame-CE is what made the SKAID model actually read real typing (per
  `RESULTS_gpu_skaid.md` §"What made it work"): CTC-alone collapsed to the
  English-letter/blank prior at ~chance. Keep frame-CE as primary
  (`LAMBDA_FRAME=1.0`, `BLANK_W=0.05`, `FRAME_WIDTH=1`).
- Frame-CE *needs* the onset alignment above — that is the tradeoff for beating
  collapse, and it is worth it.
- **Widen the labeled window and soften labels on low-confidence frames.** Set
  `FRAME_WIDTH=2` (±2 frames = ±16 ms) for the real target data so a ~1-frame
  alignment error still lands supervision on the true key, and apply
  `LABEL_SMOOTH=0.1` (already supported) to the target frames. For frames the aligner
  marked low-confidence (no onset within tolerance), either mask them from the
  frame-CE loss or down-weight them 4-8x.
- **Add a small CTC term (`LAMBDA_CTC=0.2`) as a timing-robust auxiliary.** CTC is
  alignment-free and forgives the residual offset error that frame-CE cannot; running
  both gives you frame-CE's anti-collapse gradient plus CTC's tolerance to
  mis-alignment. Keep the onset-BCE head on (`LAMBDA_ONSET=0.5`).

Verdict on Q3: **frame-CE-primary + small CTC auxiliary + HF-emphasized onset
alignment**. Do not switch to CTC-only (it collapses here) and do not rely on
frame-CE alone under noisy alignment (widen window, add CTC, mask low-confidence).

### Step 1 — AdaBN: recalibrate BatchNorm stats to the target keyboard (do this FIRST, it is nearly free)

Before any gradient step, run the SKAID base **forward-only in train mode over ALL
target audio** (both sessions + augmented copies) so every BatchNorm layer's running
mean/var re-estimates on target-domain features. This alone typically recovers a large
chunk of a pure covariate shift and costs no labels and no overfitting risk.

- Set all BN layers to `.train()`, freeze all weights (`requires_grad=False`),
  forward ~200-500 augmented target clips, then snapshot the updated running stats.
- Measure CER right here. This is your "free" baseline before fine-tuning.

### Step 2 — Freeze/adapt schedule for few-shot fine-tuning on ~200 keys (Q2)

The governing constraint: **200 keys can overfit a 2-layer BiGRU + heads instantly.**
Adapt as few parameters as possible, closest to the domain-specific part.

Recommended layer policy (ranked by safety):

1. **Freeze the CNN feature extractor entirely** (the `self.cnn` conv stack). It
   learned generic transient/spectral filters from 26 keyboards; those transfer. Only
   its BN stats update (Step 1 / AdaBN). *Do not* fine-tune conv weights on 200 keys.
2. **Adapt the output head + onset head fully** (`self.head`, `self.onset_head`) —
   these are tiny (linear) and are exactly the keyboard-specific key<->feature map.
3. **Adapt the BiGRU with LoRA / adapters, NOT full fine-tuning.** The BiGRU is the
   large parameter block and the one most prone to catastrophic overfit/forgetting on
   200 keys. Two acceptable options:
   - **LoRA on the GRU input/recurrent projections** (rank r=4-8, alpha=r,
     dropout 0.1): adds ~1-2% params, keeps the pretrained dynamics, adapts cheaply.
   - If LoRA on GRU is awkward to wire, **insert a small bottleneck adapter
     (down 2*hidden->32 -> ReLU -> up ->, residual) between the BiGRU and the head**
     and train only the adapter + heads. This is the lowest-risk option to implement.
   - Full BiGRU fine-tuning is the *fallback only if* you have the SKAID replay
     co-batch on (Step 3) and a very low LR — otherwise it forgets real-typing
     structure and regresses toward the synth failure mode.

**Hyperparameters (concrete):**

- Optimizer: `AdamW`, `weight_decay=1e-4` (higher than the 1e-5 used for from-scratch
  — we want strong regularization on tiny data).
- LR: **head/adapter/LoRA = 3e-4; BiGRU (if unfrozen) = 3e-5** (10x lower);
  CNN = 0. Discriminative LRs.
- Steps: **300-800**, not thousands. With replay co-batch, cap at ~800.
- Batch: 16 (per `KEYGUARD_BATCH`), of which a fixed fraction is SKAID replay (§3).
- Warmup 50 steps, cosine decay after (mirror the existing scheduler, shorter).
- Dropout: keep GRU dropout 0.2; add 0.1-0.2 on the adapter.
- **Early stopping on a held-out real target slice** (see §4/§5): stop when held-out
  CER stops improving for ~5 evals (eval every 25 steps). This is essential — with 200
  keys the model will start memorizing quickly.
- Gradient clip 1.0 (already present).

Ordering: **AdaBN (Step 1) -> then this fine-tune.** Consider a two-phase fine-tune:
phase A trains heads only (~150 steps) to reseat the classifier on the AdaBN'd
features; phase B unfreezes the LoRA/adapter (~300-500 steps).

### Step 3 — Anti-forgetting: SKAID replay co-batch

Every training batch = **~50% target real (augmented) + ~50% SKAID real**. Keep the
SKAID examples in the loss (frame-CE) so the model cannot drift off the real-typing
manifold while chasing 200 target keys. This is the regularizer that lets you adapt
the BiGRU at all. Weight target vs SKAID loss ~1:1 initially; you can up-weight target
to 2:1 late in training once held-out CER is improving.

### Step 4 — Data expansion & the training curriculum (Q4)

You have three sources. Combine them as a **curriculum with a shifting mixture**, not
a static pot:

| Source | Role | Realism | Note |
|---|---|---|---|
| SKAID real | anchors real-typing *dynamics* | real dynamics, wrong keyboard | replay, never remove |
| synth-from-bank (improved, §Q1) | adds target-*keyboard acoustic* coverage | right keyboard, approx dynamics | rebuild with push/release split + SKAID digraph timing + RIR + noise |
| target real (200 keys) | the ground truth to hit | real + right keyboard | augment heavily, hold out a slice |

**Mixing schedule (curriculum):**

- **Warm phase (first ~40% of steps):** mixture ~ {SKAID 40%, improved-synth 40%,
  target-real 20%}. Lets the improved synth pull the front-end toward target acoustics
  while SKAID holds dynamics.
- **Focus phase (last ~60%):** ramp target-real up to ~40-50% and synth down to
  ~10-20%, SKAID steady ~40%. Anneal the mixture linearly.
- Rationale: synth is a *scaffold* for target-keyboard acoustics; as adaptation
  proceeds you trust the real target more and lean on synth less.

**Label-preserving (timing-safe) augmentation on the 200 real keys** — every one
of these preserves onset times, so frame labels stay valid:

- **Gain** ±6 dB; **EQ / random biquad** tilt; **additive noise** (Gaussian + real
  babble/cough from the noisy session and SKAID) at SNR 0-20 dB; **reverb** via short
  RIR convolution (careful: reverb smears onsets slightly — keep RT60 < 200 ms for the
  label-preserving set, or widen `FRAME_WIDTH`); **codec/VoIP** band-limit (`_voip`).
- **SpecAugment: FREQUENCY masking ONLY.** Time masking/warping desyncs frame labels
  — this exact bug caused a collapse before (`RESULTS_gpu_skaid.md`: "SpecAugment
  time-shifted the mel but not the per-frame labels"). Freq masks (2 masks, width
  ≤ 12 mels) are label-safe and help generalization.
- **Mixup** (α≈0.2-0.4) between real and synth clips, mixing both audio and the
  frame-label targets. This is the strongest cross-distribution regularizer and
  directly attacks the sim-to-real gap.
- Do NOT time-stretch or pitch-shift the labeled real data unless you re-derive the
  onset times (time-stretch changes gaps -> must rescale labels).

Effective multiplier: 200 keys x (gain x EQ x noise x reverb x codec) can yield
thousands of *distinct* label-preserving views; combined with SKAID replay + synth
scaffold this is enough to *adapt* (not to train from scratch).

### Step 5 — Decode & the language-model layer

The acoustic model only needs to produce a decent top-k lattice; Pillar 2 (LM/LLM
rescoring) recovers English. Keep the existing greedy + n-gram / `lm_correct.py`
pipeline; feed a **top-3 candidate lattice** to the LLM rescorer. For random-password
targets there is no LM prior — that is the honest boundary and is exactly where the
shield wins.

---

## How much MORE real data is needed, and what to prioritize (Q5)

Honest expectation with **~200 keys**:

- After AdaBN + head/adapter fine-tune + SKAID replay + improved synth scaffold, a
  **plausible landing zone is ~45-65% CER** on the target's real fast typing — a large
  improvement over the ~88-93% zero-shot collapse, but **not** yet the <20% CER goal.
  The LM/LLM layer then recovers readable English words from a 50-60% CER stream (as
  already demonstrated: `HOPEEYOUSWEEKSOWSFTTTOAGGO -> HOPE YOU WEEK SOFT TO AGO`), so
  *readable* is achievable well before *20% raw CER*.
- Be honest that the 200-key CER number itself is high-variance — you are validating
  on a held-out slice of ~40-80 keys. Report a range, not a point.

**Getting to <20% real CER realistically needs more real target data.** Analogues:
SKAID reached ~37-41% with ~67k keys across 26 people; our own working isolated
attack numbers needed ~20-25 presses/key. For a single target keyboard, target:

- **Minimum useful: ~1,000-2,000 real keystrokes** (~5-10 min of continuous typing)
  to move from "adapting" to "training the head/adapter properly" — expect this to get
  you meaningfully under ~40% CER.
- **Good (<20% CER plausible): ~5,000-10,000 keystrokes** (~30-60 min) of real
  continuous typing on the target rig, spanning the conditions you'll test in.

**Prioritize collecting (in order):**

1. **More clean fast continuous typing on the exact target rig + mic** — same mic,
   same distance, same room. Volume first.
2. **Coverage of the noisy condition** (talking/coughing) — since a real attack and
   the demo happen in noise, labeled noisy data is worth more than extra clean data
   past the first few minutes.
3. **Digraph/rare-key coverage** — make sure numbers and low-frequency letters
   appear enough times (type varied text, not just prose).
4. **A few RIRs / ambient noise recordings** from the actual room, to make the synth
   scaffold and augmentation match reality.
5. Keep mic distance/type fixed between banking and testing — `REALWORLD.md` notes
   the mic matters more than the keyboard.

---

## Summary — ranked action list

1. **(Highest leverage) Adapt the SKAID real base, not the synth model.** AdaBN ->
   freeze CNN -> LoRA/adapter on BiGRU + full head -> 300-800 steps, discriminative
   LRs (head 3e-4, BiGRU 3e-5), wd 1e-4, early stop on held-out real.
2. **Fix alignment** with HF-emphasized onset detection; keep **frame-CE primary +
   CTC auxiliary (0.2) + onset-BCE (0.5)**; `FRAME_WIDTH=2`, `LABEL_SMOOTH=0.1`,
   down-weight low-confidence frames.
3. **SKAID replay co-batch (~50%)** to prevent forgetting.
4. **Improve the synth scaffold** (push/release split, SKAID digraph timing, RIR,
   noise/babble, mixup) and use it as a *scaffold* in a shifting curriculum, not the
   base.
5. **Collect ~1-2k more real keys minimum (5-10k for <20% CER)**, prioritizing the
   noisy condition and rare keys.

**Single highest-leverage action: start from the SKAID real base and close the
acoustic domain shift with AdaBN + low-rank head/BiGRU adaptation on the augmented
200 keys with SKAID replay — because SKAID already learned real-typing dynamics and
only the keyboard acoustics need fixing, which is what few-shot adaptation does well.**
The synth-from-bank path has the harder, unsolved deficit (it never saw real
dynamics) and should be demoted to a scaffold.
