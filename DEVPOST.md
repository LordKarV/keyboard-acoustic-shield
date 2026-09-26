# Keyguard — Devpost

## Inspiration
Recent papers (Harrison et al. 2023; Okada et al. 2026) showed you can read what
someone types just from the *sound* of their keyboard on a Zoom call — up to 95%
accuracy. Every remote meeting is now a keylogger. We wanted to prove the attack on
our own machines and then actually kill it — in a way that survives an attacker who
*fights back*, and where the AI is load-bearing rather than decorative.

## What it does
- **Attack (two pillars).** *Pillar 1 — per-key acoustics:* plate resonance makes keys
  distinguishable via per-key timing + spectral (plate-resonance) features; our
  CoAtNet-style classifier hits **87.2% top-1** (chance 2.8%) on the public Harrison
  dataset, and **37.8% top-1 / 57.1% top-3** on our own laptop-mic bank. *Pillar 2 — language redundancy:*
  English is ~1 bit/char, so a CTC acoustic model + an LM/LLM correction turns weak,
  ~38%-accurate keys into readable English. Random passwords are the honest hard case.
- **Defense (the centerpiece): a perceptually-constrained adversarial shield.** A
  small causal defender net **D** adds a bounded perturbation to the audio so that a
  **retraining** attacker **A** can't read the keys — while a human, PESQ/STOI, *and*
  Whisper still hear the original speech. Formally `min_D max_A Leakage(A(D(x)))`
  subject to `Perceptual(D(x), x) <= eps`.
- **Dashboard + demo.** A judge types a paragraph + fake password, watches the
  attacker reconstruct it, flips the shield on, and watches the attack collapse while
  the speech stays clear.

## The meaningful-AI story (ML/AI track)
Two places where AI is *essential*, not bolted on:

1. **The adversarially-trained shield D.** The optimal defense depends on the current
   attacker, which retrains — so **no fixed transform is optimal**, and the minimal
   speech-preserving perturbation is defined *through the attacker's neural network*
   (adversarial-example optimization + a psychoacoustic loss). There is no closed
   form. D and A co-train GAN-style: A is the discriminator (can it read keys?), D is
   the generator (make keys unreadable) under a hard STOI/PESQ + Whisper-intact
   constraint. The "arena" *is* this D-vs-A training loop.
2. **The language model on the attack side (Gemini).** Turning a weak acoustic
   stream into readable text is a language-modeling problem. We feed the attacker's
   **per-key top-3 candidate lattice** to **Gemini** (`gemini-flash-latest`), which
   multiplies those weak guesses against English redundancy to reconstruct the typed
   string — the lever that makes the attack real, and the honest reason the defense
   has to exist. This is **verified live**: from a corrupted top-1 stream
   `NAAG1MH` Gemini recovers `MEETING`, and from `OSAAEPTS` it recovers `PASSWORD`,
   using only the top-3 lattice. Crucially, **with the shield on the lattice is
   garbage, so even Gemini can't reconstruct** — in the live demo, a typed `HUNTER2`
   reads at **100% (7/7) with no shield** but collapses to **28.6% (2/7)** with the
   shield on, and Gemini cannot rescue the garbled lattice (`HUNTER2` → `B8NVA42`).
   That `HUNTER2` is a random password is the honest hard case: no language prior,
   so Pillar 2 has nothing to multiply against — and the shield still wins.
   (`keyguard/attack_lm.py`, wired live into the dashboard's `/api/demo`.)

## How we built it
Python + PyTorch (MPS) for the models, librosa for the signal front-end, FastAPI +
vanilla JS/SVG for the dashboard. The DSP shield is STFT magnitude inpainting
conditioned on OS key-event timestamps (timing only, never the key). The adversarial
shield (`keyguard/shield/adversarial.py`) is a learned bounded perturbation trained
against a differentiable attacker with adversarial co-training rounds and a perceptual
regularizer.

## The reckoning that led to the reframe
Our first shield was ~80% deterministic (keystroke removal + residue randomization +
decoys) with an LLM tuning six config knobs in an "arena." We stopped and were honest
with ourselves: you could just garble the keystroke band and it works, so the AI
wasn't doing the real work — an LLM tuning six numbers is "AI-because-AI." The fix was
to make the *constraint* the hard part. "Just garble it" is the trivial
max-perturbation solution that **breaks Whisper and destroys overlapping speech** — it
violates the perceptual constraint. Forcing transparency *and* privacy-robustness at
once is a genuine min-max no rule solves, and that's what makes the neural shield
necessary.

## Novelty, honestly
- **Attack:** a faithful reproduction/extension of prior work (Asonov & Agrawal 2004;
  Harrison et al. 2023 — up to 95% lab / 93% Zoom; Slater ACSAC'19 for overlap CTC), not
  a novel attack. Its job is to be a credible, reproducible threat that justifies the
  defense.
- **Defense:** the closest prior art is **EveGuard** (IEEE S&P 2025, arXiv 2411.10034) —
  an adversarial-audio side-channel *defense*, but for the **vibration** channel
  protecting **spoken speech**. To our knowledge Keyguard is the **first
  perceptually-constrained adversarial shield for the acoustic *keystroke* side-channel**
  (secret = typed text), adapting the adversarial-audio-privacy paradigm (EveGuard
  S&P'25, AntiFake CCS'23) to a channel and secret it hasn't been applied to. We did not
  invent adversarial-audio privacy and do not claim to have "solved" the vulnerability.

## Challenges
- **Signal strength.** Our laptop-mic keys are quiet; the isolated-key attack works
  (37.8% top-1) but the free-typing CTC transcriber, which is pipeline-verified
  (~4% CER on synthetic overlap), collapsed on the weak real signal — it emitted the
  same common letters regardless of input. Bottleneck is the sensor, not the code.
- **The deterministic-vs-AI reckoning** above — the honest realization that reframed
  the whole defense.
- **Tiny transient images.** Aggressive SpecAugment *hurt* the classifier; light
  noise + jitter is what got us to 87.2%.

## What's next
- A stronger/fixed sensor (phone mic Harrison-style, or a contact/accelerometer sensor
  via Arduino) + transfer-learning from Harrison, and a training-collapse fix
  (multi-task onset head), to make free-typing generalize.
- Push the adversarial co-training arena further. It runs on the **dashboard**
  (`/api/arena`): at an inaudible -18 dB budget (chance 2.8%, 36 classes) the clean
  attacker reads Harrison isolated keys at **87.2%**, the deployed shield drives it to
  **2.8% (= chance) every round**, and a retraining attacker fighting a frozen shield
  claws back to ~56% mid-game (**20.6% → 47.8% → 55.6% → 47.2% → 18.3%**) before D
  re-solves and re-blinds it to **2.8%** — a real bouncing arms race D wins each round,
  not a flat curve. Next: report the full leakage-vs-perceptual curve across rounds.
- Real-time BlackHole virtual mic on the victim's side.

## Prizes targeted
One build, multiple entries — the same project, honestly framed for each.

**Main tracks**
- **AI/ML (primary):** AI is load-bearing in both directions — an adversarially-trained
  min-max shield, and a language model (Gemini) that makes the acoustic attack readable.
- **Social Good:** every remote meeting is now a silent keylogger. Keyguard is a
  privacy shield that protects what people type during video calls — passwords,
  medical notes, private messages — from acoustic surveillance, running victim-side
  with zero cloud dependency. Same tech, human-impact framing.
- **Meta (human connection):** we type our most private things while talking to the
  people we trust; acoustic snooping poisons that. Keyguard makes remote conversation
  safe to be candid in again.

**Sponsor prizes (each genuinely used, not bolted on)**
- **Best Use of Gemini:** Gemini is Pillar 2 of the attack — the language model that
  reconstructs readable text from a weak per-key acoustic lattice, proving the threat
  is real and that our shield defeats even an LLM-augmented attacker.
- **Best Use of Backboard:** persistent adversarial memory for the shield's arena —
  Backboard remembers every attacker architecture the shield has faced across
  sessions, so the defense is evaluated against (and never regresses on) the full
  remembered population of adversaries.
- **Best .Tech Domain:** `keyguard.tech` for the live demo + landing page.

## Ethics
Only our own devices, consenting teammates and judges, and fake passwords. The threat
model: the attacker is **remote** (far end of a video call) or a planted nearby device
(a phone on the desk, Harrison-style). The shield runs on the *victim's* side as a
virtual mic, using the victim's **own** OS key-event timestamps (timing only, never
which key) — the asymmetry that lets it be surgical. Nothing here targets third parties.
