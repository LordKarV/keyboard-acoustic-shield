# Why AI is essential to KeyGuard's defense (and where an LLM is / isn't)

Judging-ready reasoning for "why not just garble the audio?" — written to survive
scrutiny. It separates the claim we can prove from the claim we should not overstate.

## TL;DR

- **Provable & demonstrable:** AI is *essential* to the defense. A random or
  deterministic garble cannot work — not because of engineering polish, but for
  three structural reasons below. The essential AI is a **learned, adaptive,
  adversarially-optimized perturbation network** (the min-max shield `D`).
- **Honest boundary:** the low-level defense is *not* an LLM, and we do **not**
  claim it is. An LLM producing the waveform would be the wrong tool.
- **Where the LLM is essential:** on the **attack** side (Pillar-2 text recovery)
  — which is precisely what makes the defense hard. On defense the LLM is a
  legitimate **strategist/orchestrator**, an enhancement we describe as such.

## Why "just garble it randomly/deterministically" provably fails

The intuition to kill: "take the audio, add noise in the keystroke band, done."

1. **Constrained min-max, not a transform.** The output must satisfy a hard
   perceptual budget: a human, PESQ/STOI, *and* a speech recognizer (Whisper) must
   hear the speech unchanged: `min_D max_A Leakage(A(D(x)))  s.t.  Perceptual(D(x),x) ≤ ε`.
   Keystroke transients and speech overlap in time and in the ~300–3800 Hz
   telephone band. So a garble strong enough to *reliably* destroy keystrokes also
   destroys speech and breaks Whisper — the trivial max-perturbation solution that
   **violates ε**. The useful defense lives in a narrow feasible region that only
   optimization finds; there is no closed form.

2. **The attacker retrains — this is what kills every fixed transform.**
   - A **deterministic** transform is a fixed, known function. A retraining
     attacker simply models it as part of the channel and inverts/ignores it.
     Determinism ⇒ learnable ⇒ defeated.
   - **Fixed-distribution random** noise (e.g. Gaussian in the keystroke band) is,
     from the attacker's view, just data augmentation: it retrains to noise-robust
     features and averages zero-mean noise out. Bounded random noise inside ε does
     not hold leakage down against an adaptive adversary.
   - Only a perturbation that is (a) **optimized against the current attacker's
     decision boundary** (a targeted adversarial example — maximal confusion per
     unit budget, so it fits under ε where random noise cannot) and (b)
     **re-optimized every time the attacker adapts** (the min-max loop) keeps
     leakage at chance under the budget. That is exactly what garble cannot do.

3. **Empirical proof in this repo.** A fixed DSP suppressor (RNNoise-style) moves
   an adaptive attacker 86% → 84% — essentially nothing. The learned adversarial
   shield drives 87% → **2.8% (= chance, 36 classes)** and **re-blinds a retraining
   attacker to chance every round**. That gap *is* the difference between a
   transform and an optimized, adaptive defense. (See README arena results.)

**Conclusion:** the defense is a constrained, adaptive, adversarial optimization.
That is an ML problem with no closed form and no fixed-transform solution — so AI
is essential, provably, and we can show the numbers live.

## The LLM: essential on attack, strategist on defense (stated honestly)

**Attack (LLM is essential, demonstrable).** English carries ~1 bit/char, so a
*weak* per-key acoustic guess + a language model recovers the text. Our attacker
hands its per-key top-k lattice to an LLM (Gemini / our `lm_correct`), which
reconstructs words (e.g. a corrupted `NAAG1MH` → `MEETING`). Remove the LLM and
attack accuracy on real English collapses toward the raw acoustic rate. This is a
real, load-bearing LLM use — and it **raises the defense bar**: the shield must
reduce *mutual information* below what a **language-model-augmented** adversary can
exploit, which a dumb garble cannot guarantee. Random passwords (no language prior)
are the honest hard case we state as such.

**Defense (LLM as strategist — an enhancement, not a necessity).** Two legitimate,
reasoning-shaped jobs a fixed transform cannot do:
- **Arms-race strategist** (`keyguard/arena/proposer.py`): observe the attacker
  across rounds (which keys still leak, how it adapted) and *reason* about the next
  defense strategy / budget allocation, proposing configurations the gradient
  solver then refines — generalizing to unseen attacker types faster than a blind
  restart.
- **Sensitivity-aware policy:** reason about *what* is being typed (a password
  field, a credential, PII vs idle chatter) and decide **when** and **how hard** to
  spend the perceptual budget. "Protect this password hard; ignore idle typing" is
  a reasoning task a deterministic rule handles poorly and a garble not at all.

We do **not** claim the LLM produces the perturbation. Saying so would be false and
judges would see through it.

## Reframe (chosen): the DEFENDER is an LLM agent playing a deception game

The defender — our actual product — earns "LLM-essential" via three jobs a garble
or a plain perturbation-NN cannot do, all under the perceptual budget:

1. **Coherent deception (flagship).** Against an LLM eavesdropper, the optimal use
   of a tiny perceptual budget is not noise but *misdirection*: steer the
   adversary's language model to a plausible, coherent WRONG reconstruction (read
   `HUNTER2` as `DERWENT9`). Generating a convincing counterfeit that fits the
   typing rhythm is inherently LLM generation — language-vs-language. Our arena
   showed noise gets averaged out by an adaptive attacker; a *targeted deception*
   does not, because the attacker's own LM locks onto the coherent false target.
2. **Semantic budget allocation (triage).** Victim-side, the defender can reason
   over the user's own content and protect only what's sensitive (password/card/
   SSN) hard while leaving chatter natural — spending scarce perceptual budget
   precisely. A fixed transform treats every keystroke identically.
3. **Adaptive counter-strategy under black-box.** A real eavesdropper's model is
   unknown → no gradients → no white-box optimization. The defender must reason
   about likely attacks, compose/parameterize defense tools, read leakage feedback,
   and remember (Backboard) which strategies beat which attacker families to
   generalize zero-shot to a new one.

The perturbation math is a TOOL the defender-LLM wields; the intelligence is the
reasoning about *what* false message to plant, *where* to spend budget, and *how*
to counter an unknown adversary. This is implemented as a multi-agent loop
(Attacker-LLM decoder vs Defender-LLM strategist vs Referee) in `keyguard/agents/`.

## If we want the LLM to be genuinely ESSENTIAL to the defense

Design the defense so it cannot work without reasoning — the honest way to earn the
claim:
- **Black-box attacker.** Realistically the eavesdropper's model is unknown, so the
  defense has **no gradients** to optimize against. Then it must *reason*: hypothesize
  the attacker type, propose a perturbation, observe leakage feedback, and reason
  about the next move — an LLM-driven, hypothesis-testing agentic loop where
  gradient descent alone is impossible. This makes reasoning load-bearing.
- **Cross-attacker generalization.** An LLM strategist that carries memory
  (Backboard) of every attacker faced can pick a defense for a *new* attacker
  zero-shot, where per-attacker gradient optimization would start from scratch.

Either is a real build (a few hours) that converts the LLM from "nice" to
"essential." Recommend doing this only if the submission leans on an LLM-defense
claim; otherwise the strongest honest story is: **LLM essential on attack + learned
adversarial min-max essential on defense.**

## Judge Q&A — crisp answers

- *"Why not just garble the audio?"* → It breaks speech (fails the perceptual
  budget) at strengths that work, and a retraining attacker inverts/averages out any
  fixed or fixed-random transform. Only an adversarially-optimized, adaptive
  perturbation holds leakage at chance under the budget — 87%→2.8% vs RNNoise's
  86%→84%.
- *"Is the defense an LLM?"* → No. The defense is a learned adversarial perturbation
  network (min-max). The LLM is essential on the **attack** side (text recovery),
  which is what makes the defense non-trivial; on defense the LLM is a strategist.
- *"Where is AI essential?"* → Attack: LLM recovers text from weak acoustics.
  Defense: adversarial optimization finds the minimal speech-preserving perturbation
  and re-solves as the attacker adapts. Neither has a closed form.
