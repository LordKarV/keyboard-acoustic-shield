# KeyGuard — The Arms Race ⚔️🦉

**Your keyboard is a microphone.** Every key you press has a distinct sound. We
trained **Ares** ⚔️, an AI that *listens to your typing and reads what you wrote* —
then built **Athena** 🦉, an LLM defender that fights it in a live **arms race**.
That escalating attacker-vs-defender loop is the demo.

- **Ares** (attacker): CNN+BiGRU+CTC keystroke reader + an LLM that reconstructs text.
- **Athena** (defender): an LLM strategist that triages your secret and plants a
  believable *decoy* with an inaudible adversarial perturbation.

**▶ Visual arena:** open `keyguard/web/arena.html` in a browser (it reads
`keyguard/web/arms_race_data.js`, regenerated on every run) — combatant panels, a
move-by-move feed, and an evolution timeline of speech-quality vs. budget.

---

## 60-second demo

```bash
# needs GEMINI_API_KEY (or BACKBOARD_API_KEY) in .env for the LLM agents
KEYGUARD_DEVICE=cuda uv run python -m keyguard.agents.arms_race_demo
```

Real output (one run, `runs/arms_race_demo.json`):

```
USER TYPES : "hey meet me at noon my password is hunter2 thanks"

── ROUND 0 · ATTACK ──────────────────────────────────────────────
🎙  ATTACKER (acoustic) reads : HL WEET SE AT NOLN MY ASSALRS IS HDNTEDW THANK
🧠 ATTACKER-LLM reconstructs : meet me at noon my password is hunter2 thanks   ← LLM turns noise into text

── DEFENDER · LLM PLAN ───────────────────────────────────────────
🛡  DEFENDER-LLM: sensitive = 'HUNTER2' (password)
    DECOY to plant = 'SILVER9'   ← a coherent lie, not noise (LLM-essential)

── ARMS RACE (min-max under a perceptual budget) ─────────────────
round     attacker reads secret span   STOI  budget dB   strategy
clean                        HDNTEDW   1.000     --       (no defense — secret exposed)
    0                      SSIILVVER   0.995    16.0       deceive → SILVER9
    ↳ attacker RETRAINS on the shielded audio → reads 'HUNTER2'  (broke through!)
    1                          FFLCO   0.988    13.0       deceive → FALCON4
    ↳ attacker RETRAINS → reads 'HUNTER2'  (broke through!)
    2                        FALCOON   0.977    10.0       deceive → FALCON4

── VERDICT ───────────────────────────────────────────────────────
TRUE secret : HUNTER2
Attacker's final read of the span : 'FALCOON'  (target decoy 'FALCON4')
Speech quality (STOI): 0.977   (>0.85 = speech intact, change inaudible)
Full shielded transcript: HEY MEET ME AT NOON MY PASSORD IS FALCOON THANKS
```

**The wow:** a retraining attacker reads your password `HUNTER2` *exactly*. With the
shield on, it confidently reads a **fake** password (`SILVER9`, then `FALCON4`) — and
the audio still sounds identical (STOI 0.98). Every time the attacker adapts, the
LLM-defender re-optimizes and re-blinds it. That's the arms race.

---

## What's real (no hand-waving)

**The attacker** (`runs/ctc_rich_ft.pt`) is a CNN+BiGRU+CTC keystroke transcriber
trained on this exact MacBook. Honest, held-out accuracy:

| Test | Letter accuracy |
|---|---|
| **Novel text** the model never saw (phrase-disjoint split) | **~80%** |
| **Familiar-style** phrases (demo scenario) | **~89%** |
| Per-key separability, one key at a time (slow typing) | **72% top-1**, 85% top-3 |

Slow, deliberate typing makes the attack *easier* (no acoustic overlap). It genuinely
reads unseen text: `PASSWORD IDEAS`→`PASSWORD IDEAS`, `THE CODE IS 17305`→`…IS 17305`.

**The defender** is a bounded, span-localized **adversarial perturbation** on the raw
waveform (`keyguard/agents/defense_audio.py:craft`), optimized *through* the attacker's
own network (a differentiable log-mel front-end matching the deployed model to
corr > 0.9999). It runs in two modes:
- `protect` — make the protected keystrokes unreadable (push to blank).
- `deceive` — steer them to a coherent **false secret** (the flagship).

---

## Why AI — and an LLM — is *essential* (not decoration)

The defense is a **constrained min-max**, not a filter:

$$\min_{D}\ \max_{A}\ \text{Leakage}\big(A(D(x))\big)\quad\text{s.t.}\quad \text{Perceptual}\big(D(x),x\big)\le\varepsilon$$

- **"Why not just garble the audio?"** Keystrokes and speech overlap in time and in
  the 300–3800 Hz band, so a garble strong enough to reliably kill keys also destroys
  speech — it violates ε. And a *fixed* transform (deterministic or fixed-random) is
  just learnable channel noise: **the attacker retrains and inverts it.** In-repo, a
  DSP suppressor moves an adaptive attacker 86%→84% (nothing); the adversarial shield
  drives it to chance and *re-blinds it every round*. Only an optimized, adaptive
  perturbation lives inside ε.

- **The LLM is load-bearing on BOTH sides:**
  - **Attacker-LLM** — English carries ~1 bit/char, so a weak per-key acoustic guess
    + a language model recovers the text. Remove the LLM and the read collapses to
    acoustic-rate gibberish. *(That's what makes the defense hard.)*
  - **Defender-LLM (the product)** — three reasoning jobs a transform cannot do:
    1. **Coherent deception** — against an LLM eavesdropper, the best use of a tiny
       budget isn't noise, it's *misdirection*: plant a believable wrong secret
       (`HUNTER2`→`SILVER9`). Generating a convincing counterfeit is inherently
       language generation — and an adaptive attacker's own LM *locks onto* it, so it
       doesn't average out the way noise does.
    2. **Semantic triage** — reason over the user's own text and spend the scarce
       perceptual budget only on what's sensitive (password/PIN/card), not idle chatter.
    3. **Black-box counter-strategy** — a real eavesdropper's model is unknown (no
       gradients), so the defender must *reason*: hypothesize, actuate, read the
       leakage feedback, escalate — and remember which strategies beat which attackers.

---

## Honest boundaries (so it survives scrutiny)

- The perturbation math is a **tool** the LLM wields; we do **not** claim the LLM
  emits the waveform. The intelligence is *what* to protect and *what lie* to plant.
- Each *fixed* shield is beatable by a retraining attacker (shown live as "broke
  through"); the defender's job is to **re-solve every round** and stay ahead under ε.
- Numbers are on one MacBook, one mic; the ~80% novel-text figure is a small held-out
  set (±a few pts) corroborated by 83% training-validation.
- Random passwords with no language prior are the honest hard case; deception is
  strongest exactly against a *language-model* attacker, which is the realistic one.

---

## Architecture at a glance

```
 keystroke audio ──▶ CNN+BiGRU+CTC attacker ──▶ per-key lattice ──▶ Attacker-LLM ──▶ text
                          ▲                                                    │
                          │  gradients (differentiable log-mel)                ▼
        bounded adversarial perturbation  ◀──  Defender-LLM: triage + decoy + escalate
                 (protect / deceive, ‖δ‖ ≤ ε, STOI ≥ 0.85)         (Gemini / Backboard)
```

- Attacker: `keyguard/ctc/` (model, data, training) — weights `runs/ctc_rich_ft.pt`.
- Shield: `keyguard/agents/defense_audio.py` (`craft`), `keyguard/shield/adversarial.py`
  (min-max co-training on isolated keys).
- Agents / LLM: `keyguard/agents/llm.py`, `arms_race_demo.py`, `pipeline.py`.
- Research backing every claim: `docs/research/`, `docs/DEFENSE_AI_JUSTIFICATION.md`.

*Built as an authorized, own-device security demonstration — attack to build the defense.*
