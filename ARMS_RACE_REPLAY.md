# KeyGuard — Arms Race Replay ⚔️🦉

> Move-by-move log of one live match: an acoustic keystroke **attacker** vs an LLM **defender**, on the user's own MacBook keys. Each side *evolves* in response to the other, under a hard perceptual budget (the audio must still sound identical).

**Utterance:** `hey meet me at noon my password is hunter2 thanks`  
**Secret:** `HUNTER2` (password) · **LLM backend:** `gemini`

| | | |
|---|---|---|
| ⚔️ **ARES** (attacker) | CNN+BiGRU+CTC + LLM | reads keystrokes from sound |
| 🦉 **ATHENA** (defender) | Adversarial shield + LLM | plants a believable *decoy* |

---

### Move 0 — ⚔️ ARES: Opening read (no defense)
- **Move:** Transcribe the keystroke audio, then reconstruct with the LLM.
- **Result:** acoustic: `HEY CEE T CE AY NOON MY PASSPLRS IS HUNT EFI2 T HANTS`  →  LLM: “Hey meet me at noon my password is hunter2 thanks”
- **➜ 🔓 secret region exposed**

### Move 1 — 🦉 ATHENA: Triage + deception plan (LLM)
- **Thinks:** The user explicitly stated their plaintext password in the message.
- **Move:** Mark `HUNTER2` sensitive; fabricate a coherent decoy `SHADOW3`.
- **Result:** Plan: steer the attacker to read `SHADOW3` instead of `HUNTER2`.
- **➜ 🎭 lie prepared**

### Move 2 — 🦉 ATHENA: Deploy shield (round 0)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `SHADOW3` (budget 16 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `SHHADDOOW` · STOI **0.993** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 3 — ⚔️ ARES: Evolve — retrain on the shielded audio (round 0)
- **Move:** Fine-tune on the defended clip to learn through this exact shield.
- **Result:** now reads `HUNTER2`
- **➜ 🔓 broke through!**

### Move 4 — 🦉 ATHENA: Escalate (LLM)
- **Thinks:** Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.
- **Move:** Lower SNR budget to 13 dB; new decoy `SHADOW3`→`WINTER4`
- **Result:** re-optimize next round against the adapted attacker.
- **➜ 🔁 counter-move**

### Move 5 — 🦉 ATHENA: Deploy shield (round 1)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `WINTER4` (budget 13 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `WWIINTTEER` · STOI **0.988** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 6 — ⚔️ ARES: Evolve — retrain on the shielded audio (round 1)
- **Move:** Fine-tune on the defended clip to learn through this exact shield.
- **Result:** now reads `HUNTER2`
- **➜ 🔓 broke through!**

### Move 7 — 🦉 ATHENA: Escalate (LLM)
- **Thinks:** Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.
- **Move:** Lower SNR budget to 10 dB; new decoy `WINTER4`→`RANGER4`
- **Result:** re-optimize next round against the adapted attacker.
- **➜ 🔁 counter-move**

### Move 8 — 🦉 ATHENA: Deploy shield (round 2)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `RANGER4` (budget 10 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `RRANGGERR` · STOI **0.983** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 9 — 🏁 OUTCOME: Final state
- **Result:** attacker's last read of the secret span: `RRANGGERR` (aiming for decoy `RANGER4`, true secret `HUNTER2`); speech STOI **0.983**.
- **➜ 🛡️ secret protected**

---

## Evolution timeline

| round | defender plants | attacker reads | speech (STOI) | budget | who's ahead |
|---|---|---|---|---|---|
| 0 | `SHADOW3` | `SHHADDOOW` | 0.993 | 16 dB | 🦉 Athena |
| 0·adapt | *(attacker retrains)* | `HUNTER2` | — | — | 🔓 broke through |
| 1 | `WINTER4` | `WWIINTTEER` | 0.988 | 13 dB | 🦉 Athena |
| 1·adapt | *(attacker retrains)* | `HUNTER2` | — | — | 🔓 broke through |
| 2 | `RANGER4` | `RRANGGERR` | 0.983 | 10 dB | 🦉 Athena |

**Final shielded transcript:** `HEY MEET ME AT NOON MY PASSWORD IS RRANGGERR THANKS`

---

## How to read this

- **Deception, not noise.** The defender doesn't garble — it plants a *coherent false secret* the attacker's own language model believes. Noise gets averaged out by a retraining attacker; a targeted lie does not.
- **The audio never changes audibly** — STOI stays ≈1.0 (1.0 = identical). The perturbation lives under a hard perceptual budget ε.
- **It's a min-max, live.** Each fixed shield is beatable by retraining (you can watch the attacker *break through*); the defender then re-optimizes and re-blinds it. Neither the shield nor the attack is static — they co-evolve.
- **The LLM is the brain on both sides** — reconstructing text from weak acoustics (attack) and reasoning about *what* to protect and *what lie to plant* (defense). A fixed transform can do neither.