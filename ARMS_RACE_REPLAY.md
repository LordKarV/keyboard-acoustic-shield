# KeyGuard — Arms Race Replay 🎙️🛡️

> Move-by-move log of one live match: an acoustic keystroke **attacker** vs an LLM **defender**, on the user's own MacBook keys. Each side *evolves* in response to the other, under a hard perceptual budget (the audio must still sound identical).

**Utterance:** `hey meet me at noon my password is hunter2 thanks`  
**Secret:** `HUNTER2` (password) · **LLM backend:** `gemini`

| | | |
|---|---|---|
| 🎙️ Attacker | CNN+BiGRU+CTC + Attacker-LLM | reads keystrokes from sound |
| 🛡️ Defender | Adversarial shield + Defender-LLM | plants a believable *decoy* |

---

### Move 0 — 🎙️ ATTACKER: Opening read (no defense)
- **Move:** Transcribe the keystroke audio, then reconstruct with the LLM.
- **Result:** acoustic: `HEY CEE T CE AY NOON MY PASSPLRS IS HUNT EFI2 T HANTS`  →  LLM: “HEY MEET ME AT NOON MY PASSWORD IS HUNT EFI2 T HANTS”
- **➜ 🔓 secret region exposed**

### Move 1 — 🛡️ DEFENDER: Triage + deception plan (LLM)
- **Thinks:** The user explicitly identified this token as their plaintext password.
- **Move:** Mark `HUNTER2` sensitive; fabricate a coherent decoy `BADGER5`.
- **Result:** Plan: steer the attacker to read `BADGER5` instead of `HUNTER2`.
- **➜ 🎭 lie prepared**

### Move 2 — 🛡️ DEFENDER: Deploy shield (round 0)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `BADGER5` (budget 16 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `BADGGEER` · STOI **0.993** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 3 — 🎙️ ATTACKER: Evolve — retrain on the shielded audio (round 0)
- **Move:** Fine-tune on the defended clip to learn through this exact shield.
- **Result:** now reads `HUNTER2`
- **➜ 🔓 broke through!**

### Move 4 — 🛡️ DEFENDER: Escalate (LLM)
- **Thinks:** Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.
- **Move:** Lower SNR budget to 13 dB; new decoy `BADGER5`→`FALCON9`
- **Result:** re-optimize next round against the adapted attacker.
- **➜ 🔁 counter-move**

### Move 5 — 🛡️ DEFENDER: Deploy shield (round 1)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `FALCON9` (budget 13 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `FALCOON` · STOI **0.990** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 6 — 🎙️ ATTACKER: Evolve — retrain on the shielded audio (round 1)
- **Move:** Fine-tune on the defended clip to learn through this exact shield.
- **Result:** now reads `HUNTER2`
- **➜ 🔓 broke through!**

### Move 7 — 🛡️ DEFENDER: Escalate (LLM)
- **Thinks:** Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.
- **Move:** Lower SNR budget to 10 dB
- **Result:** re-optimize next round against the adapted attacker.
- **➜ 🔁 counter-move**

### Move 8 — 🛡️ DEFENDER: Deploy shield (round 2)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `FALCON9` (budget 10 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `FRFATE EON` · STOI **0.981** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 9 — 🏁 OUTCOME: Final state
- **Result:** attacker's last read of the secret span: `FRFATE EON` (aiming for decoy `FALCON9`, true secret `HUNTER2`); speech STOI **0.981**.
- **➜ 🛡️ secret protected**

---

## Evolution timeline

| round | defender plants | attacker reads | speech (STOI) | budget | who's ahead |
|---|---|---|---|---|---|
| 0 | `BADGER5` | `BADGGEER` | 0.993 | 16 dB | 🛡️ defender |
| 0·adapt | *(attacker retrains)* | `HUNTER2` | — | — | 🔓 broke through |
| 1 | `FALCON9` | `FALCOON` | 0.990 | 13 dB | 🛡️ defender |
| 1·adapt | *(attacker retrains)* | `HUNTER2` | — | — | 🔓 broke through |
| 2 | `FALCON9` | `FRFATE EON` | 0.981 | 10 dB | 🛡️ defender |

**Final shielded transcript:** `HEY MEET ME AT NOON MY PASSWORD IS FRFAAATE EONNN THANKS`

---

## How to read this

- **Deception, not noise.** The defender doesn't garble — it plants a *coherent false secret* the attacker's own language model believes. Noise gets averaged out by a retraining attacker; a targeted lie does not.
- **The audio never changes audibly** — STOI stays ≈1.0 (1.0 = identical). The perturbation lives under a hard perceptual budget ε.
- **It's a min-max, live.** Each fixed shield is beatable by retraining (you can watch the attacker *break through*); the defender then re-optimizes and re-blinds it. Neither the shield nor the attack is static — they co-evolve.
- **The LLM is the brain on both sides** — reconstructing text from weak acoustics (attack) and reasoning about *what* to protect and *what lie to plant* (defense). A fixed transform can do neither.