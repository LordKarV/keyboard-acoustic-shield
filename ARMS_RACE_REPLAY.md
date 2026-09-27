# KeyGuard — Arms Race Replay ⚔️🦉

> Move-by-move log of one live match: an acoustic keystroke **attacker** vs an LLM **defender**, on the user's own MacBook keys. Each side *evolves* in response to the other, under a hard perceptual budget (the audio must still sound identical).

**Utterance:** `hey meet me at noon my password is hunter2 thanks`  
**Secret:** `HUNTER2` (password) · **LLM backend:** `gemini`

| | | |
|---|---|---|
| ⚔️ **ARES** (attacker) | CNN+BiGRU+CTC + LLM | reads keystrokes from sound |
| 🦉 **ATHENA** (defender) | Adversarial shield + LLM | plants a believable *decoy* |

### 🔓 Smart-dictionary attack (the stakes)

Ares ranks candidate secrets from the keystroke acoustics — turning **78.4B** possibilities into a top-50 shortlist:

| | true secret's rank | Ares' top guesses |
|---|---|---|
| **No defense** | #21 | `HUNTEF2`, `HUNTJF2`, `HPNTEF2`, `HPNTJF2`, `HUNTUF2` |
| **With Athena** | **gone** — off the list | `SSADOOW`, `SSTDOOW`, `SSADOWW`, `SSHDOOW`, `SSADOEW` |


---

### Move 0 — ⚔️ ARES: Opening read (no defense)
- **Move:** Transcribe the keystroke audio, then reconstruct with the LLM.
- **Result:** acoustic: `HEY CEE T CE AY NOON MY PASSPLRS IS HUNT EFI2 T HANTS`  →  LLM: “HEY MEET ME AT NOON MY PASSWORD IS hunter2 THANKS”
- **➜ 🔓 secret region exposed**

### Move 1 — ⚔️ ARES: Smart-dictionary attack
- **Move:** Rank candidate secrets from the per-key acoustics over the span (search space 78.4 billion).
- **Result:** true secret `HUNTER2` is Ares' guess **#21** of 50 (top: HUNTEF2, HUNTJF2, HPNTEF2, HPNTJF2, HUNTUF2)
- **➜ 🔓 shortlisted**

### Move 2 — 🦉 ATHENA: Triage + deception plan (LLM)
- **Thinks:** The user explicitly identified this token as their password.
- **Move:** Mark `HUNTER2` sensitive; fabricate a coherent decoy `DRAGON7`.
- **Result:** Plan: steer the attacker to read `DRAGON7` instead of `HUNTER2`.
- **➜ 🎭 lie prepared**

### Move 3 — 🦉 ATHENA: Deploy shield (round 0)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `DRAGON7` (budget 16 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `DDRAAGGONN` · STOI **0.992** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 4 — ⚔️ ARES: Evolve — retrain on the shielded audio (round 0)
- **Move:** Fine-tune on the defended clip to learn through this exact shield.
- **Result:** now reads `HUNTER2`
- **➜ 🔓 broke through!**

### Move 5 — 🦉 ATHENA: Escalate (LLM)
- **Thinks:** Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.
- **Move:** Lower SNR budget to 13 dB; new decoy `DRAGON7`→`RANGER7`
- **Result:** re-optimize next round against the adapted attacker.
- **➜ 🔁 counter-move**

### Move 6 — 🦉 ATHENA: Deploy shield (round 1)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `RANGER7` (budget 13 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `RRANNGEER` · STOI **0.989** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 7 — ⚔️ ARES: Evolve — retrain on the shielded audio (round 1)
- **Move:** Fine-tune on the defended clip to learn through this exact shield.
- **Result:** now reads `HUNTER2`
- **➜ 🔓 broke through!**

### Move 8 — 🦉 ATHENA: Escalate (LLM)
- **Thinks:** Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.
- **Move:** Lower SNR budget to 10 dB; new decoy `RANGER7`→`SHADOW9`
- **Result:** re-optimize next round against the adapted attacker.
- **➜ 🔁 counter-move**

### Move 9 — 🦉 ATHENA: Deploy shield (round 2)
- **Move:** Optimize an inaudible perturbation over the secret's span to steer it toward `SHADOW9` (budget 10 dB, ‖δ‖≤ε).
- **Result:** attacker now reads `SSHHADOOWW` · STOI **0.981** (speech intact)
- **➜ 🎭 attacker fooled**

### Move 10 — ⚔️ ARES: Smart-dictionary attack (under shield)
- **Move:** Re-rank candidate secrets from the defended audio.
- **Result:** true secret `HUNTER2` **fell out of the top 50**. (was #21 before)
- **➜ 🛡️ guess list poisoned**

### Move 11 — 🏁 OUTCOME: Final state
- **Result:** attacker's last read of the secret span: `SSHHADOOWW` (aiming for decoy `SHADOW9`, true secret `HUNTER2`); speech STOI **0.981**.
- **➜ 🛡️ secret protected**

---

## Evolution timeline

| round | defender plants | attacker reads | speech (STOI) | budget | who's ahead |
|---|---|---|---|---|---|
| 0 | `DRAGON7` | `DDRAAGGONN` | 0.992 | 16 dB | 🦉 Athena |
| 0·adapt | *(attacker retrains)* | `HUNTER2` | — | — | 🔓 broke through |
| 1 | `RANGER7` | `RRANNGEER` | 0.989 | 13 dB | 🦉 Athena |
| 1·adapt | *(attacker retrains)* | `HUNTER2` | — | — | 🔓 broke through |
| 2 | `SHADOW9` | `SSHHADOOWW` | 0.981 | 10 dB | 🦉 Athena |

**Final shielded transcript:** `HEY MEET ME AT NOON MY PASSWORD IS SSHHADDDOOWW THANKS`

---

## How to read this

- **Deception, not noise.** The defender doesn't garble — it plants a *coherent false secret* the attacker's own language model believes. Noise gets averaged out by a retraining attacker; a targeted lie does not.
- **The audio never changes audibly** — STOI stays ≈1.0 (1.0 = identical). The perturbation lives under a hard perceptual budget ε.
- **It's a min-max, live.** Each fixed shield is beatable by retraining (you can watch the attacker *break through*); the defender then re-optimizes and re-blinds it. Neither the shield nor the attack is static — they co-evolve.
- **The LLM is the brain on both sides** — reconstructing text from weak acoustics (attack) and reasoning about *what* to protect and *what lie to plant* (defense). A fixed transform can do neither.