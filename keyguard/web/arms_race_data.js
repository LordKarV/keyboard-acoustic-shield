window.ARMS_RACE = {
  "line": "hey meet me at noon my password is hunter2 thanks",
  "secret": "HUNTER2",
  "decoy": "SHADOW3",
  "kind": "password",
  "attacker": "Ares",
  "defender": "Athena",
  "backend": "gemini",
  "clean_span_read": "HUNT EFI2",
  "rounds": [
    {
      "round": 0,
      "mode": "deceive",
      "span_read": "SSSHHAAADDOOWWW",
      "reads_true_secret": false,
      "stoi": 0.9928792325866419,
      "snr_db": 16.0,
      "decoy": "SHADOW3",
      "after_retrain_read": "HUNTER2",
      "after_retrain_reads_true": true
    },
    {
      "round": 1,
      "mode": "deceive",
      "span_read": "WWIINNNTTEER",
      "reads_true_secret": false,
      "stoi": 0.9880051419866269,
      "snr_db": 13.0,
      "decoy": "WINTER4",
      "after_retrain_read": "HUNTER2",
      "after_retrain_reads_true": true
    },
    {
      "round": 2,
      "mode": "deceive",
      "span_read": "RRANGGERR",
      "reads_true_secret": false,
      "stoi": 0.9828523498463666,
      "snr_db": 9.999999046325684,
      "decoy": "RANGER4"
    }
  ],
  "moves": [
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Opening read (no defense)",
      "reasoning": "",
      "action": "Transcribe the keystroke audio, then reconstruct with the LLM.",
      "result": "acoustic: `HEY CEE T CE AY NOON MY PASSPLRS IS HUNT EFI2 T HANTS`  \u2192  LLM: \u201cHey meet me at noon my password is hunter2 thanks\u201d",
      "tag": "\ud83d\udd13 secret region exposed"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Triage + deception plan (LLM)",
      "reasoning": "The user explicitly stated their plaintext password in the message.",
      "action": "Mark `HUNTER2` sensitive; fabricate a coherent decoy `SHADOW3`.",
      "result": "Plan: steer the attacker to read `SHADOW3` instead of `HUNTER2`.",
      "tag": "\ud83c\udfad lie prepared"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Deploy shield (round 0)",
      "reasoning": "",
      "action": "Optimize an inaudible perturbation over the secret's span to steer it toward `SHADOW3` (budget 16 dB, \u2016\u03b4\u2016\u2264\u03b5).",
      "result": "attacker now reads `SHHADDOOW` \u00b7 STOI **0.993** (speech intact)",
      "tag": "\ud83c\udfad attacker fooled"
    },
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Evolve \u2014 retrain on the shielded audio (round 0)",
      "reasoning": "",
      "action": "Fine-tune on the defended clip to learn through this exact shield.",
      "result": "now reads `HUNTER2`",
      "tag": "\ud83d\udd13 broke through!"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Escalate (LLM)",
      "reasoning": "Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.",
      "action": "Lower SNR budget to 13 dB; new decoy `SHADOW3`\u2192`WINTER4`",
      "result": "re-optimize next round against the adapted attacker.",
      "tag": "\ud83d\udd01 counter-move"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Deploy shield (round 1)",
      "reasoning": "",
      "action": "Optimize an inaudible perturbation over the secret's span to steer it toward `WINTER4` (budget 13 dB, \u2016\u03b4\u2016\u2264\u03b5).",
      "result": "attacker now reads `WWIINTTEER` \u00b7 STOI **0.988** (speech intact)",
      "tag": "\ud83c\udfad attacker fooled"
    },
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Evolve \u2014 retrain on the shielded audio (round 1)",
      "reasoning": "",
      "action": "Fine-tune on the defended clip to learn through this exact shield.",
      "result": "now reads `HUNTER2`",
      "tag": "\ud83d\udd13 broke through!"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Escalate (LLM)",
      "reasoning": "Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.",
      "action": "Lower SNR budget to 10 dB; new decoy `WINTER4`\u2192`RANGER4`",
      "result": "re-optimize next round against the adapted attacker.",
      "tag": "\ud83d\udd01 counter-move"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Deploy shield (round 2)",
      "reasoning": "",
      "action": "Optimize an inaudible perturbation over the secret's span to steer it toward `RANGER4` (budget 10 dB, \u2016\u03b4\u2016\u2264\u03b5).",
      "result": "attacker now reads `RRANGGERR` \u00b7 STOI **0.983** (speech intact)",
      "tag": "\ud83c\udfad attacker fooled"
    },
    {
      "agent": "\ud83c\udfc1 OUTCOME",
      "title": "Final state",
      "reasoning": "",
      "action": "",
      "result": "attacker's last read of the secret span: `RRANGGERR` (aiming for decoy `RANGER4`, true secret `HUNTER2`); speech STOI **0.983**.",
      "tag": "\ud83d\udee1\ufe0f secret protected"
    }
  ]
};
