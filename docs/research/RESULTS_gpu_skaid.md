# GPU box results — overlap attacker on REAL continuous typing (SKAID)

Run on the CUDA GPU laptop (RTX 5050, Blackwell/sm_120, torch 2.11+cu128).

## Headline

The overlap keystroke transcriber now **reads real, continuous free typing** from
audio. Evaluated on **586 held-out chunks from unseen typists** (SKAID, 26 people,
~67k keystrokes; every 5th session held out):

| Model | Real held-out MEAN CER | vs chance |
|-------|------------------------|-----------|
| `runs/ctc_skaid_crnn.pt` (CRNN + dense frame-CE) | **~40.7%** (median 37.5%) | chance ≈ 95–100% |

Legible transcriptions of real typing (raw greedy, unseen typist):

```
REF : OVERTHEWEEKENDITOOKSOME     RAW : OVERTEWECKENDHITOOKSRBE
REF : HOPEYOURWEEKISOFFTOAGO      RAW : GOPEYPOUSWEEKIOSSTOAGGO
REF : MAYBESOMETHINGWORTHAQUICKREAD  RAW : MAABSSSMARHINWOTAWWUIECMRADA
```

An LM/spell-correction pass (`keyguard/ctc/lm_correct.py`, `correct()`) re-inserts
word boundaries and fixes English words (Pillar-2 layer): e.g.
`HOPEEYOUSWEEKSOWSFTTTOAGGO` → `HOPE YOU WEEK SOFT TO AGO`.

## What made it work (the project had been stuck at ~chance / collapse)

1. **Real data pipeline (SKAID).** `keyguard/continuous_convert.py` fixes for this box:
   Windows-safe ffmpeg temp handling, Excel-scientific-notation timestamp parse, and
   a per-session onset-alignment quality gate (drops sessions with <55% onset hits).
   → `data/continuous/skaid/labels.jsonl` (51 sessions, 67,747 keystrokes).
2. **Diagnosed the collapse.** CTC alone collapsed to the English letter-prior / blank
   (~95% CER = chance, matching Slater's random baseline). Root causes found:
   (a) an MPS-era `.cpu()` shipped logits off-GPU every step (25× slowdown — fixed,
   CTC now on-device for CUDA); (b) **SpecAugment time-shifted the mel but not the
   per-frame labels**, desyncing them (disabled for the frame-aligned path); (c) the
   gap-sorted **curriculum** fed near-empty chunks first, reinforcing blank (disabled).
3. **Dense class-weighted frame-CE (the fix).** Turn the CTC head into a per-frame
   keystroke classifier: supervise every frame (blank elsewhere, key at each onset),
   with blank down-weighted so the ~4% keystroke frames dominate. This defeats the
   blank/prior collapse that a decoupled onset head + entropy bonus could not.
4. **CRNN (BiGRU) encoder** with an onset head (`MtlCRNN`) — the Slater-style
   recurrent recipe, which reads overlapping clicks better than the conv-only encoder.

## Reproduce

```bash
# 1) get + convert SKAID (needs ffmpeg; imageio-ffmpeg provides one)
#    two zips from https://zenodo.org/records/17282184 ->
#      data/external/skaid/recordings/Participant Recordings/...
#      data/external/skaid/keylogs/Keystroke Logs/...
uv run python3 -m keyguard.continuous_convert         # -> data/continuous/skaid/labels.jsonl

# 2) train + eval on real held-out typists (CRNN + dense frame-CE are the defaults)
KEYGUARD_DEVICE=cuda KEYGUARD_SKAID=data/continuous/skaid/labels.jsonl \
  KEYGUARD_CURRICULUM=0 KEYGUARD_SPEC_AUG=0 \
  uv run python3 -m keyguard.ctc.train_overlap 6000
```

Key knobs (env): `KEYGUARD_MODEL=crnn|conv`, `KEYGUARD_BLANK_W` (blank CE weight,
default 0.05), `KEYGUARD_FRAME_WIDTH` (±frames labeled per onset, default 1),
`KEYGUARD_LAMBDA_FRAME`, `KEYGUARD_LAMBDA_CTC` (0=off; >0 adds CTC timing),
`KEYGUARD_LAMBDA_ONSET`, `KEYGUARD_CURRICULUM`, `KEYGUARD_SPEC_AUG`.

## Next levers (not yet done)
- Reduce substitution errors (dominant now): more capacity / data / careful augmentation.
- Stronger decoder: char/word n-gram or LLM rescoring over an n-best (beyond the
  unigram segmentation in `lm_correct.py`).
- Wire this CTC attacker into the shield min-max arena (`keyguard/shield/adversarial.py`
  is currently built around the isolated-key KeyNet) to test attacker-vs-defense.
