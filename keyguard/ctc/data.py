"""Overlap data engine: synthesize free-typing audio with exact string labels.

Per-key Harrison samples are the "acoustic alphabet". To make a training example
we overlap-add those samples at realistic inter-key intervals (sampled from a
lognormal typing-dynamics distribution), so consecutive clicks physically overlap
just like fast typing -- and the typed string is the exact CTC target. Sound adds
linearly, so overlap-add at digraph-timed offsets is a faithful model of
overlapping typing. Unlimited labeled data for a sequence model.

Vocab: index 0 = CTC blank, 1..36 = config.CLASSES (a-z, 0-9). Spaces are gaps
(no emitted symbol) in v1.
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from .. import audio, segment
from ..config import SR, CLASSES, CLS_IDX

BLANK = 0
VOCAB = ["<blank>"] + list(CLASSES)          # 37 symbols
SYM_OF_KEY = {k: i + 1 for i, k in enumerate(CLASSES)}   # key -> ctc index
CLIP = int(0.12 * SR)                          # per-key press clip length (~120ms)


@lru_cache(maxsize=2)
def sample_bank(root="data/harrison/MBPWavs"):
    """{key: (n, CLIP) float32} press clips for each key.

    root ending in .npz = a captured per-key bank (from capture.py bank), loaded
    directly. Otherwise a Harrison-style dir of <KEY>.wav files (25 presses each).
    """
    if str(root).endswith(".npz"):
        d = np.load(root)
        return {k: d[k].astype(np.float32) for k in d.files}
    bank = {}
    for k in CLASSES:
        p = f"{root}/{k}.wav"
        try:
            y = audio.load(p)
        except Exception:
            continue
        on = segment.onsets_n(y, 25)
        clips = []
        for o in on:
            a = max(0, o - int(0.02 * SR))
            seg = y[a:a + CLIP]
            c = np.zeros(CLIP, np.float32)
            c[:len(seg)] = seg
            clips.append(c)
        if clips:
            bank[k] = np.stack(clips)
    return bank


def synth_line(text, rng, wpm=(200, 520), space_gap=(0.15, 0.35),
               noise=0.002, root="data/harrison/MBPWavs"):
    """Return (audio, label_ids, onset_samples) for `text`.

    wpm range controls typing speed -> inter-key interval; higher wpm => more
    overlap (5 chars/word, so interval = 60/(wpm*5) s). Non-letter/digit chars
    become gaps. label_ids excludes blanks/spaces (CTC targets)."""
    bank = sample_bank(root)
    speed = rng.uniform(*wpm)
    base_iki = 60.0 / (speed * 5)                # seconds between keys
    buf = np.zeros(int(0.1 * SR), np.float32)    # leading pad
    cursor = int(0.05 * SR)
    labels, onsets = [], []
    for ch in text.upper():
        if ch not in CLS_IDX:                     # space / punctuation -> gap
            cursor += int(rng.uniform(*space_gap) * SR)
            continue
        if ch not in bank:
            continue
        clip = bank[ch][rng.integers(len(bank[ch]))]
        clip = clip * rng.uniform(0.7, 1.2)       # amplitude jitter
        end = cursor + CLIP
        if end + SR > len(buf):
            buf = np.concatenate([buf, np.zeros(SR, np.float32)])
        buf[cursor:cursor + CLIP] += clip         # overlap-add
        onsets.append(cursor + int(0.02 * SR))
        labels.append(SYM_OF_KEY[ch])
        iki = max(0.045, rng.lognormal(np.log(base_iki), 0.35))
        cursor += int(iki * SR)
    buf = buf[:cursor + CLIP + int(0.05 * SR)]
    if noise > 0:
        buf = buf + noise * rng.standard_normal(len(buf)).astype(np.float32)
    return buf, np.array(labels, dtype=np.int64), np.array(onsets, dtype=np.int64)


WORDS = ("the quick brown fox jumps over a lazy dog my password is hunter two "
         "meeting notes are ready send the report before five pm thanks team "
         "login admin secret code four two zero seven nine one three").split()


def random_text(rng, n_words=(4, 9)):
    k = rng.integers(*n_words)
    return " ".join(rng.choice(WORDS) for _ in range(k))


def batch(rng, n=8, **kw):
    return [synth_line(random_text(rng), rng, **kw) for _ in range(n)]


def demo():
    rng = np.random.default_rng(0)
    bank = sample_bank()
    assert len(bank) == 36, f"expected 36 keys, got {len(bank)}"
    y, lab, on = synth_line("the fox 42", rng, wpm=(450, 500))  # fast -> overlap
    non_space = len("thefox42")
    assert len(lab) == non_space == len(on), (len(lab), non_space, len(on))
    gaps = np.diff(on)
    overlap_rate = float(np.mean(gaps < CLIP)) if len(gaps) else 0.0
    print(f"data engine ok: 'the fox 42' -> {len(lab)} keys, "
          f"audio {len(y)/SR:.2f}s, overlap_rate {overlap_rate:.0%}, vocab {len(VOCAB)}")


if __name__ == "__main__":
    demo()
