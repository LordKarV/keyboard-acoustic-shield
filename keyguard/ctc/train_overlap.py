"""GPU-ready trainer for the OVERLAP keystroke transcriber (the novel attacker).

Combines the fixes the literature says are needed to read fast/overlapping typing
(Slater ACSAC'19 end-to-end CTC; EnCTC/label-prior anti-collapse; SED onset head;
sim-to-real augmentation):

  * MtlCTC  = ConvCTC encoder + per-frame ONSET-BCE head (dense gradient that
              stops CTC blank-collapse and localises events without hard segmentation).
  * loss    = CTC(zero_infinity) + lambda_onset * BCE + beta_ent * (-mean entropy)
              (entropy bonus keeps the posterior from going peaky/overconfident).
  * curriculum = train slow (well-separated) typing first, widen to fast/overlapping.
  * augmentation = channel/EQ + noise + gain (augment.py) + a cheap VoIP band-limit,
              so a model trained on synth reads real, over-a-call audio.

Runs on CUDA automatically on a GPU box (no MPS crashes there). Score = sequence
CER vs overlap level (edit distance on the key string; chance ~1.0, lower better).

  GPU box:  uv run python3 -m keyguard.ctc.train_overlap 4000
  Smoke:    KEYGUARD_DEVICE=cpu uv run python3 -m keyguard.ctc.train_overlap demo
  Bank:     KEYGUARD_BANK=data/harrison/MBPWavs (dir) or a .npz per-key bank
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..config import SR
from . import augment
from .data import synth_line, random_text, sample_bank, VOCAB, BLANK, CLIP
from .model import logmel, greedy_decode, DEVICE, HOP, N_MELS
from .mtl_probe import MtlCTC, onset_target
from .overlap_eval import cer_str, _ids_to_keys, SPEED_BINS

BANK = os.environ.get("KEYGUARD_BANK", "data/harrison/MBPWavs")
CKPT = os.environ.get("KEYGUARD_CKPT", "runs/ctc_overlap.pt")
STEPS = int(os.environ.get("KEYGUARD_STEPS", "4000"))
POOL = int(os.environ.get("KEYGUARD_POOL", "2000"))
BATCH = int(os.environ.get("KEYGUARD_BATCH", "16"))
LR = float(os.environ.get("KEYGUARD_LR", "3e-4"))
WARMUP = 200
LAMBDA_ONSET = 0.5            # onset-BCE weight (research: 0.3-1.0)
BETA_ENT = 0.01              # entropy bonus weight (anti-peaky-collapse)
WPM_FULL = (140, 580)        # full speed range; curriculum reveals fast last


def _voip(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap over-a-call sim: band-limit ~300-3800 Hz + mild quantization (p=0.5)."""
    if rng.random() > 0.5:
        return y
    from scipy import signal
    b, a = signal.butter(4, [300 / (SR / 2), 3800 / (SR / 2)], btype="band")
    z = signal.lfilter(b, a, y).astype(np.float32)
    return (np.round(z * 64) / 64).astype(np.float32)     # coarse quantize


def build_pool(n: int, seed: int = 0, aug: bool = True) -> list:
    """Precompute (logmel, label_ids, onset_target, mean_gap_samples) across speeds."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n):
        y, lab, on = synth_line(random_text(rng), rng, wpm=WPM_FULL, root=BANK)
        if len(lab) == 0:
            continue
        if aug:
            y = augment.augment_waveform(y, rng)
            y = _voip(y, rng)
        m = logmel(y)
        gap = float(np.mean(np.diff(on))) if len(on) > 1 else float(len(y))
        pool.append((m, lab, onset_target(m.shape[0], on), gap))
    pool.sort(key=lambda p: -p[3])           # easy (large gap) first for curriculum
    return pool


def _collate(items, rng, spec_aug=True):
    maxT = max(m.shape[0] for m, _, _, _ in items)
    B = len(items)
    mel = torch.zeros(B, maxT, N_MELS)
    onset = torch.zeros(B, maxT)
    in_len = torch.zeros(B, dtype=torch.long)
    tgts, tlen = [], []
    for i, (m, lab, on, _) in enumerate(items):
        if spec_aug:
            m = augment.spec_augment(m, rng)
        t = m.shape[0]
        mel[i, :t] = torch.from_numpy(np.ascontiguousarray(m))
        onset[i, :t] = torch.from_numpy(on)
        in_len[i] = t
        tgts.append(torch.from_numpy(lab)); tlen.append(len(lab))
    return (mel.to(DEVICE), onset.to(DEVICE), in_len,
            torch.cat(tgts), torch.tensor(tlen, dtype=torch.long))


def _curriculum_slice(pool, step):
    """Fraction of the (gap-sorted) pool visible so far: 40% -> 100% over training."""
    frac = min(1.0, 0.4 + 0.6 * step / max(1, STEPS * 0.6))
    return pool[: max(BATCH, int(frac * len(pool)))]


@torch.no_grad()
def evaluate(net, seed=100, lines=12) -> dict:
    net.eval()
    rows = []
    for lo, hi in SPEED_BINS:
        rng = np.random.default_rng(seed + lo)
        ov, ce = [], []
        for _ in range(lines):
            y, lab, on = synth_line(random_text(rng), rng, wpm=(lo, hi), root=BANK)
            if len(lab) == 0:
                continue
            gaps = np.diff(on)
            ov.append(float(np.mean(gaps < CLIP)) if len(gaps) else 0.0)
            logits, _ = net(torch.from_numpy(logmel(y))[None].to(DEVICE))
            hyp = _ids_to_keys(greedy_decode(logits)[0])
            ce.append(cer_str(_ids_to_keys(list(lab)), hyp))
        rows.append({"wpm": (lo + hi) // 2, "overlap": float(np.mean(ov)),
                     "cer": float(np.mean(ce))})
    return {"rows": rows, "mean_cer": float(np.mean([r["cer"] for r in rows]))}


def _print_eval(e, tag=""):
    print(f"  eval{tag}: " + "  ".join(
        f"{r['wpm']}wpm(ov{r['overlap']:.0%}) CER {r['cer']:.1%}" for r in e["rows"])
        + f"  | MEAN {e['mean_cer']:.1%}", flush=True)


def train():
    import json
    print(f"device={DEVICE} bank={BANK} steps={STEPS} pool={POOL} "
          f"batch={BATCH} onset={LAMBDA_ONSET} ent={BETA_ENT}", flush=True)
    assert len(sample_bank(BANK)) >= 20, f"bank {BANK} has too few keys"
    pool = build_pool(POOL)
    print(f"pool {len(pool)} lines (gap-sorted for curriculum); training...", flush=True)

    torch.manual_seed(0)
    net = MtlCTC().to(DEVICE).train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    brng = np.random.default_rng(1234)
    best = 1e9
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        sub = _curriculum_slice(pool, step)
        idx = brng.integers(0, len(sub), size=BATCH)
        mel, onset, in_len, tgt, tlen = _collate([sub[i] for i in idx], brng)
        logits, onset_logit = net(mel)
        logp = logits.log_softmax(-1).permute(1, 0, 2).cpu()
        loss = ctc(logp, tgt, in_len.clamp(max=logits.shape[1]), tlen)
        mask = (torch.arange(mel.shape[1], device=DEVICE)[None] < in_len.to(DEVICE)[:, None]).float()
        loss = loss + LAMBDA_ONSET * (bce(onset_logit, onset) * mask).sum() / mask.sum()
        p = logits.softmax(-1).clamp_min(1e-8)
        ent = -(p * p.log()).sum(-1).mean()                 # posterior entropy
        loss = loss - BETA_ENT * ent                        # bonus (subtract)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if DEVICE == "mps" and step % 25 == 0:
            torch.mps.empty_cache()
        if step % 250 == 0 or step == STEPS - 1:
            e = evaluate(net)
            tag = ""
            if e["mean_cer"] < best:
                best = e["mean_cer"]; torch.save(net.state_dict(), CKPT); tag = "  <-best"
            print(f"step {step:4d} loss {float(loss):.3f} ent {float(ent):.2f}"
                  f" ({(time.time()-t0)/max(1,step+1)*1000:.0f}ms/step)", flush=True)
            _print_eval(e, tag)
            net.train()
    print(f"done. best mean CER {best:.1%}; weights {CKPT}", flush=True)
    (config.RUNS / "overlap_train_result.json").write_text(
        json.dumps({"best_mean_cer": best, "steps": STEPS, "device": DEVICE,
                    "final_eval": evaluate(net)}, indent=2))


def demo():
    """CPU smoke: pool builds, one train step runs, eval returns in-range CER."""
    global STEPS, POOL
    STEPS, POOL = 2, 12
    pool = build_pool(6, aug=True)
    assert pool and pool[0][0].shape[1] == N_MELS
    net = MtlCTC().to(DEVICE).train()
    rng = np.random.default_rng(0)
    mel, onset, in_len, tgt, tlen = _collate(pool[:2], rng)
    logits, onset_logit = net(mel)
    assert logits.shape[0] == 2 and logits.shape[2] == len(VOCAB)
    assert onset_logit.shape == onset.shape
    e = evaluate(net, lines=2)
    assert 0.0 <= e["mean_cer"] <= 2.0
    print(f"train_overlap demo ok on {DEVICE}: pool {len(pool)}, "
          f"logits {tuple(logits.shape)}, eval mean CER {e['mean_cer']:.1%} "
          f"(untrained). Full run: uv run python3 -m keyguard.ctc.train_overlap 4000")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "demo":
        demo()
    else:
        if arg.isdigit():
            STEPS = int(arg)
        train()
