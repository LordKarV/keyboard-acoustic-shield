"""Train the ConvCTC keystroke transcriber on synthesized overlapping typing.

Configurable by env:
  KEYGUARD_BANK   sample-clip source (default Harrison MacBook dir; set to
                  data/live_bank.npz for a captured keyboard)
  KEYGUARD_CKPT   output checkpoint (default runs/ctc.pt)
  KEYGUARD_LIGHT  '1' -> lighter augmentation (known keyboard converges faster)

Pipeline: overlap data engine + augmentation baked into a cached feature pool +
per-step SpecAugment; MPS-stable (small batch + periodic cache clear); best-CER
checkpoint; final CER-vs-overlap sweep with greedy + LM decode.
Usage: uv run python3 train_ctc.py [steps]
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from keyguard.ctc import data, augment
from keyguard.ctc.model import ConvCTC, logmel, greedy_decode, cer, DEVICE

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
BANK = os.environ.get("KEYGUARD_BANK", "data/harrison/MBPWavs")
OUT = os.environ.get("KEYGUARD_CKPT", "runs/ctc.pt")
LIGHT = os.environ.get("KEYGUARD_LIGHT") == "1"
NOAUG = os.environ.get("KEYGUARD_NOAUG") == "1"   # same kbd+mic -> no aug needed
BS = 8 if DEVICE == "mps" else 16          # smaller batch is MPS-stabler
POOL = 2500
CLIP = data.CLIP
LIGHT_P = {"style": 0.4, "noise": 0.3, "gain": 0.5}   # gentle aug for known kbd


def make_pool(rng, n):
    pool = []
    for i in range(n):
        y, lab, _ = data.synth_line(data.random_text(rng), rng, wpm=(150, 560),
                                    root=BANK)
        if not len(lab):
            continue
        if not NOAUG:
            y = augment.augment_waveform(y, rng, p=LIGHT_P if LIGHT else None)
        pool.append((logmel(y), lab))
        if (i + 1) % 1000 == 0:
            print(f"  pool {i+1}/{n}", flush=True)
    return pool


def collate(items, rng):
    feats = [f if NOAUG or (LIGHT and rng.random() < 0.5)
             else augment.augment_logmel(f, rng) for f, _ in items]
    T = max(f.shape[0] for f in feats)
    X = np.zeros((len(feats), T, feats[0].shape[1]), np.float32)
    in_len = []
    for i, f in enumerate(feats):
        X[i, :f.shape[0]] = f
        in_len.append(f.shape[0])
    tgt = np.concatenate([lab for _, lab in items])
    tgt_len = [len(lab) for _, lab in items]
    return (torch.from_numpy(X).to(DEVICE), torch.tensor(in_len),
            torch.from_numpy(tgt).long(), torch.tensor(tgt_len))


def make_eval(rng, wpm, n=30):
    return [(logmel(y), lab) for y, lab, _ in
            (data.synth_line(data.random_text(rng), rng, wpm=wpm, root=BANK)
             for _ in range(n)) if len(lab)]


def greedy_cer(net, evset):
    net.eval()
    errs = []
    with torch.no_grad():
        for f, lab in evset:
            x = torch.from_numpy(f)[None].to(DEVICE)
            errs.append(cer(list(lab), greedy_decode(net(x))[0]))
    return float(np.mean(errs)) if errs else 1.0


def overlap_sweep(net):
    from keyguard.ctc.lm_decode import LMDecoder, decode_ids
    dec = LMDecoder()
    print(f"\n=== CER vs overlap (bank={BANK}, LM={dec.backend}) ===", flush=True)
    print(f"{'speed(wpm)':>12}{'overlap%':>10}{'greedyCER':>11}{'lmCER':>8}", flush=True)
    net.eval()
    for lo, hi in [(150, 220), (250, 340), (360, 460), (470, 580)]:
        rng = np.random.default_rng(hash((lo, hi)) % 2**31)
        ov, g, l = [], [], []
        with torch.no_grad():
            for _ in range(8):
                y, lab, on = data.synth_line(data.random_text(rng), rng,
                                             wpm=(lo, hi), root=BANK)
                if not len(lab):
                    continue
                gaps = np.diff(on)
                ov.append(np.mean(gaps < CLIP) if len(gaps) else 0.0)
                logits = net(torch.from_numpy(logmel(y))[None].to(DEVICE))
                g.append(cer(list(lab), greedy_decode(logits)[0]))
                lg = logits[0].log_softmax(-1).cpu().numpy()
                l.append(cer(list(lab), decode_ids(lg, dec)))
        print(f"{(lo+hi)//2:>12}{np.mean(ov):>9.0%}{np.mean(g):>11.1%}"
              f"{np.mean(l):>8.1%}", flush=True)


def main():
    rng = np.random.default_rng(0)
    print(f"device {DEVICE}, bank {BANK}, light_aug {LIGHT}, "
          f"building pool ({POOL})...", flush=True)
    pool = make_pool(rng, POOL)
    ev_slow = make_eval(np.random.default_rng(1), (180, 260), 30)
    ev_fast = make_eval(np.random.default_rng(2), (440, 560), 30)
    print(f"pool {len(pool)} lines; training {STEPS} steps on {DEVICE}", flush=True)

    net = ConvCTC().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, STEPS - warm))))
    ctc = nn.CTCLoss(blank=data.BLANK, zero_infinity=True)
    idx = np.arange(len(pool))
    best = 1.0
    for step in range(STEPS):
        net.train()
        pick = rng.choice(idx, BS, replace=False)
        X, in_len, tgt, tgt_len = collate([pool[i] for i in pick], rng)
        opt.zero_grad()
        logits = net(X)
        logp = logits.log_softmax(-1).permute(1, 0, 2)
        loss = ctc(logp, tgt, in_len.clamp(max=logits.shape[1]), tgt_len)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if DEVICE == "mps" and step % 100 == 0:
            torch.mps.empty_cache()          # reduce MPS command-buffer crashes
        if step % 100 == 0 or step == STEPS - 1:
            fast = greedy_cer(net, ev_fast)
            slow = greedy_cer(net, ev_slow)
            tag = ""
            if fast < best:
                best = fast
                torch.save(net.state_dict(), OUT)
                tag = "  <- saved best"
            print(f"  step {step:4d} loss {float(loss):.3f}  "
                  f"CER slow {slow:.1%}  fast {fast:.1%}{tag}", flush=True)
    print(f"best fast-CER {best:.1%}; weights {OUT}", flush=True)
    overlap_sweep(net)


if __name__ == "__main__":
    main()
