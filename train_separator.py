"""Train the causal neural keystroke separator on synthetic mixtures.

Supervision comes free from synth.py: we have the clean speech, the mixture, and
the exact keystroke onsets. Target is the ideal ratio mask clean_mag/mix_mag; the
net learns to reproduce it *conditioned on the key mask*, so it suppresses
keystroke energy while passing speech.

Loss = L1(mask, ideal_mask) + L1(mask*mix_mag, clean_mag)  [magnitude recon]
       + keystroke-suppression term (drive masked energy at key frames toward the
         clean speech floor).

Usage: uv run python3 train_separator.py [epochs]
"""
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa

from keyguard import audio
from keyguard.config import N_FFT, HOP
from keyguard.shield.separator import SepUNet, DEVICE

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MANIFEST = "data/synth_train/manifest.json"


def _featurize(item):
    mix = audio.load(item["mix_wav"])
    clean = audio.load(item["clean_wav"])
    n = min(len(mix), len(clean))
    mix, clean = mix[:n], clean[:n]
    Sm = np.abs(librosa.stft(mix, n_fft=N_FFT, hop_length=HOP))
    Sc = np.abs(librosa.stft(clean, n_fft=N_FFT, hop_length=HOP))
    T = Sm.shape[1]
    km = np.zeros(T, np.float32)
    for o in item["onsets"]:
        c = int(o / HOP)
        km[max(0, c - 2):min(T, c + 6)] = 1.0
    ideal = np.clip(Sc / (Sm + 1e-6), 0, 1).astype(np.float32)
    return (np.log1p(Sm).astype(np.float32), Sm.astype(np.float32),
            Sc.astype(np.float32), ideal, km)


def main():
    items = json.load(open(MANIFEST))
    data = [_featurize(it) for it in items]
    print(f"train clips: {len(data)}, device {DEVICE}")
    net = SepUNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def to_t(a):
        return torch.from_numpy(a).float().to(DEVICE)

    for ep in range(EPOCHS):
        net.train()
        total = 0.0
        order = np.random.permutation(len(data))
        for i in order:
            logmag, mixmag, cleanmag, ideal, km = data[i]
            lm = to_t(logmag)[None]              # (1,F,T)
            mm = to_t(mixmag)[None]
            cm = to_t(cleanmag)[None]
            idl = to_t(ideal)[None]
            kt = to_t(km)[None, None]            # (1,1,T)
            mask = net(lm, kt)
            recon = mask * mm
            # key frames are ~5% of the clip; without heavy weighting the net just
            # learns mask==1 everywhere (pass-through) and never suppresses. Weight
            # the loss ~20x on key frames so it actually learns to separate there.
            w = 1.0 + 20.0 * kt.squeeze(1)               # (1,T)
            w = w[:, None, :]                            # (1,1,T) broadcast over F
            loss_mask = (w * (mask - idl).abs()).mean()
            loss_rec = (w * (torch.log1p(recon) - torch.log1p(cm)).abs()).mean()
            keyf = kt.squeeze(1) > 0
            supp = F.relu(recon - cm)[:, :, keyf[0]].mean() if keyf.any() else 0.0
            loss = loss_mask + loss_rec + 0.5 * supp
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        sched.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep{ep:3d} loss {total/len(data):.4f}", flush=True)

    torch.save(net.state_dict(), "runs/separator.pt")
    print("saved runs/separator.pt", flush=True)


if __name__ == "__main__":
    main()
