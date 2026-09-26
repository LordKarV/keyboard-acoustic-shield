"""Audio-level defender actuation: a bounded, span-localized adversarial
perturbation on the raw waveform, optimized against the REAL CRNN attacker so it
misreads the protected keystrokes — while speech stays intact (high STOI).

This is the tool the Defender-LLM wields: the LLM decides WHICH span to protect
and (for deception) WHAT false text to plant; this module realizes that decision
as an actual, inaudible waveform change. The differentiable log-mel matches
model.logmel to corr>0.9999, so gradients transfer to the deployed attacker.
"""
from __future__ import annotations
import numpy as np
import torch
import librosa

from ..config import SR
from ..ctc.model import HOP, N_FFT, N_MELS, greedy_decode
from ..ctc.decode_onset import onset_gated_decode
from ..ctc.data import VOCAB, SYM_OF_KEY

_MEL = torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS), dtype=torch.float32)
_WIN = torch.hann_window(N_FFT)


def torch_logmel(y: torch.Tensor) -> torch.Tensor:
    """Differentiable (T, mel) log-mel matching keyguard.ctc.model.logmel."""
    S = torch.stft(y, N_FFT, hop_length=HOP, window=_WIN.to(y.device), center=True,
                   pad_mode="constant", return_complex=True)
    power = S.real ** 2 + S.imag ** 2
    mel = _MEL.to(y.device) @ power
    db = 10 * torch.log10(mel.clamp_min(1e-10))
    db = db - db.amax()
    db = torch.maximum(db, db.amax() - 80.0)
    db = (db - db.mean()) / (db.std() + 1e-6)
    return db.T


def decode(net, y: np.ndarray):
    with torch.no_grad():
        lg, onl = net(torch_logmel(torch.tensor(y, dtype=torch.float32))[None])
    return onset_gated_decode(lg, onl)


def craft(net, y: np.ndarray, lo: int, hi: int, mode: str = "protect",
          target: str | None = None, snr_db: float = 20.0, steps: int = 250,
          lr: float = 0.02):
    """Optimize a perturbation confined to samples [lo:hi].
      mode='protect' : make the protected keystrokes unreadable (push to blank).
      mode='deceive' : steer the protected frames to `target` keys (a false secret).
    Returns (y_defended, info). Perturbation power is projected to >= snr_db (in-span)."""
    y_t = torch.tensor(y, dtype=torch.float32)
    delta = torch.zeros_like(y_t, requires_grad=True)
    mask = torch.zeros_like(y_t); mask[lo:hi] = 1.0
    opt = torch.optim.Adam([delta], lr=lr)
    f_lo, f_hi = lo // HOP, max(lo // HOP + 1, hi // HOP)
    sig_pow = float(np.mean(y[lo:hi] ** 2)) + 1e-12
    budget = (hi - lo) * sig_pow / (10 ** (snr_db / 10.0))   # total allowed ||delta||^2 in span

    tgt_ids = None
    if mode == "deceive" and target:
        ids = [SYM_OF_KEY.get(c) for c in target.upper() if c in SYM_OF_KEY]
        if ids:
            tgt_ids = torch.tensor(ids)

    for _ in range(steps):
        yp = y_t + delta * mask
        lg, _ = net(torch_logmel(yp)[None])
        span = lg[0, f_lo:f_hi, :]
        if tgt_ids is not None:
            # steer the span's non-blank frames toward the target keys (spread evenly)
            logp = span.log_softmax(-1)
            k = tgt_ids[torch.linspace(0, len(tgt_ids) - 1, span.shape[0]).long()]
            loss_atk = -logp[torch.arange(span.shape[0]), k].mean()
        else:
            loss_atk = span.log_softmax(-1)[:, 1:].max(-1).values.mean()  # kill confident keys
        loss = loss_atk + 30.0 * (delta * mask).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cur = (delta * mask).pow(2).sum().item()
            if cur > budget:
                delta.data *= (budget / cur) ** 0.5

    yp = (y_t + delta * mask).detach().numpy().astype(np.float32)
    d = yp[lo:hi] - y[lo:hi]
    snr = 10 * np.log10(sig_pow / (np.mean(d ** 2) + 1e-12))
    try:
        from pystoi import stoi
        st = float(stoi(y, yp, SR, extended=False))
    except Exception:
        st = float("nan")
    return yp, {"snr_db": float(snr), "stoi": st, "span_s": (lo / SR, hi / SR)}
