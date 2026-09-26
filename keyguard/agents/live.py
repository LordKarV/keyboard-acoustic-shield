"""Test KeyGuard on YOUR computer: type live, see what the attacker reads, then
watch the defender protect it. Records the mic while logging your real keystrokes
(ground truth) via pynput.

IMPORTANT: per-key acoustics are keyboard-specific, so the SKAID-trained model
reads a NEW keyboard poorly zero-shot. Run `calibrate` once (~2 min) to adapt it
to your keyboard, then `record`.

    uv run python3 -m keyguard.agents.live calibrate     # adapt to your keyboard
    uv run python3 -m keyguard.agents.live record --secs 20
    uv run python3 -m keyguard.agents.live record --secs 20 --defend

Needs a working mic + permission for global key capture (pynput).
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import torch

from .. import config, audio
from ..ctc import train_overlap as T
from ..ctc.model import logmel, DEVICE
from ..ctc.data import VOCAB, SYM_OF_KEY, BLANK
from ..ctc.overlap_eval import cer_str
from ..config import SR, RUNS
from . import defense_audio as DA

CAL_PROMPTS = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS 1234567890",
    "HOW QUICKLY DAFT JUMPING ZEBRAS VEX THE 0 9 8 7",
]
LIVE_CKPT = str(RUNS / "ctc_live.pt")


def _record(secs: float):
    """Record mic (device default sr -> 16k) + log real keystrokes. Returns
    (y16, events[list of {t,key}])."""
    import sounddevice as sd
    from pynput import keyboard
    import librosa
    dev = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    sr_dev = int(sd.query_devices(dev)["default_samplerate"]) if dev is not None else 48000
    chunks, events = [], []
    state = {"t0": None}

    def cb(indata, frames, tinfo, status):
        chunks.append(indata[:, 0].copy())

    def on_press(key):
        if state["t0"] is None:
            return
        ch = " " if key == keyboard.Key.space else getattr(key, "char", None)
        if ch and (ch.isalnum() or ch == " "):
            events.append({"t": time.monotonic() - state["t0"], "key": ch.upper()})

    stream = sd.InputStream(samplerate=sr_dev, channels=1, dtype="float32", callback=cb)
    kl = keyboard.Listener(on_press=on_press)
    print(f"\n>>> RECORDING for {secs:.0f}s — start typing now (talk too, to test noise)! <<<")
    state["t0"] = time.monotonic(); stream.start(); kl.start()
    time.sleep(secs)
    stream.stop(); stream.close(); kl.stop()
    y = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    y16 = librosa.resample(y, orig_sr=sr_dev, target_sr=SR).astype(np.float32)
    print(f"    captured {len(y16)/SR:.1f}s audio, {len(events)} keys")
    return y16, events


def _load_net(prefer_live=True):
    ck = LIVE_CKPT if (prefer_live and __import__("os").path.exists(LIVE_CKPT)) else "runs/ctc_skaid_crnn.pt"
    net = (T.MtlCRNN if T.MODEL == "crnn" else T.MtlCTC)().to(DEVICE)
    net.load_state_dict(torch.load(ck, map_location=DEVICE)); net.eval()
    return net, ck


def calibrate(rounds=1, secs=45):
    """Collect labeled typing on YOUR keyboard and fine-tune the attacker to it."""
    import os
    from scipy.signal import find_peaks
    net, base = _load_net(prefer_live=False)
    print(f"Calibrating from {base}. You'll type prompts; press keys clearly.")
    samples = []  # (logmel, frame_key_target)
    for r in range(rounds):
        for pr in CAL_PROMPTS:
            print(f"\nTYPE THIS (then keep still ~1s):\n   {pr}")
            y, ev = _record(secs=max(8, len(pr) * 0.4))
            keys = [e for e in ev if e["key"] in SYM_OF_KEY]
            if len(keys) < 8:
                print("   (too few keys captured, skipping)"); continue
            m = logmel(y)
            onset_samp = np.array([int(e["t"] * SR) for e in keys])
            ids = np.array([SYM_OF_KEY[e["key"]] for e in keys], np.int64)
            samples.append((m, T.frame_key_target(m.shape[0], onset_samp, ids)))
    if not samples:
        print("No calibration data captured."); return
    import torch.nn as nn
    w = torch.ones(len(VOCAB), device=DEVICE); w[BLANK] = 0.05
    fce = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    net.train()
    rng = np.random.default_rng(0)
    for step in range(400):
        m, fk = samples[int(rng.integers(len(samples)))]
        logits, _ = net(torch.from_numpy(m)[None].to(DEVICE))
        loss = fce(logits.reshape(-1, logits.shape[-1]), torch.from_numpy(fk).to(DEVICE).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); os.makedirs(RUNS, exist_ok=True); torch.save(net.state_dict(), LIVE_CKPT)
    print(f"\nCalibrated attacker saved -> {LIVE_CKPT}. Now run: record")


def record(secs, defend=False):
    net, ck = _load_net()
    y, ev = _record(secs)
    typed = "".join(e["key"] for e in ev if e["key"] in SYM_OF_KEY)
    read = DA.decode(net, y)
    print(f"\nattacker model: {ck}")
    print(f"YOU TYPED : {typed}")
    print(f"ATTACKER  : {read}   (CER {cer_str(typed, read):.0%})")
    if "skaid" in ck:
        print("  NOTE: using the SKAID model on your keyboard (zero-shot) -> expect high CER."
              "\n  Run `calibrate` first to adapt to your keyboard.")
    if defend and len(ev) >= 8:
        keys = [e for e in ev if e["key"] in SYM_OF_KEY]
        s = len(keys) // 3; e = min(s + 7, len(keys))
        lo = max(0, int(keys[s]["t"] * SR) - int(0.03 * SR))
        hi = min(len(y), int(keys[e - 1]["t"] * SR) + int(0.12 * SR))
        yp, info = DA.craft(net, y, lo, hi, mode="protect", snr_db=18.0)
        read2 = DA.decode(net, yp)
        print(f"\nDEFENDER protects keystrokes [{s}:{e}] "
              f"(t {info['span_s'][0]:.2f}-{info['span_s'][1]:.2f}s), "
              f"perturbation SNR {info['snr_db']:.1f}dB, STOI {info['stoi']:.3f} (inaudible)")
        print(f"ATTACKER (defended): {read2}   (CER {cer_str(typed, read2):.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    rp = sub.add_parser("record"); rp.add_argument("--secs", type=float, default=20); rp.add_argument("--defend", action="store_true")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate()
    else:
        record(a.secs, a.defend)
