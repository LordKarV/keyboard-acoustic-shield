"""Live keystroke capture + on-device calibration + demo attack/defense.

Records the laptop mic while a keylogger records key-press timestamps AND
identities (ground truth -- OUR OWN consenting machine only). That gives labeled
keystroke windows to (a) fine-tune the attacker to *this* keyboard and (b) drive
the live judge demo: type -> attacker reads it -> turn the shield on -> attack
collapses while speech stays clear.

Ethics: only your own device and consenting people. The keylogger identity is used
purely as ground-truth labels; the *shield* never sees key identity, only timing.

Modes (run on the demo machine, needs mic + Accessibility permission):
  uv run python3 capture.py calibrate   # type the shown charset a few times -> fine-tune
  uv run python3 capture.py demo        # type freely -> attack, then shield, then attack
  uv run python3 capture.py selfcheck   # offline: exercise alignment logic, no hardware
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np

from keyguard.config import SR, CLS_IDX
from keyguard import segment, features

DATA_DIR = "data/live"


def find_builtin_mic():
    """Return the built-in MacBook mic's device index, never AirPods/BlackHole.
    Override with env KEYGUARD_MIC=<index>. Falls back to the system default."""
    import os
    import sounddevice as sd
    env = os.environ.get("KEYGUARD_MIC")
    if env is not None and env.strip() != "":
        return int(env)
    devs = sd.query_devices()
    # prefer an input device whose name looks like the internal mic
    for i, d in enumerate(devs):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and (
                "macbook" in name or "built-in" in name or "builtin" in name):
            return i
    # otherwise the first real input that isn't AirPods/BlackHole/virtual
    for i, d in enumerate(devs):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and not any(
                x in name for x in ("airpod", "blackhole", "aggregate", "multi")):
            return i
    return None


class Session:
    """Concurrent mic recorder + key-timestamp/identity logger."""

    def __init__(self):
        self.audio_chunks = []
        self.events = []          # {"t": rel_seconds, "key": "A"}
        self.t0 = None
        self._stream = None
        self._klistener = None

    def _on_press(self, key):
        from pynput import keyboard
        if key == keyboard.Key.space:
            ch = " "
        else:
            ch = getattr(key, "char", None)
            ch = ch.upper() if ch else None
        if ch is not None and self.t0 is not None:
            self.events.append({"t": time.monotonic() - self.t0, "key": ch})

    def start(self):
        import sounddevice as sd
        from pynput import keyboard
        self.t0 = time.monotonic()

        def cb(indata, frames, tinfo, status):
            self.audio_chunks.append(indata[:, 0].copy())

        dev = find_builtin_mic()
        if dev is not None:
            print(f"   [mic] using input device #{dev}: "
                  f"{sd.query_devices(dev)['name']}")
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                      device=dev, callback=cb)
        self._stream.start()
        self._klistener = keyboard.Listener(on_press=self._on_press)
        self._klistener.start()
        return self

    def stop(self):
        if self._stream:
            self._stream.stop(); self._stream.close()
        if self._klistener:
            self._klistener.stop()
        y = np.concatenate(self.audio_chunks) if self.audio_chunks \
            else np.zeros(1, np.float32)
        return y, self.events

    def save(self, name):
        import soundfile as sf
        import os
        os.makedirs(DATA_DIR, exist_ok=True)
        y, ev = self.stop()
        sf.write(f"{DATA_DIR}/{name}.wav", y, SR)
        json.dump({"sr": SR, "events": ev},
                  open(f"{DATA_DIR}/{name}.keys.json", "w"))
        return y, ev


def labeled_windows(y, events, snap_s=0.15):
    """Align key timestamps to audio and cut labeled mel windows.

    The keylog timestamp is offset from the real click by audio input latency, so
    we snap each event to the nearest *detected audio onset* (within snap_s) and
    window on that -- matching how the Harrison/synth models were trained. The
    keylog still supplies the label and ordering.
    Returns (mels, labels, onsets) for keys in CLASSES (letters/digits only).
    """
    audio_on = segment.onsets(y)
    onsets, labels = [], []
    for e in events:
        k = e["key"]
        if k not in CLS_IDX:
            continue
        t = int(e["t"] * SR)
        if len(audio_on):
            j = int(np.argmin(np.abs(audio_on - t)))
            if abs(audio_on[j] - t) < snap_s * SR:
                t = int(audio_on[j])
        onsets.append(t)
        labels.append(CLS_IDX[k])
    if not onsets:
        return np.zeros((0, 64, 1), np.float32), np.array([], int), np.array([], int)
    onsets = np.array(onsets)
    mels = features.mel(segment.windows(y, onsets))
    return mels, np.array(labels), onsets


def _record_prompt(prompt, seconds=None):
    print(prompt)
    print(">> recording... press ENTER when done." if seconds is None
          else f">> recording for {seconds}s...")
    s = Session().start()
    if seconds is None:
        input()
    else:
        time.sleep(seconds)
    return s


def calibrate():
    """Type the charset a few times to fine-tune the attacker to this keyboard."""
    from keyguard.attackers.supervised import SupervisedAttacker
    charset = "the quick brown fox jumps over the lazy dog 1234567890"
    ROUNDS = 6
    Xs, ys = [], []
    for r in range(ROUNDS):
        s = _record_prompt(
            f"[round {r+1}/{ROUNDS}] Type slowly, then ENTER:\n  {charset}")
        y, ev = s.stop()
        m, lab, _ = labeled_windows(y, ev)
        print(f"   captured {len(lab)} labeled keystrokes")
        if len(lab):
            Xs.append(m); ys.append(lab)
    X = np.concatenate(Xs); Y = np.concatenate(ys)
    print(f"total {len(Y)} samples across {len(set(Y.tolist()))} keys")
    atk = SupervisedAttacker()
    try:
        atk.load("runs/supervised_mbp.pt"); print("warm-started from Harrison model")
    except Exception:
        pass
    # few-shot transfer: freeze the conv stem, adapt the deeper block + head only
    for p in atk.net.b1.parameters():
        p.requires_grad = False
    for p in atk.net.b2.parameters():
        p.requires_grad = False
    atk.fit(X, Y, epochs=150, lr=5e-4,
            log=lambda e, l: print(f"   ft ep{e} loss {l:.3f}"))
    atk.save("runs/live_attacker.pt")
    print("saved runs/live_attacker.pt")


def demo():
    from keyguard.attackers.supervised import SupervisedAttacker
    from keyguard.shield.separator import NeuralShield
    atk = SupervisedAttacker()
    for w in ("runs/live_attacker.pt", "runs/demo_attacker.pt",
              "runs/supervised_mbp.pt"):
        try:
            atk.load(w); print(f"attacker: {w}"); break
        except Exception:
            continue
    s = _record_prompt("Type a sentence + a FAKE password, then ENTER:")
    y, ev = s.stop()
    truth = "".join(e["key"] for e in ev)
    m, _, onsets = labeled_windows(y, ev)

    def read(mels):
        return "".join(atk.predict(mels)) if len(mels) else ""

    print(f"\n  ground truth : {truth}")
    print(f"  ATTACK (raw) : {read(m)}")
    shield = NeuralShield(randomize=0.6)
    y_s = shield.apply(y, onsets=onsets)
    m_s = features.mel(segment.windows(y_s, onsets))
    print(f"  ATTACK (shielded): {read(m_s)}   <- should be garbage")
    import soundfile as sf
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    sf.write(f"{DATA_DIR}/demo_raw.wav", y, SR)
    sf.write(f"{DATA_DIR}/demo_shielded.wav", y_s, SR)
    print(f"  saved {DATA_DIR}/demo_raw.wav and demo_shielded.wav")


def bank(out="data/live_bank.npz", rounds=2):
    """Capture an 'acoustic alphabet' for THIS keyboard: isolated per-key clips.
    Type the shown paragraph a couple of times; the keylogger gives exact labels,
    we snap each to its audio onset and save 0.12s clips per key to an .npz the
    data engine trains on (synth_line(root='data/live_bank.npz'))."""
    import os
    from collections import defaultdict
    from keyguard.ctc.data import CLIP
    line = ("the quick brown fox jumps over the lazy dog. pack my box with five "
            "dozen liquor jugs. amazingly few discotheques provide jukeboxes. "
            "sphinx of black quartz judge my vow. the wizard quickly zapped the "
            "jinxed gremlin. my phone is 0123456789 and the backup code is "
            "9876543210. we typed every letter many extra times just to be sure.")
    clips = defaultdict(list)
    for r in range(rounds):
        s = _record_prompt(f"[bank {r+1}/{rounds}] Type this at a steady pace, "
                           f"then ENTER:\n  {line}")
        y, ev = s.stop()
        au = segment.onsets(y)
        for e in ev:
            k = e["key"]
            if k not in CLS_IDX:
                continue
            t = int(e["t"] * SR)
            if len(au):
                j = int(np.argmin(np.abs(au - t)))
                if abs(au[j] - t) < 0.15 * SR:
                    t = int(au[j])
            a = max(0, t - int(0.02 * SR))
            c = np.zeros(CLIP, np.float32)
            seg = y[a:a + CLIP]
            c[:len(seg)] = seg
            if np.abs(c).max() > 1e-3:
                clips[k].append(c)
        got = {k: len(v) for k, v in sorted(clips.items())}
        print(f"   running totals: {got}")
    missing = [k for k in CLS_IDX if k not in clips]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, **{k: np.stack(v) for k, v in clips.items()})
    print(f"\nsaved {out}: {len(clips)}/36 keys, "
          f"{sum(len(v) for v in clips.values())} clips total")
    if missing:
        print(f"MISSING keys (type them next round or re-run): {missing}")
    else:
        print("all 36 keys captured -> retrain with root='data/live_bank.npz'")


def selfcheck():
    """Offline: timestamp->window alignment works without any hardware."""
    y = np.zeros(SR * 2, np.float32)
    ev = []
    for i, k in enumerate("ABC123"):
        t = 0.2 + i * 0.25
        y[int(t * SR):int(t * SR) + 200] += np.hanning(200)   # fake click
        ev.append({"t": t, "key": k})
    m, lab, ons = labeled_windows(y, ev)
    assert len(lab) == 6 and list(lab) == [CLS_IDX[c] for c in "ABC123"]
    assert m.shape[0] == 6 and len(ons) == 6
    print(f"capture selfcheck ok: aligned {len(lab)} keystrokes -> mels {m.shape}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    {"calibrate": calibrate, "demo": demo, "bank": bank, "selfcheck": selfcheck}.get(
        mode, selfcheck)()
