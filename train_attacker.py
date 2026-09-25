"""Train + eval the supervised attacker on clean Harrison MBP data. Baseline."""
import numpy as np
from keyguard.dataset import build_aug as build
from keyguard.attackers.supervised import SupervisedAttacker
from keyguard.config import RUNS

Xtr, ytr, Xte, yte = build("data/harrison/MBPWavs")
print("train", Xtr.shape, "test", Xte.shape)
atk = SupervisedAttacker()
atk.fit(Xtr, ytr, epochs=150, log=lambda e, l: print(f"  ep{e} loss {l:.3f}"))
pred = atk.predict(Xte)
acc = np.mean([atk.classes[t] == p for t, p in zip(yte, pred)])
print(f"\nTOP-1 test accuracy: {acc:.1%}  (chance {1/36:.1%})")
RUNS.mkdir(exist_ok=True)
atk.save(RUNS / "supervised_mbp.pt")
print("saved", RUNS / "supervised_mbp.pt")
