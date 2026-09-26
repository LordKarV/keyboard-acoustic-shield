"""Build the results grid the dashboard shows: app x protection, supervised
attacker (adaptive). Writes runs/grid.json.

Metrics per cell: text_accuracy (top-1 proxy), mi_bits (leakage lower bound),
pesq/stoi (speech quality of that protection on synthetic mixtures).
Usage: uv run python3 build_grid.py [epochs]
"""
import json
import sys
import time
from pathlib import Path
from keyguard.eval import harness, baselines
from keyguard.shield.shield import Shield, ShieldConfig

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

APPS = {"phone": "data/harrison/MBPWavs"}      # +"zoom" if labeled per-key
PROTECTIONS = {
    "none": baselines.none,
    "spectral(RNNoise-like)": baselines.spectral_gate,
    "keyguard": Shield(ShieldConfig()).apply,
}

cells = []
for app, root in APPS.items():
    for pname, fn in PROTECTIONS.items():
        print(f"[grid] {app} / {pname} ...")
        m = harness.evaluate(protect=fn, root=root, adaptive=True, epochs=EPOCHS)
        q = harness.speech_quality(fn)
        cells.append({
            "laptop": "MacBook", "app": app, "protection": pname,
            "attacker": "supervised",
            "text_accuracy": round(m["attack_acc"], 4),
            "mi_bits": round(m["mi_bits"], 3),
            "cluster_purity": None,
            "pesq": q.get("pesq"), "stoi": q.get("stoi"),
        })
        print(f"       acc={m['attack_acc']:.1%} mi={m['mi_bits']:.2f} "
              f"stoi={q.get('stoi')}")

Path("runs").mkdir(exist_ok=True)
Path("runs/grid.json").write_text(json.dumps(
    {"created": time.time(), "cells": cells}, indent=2))
print(f"\nwrote runs/grid.json ({len(cells)} cells)")
