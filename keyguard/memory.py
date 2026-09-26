"""Backboard: persistent memory of the adversarial arena across sessions.

Each co-training run is a fact the shield remembers -- "faced a retraining attacker,
held leakage to Y% at an inaudible -18 dB budget." Storing runs in Backboard means
the defense's track record survives across sessions and machines, so the arena is
evaluated against (and never silently regresses on) its whole history of adversaries.

A local JSONL is always written (the dashboard's source of truth, so the demo has
history offline); Backboard is mirrored best-effort when BACKBOARD_API_KEY is set.
REST via httpx -- header is X-API-Key (not Bearer), base https://app.backboard.io/api.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

from .config import RUNS

BASE = "https://app.backboard.io/api"
LOCAL = RUNS / "arena_memory.jsonl"
ASSISTANT_FILE = RUNS / "backboard_assistant.txt"


def _headers(key: str) -> dict:
    return {"X-API-Key": key, "Content-Type": "application/json"}


def _assistant_id(key: str):
    """Reuse a cached assistant so all runs share one memory; None on failure."""
    if ASSISTANT_FILE.exists():
        return ASSISTANT_FILE.read_text().strip() or None
    try:
        import httpx
        r = httpx.post(f"{BASE}/threads/messages", headers=_headers(key),
                       json={"content": "Keyguard adversarial-arena memory.",
                             "memory": "Auto"}, timeout=30)
        aid = r.json().get("assistant_id")
        if aid:
            ASSISTANT_FILE.write_text(aid)
        return aid
    except Exception:
        return None


def _summary_text(run: dict) -> str:
    return (f"Arena run {run['created']}: clean attack {run['clean_acc']:.1%}, "
            f"shielded to {run['final_shielded']:.1%} after {run['rounds']} "
            f"retraining rounds at {run['snr_db']:.1f} dB (chance {run['chance']:.1%}).")


def remember_run(run: dict) -> None:
    """Append the run locally (always) and mirror to Backboard (best-effort)."""
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL.open("a") as f:
        f.write(json.dumps(run) + "\n")
    key = os.environ.get("BACKBOARD_API_KEY")
    if not key:
        return
    aid = _assistant_id(key)
    if not aid:
        return
    try:  # best-effort; local JSONL is the source of truth
        import httpx
        httpx.post(f"{BASE}/assistants/{aid}/memories", headers=_headers(key),
                   json={"content": _summary_text(run),
                         "metadata": {"source": "keyguard-arena"}}, timeout=30)
    except Exception:
        pass


def recall_runs(limit: int = 20) -> list[dict]:
    """Past arena runs, newest first, from the local mirror."""
    if not LOCAL.exists():
        return []
    rows = []
    for line in LOCAL.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[::-1][:limit]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def demo():
    import tempfile, pathlib
    global LOCAL
    orig = LOCAL
    LOCAL = pathlib.Path(tempfile.mkdtemp()) / "arena_memory.jsonl"
    try:
        assert recall_runs() == []                       # empty is safe
        run = {"created": now_iso(), "clean_acc": 0.88, "final_shielded": 0.05,
               "snr_db": -18.0, "rounds": 4, "chance": 0.028}
        remember_run(run)                                # no key -> local only, no raise
        got = recall_runs()
        assert len(got) == 1 and got[0]["clean_acc"] == 0.88
        assert "shielded to 5.0%" in _summary_text(run)
        print(f"memory ok: local JSONL write/recall works ({LOCAL.name}); "
              f"Backboard mirror wired for BACKBOARD_API_KEY")
    finally:
        LOCAL = orig


if __name__ == "__main__":
    demo()
