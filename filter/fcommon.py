"""Shared helpers for the filter stages: clip listing, worker sharding, frame loading, scores.json merge, stage logs."""

import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np

H = Path("/data/home/z00533370")
POOL = Path(os.environ.get("FILTER_POOL", H / "data/filter_pool"))
LOGS = H / "outputs/logs/filter"


def clips(pool=None, wid=0, nw=1):
    """Clip dirs from pool.json (or a bench-style clips.json with frames/<clip>), sharded wid::nw."""
    pool = Path(pool or POOL)
    if (pool / "pool.json").exists():
        names = [m["clip"] for m in json.loads((pool / "pool.json").read_text())]
        return [pool / n for n in names][wid::nw]
    return sorted(p.parent for p in pool.glob("*/meta.json"))[wid::nw]


def frame_paths(d):
    return sorted((d / "frames").glob("*.jpg"))


def load_frames(d, idx=None, rgb=True):
    import cv2
    ps = frame_paths(d)
    if idx is not None:
        ps = [ps[i] for i in idx]
    out = [cv2.imread(str(p)) for p in ps]
    return [f[..., ::-1].copy() for f in out] if rgb else out


def even_idx(n, k):
    return np.linspace(0, n - 1, k).round().astype(int).tolist()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_scores(d, stage, data):
    """Merge {stage: data} into d/scores.json under an exclusive lock (stages may run concurrently)."""
    p = d / "scores.json"
    with open(d / ".scores.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = json.loads(p.read_text()) if p.exists() else {}
        s[stage] = _clean(data)
        tmp = d / ".scores.json.tmp"
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(p)


def has_stage(d, stage):
    p = d / "scores.json"
    return p.exists() and stage in json.loads(p.read_text())


class StageLog:
    """Append one JSON line per clip to outputs/logs/filter/<stage>.jsonl with its runtime."""

    def __init__(self, stage, wid):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.f = open(LOGS / f"{stage}.w{wid}.jsonl", "a")
        self.stage = stage

    def __call__(self, d, sec, **kw):
        self.f.write(json.dumps({"clip": d.name, "sec": round(sec, 3), **_clean(kw)}) + "\n")
        self.f.flush()


def timer():
    t = time.time()
    return lambda: time.time() - t
