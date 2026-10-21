"""Collect timing and results of a pipeline run: OUT/timing.json and a printed summary.

Usage: python summarize.py OUT WALL_SEC
timing.json: wall time of the whole run, wall seconds per step (each step includes its model loading, so for a single
clip the loading dominates; per-clip model time is in the step logs / pred.npz `sec`), and per-clip model seconds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT, WALL = Path(sys.argv[1]), float(sys.argv[2])
steps = [json.loads(x) for x in (OUT / "timing.jsonl").read_text().splitlines()] if (OUT / "timing.jsonl").exists() else []
B = OUT / "bench"
per_clip = {}
for c in json.loads((B / "clips.json").read_text()):
    r = {}
    st = B / "vipe" / "logs" / "status.jsonl"
    if st.exists():
        for line in st.read_text().splitlines():
            x = json.loads(line)
            if x.get("clip") == c["clip"]:
                r["vipe_sec"] = x.get("sec")
    for m in ("pi3x", "da3", "pi3x_kgate", "da3_pkvipe"):
        p = B / m / c["clip"] / "pred.npz"
        if p.exists():
            r[f"{m}_sec"] = round(float(np.load(p)["sec"]), 2)
    g = B / "geocalib" / f"{c['clip']}.json"
    if g.exists():
        r["geocalib_sec"] = round(json.loads(g.read_text())["sec"], 2)
    per_clip[c["clip"]] = r
agg = {}
for s in steps:  # a step can run twice (filter stages again on split shots): sum
    agg[s["step"]] = round(agg.get(s["step"], 0.0) + s["sec"], 2)
t = {"wall_sec": WALL, "steps": agg, "per_clip_model_sec": per_clip}
(OUT / "timing.json").write_text(json.dumps(t, indent=1))
print(json.dumps(t["steps"]))
if (OUT / "results.csv").exists():
    print(pd.read_csv(OUT / "results.csv").to_string(index=False))
if (OUT / "filter" / "decisions.csv").exists():
    d = pd.read_csv(OUT / "filter" / "decisions.csv")
    print(d[["clip", "decision", "reason"]].to_string(index=False))
