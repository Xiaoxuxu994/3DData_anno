"""Build the end-to-end check bench: ~100 passed + ~100 rejected pool clips (rejects stratified by reason).

Usage: python fc_setup.py decisions.csv POOL_DIR BENCH_DIR [N=100]
BENCH_DIR/frames/<clip> -> POOL_DIR/<clip>/frames ; BENCH_DIR/clips.json (cat = pass | reject, reason kept).
"""

import json
import sys
from pathlib import Path

import pandas as pd

csv, pool, B = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
df = pd.read_csv(csv)
df["reason"] = df.reason.fillna("")
pas = df[df.decision == "pass"].sample(min(N, (df.decision == "pass").sum()), random_state=1)
rej = df[df.decision == "reject"].copy()
rej["main"] = rej.reason.str.split(";").str[0]
# round-robin over primary reasons so rare reasons (zoom, cuts, ...) are represented
groups = [g.sample(frac=1, random_state=1) for _, g in rej.groupby("main")]
pick, k = [], 0
while len(pick) < min(N, len(rej)):
    for g in groups:
        if k < len(g) and len(pick) < N:
            pick.append(g.iloc[k])
    k += 1
rej = pd.DataFrame(pick)
(B / "frames").mkdir(parents=True, exist_ok=True)
clips = []
for kind, sub in (("pass", pas), ("reject", rej)):
    for _, r in sub.iterrows():
        link = B / "frames" / r["clip"]
        if not link.exists():
            link.symlink_to(pool / r["clip"] / "frames")
        n = len(list((pool / r["clip"] / "frames").glob("*.jpg")))
        clips.append({"clip": r["clip"], "cat": kind, "reason": r.reason, "n_frames": n, "cat_hint": r.cat_hint})
(B / "clips.json").write_text(json.dumps(clips, indent=1))
print(len(pas), "pass", len(rej), "reject", rej.main.value_counts().to_dict())
