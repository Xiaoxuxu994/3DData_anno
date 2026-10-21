"""Extend the end-to-end bench with clips from reject reasons / categories that the first random draw under-covers.

Usage: python fc_extend.py decisions.csv POOL_DIR BENCH_DIR N_EXTRA [PER_GROUP=12]
Groups = primary reject reason for rejected clips and cat_hint for passed clips; each group is topped up to
PER_GROUP clips (already-benched clips count), round-robin, until N_EXTRA new clips are added.
"""

import json
import sys
from pathlib import Path

import pandas as pd

csv, pool, B, N = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 12
df = pd.read_csv(csv)
df["reason"] = df.reason.fillna("")
df["group"] = [("reject:" + r.split(";")[0]) if d == "reject" else ("pass:" + c)
               for d, r, c in zip(df.decision, df.reason, df.cat_hint)]
cl = json.loads((B / "clips.json").read_text())
have = {c["clip"] for c in cl}
cnt = df[df["clip"].isin(have)].group.value_counts().to_dict()
pools = {g: sub[~sub["clip"].isin(have)].sample(frac=1, random_state=2) for g, sub in df.groupby("group")}
added, k = [], 0
while len(added) < N and any(len(v) > k for v in pools.values()):
    for g, sub in sorted(pools.items()):
        if len(added) >= N:
            break
        if cnt.get(g, 0) < PER and k < len(sub):
            r = sub.iloc[k]
            added.append(r)
            cnt[g] = cnt.get(g, 0) + 1
    k += 1
for r in added:
    link = B / "frames" / r["clip"]
    if not link.exists():
        link.symlink_to(pool / r["clip"] / "frames")
    cl.append({"clip": r["clip"], "cat": r["cat_hint"], "n_frames": len(list((pool / r["clip"] / "frames").glob("*.jpg"))),
               "added_for": r["group"]})
(B / "clips.json").write_text(json.dumps(cl, indent=1))
print(len(added), "added; bench now", len(cl), pd.Series([r["group"] for r in added]).value_counts().to_dict())
