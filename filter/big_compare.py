"""Large-VLM comparison against Qwen3-VL-32B: agreement, known blind spots, pass-rate change.

Usage: python big_compare.py OUT_DIR REF_TAG TAG [TAG ...]
Writes OUT_DIR/big_compare.json and big_blindspots.csv.
Blind spots of the reference found on group C: far aerial (GyKAR7bmUD4, nNjYd1Qh1Ws), synthetic zoom
(hard__zoom__*), "sees the cut but says yes" (has_cuts true and reconstructable yes).
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decide import apply_rules, load  # noqa: E402

H = Path("/data/home/z00533370")


def cohen(a, b):  # same as analyze.py
    m = a.notna() & b.notna()
    a, b = a[m].astype(str), b[m].astype(str)
    if not len(a):
        return float("nan"), float("nan"), 0
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
    return float(po), float((po - pe) / (1 - pe)) if pe < 1 else float("nan"), int(len(a))
OUT, REF, TAGS = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
C = apply_rules(load(H / "data/filter_pool/_calibC"))
P = apply_rules(load(H / "data/filter_pool"))
fc = {c["clip"] for c in json.loads((H / "outputs/bench/filter_check/clips.json").read_text())}
BLIND = ["hard__yt__GyKAR7bmUD4", "hard__yt__nNjYd1Qh1Ws", "hard__zoom__b06ThXzoqRI_10369", "hard__zoom__uO99uLuqisc_1261",
         "hard__yt__PXDArVNjIhs", "hard__yt__W7c6OGrUM-4", "driving__youtube_cc__uRbDaKgz1_w"]
rows = []
for c in BLIND:
    r = C[C["clip"] == c].iloc[0]
    row = {"clip": c}
    for t in [REF] + TAGS:
        row[t] = f'{r.get(f"vlm_{t}.reconstructable")} / motion={r.get(f"vlm_{t}.camera_motion")} / cuts={r.get(f"vlm_{t}.has_cuts")}'
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "big_blindspots.csv", index=False)
res = {}
for t in [REF] + TAGS:
    rc, hc = f"vlm_{t}.reconstructable", f"vlm_{t}.has_cuts"
    d = {}
    for name, df in (("pool", P), ("C", C), ("filter_check", P[P["clip"].isin(fc)])):
        if rc not in df:
            continue
        d[name] = {"n_judged": int(df[rc].notna().sum()), "vlm_yes_rate": float((df[rc] == "yes").mean()),
                   "final_pass_rate": float(((df[rc] == "yes") & df.rule_pass).mean()),
                   "cut_seen_but_yes": int(((df[hc] == True) & (df[rc] == "yes")).sum()),  # noqa: E712
                   "cut_seen": int((df[hc] == True).sum())}  # noqa: E712
        if t != REF:
            po, k, n = cohen(df[f"vlm_{REF}.reconstructable"], df[rc])
            d[name].update({"agree_vs_ref": po, "kappa_vs_ref": k, "n_pair": n})
    res[t] = d
(OUT / "big_compare.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
print(pd.DataFrame(rows).to_string())
