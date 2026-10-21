"""E7: cheap-first cascade — can feed-forward-only QC drop clips BEFORE the expensive ViPE run? (offline, CPU)

python e7_cascade.py -> outputs/evolution/E7/cascade.csv + printed summary
Pre-ViPE score per clip from the image-only feed-forward models (outputs/qc/cross_<G>.csv, method = pi3x):
  ff = max(xff_traj / T1, xff_depth / T2, traj_spread_ff / T3)   (Pi3X vs VGGT / VGGT-Omega / DA3 / Voyager)
Thresholds T = 90th percentile on A+B clips whose v1.1 label is good. A clip with ff > s is not sent to ViPE.
Targets: A+B — v1.1 geometric failure (E3b raw, pi3x_kgate@vipe_td: ATE-Sim3 / FOV / metric depth) and the v1b QC
decision (E1/qc_rule_v1b.csv); C — v1b QC decision only (no GT). Reports, per s: share of clips skipped (ViPE GPU
saved), share of finally-accepted clips lost, failure rate among the finally-kept clips.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path("/data/home/z00533370/outputs")
E = R / "evolution"
cx = pd.concat([pd.read_csv(R / "qc" / f"cross_{g}.csv") for g in "ABC"])
cx = cx[cx["method"] == "pi3x"][["group", "clip", "xff_traj", "xff_depth", "xff_fov", "traj_spread_ff"]]
q = pd.read_csv(E / "E1" / "qc_rule_v1b.csv")[["group", "clip", "v1b_traj_depth_kdis"]].rename(columns={"v1b_traj_depth_kdis": "qc"})
gt = []
for g in "AB":
    x = pd.read_csv(E / "E3b" / "raw" / f"{g}.csv")
    x = x[x["method"] == "pi3x_kgate@vipe_td"].copy()
    x["geom_fail"] = ((x["fail_ate_sim3"] > 0) | (x["fail_fov"] > 0) | (x["fail_depth"].fillna(0) > 0)).astype(int)
    gt.append(x[["clip", "geom_fail"]].assign(group=g))
d = cx.merge(q, on=["group", "clip"]).merge(pd.concat(gt), on=["group", "clip"], how="left")
d["accepted"] = d["qc"] <= 1
ab = d[d["group"].isin(["A", "B"])]
good = ab[(ab["geom_fail"] == 0)]
T = {c: float(good[c].quantile(0.9)) for c in ("xff_traj", "xff_depth", "traj_spread_ff")}
d["ff"] = np.nanmax(np.column_stack([d[c] / T[c] for c in T]), 1)
ab = d[d["group"].isin(["A", "B"])]
print("thresholds", {k: round(v, 3) for k, v in T.items()})
y = ab["geom_fail"]
print("A+B AUROC of pre-ViPE ff score for v1.1 failure:", round(roc_auc_score(y, ab["ff"].fillna(9)), 3),
      "| for v1b QC reject:", round(roc_auc_score((~ab["accepted"]).astype(int), ab["ff"].fillna(9)), 3))
rows = []
for s in (1.0, 1.5, 2.0, 3.0, 5.0):
    for g, x in (("A+B", ab), ("C", d[d["group"] == "C"])):
        skip = x["ff"] > s
        acc = x["accepted"]
        r = {"s": s, "set": g, "n": len(x), "skipped": round(skip.mean(), 3),
             "accepted_lost": round(float((skip & acc).sum() / max(acc.sum(), 1)), 3),
             "accepted_rate_after": round(float((acc & ~skip).mean()), 3)}
        if g == "A+B":
            kept = acc & ~skip
            r["fail_among_kept"] = round(float(x.loc[kept, "geom_fail"].mean()), 3)
            r["fail_among_kept_no_cascade"] = round(float(x.loc[acc, "geom_fail"].mean()), 3)
        rows.append(r)
out = pd.DataFrame(rows)
(E / "E7").mkdir(exist_ok=True)
out.to_csv(E / "E7" / "cascade.csv", index=False)
d.to_csv(E / "E7" / "cascade_per_clip.csv", index=False, float_format="%.4g")
print(out.to_string())
c = d[d["group"] == "C"]
print("C skipped at s=2 by category:", c.assign(k=c["ff"] > 2).groupby(c["clip"].str.split("__").str[0])["k"].mean().round(2).to_dict())
