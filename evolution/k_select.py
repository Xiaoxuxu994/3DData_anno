"""E1 helper: which intrinsics-selection rule is best on GT benches? (CPU, uses fov_sources.csv)

Rules evaluated (HFOV error vs GT, fail = error > 5 deg):
  vipe | geo | pi3x | da3 | med4 = median(vipe, geo, pi3x, da3)
  sel_geo3  : vipe if |vipe-geo| < 3 deg else geo           (the rule proposed before E1)
  sel_vote  : vipe unless the three others agree (spread < S) and vipe is > T away from their median
Also: how well does each disagreement predict a ViPE failure (AUROC)?  -> outputs/evolution/E1/k_select.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/home/z00533370/outputs/evolution/E1")
df = pd.read_csv(ROOT / "fov_sources.csv")
gt = df[df["hfov_gt"].notna()].copy()
o3 = gt[["hfov_geo", "hfov_pi3x", "hfov_da3"]]
gt["others_med"], gt["others_spread"] = o3.median(axis=1), o3.max(axis=1) - o3.min(axis=1)
gt["hfov_med4"] = gt[["hfov_vipe", "hfov_geo", "hfov_pi3x", "hfov_da3"]].median(axis=1)
gt["hfov_sel_geo3"] = np.where((gt["hfov_vipe"] - gt["hfov_geo"]).abs() < 3, gt["hfov_vipe"], gt["hfov_geo"])
res = {"n": int(len(gt))}
for T in (5, 8, 12):
    for S in (5, 8, 12):
        k = f"hfov_sel_vote_T{T}_S{S}"
        sw = ((gt["hfov_vipe"] - gt["others_med"]).abs() > T) & (gt["others_spread"] < S)
        gt[k] = np.where(sw, gt["others_med"], gt["hfov_vipe"])
def gate(d, lo=25, hi=130, T=None, S=None):
    """ViPE K unless implausible (HFOV outside [lo, hi]) -> median of the other sources; optional vote on top."""
    o = d[["hfov_geo", "hfov_pi3x", "hfov_da3"]]
    om, osp = o.median(axis=1), o.max(axis=1) - o.min(axis=1)
    bad = (d["hfov_vipe"] < lo) | (d["hfov_vipe"] > hi) | d["hfov_vipe"].isna()
    if T is not None:
        bad |= ((d["hfov_vipe"] - om).abs() > T) & (osp < S)
    return np.where(bad, om, d["hfov_vipe"]), bad


for T, S in ((None, None), (12, 8), (8, 5), (15, 10)):
    gt[f"hfov_gate_T{T}_S{S}"] = gate(gt, T=T, S=S)[0]
rules = [c for c in gt.columns if c.startswith("hfov_") and c != "hfov_gt"]
tab = {}
for r in rules:
    e = (gt[r] - gt["hfov_gt"]).abs()
    tab[r[5:]] = {"med_err": round(float(e.median()), 2), "mean_err": round(float(e.mean()), 2),
                  "fail5": round(float((e > 5).mean()), 3),
                  "fail5_by_ds": {d: round(float((g[r] - g["hfov_gt"]).abs().gt(5).mean()), 2) for d, g in gt.groupby("dataset")}}
res["rules"] = dict(sorted(tab.items(), key=lambda kv: (kv[1]["fail5"], kv[1]["mean_err"])))
y = ((gt["hfov_vipe"] - gt["hfov_gt"]).abs() > 5).astype(int)
au = {}
for nm, s in {"vipe_vs_geo": (gt["hfov_vipe"] - gt["hfov_geo"]).abs(),
              "vipe_vs_pi3x": (gt["hfov_vipe"] - gt["hfov_pi3x"]).abs(),
              "vipe_vs_others_med": (gt["hfov_vipe"] - gt["others_med"]).abs(),
              "geo_fcv": gt["geo_fcv"], "geo_unc": gt["geo_unc"]}.items():
    m = s.notna()
    au[nm] = round(float(roc_auc_score(y[m], s[m])), 3) if y[m].nunique() == 2 else None
res["auroc_predict_vipe_fov_fail"] = au
res["vipe_fail_clips"] = gt.loc[y == 1, ["clip", "hfov_gt", "hfov_vipe", "hfov_geo", "hfov_pi3x", "hfov_da3"]].round(1).to_dict("records")
wild = df[df["bench"] == "wild"].copy()
res["wild_switch_rate"] = {}
for T, S in ((None, None), (12, 8), (8, 5), (15, 10)):
    v, bad = gate(wild, T=T, S=S)
    res["wild_switch_rate"][f"gate_T{T}_S{S}"] = {"rate": round(float(bad.mean()), 3),
                                                   "by_cat": wild.assign(b=bad).groupby("dataset")["b"].mean().round(2).to_dict()}
(ROOT / "k_select.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for k, v in list(res["rules"].items())[:8]:
    print(k, v["med_err"], v["mean_err"], v["fail5"])
print(au)
print(json.dumps(res["wild_switch_rate"]))
print(pd.DataFrame(res["vipe_fail_clips"]).to_string())
