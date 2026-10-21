"""Aggregate PURE comparison results into one JSON for the report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = Path("/data/home/z00533370/outputs/bench")
OUT = O / "pure_report" / "results.json"
PAIRS = [("pi3", "pure_pi3"), ("vggt_b", "pure_vggt"), ("da3g", "pure_da3")]
NEW = ["pi3", "pure_pi3", "vggt_b", "pure_vggt", "da3g", "pure_da3", "vggt_omega", "vggt_omega512"]
REF = ["pi3x", "da3", "vggt", "vipe"]

A = pd.read_csv(O / "eval3d_A/eval/per_clip_pure.csv")
B = pd.read_csv(O / "eval3d_B/eval/per_clip_pure.csv")
S = pd.concat([pd.read_csv(O / f"scannetpp/eval/{m}.csv") for m in NEW + REF if (O / f"scannetpp/eval/{m}.csv").exists()])
S["dataset"] = "scannetpp"
S["metric"] = True
gt = pd.concat([A, B, S], ignore_index=True)
gt["ate_pct"] = gt["ate_sim3_cm"] / 100 / gt["traj_len_m"].clip(lower=1e-6) * 100
DS = ["sintel", "bonn", "kitti", "tum", "7scenes", "eth3d_mvs", "tartanair", "vkitti2", "scannetpp"]
METRIC_DS = ["bonn", "kitti", "tum", "7scenes", "eth3d_mvs", "tartanair", "vkitti2", "scannetpp"]  # Sintel GT not metric
res = {"n_clips": {d: int(gt[(gt.method == "pi3") & (gt.dataset == d)].shape[0]) for d in DS}}

# 1) paper-protocol-like table: mean over sequences on the paper's overlapping benchmarks
paper = {}
for d in ["sintel", "bonn", "kitti"]:
    for m in NEW:
        g = gt[(gt.method == m) & (gt.dataset == d)]
        paper.setdefault(d, {})[m] = {k: round(float(g[k].mean()), 4) for k in
                                       ["depth_seq_absrel", "depth_seq_delta1", "ate_sim3_cm", "rre_deg"]}
res["paper_overlap_mean"] = paper

# 2) per-dataset medians, all methods
KEYS = ["depth_seq_absrel", "depth_seq_delta1", "depth_frame_absrel", "depth_metric_absrel", "ate_pct", "ate_sim3_cm",
        "rre_deg", "fov_err_deg", "sim3_scale", "pc_fscore_5cm"]
med = {}
for d in DS:
    for m in NEW + REF:
        g = gt[(gt.method == m) & (gt.dataset == d)]
        if len(g):
            med.setdefault(d, {})[m] = {k: (round(float(g[k].median()), 4) if k in g and g[k].notna().any() else None) for k in KEYS}
res["median_by_dataset"] = med

# 3) paired statistics over all GT clips (base vs PURE): win rate, median relative change, Wilcoxon p
LOWER = {"depth_seq_absrel": True, "depth_seq_delta1": False, "depth_frame_absrel": True, "ate_pct": True,
         "rre_deg": True, "fov_err_deg": True, "depth_metric_absrel": True}
paired = {}
for base, pure in PAIRS:
    for scope, dss in [("all", DS), ("dynamic", ["sintel", "bonn"]), ("static_indoor", ["tum", "7scenes", "scannetpp", "eth3d_mvs"]),
                       ("driving", ["kitti", "vkitti2"]), ("synthetic", ["sintel", "tartanair", "vkitti2"])]:
        a = gt[(gt.method == base) & gt.dataset.isin(dss)].set_index("clip")
        b = gt[(gt.method == pure) & gt.dataset.isin(dss)].set_index("clip")
        common = a.index.intersection(b.index)
        for k, low in LOWER.items():
            x, y = a.loc[common, k].astype(float), b.loc[common, k].astype(float)
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            if len(x) < 3:
                continue
            better = (y < x) if low else (y > x)
            try:
                p = float(wilcoxon(x, y).pvalue)
            except ValueError:
                p = float("nan")
            paired.setdefault(f"{base}->{pure}", {}).setdefault(scope, {})[k] = {
                "n": int(len(x)), "win_rate": round(float(better.mean()), 3),
                "median_base": round(float(x.median()), 4), "median_pure": round(float(y.median()), 4),
                "median_rel_change": round(float(((y - x) / x.abs().clip(lower=1e-9)).median()), 4), "wilcoxon_p": round(p, 5)}
res["paired"] = paired

# 4) overall ranking across the 9 GT datasets (mean of per-dataset ranks; lower = better)
rank_keys = {"depth_seq_absrel": True, "ate_pct": True, "rre_deg": True, "fov_err_deg": True}
ranks = {}
for k, low in rank_keys.items():
    rk = []
    for d in DS:
        s = pd.Series({m: med[d][m][k] for m in NEW + REF if m in med[d] and med[d][m][k] is not None})
        rk.append(s.rank(ascending=low))
    ranks[k] = pd.concat(rk, axis=1).mean(axis=1).round(2).to_dict()
res["mean_rank"] = ranks

# 5) wild (no GT)
W = pd.read_csv(O / "wild/eval_pure/per_clip.csv")
W["subset"] = np.where(W["clip"].str.contains("__sekai__"), "sekai", np.where(W["clip"].str.contains("droid"), "droid", "youtube"))
wild = {}
for m in NEW + REF:
    g = W[W.method == m]
    wild[m] = {s: {k: round(float(g[g.subset == s][k].median()), 4) for k in ["g5_selfcons", "g20_selfcons", "g5_photo", "g20_photo"]}
               for s in ["youtube", "sekai", "droid"]}
    wild[m]["all"] = {k: round(float(g[k].median()), 4) for k in ["g5_selfcons", "g20_selfcons", "g5_photo", "g20_photo", "hfov_deg"]}
res["wild_median"] = wild
wp = {}
for base, pure in PAIRS:
    a = W[W.method == base].set_index("clip")
    b = W[W.method == pure].set_index("clip")
    for s in ["youtube", "sekai", "droid", "all"]:
        cl = a.index if s == "all" else a.index[a.subset == s]
        cl = cl.intersection(b.index)
        for k, low in [("g20_selfcons", False), ("g20_photo", True), ("g5_selfcons", False)]:
            x, y = a.loc[cl, k].astype(float), b.loc[cl, k].astype(float)
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            better = (y < x) if low else (y > x)
            try:
                p = float(wilcoxon(x, y).pvalue)
            except ValueError:
                p = float("nan")
            wp.setdefault(f"{base}->{pure}", {}).setdefault(s, {})[k] = {
                "n": int(len(x)), "win_rate": round(float(better.mean()), 3), "median_base": round(float(x.median()), 4),
                "median_pure": round(float(y.median()), 4), "wilcoxon_p": round(p, 5)}
res["wild_paired"] = wp
res["wild_consensus"] = json.loads((O / "wild/eval_pure/summary.json").read_text())["consensus_by_method"]
res["wild_selfcons_by_cat"] = json.loads((O / "wild/eval_pure/summary.json").read_text())["selfcons_by_cat"]

# 6) clean timing
tim = {}
for m in NEW:
    r = [json.loads(line) for line in (O / "pure_timing" / m / "status.jsonl").read_text().splitlines()]
    tim[m] = {"sec_per_200f": round(float(np.median([x["sec"] for x in r])), 2), "peak_gb": max(x.get("peak_gb", 0) for x in r),
              "hw": r[0]["hw"]}
res["timing"] = tim
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ["n_clips", "paper_overlap_mean", "timing"]}, indent=1))
for pair, d in paired.items():
    print(pair)
    for scope, v in d.items():
        print("  ", scope, {k: (x["win_rate"], x["median_base"], x["median_pure"], x["wilcoxon_p"]) for k, x in v.items()})
for pair, d in wp.items():
    print("WILD", pair, {s: {k: (x["win_rate"], x["median_base"], x["median_pure"], x["wilcoxon_p"]) for k, x in v.items()} for s, v in d.items()})
print("RANK", json.dumps(ranks))
