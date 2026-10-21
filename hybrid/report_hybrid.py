"""Aggregate raw/{A,B,scannetpp,wild}.csv into outputs/hybrid/{tables.md, summary.json, ranking_A.csv, *_per_clip.csv}.

Ranking follows eval3d_report.py: per A dataset, rank every combo's median per metric, average over metrics,
then average over datasets. Two extra rankings for hybrids: 'metric' (scale / metric depth / traj-aligned cloud)
and 'consistency' (traj-aligned cloud F, reprojection EPE).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path("/data/home/z00533370/outputs/hybrid")
RAW = H / "raw"
RANK_DATASETS = ["7scenes", "tum", "bonn", "sintel", "kitti"]
UNSEEN = ["1c7a683c92", "6c14d5fd01", "a4c043ac48"]
RANK_GEOM = [("ate_sim3_pct", False), ("rre_deg", False), ("fov_err_deg", False), ("depth_seq_absrel", False),
             ("pc_fscore_5cm", True), ("g5_epe_med_px", False)]
RANK_POSE = [("ate_sim3_pct", False), ("rre_deg", False), ("fov_err_deg", False)]
RANK_DEPTH = [("depth_seq_absrel", False), ("pc_fscore_5cm", True), ("g5_epe_med_px", False)]
RANK_METRIC = [("sim3_scale_err", False), ("depth_metric_absrel", False)]
RANK_CONS = [("pc_traj_fscore_5cm", True), ("g5_epe_med_px", False), ("g20_epe_med_px", False)]
COLS = ["n_ok", "ate_sim3_pct", "sim3_scale", "fov_err_deg", "depth_metric_absrel", "depth_seq_absrel",
        "depth_seq_delta1", "pc_fscore_5cm", "pc_traj_fscore_5cm", "g5_epe_med_px", "g20_epe_med_px",
        "rate_fail_ate_sim3", "rate_fail_ate", "rate_fail_scale", "rate_fail_depth", "rate_fail_fov"]
FAILS = ["fail_ate_sim3", "fail_ate", "fail_scale", "fail_depth", "fail_fov"]
FOCUS = ["vipe", "spatialvid_metric", "pi3x", "da3", "vggt_omega",
         "H:vipe+pi3x+raw", "H:vipe+pi3x+med", "H:vipe+pi3x+geo", "H:vipe+da3+raw", "H:vipe+da3+med", "H:vipe+da3+geo",
         "H:vipe+vggt_omega+med", "H:vipe+vggt_omega+geo", "H:vipe+vote3+med", "H:vipe+vote3+geo",
         "H:vipe+vote4+med", "H:vipe+vote4+geo", "H:spatialvid_metric+pi3x+raw", "H:spatialvid_metric+pi3x+med",
         "H:spatialvid_metric+pi3x+geo", "H:spatialvid_metric+vote3+geo", "H:pi3x+da3+raw", "H:pi3x+vote3+med",
         "H:da3+pi3x+raw", "H:vipe+own+geo"]


KEY = ["vipe", "spatialvid_metric", "pi3x", "da3", "H:vipe+pi3x+raw", "H:vipe+pi3x+med", "H:vipe+pi3x+geo",
       "H:vipe+da3+med", "H:vipe+vote3+med", "H:vipe+vote4+med", "H:spatialvid_metric+pi3x+med",
       "H:spatialvid_metric+pi3x+geo", "H:pi3x+vote3+med"]


def to_md(t, fmt="{:.3g}"):
    cell = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))  # noqa: E731
    head = "| | " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * (len(t.columns) + 1) + "\n"
    return head + "\n".join(f"| {i} | " + " | ".join(cell(v) for v in row) + " |" for i, row in zip(t.index, t.values))


def prep(df):
    df = df.copy()
    if "error" not in df:
        df["error"] = np.nan
    df["ate_sim3_pct"] = df["ate_sim3_cm"] / df["traj_len_m"].clip(lower=1e-6)
    df["sim3_scale_err"] = (df["sim3_scale"] - 1).abs()
    for c in [c for c in df if c.startswith("pc_traj_")]:  # metric GT only (Sintel units are not metres)
        df.loc[df["sim3_scale"].isna(), c] = np.nan
    return df


def flags(df):  # ScanNet++ rows come from eval_bench.py without flags (same thresholds as eval3d_bench.py)
    L = df["traj_len_m"].clip(lower=1e-6)
    df["fail_ate_sim3"] = (df["ate_sim3_cm"] / 100 > 0.05 * L).astype(float)
    df["fail_ate"] = (df["ate_se3_cm"] / 100 > 0.05 * L).astype(float)
    df["fail_scale"] = ((df["sim3_scale"] - 1).abs() > 0.1).astype(float)
    df["fail_depth"] = (df["depth_metric_absrel"] > 0.2).astype(float)
    df["fail_fov"] = (df["fov_err_deg"] > 5).astype(float)
    return df


def table(df):
    g = df.groupby("method")
    med = g[[c for c in COLS if c in df]].median()
    fr = g[[c for c in FAILS if c in df]].mean().add_prefix("rate_")
    t = pd.concat([g.size().rename("n_ok"), med, fr], axis=1)
    return t[[c for c in COLS if c in t]]


def rank(df, metrics):
    med = df.groupby("method")[[m for m, _ in metrics if m in df]].median()
    rk = [med[m].rank(ascending=not hib) for m, hib in metrics if m in med and med[m].notna().sum() >= 2]
    return pd.concat(rk, axis=1).mean(axis=1) if rk else pd.Series(dtype=float)


def retention_table(df):
    out = {}
    for src, pre in (("vote", "vote_"), ("own", "own_")):
        cols = [c for c in df if c.startswith(pre + "ret") and c.endswith("seq_absrel")]
        if cols:
            t = df.groupby("method")[cols + [c.replace("seq_absrel", "metric_absrel") for c in cols if c.replace("seq_absrel", "metric_absrel") in df]].median().dropna(how="all")
            out[src] = t
    return pd.concat(out.values()) if out else pd.DataFrame()


def main():
    summ, md = {}, ["# Hybrid annotation evaluation — tables\n"]
    # ---------------- A
    A = prep(pd.read_csv(RAW / "A.csv"))
    okA = A[A["error"].isna()]
    okA.to_csv(H / "A_per_clip.csv", index=False, float_format="%.4g")
    R = {}
    for name, mets in (("geom", RANK_GEOM), ("pose", RANK_POSE), ("depth", RANK_DEPTH), ("metric", RANK_METRIC), ("consistency", RANK_CONS)):
        per = {}
        for ds, d in okA[okA.dataset.isin(RANK_DATASETS)].groupby("dataset"):
            if name == "metric" and not d["sim3_scale"].notna().any():
                continue
            if name == "consistency" and not d.get("pc_traj_fscore_5cm", pd.Series(dtype=float)).notna().any():
                continue
            per[ds] = rank(d, mets)
        R[name] = pd.DataFrame(per).mean(axis=1)
    RK = {}
    for name, mets in (("geom", RANK_GEOM), ("pose", RANK_POSE), ("depth", RANK_DEPTH), ("metric", RANK_METRIC), ("consistency", RANK_CONS)):
        per = {}
        for ds, d in okA[okA.dataset.isin(RANK_DATASETS) & okA.method.isin(KEY)].groupby("dataset"):
            if name == "metric" and not d["sim3_scale"].notna().any():
                continue
            if name == "consistency" and not d["pc_traj_fscore_5cm"].notna().any():
                continue
            per[ds] = rank(d, mets)
        RK[name] = pd.DataFrame(per).mean(axis=1)
    rkk = pd.DataFrame(RK).add_suffix("_rank")
    rkk["mean_rank"] = rkk.mean(axis=1)
    rkk = rkk.sort_values("mean_rank")
    rkk.round(2).to_csv(H / "ranking_A_key.csv")
    summ["A_key_ranking"] = json.loads(rkk.round(2).to_json(orient="index"))
    md.append("## A group ranking among the %d key combos (lower = better)\n" % len(rkk))
    md.append(to_md(rkk.round(2)))
    md.append("")
    rk = pd.DataFrame(R).add_suffix("_rank")
    rk["mean_rank"] = rk.mean(axis=1)
    rk = rk.sort_values("mean_rank")
    rk.round(2).to_csv(H / "ranking_A.csv")
    tA = table(okA).loc[lambda t: t.index.isin(rk.index)]
    summ["A"] = {"ranking": json.loads(rk.round(2).to_json(orient="index")),
                 "table": json.loads(tA.round(4).to_json(orient="index")),
                 "errors": A.loc[A["error"].notna(), ["method", "clip", "error"]].to_dict("records")}
    md.append("## A group ranking (mean rank over 7scenes/tum/bonn/sintel/kitti; lower = better; ranks among all %d combos)\n" % len(rk))
    md.append(to_md(rk.round(2)))
    md.append("\n## A group — focus combos, medians over all A clips (excl. ETH3D) and failure rates\n")
    md.append(to_md(tA.reindex([f for f in FOCUS if f in tA.index])))
    for ds in RANK_DATASETS:
        d = okA[okA.dataset == ds]
        if len(d):
            md.append(f"\n### A / {ds} — focus combos\n")
            md.append(to_md(table(d).reindex([f for f in FOCUS if f in d.method.unique()])))
    rt = retention_table(okA)
    if len(rt):
        rt.round(4).to_csv(H / "retention_A.csv")
        md.append("\n## A group — confidence retention (depth seq AbsRel / metric AbsRel on the most-confident X% of pixels; vote = spread, own = model conf)\n")
        md.append(to_md(rt))
    gcols = [c for c in okA if c.startswith("geo_fallback_") or c.startswith("scale_")]
    # (a) vs (b) scale comparison
    ab = []
    for pose in ("vipe", "spatialvid_metric"):
        for dep in ("pi3x", "da3", "vggt_omega"):
            a_, b_ = okA[okA.method == f"H:{pose}+{dep}+med"], okA[okA.method == f"H:{pose}+{dep}+geo"]
            if len(a_) and len(b_):
                m = a_.merge(b_, on="clip", suffixes=("_a", "_b"))
                ab.append({"combo": f"{pose}+{dep}", "n": len(m),
                           "median_ratio_b_over_a": float(np.median(m[f"scale_{dep}_b"] / m[f"scale_{dep}_a"])),
                           "geo_fallback_rate": float(m[f"geo_fallback_{dep}_b"].mean()),
                           "pc_traj_F_a": float(m["pc_traj_fscore_5cm_a"].median()), "pc_traj_F_b": float(m["pc_traj_fscore_5cm_b"].median()),
                           "metric_absrel_a": float(m["depth_metric_absrel_a"].median()), "metric_absrel_b": float(m["depth_metric_absrel_b"].median()),
                           "g20_epe_a": float(m["g20_epe_med_px_a"].median()), "g20_epe_b": float(m["g20_epe_med_px_b"].median())})
    if ab:
        ab = pd.DataFrame(ab).set_index("combo")
        summ["A_align_a_vs_b"] = json.loads(ab.round(4).to_json(orient="index"))
        md.append("\n## A group — alignment (a) median vs (b) multi-view geometry\n")
        md.append(to_md(ab))
    # ---------------- B + ScanNet++
    for kind in ("B", "scannetpp"):
        f = RAW / f"{kind}.csv"
        if not f.exists():
            continue
        d = prep(pd.read_csv(f))
        if kind == "scannetpp":
            d = flags(d)
            d["dataset"] = np.where(d["scene"].astype(str).isin(UNSEEN), "scannetpp_unseen3", "scannetpp_seen25")
        ok = d[d["error"].isna()]
        ok.to_csv(H / f"{kind}_per_clip.csv", index=False, float_format="%.4g")
        summ[kind] = {}
        for ds, g in ok.groupby("dataset"):
            t = table(g)
            summ[kind][ds] = json.loads(t.round(4).to_json(orient="index"))
            md.append(f"\n## {kind} / {ds} — focus combos\n")
            md.append(to_md(t.reindex([x for x in FOCUS if x in t.index])))
        rt = retention_table(ok)
        if len(rt):
            rt.round(4).to_csv(H / f"retention_{kind}.csv")
            md.append(f"\n## {kind} — confidence retention\n")
            md.append(to_md(rt))
    # ---------------- C wild
    f = RAW / "wild.csv"
    if f.exists():
        w = pd.read_csv(f)
        if "error" not in w:
            w["error"] = np.nan
        ok = w[w["error"].isna()]
        ok.to_csv(H / "wild_per_clip.csv", index=False, float_format="%.4g")
        cols = [c for c in ["g5_selfcons", "g20_selfcons", "g5_photo", "g20_photo", "depth_agree_absrel", "metric_scale_dev",
                            "vote_spread_med", "vote_conf_frac5", "speed_mps", "depth_p50", "implaus_speed", "implaus_depth"] if c in ok]
        t = ok.groupby("method")[cols].median()
        for c in ("implaus_speed", "implaus_depth"):
            if c in ok:
                t[c] = ok.groupby("method")[c].mean()
        summ["wild"] = json.loads(t.round(4).to_json(orient="index"))
        md.append("\n## C wild (72 clips, no GT) — medians; implaus_* = rate\n")
        md.append(to_md(t.reindex([x for x in FOCUS if x in t.index])))
        bycat = ok.pivot_table(index="cat", columns="method", values="g5_selfcons", aggfunc="median")
        bycat = bycat[[x for x in FOCUS[:17] if x in bycat]]
        md.append("\n### C wild — g5 self-consistency by category\n")
        md.append(to_md(bycat.T))
    (H / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    (H / "tables.md").write_text("\n".join(md) + "\n")
    print(rk.head(15).round(2).to_string())


if __name__ == "__main__":
    main()
