"""Aggregate eval3d A/B (GT), ScanNet++ unseen-3 (GT, reused) and wild C (no GT) into one report.

  python eval3d_report.py  -> outputs/bench/eval3d_report/{summary.json, A_<ds>.csv, B_<ds>.csv, scannetpp_unseen3.csv,
                                                          wild_per_clip.csv, ranking_A.csv, tables.md}
Ranking (A group): per dataset, each method's median of every rank metric is ranked across methods (1 = best);
the dataset rank is the mean over metrics; the cross-dataset rank is the mean of the dataset ranks.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/data/home/z00533370/outputs/bench")
R = OUT / "eval3d_report"
METHODS = ["vipe", "spatialvid", "spatialvid_metric", "voyager", "vggt", "vggt_omega", "da3", "pi3x"]
RANK_DATASETS = ["7scenes", "tum", "bonn", "sintel", "kitti"]  # eth3d_mvs (unordered photos) reported separately
UNSEEN = ["1c7a683c92", "6c14d5fd01", "a4c043ac48"]
# (column, higher_is_better); ate is normalised by trajectory length so datasets with long drives are comparable
RANK_GEOM = [("ate_sim3_pct", False), ("rre_deg", False), ("fov_err_deg", False), ("depth_seq_absrel", False),
             ("pc_fscore_5cm", True), ("g5_epe_med_px", False)]
RANK_POSE = [("ate_sim3_pct", False), ("rre_deg", False), ("fov_err_deg", False)]
RANK_DEPTH = [("depth_seq_absrel", False), ("pc_fscore_5cm", True), ("g5_epe_med_px", False)]
RANK_METRIC = [("sim3_scale_err", False), ("depth_metric_absrel", False)]
SHOW = ["ate_sim3_cm", "ate_sim3_pct", "ate_se3_cm", "sim3_scale", "rte_cm", "rre_deg", "fov_err_deg",
        "depth_metric_absrel", "depth_seq_absrel", "depth_seq_delta1", "depth_frame_absrel",
        "pc_acc_m", "pc_comp_m", "pc_fscore_5cm", "g5_epe_med_px", "g5_vipe_selfcons_5pct", "sec"]
FAILS = ["fail_ate_sim3", "fail_ate", "fail_scale", "fail_depth", "fail_fov"]


def to_md(t, fmt="{:.3g}"):
    cell = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))  # noqa: E731
    head = "| | " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * (len(t.columns) + 1) + "\n"
    return head + "\n".join(f"| {i} | " + " | ".join(cell(v) for v in row) + " |" for i, row in zip(t.index, t.values))


def prep(df):
    df = df.copy()
    df["ate_sim3_pct"] = 100 * df["ate_sim3_cm"] / 100 / df["traj_len_m"].clip(lower=1e-6)
    df["sim3_scale_err"] = (df["sim3_scale"] - 1).abs()
    return df


def flags(df, metric):
    """Failure flags for CSVs written by eval_bench.py (ScanNet++), same thresholds as eval3d_bench.py."""
    L = df["traj_len_m"].clip(lower=1e-6)
    df["fail_ate_sim3"] = (df["ate_sim3_cm"] / 100 > 0.05 * L).astype(float)
    if metric:
        df["fail_ate"] = (df["ate_se3_cm"] / 100 > 0.05 * L).astype(float)
        df["fail_scale"] = ((df["sim3_scale"] - 1).abs() > 0.1).astype(float)
        df["fail_depth"] = (df["depth_metric_absrel"] > 0.2).astype(float)
    df["fail_fov"] = (df["fov_err_deg"] > 5).astype(float)
    return df


def table(df, by):
    g = df.groupby(by)
    med = g[[c for c in SHOW if c in df]].median()
    fr = g[[c for c in FAILS if c in df]].mean().add_prefix("rate_")
    n = g.size().rename("n_ok")
    return pd.concat([n, med, fr], axis=1)


def rank(df, metrics):
    """df: rows = clip x method for one dataset -> mean rank per method over metrics."""
    med = df.groupby("method")[[m for m, _ in metrics if m in df]].median()
    rk = []
    for m, hib in metrics:
        if m in med and med[m].notna().sum() >= 2:
            rk.append(med[m].rank(ascending=not hib, na_option="keep"))
    return pd.concat(rk, axis=1).mean(axis=1) if rk else pd.Series(dtype=float)


def main():
    R.mkdir(exist_ok=True)
    summ = {"methods": METHODS, "groups": {}}
    # ---------------- A / B (GT)
    for grp, bench in (("A", "eval3d_A"), ("B", "eval3d_B")):
        clips = json.loads((OUT / bench / "clips.json").read_text())
        df = prep(pd.read_csv(OUT / bench / "eval" / "per_clip.csv"))
        if "error" not in df:
            df["error"] = np.nan
        err, ok = df[df["error"].notna()], df[df["error"].isna()]
        G = {"n_clips": len(clips), "datasets": {}, "success": {}, "errors": err[["method", "clip", "error"]].to_dict("records")}
        for m in METHODS:
            G["success"][m] = f"{int((ok.method == m).sum())}/{len(clips)}"
        for ds, d in ok.groupby("dataset"):
            d.sort_values(["method", "clip"]).to_csv(R / f"{grp}_{ds}.csv", index=False, float_format="%.4f")
            t = table(d, "method").round(4)
            G["datasets"][ds] = {"n_clips": sum(c["dataset"] == ds for c in clips),
                                 "table": json.loads(t.to_json(orient="index"))}
        G["group_table"] = json.loads(table(ok, "method").round(4).to_json(orient="index"))
        if grp == "A":
            per_ds_geom, per_ds_metric, per_ds_pose, per_ds_depth = {}, {}, {}, {}
            for ds, d in ok[ok.dataset.isin(RANK_DATASETS)].groupby("dataset"):
                # spatialvid_metric only rescales spatialvid -> one entry per ranking (raw for geometry, rescaled for metric)
                dg = d[d.method != "spatialvid_metric"]
                per_ds_geom[ds] = rank(dg, RANK_GEOM)
                per_ds_pose[ds] = rank(dg, RANK_POSE)
                per_ds_depth[ds] = rank(dg, RANK_DEPTH)
                if d["sim3_scale"].notna().any():
                    per_ds_metric[ds] = rank(d[d.method != "spatialvid"], RANK_METRIC)
            rg, rm = pd.DataFrame(per_ds_geom), pd.DataFrame(per_ds_metric)
            rank_tab = pd.concat([rg.add_prefix("geom_"), rm.add_prefix("metric_")], axis=1)
            rank_tab["geom_rank_mean"] = rg.mean(axis=1)
            rank_tab["metric_rank_mean"] = rm.mean(axis=1)
            rank_tab["pose_rank_mean"] = pd.DataFrame(per_ds_pose).mean(axis=1)
            rank_tab["depth_rank_mean"] = pd.DataFrame(per_ds_depth).mean(axis=1)
            rank_tab = rank_tab.sort_values("geom_rank_mean")
            rank_tab.round(3).to_csv(R / "ranking_A.csv")
            G["ranking"] = json.loads(rank_tab.round(3).to_json(orient="index"))
        summ["groups"][grp] = G
    # ---------------- ScanNet++ unseen-3 (reused results of the ScanNet++ bench)
    rows = []
    for m in METHODS:
        f = OUT / "scannetpp" / "eval" / f"{m}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["method"] = m
            rows.append(d)
    if rows:
        sp = flags(prep(pd.concat(rows, ignore_index=True)), metric=True)
        sp["unseen"] = sp["scene"].astype(str).isin(UNSEEN)
        sp[sp.unseen].to_csv(R / "scannetpp_unseen3.csv", index=False, float_format="%.4f")
        summ["scannetpp"] = {
            "unseen3": json.loads(table(sp[sp.unseen], "method").round(4).to_json(orient="index")),
            "all30": json.loads(table(sp, "method").round(4).to_json(orient="index")),
            "unseen3_rank_geom": rank(sp[sp.unseen], RANK_GEOM).round(2).to_dict()}
    # ---------------- C (wild, no GT)
    W = OUT / "wild" / "eval"
    if (W / "summary.json").exists():
        summ["wild"] = json.loads((W / "summary.json").read_text())
        for f in ("per_clip.csv", "agreement.csv", "consensus.csv"):
            shutil.copy(W / f, R / f"wild_{f}")
    (R / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    # ---------------- markdown tables for the README
    md = []
    for grp in ("A", "B"):
        G = summ["groups"][grp]
        md.append(f"## {grp} group success\n\n" + " | ".join(f"{m} {s}" for m, s in G["success"].items()))
        for ds, v in G["datasets"].items():
            t = pd.DataFrame(v["table"]).T
            cols = [c for c in ["n_ok", "ate_sim3_pct", "ate_sim3_cm", "sim3_scale", "rre_deg", "fov_err_deg",
                                "depth_metric_absrel", "depth_seq_absrel", "pc_fscore_5cm", "g5_epe_med_px",
                                "rate_fail_ate_sim3", "rate_fail_ate", "rate_fail_scale", "rate_fail_depth",
                                "rate_fail_fov", "sec"] if c in t]
            md.append(f"\n### {grp} / {ds} ({v['n_clips']} clips) — median, rate_* = failure rate\n\n" + to_md(t[cols]))
        if "ranking" in G:
            md.append("\n### A ranking (mean of per-dataset ranks, 1 = best)\n\n" + to_md(pd.DataFrame(G["ranking"]).T, "{:.2f}"))
    if "scannetpp" in summ:
        t = pd.DataFrame(summ["scannetpp"]["unseen3"]).T
        md.append("\n## ScanNet++ unseen-3 (" + ", ".join(UNSEEN) + ")\n\n" + to_md(t))
    (R / "tables.md").write_text("\n".join(md) + "\n")
    print((R / "tables.md").read_text()[:6000])


if __name__ == "__main__":
    main()
