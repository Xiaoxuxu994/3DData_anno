"""Summarise an evolution experiment's raw CSVs with the line-H table definitions.

python evo_report.py EXP_DIR  (EXP_DIR/raw/{A,B,scannetpp,wild}.csv) -> EXP_DIR/tables.md, EXP_DIR/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import report_hybrid as R  # noqa: E402

EXP = Path(sys.argv[1])
RAW = EXP / "raw"
md, summ = [f"# {EXP.name} — tables (same metric definitions as outputs/hybrid/tables.md)\n"], {}
WILD_COLS = ["g5_selfcons", "g20_selfcons", "g5_photo", "hfov_deg", "depth_agree_absrel", "metric_scale_dev",
             "implaus_speed", "implaus_depth"]


def gt_tables(kind, df):
    df = R.prep(df[df["error"].isna()] if "error" in df else df)
    if kind == "scannetpp":
        df = R.flags(df)
    t = R.table(df)
    md.append(f"\n## {kind} — all clips (median; rate_* = failure rate)\n\n" + R.to_md(t))
    summ[kind] = json.loads(t.to_json(orient="index"))
    if kind == "A":
        rk = pd.concat({ds: pd.concat({"geom": R.rank(g, R.RANK_GEOM), "metric": R.rank(g, R.RANK_METRIC),
                                       "cons": R.rank(g, R.RANK_CONS)}, axis=1)
                        for ds, g in df.groupby("dataset") if ds in R.RANK_DATASETS})
        mean = rk.groupby(level=1).mean()
        mean["mean"] = mean.mean(1)
        md.append("\n## A — mean rank over 7scenes/tum/bonn/sintel/kitti among the methods of this experiment\n\n"
                  + R.to_md(mean.sort_values("mean")))
        summ["A_rank"] = json.loads(mean.to_json(orient="index"))
        for ds, g in df.groupby("dataset"):
            md.append(f"\n### A / {ds}\n\n" + R.to_md(R.table(g)))
    if kind == "B":
        for ds, g in df.groupby("dataset"):
            md.append(f"\n### B / {ds}\n\n" + R.to_md(R.table(g)))
    if kind == "scannetpp":
        u = df[df["scene"].isin(R.UNSEEN)]
        md.append("\n### scannetpp unseen-3 (not in MapAnything train split)\n\n" + R.to_md(R.table(u)))
        summ["scannetpp_unseen3"] = json.loads(R.table(u).to_json(orient="index"))


for kind in ("A", "B", "scannetpp", "wild"):
    f = RAW / f"{kind}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    if "error" not in df:
        df["error"] = np.nan
    if kind == "wild":
        ok = df[df["error"].isna()]
        cols = [c for c in WILD_COLS if c in ok]
        t = ok.groupby("method")[cols].median()
        md.append("\n## wild (C) — medians (no GT)\n\n" + R.to_md(t))
        tc = ok.groupby(["cat", "method"])[["g5_selfcons", "depth_agree_absrel"]].median().unstack(0)
        md.append("\n### wild by category: g5 self-consistency / depth agreement\n\n" + R.to_md(tc.round(3)))
        summ["wild"] = json.loads(t.to_json(orient="index"))
    else:
        gt_tables(kind, df)
(EXP / "tables.md").write_text("\n".join(md))
(EXP / "summary.json").write_text(json.dumps(summ, indent=1))
print("\n".join(md)[:6000])
