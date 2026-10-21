"""E12 finalize: merge evo_report tables (tables.md / summary.json), the analytic scale tables and the cost stats.

  python e12_finalize.py   (after evo_report.py E12 and e12_scale_report.py)
-> E12/tables.md (key table + cost + scale tables + full method tables), E12/summary.json, E12/raw/cost_per_clip.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

E12 = Path("/data/home/z00533370/outputs/evolution/E12")
BENCH = Path("/data/home/z00533370/outputs/bench")
evo_md = (E12 / "tables.md").read_text()
if evo_md.startswith("# E12 — summary"):  # idempotent: keep only the evo_report part
    evo_md = evo_md[evo_md.index("<!-- evo_report -->") + len("<!-- evo_report -->"):]
evo = json.loads((E12 / "summary.json").read_text())
evo = evo.get("evo_report", evo)
scale = json.loads((E12 / "scale_summary.json").read_text())
scale_md = (E12 / "scale_tables.md").read_text()

rows = []
for b in ("eval3d_A", "eval3d_B", "scannetpp", "wild"):
    for f in sorted((BENCH / b).glob("moge_*/status.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                rows.append({"bench": b, **{k: r[k] for k in ("clip", "model", "mode", "n_frames", "sec", "peak_gb", "mask_frac", "hfov_out", "hfov_in")}})
cost = pd.DataFrame(rows)
cost.to_csv(E12 / "raw" / "cost_per_clip.csv", index=False)
big = cost[cost.n_frames >= 100]
ct = big.assign(ms_per_frame=1000 * big.sec / big.n_frames).groupby(["model", "mode"]).agg(
    clips=("clip", "count"), ms_per_frame_med=("ms_per_frame", "median"), peak_gb=("peak_gb", "max"),
    mask_frac_med=("mask_frac", "median"))
ct["sec_per_180f_clip"] = ct["ms_per_frame_med"] * 0.18
ct["total_inference_gpu_h_all_benches"] = cost.groupby(["model", "mode"])["sec"].sum() / 3600


def g(kind, src, col):
    try:
        return evo[kind][src][col]
    except KeyError:
        return np.nan


def s(kind, src, col="fail_scale"):
    try:
        return scale[kind][src][col]
    except KeyError:
        return np.nan


key_scale = pd.DataFrame({
    src_name: {f"{k} fail_scale": s(k, src) for k in ("A", "B", "scannetpp")}
    | {f"{k} traj_err_med": s(k, src, "traj_err_med") for k in ("A", "B", "scannetpp")}
    for src_name, src in [("v1.3 (ViPE sqrt d*t)", "v1.3 sqrt(ViPE d*t)"), ("ViPE own trajectory", "ViPE own trajectory"),
                          ("Pi3X raw", "own (raw Pi3X)"), ("DA3 own trajectory", "DA3 own trajectory"),
                          ("DA3 depth ratio", "DA3 depth ratio"),
                          ("MoGe-2 L kgate", "moge_m2l_kgate"), ("MoGe-3 L kgate", "moge_m3l_kgate"),
                          ("MoGe-3 g free", "moge_m3g_free"), ("MoGe-3 g kgate", "moge_m3g_kgate"),
                          ("med3(v13, MoGe-3g-k, DA3)", "med3(v13,moge_m3g_kgate,da3)"),
                          ("gm(v13, MoGe-3g-k)", "gm(v13,moge_m3g_kgate)"),
                          ("med3(ViPE d, ViPE t, MoGe-3g-k)", "med3(vd,vt,moge_m3g_kgate)")]}).T
agree = pd.DataFrame({f"agree(MoGe-3g-k, {t}%)": {
    f"{k} {c}": s(k, f"agree(moge_m3g_kgate,{t}%)", c) for k in ("A", "B", "scannetpp") for c in ("flag_rate", "fail_scale")}
    | {"C flag_rate": scale["wild"]["moge_m3g_kgate"][f"flag@{t}%"]} for t in (5, 10, 15, 20)}).T
DCOLS = ["depth_metric_absrel", "depth_seq_absrel", "pc_fscore_5cm", "pc_traj_fscore_5cm", "g5_epe_med_px",
         "g20_epe_med_px", "rate_fail_depth"]
dsrc = {}
for kind in ("A", "scannetpp"):
    for m in ["pi3x_kgate@v13", "D:moge_m3g_kgate", "Df:moge_m2l_kgate", "Df:moge_m3l_kgate", "Df:moge_m3g_kgate",
              "vipe", "VD:moge_m3g_kgate", "VD:moge_m2l_kgate"]:
        dsrc[(kind, m)] = {c: g(kind, m, c) for c in DCOLS}
dsrc = pd.DataFrame(dsrc).T
wild = pd.DataFrame({m: {c: evo["wild"][m][c] for c in ("g5_selfcons", "g20_selfcons", "g5_photo", "depth_agree_absrel", "metric_scale_dev")}
                     for m in ["pi3x_kgate@v13", "D:moge_m3g_kgate", "Df:moge_m3g_kgate", "Df:moge_m2l_kgate", "vipe",
                               "VD:moge_m3g_kgate", "pi3x_kgate@moge_m3g_kgate", "pi3x_kgate@med3_moge_m3g_kgate"]}).T
raw = {}
for kind in ("A", "B", "scannetpp"):
    for m in ["pi3x_kgate", "da3", "vipe"] + [f"moge_{a}_{b}" for a in ("m2l", "m3l", "m3g") for b in ("free", "kgate")]:
        raw[(kind, m)] = {c: g(kind, m, c) for c in ("depth_metric_absrel", "depth_seq_absrel", "depth_seq_delta1", "fov_err_deg", "rate_fail_fov", "rate_fail_depth")}
raw = pd.DataFrame(raw).T

import sys  # noqa: E402
sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import report_hybrid as R  # noqa: E402

f = lambda t: R.to_md(t.astype(float).round(3))  # noqa: E731
md = ["# E12 — summary (MoGe-2 / MoGe-3 on A (no ETH3D) / B / ScanNet++ / C)\n",
      "Scale fail = |Sim3 scale - 1| > 10% on metric-GT clips (A 28, B 13, ScanNet++ 28). MoGe as scale source = "
      "pi3x_kgate x median over frames of median(D_moge)/median(D_pi3x). kgate = fov_x from the v1.3 gated K.\n",
      "\n## 1. Metric scale: sources and votes (fail rate / median traj error)\n\n" + f(key_scale),
      "\n## 2. 'Agree' gate: use sqrt(v13 * MoGe-3g) when within T of v1.3, else flag\n\n" + f(agree),
      "\n## 3. Single-image depth quality (median over clips) and FOV error\n\n" + f(raw),
      "\n## 4. MoGe as depth source (v1.3 cameras, gated K; D = one factor per clip, Df = per frame; VD = ViPE cameras)\n\n" + f(dsrc),
      "\n## 5. C (wild) self-consistency / agreement\n\n" + f(wild),
      "\n## 6. Cost on H200 (batch 16, ~250k px input = pi3x_kgate grid, resolution_level 9, fp16)\n\n" + f(ct),
      "\n---\n" + scale_md.replace("# E12 — metric-scale tables", "# Appendix A — metric-scale tables"),
      "\n---\n# Appendix B — full method tables (evo_report)\n<!-- evo_report -->", evo_md]
(E12 / "tables.md").write_text("\n".join(md))
summary = {"key_scale": json.loads(key_scale.to_json(orient="index")), "agree": json.loads(agree.to_json(orient="index")),
           "depth_source": {f"{k}|{m}": dsrc.loc[(k, m)].to_dict() for k, m in dsrc.index},
           "raw_depth": {f"{k}|{m}": raw.loc[(k, m)].to_dict() for k, m in raw.index},
           "wild": json.loads(wild.to_json(orient="index")), "cost": {f"{a}_{b}": v for (a, b), v in ct.to_dict(orient="index").items()},
           "scale_tables": scale, "evo_report": evo}
(E12 / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print("\n".join(md[:8]))
