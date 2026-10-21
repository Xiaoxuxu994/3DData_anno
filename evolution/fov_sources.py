"""E1 helper: per-clip HFOV of every intrinsics source (GT, ViPE, GeoCalib, Pi3X/DA3 self-estimate).

python fov_sources.py -> outputs/evolution/E1/fov_sources.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/home/z00533370")
BB = ROOT / "outputs/bench"
GEO = ROOT / "outputs/evolution/E1/geocalib"
SPP = ROOT / "data/scannetpp_sample30"


def hfov(fx, W):
    return float(np.degrees(2 * np.arctan(W / 2 / fx)))


rows = []
for b in ("eval3d_A", "eval3d_B", "scannetpp", "wild"):
    for c in json.loads((BB / b / "clips.json").read_text()):
        clip = c["clip"]
        if c.get("dataset") == "eth3d_mvs":
            continue
        r = {"bench": b, "clip": clip, "dataset": c.get("dataset", c.get("cat", "scannetpp")),
             "hard_case": c.get("hard_case", False)}
        gf = GEO / b / f"{clip}.json"
        if not gf.exists():
            continue
        g = json.loads(gf.read_text())
        W = g["W"]
        r["hfov_geo"], r["geo_fcv"], r["geo_unc"] = g["hfov"], g["f_cv"], g["unc_rel"]
        if b == "scannetpp":
            m = np.load(SPP / c["scene"] / "scene_metadata.npz")
            idx = {k: i for i, k in enumerate(m["images"])}
            r["hfov_gt"] = hfov(m["intrinsics"][idx[c["frames"][0]]][0, 0], W)
        elif b != "wild":
            r["hfov_gt"] = hfov(np.load(BB / b / "gt" / f"{clip}.npz", allow_pickle=True)["K"][0, 0, 0], W)
        iz = np.load(BB / b / "vipe" / "intrinsics" / f"{clip}.npz")["data"]
        r["hfov_vipe"] = hfov(float(np.median(iz[:, 0])), W)
        for m in ("pi3x", "da3", "vggt_omega"):
            p = BB / b / m / clip / "pred.npz"
            if p.exists():
                z = np.load(p)
                w = z["depth"].shape[2]
                r[f"hfov_{m}"] = hfov(float(np.median(z["K"][:, 0, 0])), w)
        rows.append(r)
df = pd.DataFrame(rows)
for s in ("vipe", "geo", "pi3x", "da3"):
    if "hfov_gt" in df:
        df[f"err_{s}"] = (df[f"hfov_{s}"] - df["hfov_gt"]).abs()
df["vipe_geo_diff"] = (df["hfov_vipe"] - df["hfov_geo"]).abs()
out = ROOT / "outputs/evolution/E1/fov_sources.csv"
df.to_csv(out, index=False, float_format="%.3f")
gt = df[df["hfov_gt"].notna()] if "hfov_gt" in df else df.iloc[:0]
print(gt.groupby("dataset")[[c for c in df.columns if c.startswith("err_")]].median().round(2).to_string())
print("fail>5deg rate:", {s: round(float((gt[f"err_{s}"] > 5).mean()), 3) for s in ("vipe", "geo", "pi3x", "da3")})
print("vipe-geo diff median by bench:", df.groupby("bench")["vipe_geo_diff"].median().round(2).to_dict())
