"""E6c: since v1.x uses ViPE only for K and metric scale, is half-frame-rate ViPE good enough for that role? (CPU)

python e6c_sub2_role.py -> outputs/evolution/E6/sub2_role.csv + summary
Per clip (A, B, C): ViPE full vs ViPE every-2nd-frame (outputs/qc/variants/*_sub2/vipe):
  dhfov  |HFOV_sub2 - HFOV_full|  (the K handed to Pi3X)
  k_full / k_sub2  v1.1 scale sqrt(k_depth * k_traj) of pi3x_kgate w.r.t. each ViPE run (sub2: on its frame subset)
A/B: scale failure of the v1.1 label when k_sub2 replaces k_full: s_new = s_old * k_full / k_sub2
(s_old = Sim3 scale to GT of pi3x_kgate@vipe_td from E3b).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/bench")
sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
sys.path.insert(0, "/data/home/z00533370/code/evolution")
import eval3d_bench as M  # noqa: E402
from e5_eval import load_vipe_dir  # noqa: E402  (imports run_hybrid; argv[1] must be a kind)
from eval_vipe import umeyama  # noqa: E402

R = Path("/data/home/z00533370/outputs")
E = R / "evolution"
VAR = R / "qc/variants"
BEN = {"A": R / "bench/eval3d_A", "B": R / "bench/eval3d_B", "C": R / "bench/wild"}


def kscale(c2w, D, cv, Dv):
    ks = np.linspace(0, len(D) - 1, 10).astype(int)
    a, b = Dv[ks], D[ks]
    mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    kd = float(np.median(a[mk]) / np.median(b[mk])) if mk.sum() > 1000 else 1.0
    pv, pm = cv[:, :3, 3], c2w[:, :3, 3]
    Lv = np.linalg.norm(np.diff(pv, axis=0), axis=1).sum()
    s, Rr, t = umeyama(pm, pv, True)
    res = np.sqrt(((s * pm @ Rr.T + t - pv) ** 2).sum(1).mean()) / max(Lv, 1e-9)
    dv = float(np.median(a[mk])) if mk.sum() > 1000 else np.inf
    kt = s if (Lv > 0.02 * dv and res < 0.2 and s > 0) else kd
    return float(np.sqrt(kd * kt))


rows = []
for G, B in BEN.items():
    sub = {c["clip"]: c for c in json.loads((VAR / f"{G}_sub2" / "clips.json").read_text())}
    for c in json.loads((B / "clips.json").read_text()):
        if c.get("dataset") == "eth3d_mvs" or c["clip"] not in sub:
            continue
        clip, n = c["clip"], c["n_frames"]
        idx = np.array(sub[clip]["src_index"])
        H, W = (540, 960) if G == "C" else (c["hw"][0], c["hw"][1])
        try:
            T, K, D, _ = M.load_method(B, "pi3x_kgate", clip, n, H, W)
            cf, Kf, Df, _ = load_vipe_dir(B, "vipe", clip, n, H, W)
            cs, Ks, Ds, _ = load_vipe_dir(VAR / f"{G}_sub2", "vipe", clip, len(idx), H, W)
        except Exception as e:  # noqa: BLE001
            rows.append({"group": G, "clip": clip, "error": str(e)[:120]})
            continue
        hf = lambda Kx: float(np.degrees(2 * np.arctan(W / 2 / np.median(Kx[:, 0, 0]))))  # noqa: E731
        rows.append({"group": G, "clip": clip, "cat": c.get("cat", c.get("dataset")),
                     "hfov_full": hf(Kf), "hfov_sub2": hf(Ks), "dhfov": abs(hf(Kf) - hf(Ks)),
                     "k_full": kscale(T, D, cf, Df), "k_sub2": kscale(T[idx], D[idx], cs, Ds)})
d = pd.DataFrame(rows)
d["k_ratio"] = d["k_full"] / d["k_sub2"]
gt = []
for g in "AB":
    x = pd.read_csv(E / "E3b" / "raw" / f"{g}.csv")
    x = x[x["method"] == "pi3x_kgate@vipe_td"][["clip", "sim3_scale"]].assign(group=g)
    gt.append(x)
d = d.merge(pd.concat(gt), on=["group", "clip"], how="left")
d["s_new"] = d["sim3_scale"] * d["k_ratio"]
d.to_csv(E / "E6" / "sub2_role.csv", index=False, float_format="%.4g")
for G, g in d.groupby("group"):
    print(G, "n", len(g), "dHFOV median/p90", round(g["dhfov"].median(), 2), round(g["dhfov"].quantile(0.9), 2),
          "| |log k ratio| median/p90", round(np.abs(np.log(g["k_ratio"])).median(), 3), round(np.abs(np.log(g["k_ratio"])).quantile(0.9), 3),
          "| >10% scale change", round((np.abs(np.log(g["k_ratio"])) > np.log(1.1)).mean(), 3))
    m = g["sim3_scale"].notna()
    if m.any():
        print("   scale fail full", round((np.abs(g.loc[m, "sim3_scale"] - 1) > 0.1).mean(), 3),
              "sub2", round((np.abs(g.loc[m, "s_new"] - 1) > 0.1).mean(), 3), "n", int(m.sum()))
c = d[d["group"] == "C"]
print("C by cat dHFOV>3deg:", c.assign(x=c["dhfov"] > 3).groupby("cat")["x"].mean().round(2).to_dict())
