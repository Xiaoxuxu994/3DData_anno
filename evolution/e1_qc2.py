"""E1d: QC signals re-defined for K-conditioned labels (partners that share the same intrinsics).

python e1_qc2.py -> outputs/evolution/E1/qc_rule_v1b.csv (+ summary printed by e1_qc2_summary in qc-venv)
Label v1 = pi3x_kgate@vipe_d. Signals:
  xtraj_ba   Sim3 ATE % vs ViPE and SpatialVID (median)      xtraj_ff  vs DA3 / VGGT-Omega (median)
  xdepth_k   aligned AbsRel vs K-consistent partners da3_pkvipe and pi3x_pkvipe (median)
  kdis       |HFOV ViPE - median(GeoCalib, Pi3X, DA3)| (E1a disagreement, AUROC 0.87 for ViPE FOV failure)
  fov_sv     |HFOV label - HFOV SpatialVID|
"""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/qc")
sys.path.insert(0, "/data/home/z00533370/code/bench")
import qc_cross as QX  # noqa: E402
from eval3d_bench import load_method  # noqa: E402
from qc_common import BENCH, CACHE, vipe_static_mask  # noqa: E402

ROOT = Path("/data/home/z00533370/outputs/evolution")
SCALES = {g: pd.read_csv(ROOT / "E3" / f"scales_{k}.csv").set_index("clip") for g, k in (("A", "A"), ("B", "B"), ("C", "wild"))}
BN = {"A": "eval3d_A", "B": "eval3d_B", "C": "wild"}
KDIS = {}
for r in csv.DictReader(open(ROOT / "E1" / "fov_sources.csv")):
    f = lambda k: float(r[k]) if r[k] else np.nan  # noqa: E731
    KDIS[(r["bench"], r["clip"])] = abs(f("hfov_vipe") - np.nanmedian([f("hfov_geo"), f("hfov_pi3x"), f("hfov_da3")]))


def one(args):
    G, c = args
    clip, n = c["clip"], c["n_frames"]
    hw = tuple(int(x) for x in np.load(CACHE / G / f"{clip}.npz")["hw"])
    ks = np.linspace(0, n - 1, 8).astype(int)
    lab = vipe_static_mask(BENCH[G], clip, n, hw)
    static = None if lab is None else lab[ks] == 0
    L = lambda m: load_method(BENCH[G], m, clip, n, *hw)  # noqa: E731
    r = {"group": G, "clip": clip, "label": "v1", "cat": c.get("cat", c.get("dataset"))}
    try:
        T, K, D, _ = L("pi3x_kgate")
        k = float(SCALES[G].loc[clip, "k_vipe_d"])
        T = T.copy()
        T[:, :3, 3] *= k
        D = D * k
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:150]
        return r
    p, fv = T[:, :3, 3], QX.hfov(K, hw[1])
    tb, tf, dk = [], [], []
    for m, bucket in (("vipe", tb), ("spatialvid_metric", tb), ("da3", tf), ("vggt_omega", tf)):
        try:
            Tm, Km, Dm, _ = L(m)
            bucket.append(QX.traj_pct(p, Tm[:, :3, 3]))
            if m == "spatialvid_metric":
                r["fov_sv"] = abs(fv - QX.hfov(Km, hw[1]))
        except Exception:  # noqa: BLE001
            pass
    for m in ("da3_pkvipe", "pi3x_pkvipe"):
        try:
            dk.append(QX.depth_cmp(D[ks], L(m)[2][ks], static)[0])
        except Exception:  # noqa: BLE001
            pass
    r["xtraj_ba"] = float(np.nanmedian(tb)) if tb else np.nan
    r["xtraj_ff"] = float(np.nanmedian(tf)) if tf else np.nan
    r["xdepth_k"] = float(np.nanmedian(dk)) if dk else np.nan
    r["kdis"] = KDIS.get((BN[G], clip), np.nan)
    return r


if __name__ == "__main__":
    jobs = [(G, c) for G in ("A", "B", "C") for c in json.loads((BENCH[G] / "clips.json").read_text())
            if c.get("dataset") != "eth3d_mvs"]
    with ProcessPoolExecutor(24) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs)))
    q = pd.read_csv(ROOT / "E1" / "qc_rule.csv")
    q = q[q["label"] == "v1"][["group", "clip", "geom_fail", "fail_scale", "rule", "xall_traj", "xba_fov", "xff_depth"]]
    df.merge(q, on=["group", "clip"], how="left").to_csv(ROOT / "E1" / "qc_rule_v1b.csv", index=False, float_format="%.5g")
    print("saved", len(df))
