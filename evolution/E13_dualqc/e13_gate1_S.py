"""E13: gate 1 (cross-model) on ScanNet++ for the composites, which line G / E1 / E8 never scored.
  v0 rule  = max(xall_traj/3.5%, xba_fov/4.1deg, xff_depth/0.157) of the v0 label vs the 7 annotators (e1_qc.py)
  v1c rule = max(xtraj vs ViPE/3.27%, depth vs DA3(ViPE cams)/0.194, K disagreement/13.2deg) of the v13 label (e8_qc_cheap.py)
  -> outputs/evolution/E13/raw/gate1_S.csv
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
import e13_signals as E  # noqa: E402
import qc_cross as QX  # noqa: E402

KD = {}
for r in csv.DictReader(open(E.H0 / "outputs/evolution/E1/fov_sources.csv")):
    if r["bench"] == "scannetpp":
        f = lambda k: float(r[k]) if r[k] else np.nan  # noqa: E731
        KD[r["clip"]] = abs(f("hfov_vipe") - np.nanmedian([f("hfov_geo"), f("hfov_pi3x"), f("hfov_da3")]))


def one(c):
    G, B, clip, n = "S", E.QC.BENCH["S"], c["clip"], c["n_frames"]
    hw = tuple(int(x) for x in np.load(E.LINKS / G / f"{clip}.npz")["hw"])
    ks = np.linspace(0, n - 1, 8).astype(int)
    lab = E.QC.vipe_static_mask(B, clip, n, hw)
    static = None if lab is None else lab[ks] == 0
    L = lambda m: E.load_label(B, m, clip, n, *hw)  # noqa: E731
    data = {}
    for m in E.QC.METHODS:
        try:
            T, K, D, _ = L(m)
            if np.isfinite(T).all() and np.isfinite(K).all():
                data[m] = (T[:, :3, 3], QX.hfov(K, hw[1]), D[ks])
        except Exception:  # noqa: BLE001
            pass
    rows = []
    T, K, D, _ = L("v0")
    me = (T[:, :3, 3], QX.hfov(K, hw[1]), D[ks])
    tr = [QX.traj_pct(me[0], v[0]) for v in data.values()]
    fb = [abs(me[1] - v[1]) for o, v in data.items() if o in E.QC.BA_CLASS]
    fd = [QX.depth_cmp(me[2], v[2], static)[0] for o, v in data.items() if o not in E.QC.BA_CLASS]
    r = {"group": G, "clip": clip, "label": "v0", "xall_traj": np.nanmedian(tr), "xba_fov": np.nanmedian(fb), "xff_depth": np.nanmedian(fd)}
    r["gate1"] = np.nanmax([r["xall_traj"] / 3.5, r["xba_fov"] / 4.1, r["xff_depth"] / 0.157])
    rows.append(r)
    T, K, D, _ = L("v13")
    r = {"group": G, "clip": clip, "label": "v13", "xtraj_ba": QX.traj_pct(T[:, :3, 3], data["vipe"][0]) if "vipe" in data else np.nan,
         "xdepth_k": QX.depth_cmp(D[ks], L("da3_pkvipe")[2][ks], static)[0], "kdis": KD.get(clip, np.nan)}
    r["gate1"] = np.nanmax([r["xtraj_ba"] / 3.27, r["xdepth_k"] / 0.194, r["kdis"] / 13.2])
    rows.append(r)
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(14) as ex:
        df = pd.DataFrame([r for rs in ex.map(one, E.QC.clips("S")) for r in rs])
    df.to_csv(E.E13 / "raw/gate1_S.csv", index=False, float_format="%.5g")
    print(df.groupby("label")["gate1"].describe())
