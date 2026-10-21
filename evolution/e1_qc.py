"""E1c: does the line-G QC rule still work on the new label, and how many wild labels does it keep?

python e1_qc.py -> outputs/evolution/E1/qc_rule.csv + printed summary
Labels: v0 = H:vipe+pi3x+med ; v1 = pi3x_kgate@vipe_d (K-conditioned Pi3X, metric scale from ViPE depth).
Per label and clip, the outputs/qc cross-model signals against the 7 original annotators (qc_cross.py definitions):
xall_traj (Sim3 ATE %), xba_fov (deg vs ViPE / SpatialVID), xff_depth (aligned AbsRel vs feed-forward family);
rule score = max(xall_traj/3.5, xba_fov/4.1, xff_depth/0.157), reject if > 1 (outputs/qc/README.md).
New no-GT signals for the conditioned label: cond_dfov = |HFOV(Pi3X image-only) - HFOV(label)|,
cond_ddepth = aligned AbsRel between Pi3X image-only and the label depth.
GT groups: geometric failure = fail_ate_sim3 | fail_fov | fail_depth (metric GT only) from the E1b/E3 raw tables.
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/qc")
sys.path.insert(0, "/data/home/z00533370/code/hybrid")
sys.path.insert(0, "/data/home/z00533370/code/bench")
import hybrid as HY  # noqa: E402
import qc_cross as QX  # noqa: E402
from eval3d_bench import load_method  # noqa: E402
from qc_common import BA_CLASS, BENCH, CACHE, METHODS, vipe_static_mask  # noqa: E402

ROOT = Path("/data/home/z00533370/outputs/evolution")
SCALES = {g: pd.read_csv(ROOT / "E3" / f"scales_{k}.csv").set_index("clip") for g, k in (("A", "A"), ("B", "B"), ("C", "wild"))}


def label(G, name, clip, n, hw):
    B = BENCH[G]
    base = lambda m: load_method(B, m, clip, n, *hw)  # noqa: E731
    if name == "v0":
        return HY.compose("H:vipe+pi3x+med", base)[:4]
    c2w, K, D, s = base("pi3x_kgate")
    k = float(SCALES[G].loc[clip, "k_vipe_d"]) if clip in SCALES[G].index else 1.0
    c2w = c2w.copy()
    c2w[:, :3, 3] *= k
    return c2w, K, D * k, s


def one(args):
    G, c = args
    clip, n = c["clip"], c["n_frames"]
    hw = tuple(int(x) for x in np.load(CACHE / G / f"{clip}.npz")["hw"])
    ks = np.linspace(0, n - 1, 8).astype(int)
    lab = vipe_static_mask(BENCH[G], clip, n, hw)
    static = None if lab is None else lab[ks] == 0
    data = {}
    for m in METHODS:
        try:
            T, K, D, _ = load_method(BENCH[G], m, clip, n, *hw)
            if np.isfinite(T).all() and np.isfinite(K).all():
                data[m] = (T[:, :3, 3], QX.hfov(K, hw[1]), D[ks])
        except Exception:  # noqa: BLE001
            pass
    rows = []
    for name in ("v0", "v1"):
        r = {"group": G, "clip": clip, "label": name, "cat": c.get("cat", c.get("dataset"))}
        try:
            T, K, D, _ = label(G, name, clip, n, hw)
        except Exception as e:  # noqa: BLE001
            r["error"] = str(e)[:150]
            rows.append(r)
            continue
        me = (T[:, :3, 3], QX.hfov(K, hw[1]), D[ks])
        P = {o: data[o] for o in data}
        tr = [QX.traj_pct(me[0], v[0]) for v in P.values()]
        fb = [abs(me[1] - v[1]) for o, v in P.items() if o in BA_CLASS]
        fd = [QX.depth_cmp(me[2], v[2], static)[0] for o, v in P.items() if o not in BA_CLASS]
        r["xall_traj"] = float(np.nanmedian(tr)) if tr else np.nan
        r["xba_fov"] = float(np.nanmedian(fb)) if fb else np.nan
        r["xff_depth"] = float(np.nanmedian(fd)) if fd else np.nan
        r["rule"] = np.nanmax([r["xall_traj"] / 3.5, r["xba_fov"] / 4.1, r["xff_depth"] / 0.157])
        if "pi3x" in data:
            r["cond_dfov"] = abs(data["pi3x"][1] - me[1])
            r["cond_ddepth"] = QX.depth_cmp(data["pi3x"][2], me[2], static)[0]
        rows.append(r)
    return rows


if __name__ == "__main__":
    jobs = [(G, c) for G in ("A", "B", "C") for c in json.loads((BENCH[G] / "clips.json").read_text())
            if c.get("dataset") != "eth3d_mvs"]
    with ProcessPoolExecutor(24) as ex:
        df = pd.DataFrame([r for rs in ex.map(one, jobs) for r in rs])
    # GT failures for A/B
    gtf = []
    for G, kind, raw, names in (("A", "A", "E3", {"H:vipe+pi3x+med": "v0", "pi3x_kgate@vipe_d": "v1"}),
                                ("B", "B", "E3", {"H:vipe+pi3x+med": "v0", "pi3x_kgate@vipe_d": "v1"})):
        x = pd.read_csv(ROOT / raw / "raw" / f"{kind}.csv")
        x = x[x["method"].isin(names)].copy()
        x["label"] = x["method"].map(names)
        x["geom_fail"] = ((x["fail_ate_sim3"] > 0) | (x["fail_fov"] > 0) | (x["fail_depth"].fillna(0) > 0)).astype(int)
        gtf.append(x[["clip", "label", "geom_fail", "fail_scale"]].assign(group=G))
    df = df.merge(pd.concat(gtf), on=["group", "clip", "label"], how="left")
    df.to_csv(ROOT / "E1" / "qc_rule.csv", index=False, float_format="%.5g")
    from sklearn.metrics import roc_auc_score
    for name, g in df[df["group"].isin(["A", "B"])].groupby("label"):
        y = g["geom_fail"]
        keep = g["rule"] <= 1
        print(name, "AB n", len(g), "geom fail", round(y.mean(), 3), "| rule keeps", round(keep.mean(), 3),
              "fail among kept", round(y[keep].mean(), 3), "| AUROC rule", round(roc_auc_score(y, g["rule"].fillna(9)), 3),
              "cond_dfov", round(roc_auc_score(y, g["cond_dfov"].fillna(0)), 3) if "cond_dfov" in g else "",
              "cond_ddepth", round(roc_auc_score(y, g["cond_ddepth"].fillna(0)), 3) if "cond_ddepth" in g else "")
    for name, g in df[df["group"] == "C"].groupby("label"):
        print(name, "C n", len(g), "rule reject", round((g["rule"] > 1).mean(), 3),
              "| why:", {k: round(float((g[k] / t > 1).mean()), 3) for k, t in (("xall_traj", 3.5), ("xba_fov", 4.1), ("xff_depth", 0.157))})
        print("   by cat reject:", g.assign(r=g["rule"] > 1).groupby("cat")["r"].mean().round(2).to_dict())
