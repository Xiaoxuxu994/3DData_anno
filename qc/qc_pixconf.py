"""Per-pixel label confidence from multi-view depth agreement, and its calibration against dense GT depth.

conf(pixel of frame k) = share of neighbour frames k+-1, k+-5 in which the back-projected point lands on a surface
within 5 % depth (only neighbours where it projects in-bounds onto valid depth). Every 5th frame, QC grid.

  python qc_pixconf.py calib            -> outputs/qc/pixconf_calib.csv  (A/B dense-GT clips: AUROC of 1-conf for
                                           pixels with seq-aligned |depth error| > 10 %, AbsRel before / after keeping conf>=0.75)
  python qc_pixconf.py export G M [M..] -> outputs/qc/pixconf/<G>/<method>/<clip>.npz (frames, conf uint8 0-100)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data/home/z00533370/code/bench")
from eval3d_bench import load_method  # noqa: E402
from qc_common import BENCH, QC, clips, qc_grid  # noqa: E402

OFFS = (-5, -1, 1, 5)


def pixconf(c2w, K, D, k):
    n, h, w = D.shape
    v, u = np.mgrid[0:h, 0:w]
    z = D[k]
    sel = np.isfinite(z) & (z > 0)
    X = np.stack([(u[sel] - K[k, 0, 2]) / K[k, 0, 0] * z[sel], (v[sel] - K[k, 1, 2]) / K[k, 1, 1] * z[sel], z[sel]], 1)
    good, vis = np.zeros(sel.sum()), np.zeros(sel.sum())
    for o in OFFS:
        j = k + o
        if not 0 <= j < n:
            continue
        T = np.linalg.inv(c2w[j]) @ c2w[k]
        Xc = X @ T[:3, :3].T + T[:3, 3]
        zj = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = np.round(K[j, 0, 0] * Xc[:, 0] / zj + K[j, 0, 2])
            vv = np.round(K[j, 1, 1] * Xc[:, 1] / zj + K[j, 1, 2])
        ok = np.isfinite(uu) & np.isfinite(vv) & (zj > 1e-6) & (uu >= 0) & (uu <= w - 1) & (vv >= 0) & (vv <= h - 1)
        dj = np.full(len(zj), np.nan)
        dj[ok] = D[j][vv[ok].astype(int), uu[ok].astype(int)]
        ok &= np.isfinite(dj) & (dj > 0)
        vis += ok
        good += ok & (np.abs(zj - np.nan_to_num(dj, nan=1)) / np.nan_to_num(dj, nan=1) < 0.05)
    conf = np.full((h, w), np.nan, np.float32)
    conf[sel] = np.where(vis > 0, good / np.maximum(vis, 1), np.nan)
    return conf


def calib_clip(args):
    G, m, c = args
    clip, n = c["clip"], c["n_frames"]
    r = {"group": G, "method": m, "clip": clip, "dataset": c["dataset"]}
    try:
        z = np.load(BENCH[G] / "gt" / f"{clip}.npz", allow_pickle=True)
        if not str(json.loads(str(z["meta"])).get("depth_type", "")).startswith("dense"):
            return None
        hw = qc_grid(G, clip)
        c2w, K, D, _ = load_method(BENCH[G], m, clip, n, *hw)
        GD = np.stack([cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) for d in z["depth"].astype(np.float32)])
        ks = range(0, n, 5)
        C = np.stack([pixconf(c2w, K, D, k) for k in ks])
        P, Gt = D[list(ks)], GD[list(ks)]
        msk = np.isfinite(C) & np.isfinite(P) & (P > 0) & (Gt > 0) & (Gt < 80)
        s = np.median(Gt[msk]) / np.median(P[msk])
        e = np.abs(P[msk] * s - Gt[msk]) / Gt[msk]
        cf = C[msk]
        bad = e > 0.10
        r["bad_frac"] = float(bad.mean())
        r["auroc"] = float(roc_auc_score(bad, 1 - cf)) if 0 < bad.mean() < 1 else np.nan
        r["absrel_all"] = float(e.mean())
        keep = cf >= 0.75
        r["keep_frac"] = float(keep.mean())
        r["absrel_kept"] = float(e[keep].mean()) if keep.any() else np.nan
        r["bad_kept"] = float(bad[keep].mean()) if keep.any() else np.nan
    except Exception as ex:  # noqa: BLE001
        r["error"] = str(ex)[:200]
    return r


def export_clip(args):
    G, m, c = args
    clip, n = c["clip"], c["n_frames"]
    out = QC / "pixconf" / G / m / f"{clip}.npz"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    hw = qc_grid(G, clip)
    c2w, K, D, _ = load_method(BENCH[G], m, clip, n, *hw)
    ks = np.arange(0, n, 5)
    C = np.stack([pixconf(c2w, K, D, int(k)) for k in ks])
    np.savez_compressed(out, frames=ks, conf=np.where(np.isfinite(C), np.round(C * 100), 255).astype(np.uint8))


if __name__ == "__main__":
    MS = ["vipe", "spatialvid_metric", "vggt_omega", "da3", "pi3x"]
    with ProcessPoolExecutor(16) as ex:
        if sys.argv[1] == "calib":
            jobs = [(G, m, c) for G in ("A", "B") for c in clips(G) for m in MS]
            df = pd.DataFrame([r for r in ex.map(calib_clip, jobs) if r])
            df.to_csv(QC / "pixconf_calib.csv", index=False, float_format="%.4f")
            print(df.groupby("method")[["auroc", "bad_frac", "absrel_all", "keep_frac", "absrel_kept", "bad_kept"]].median().round(3))
        else:
            G = sys.argv[2]
            list(ex.map(export_clip, [(G, m, c) for c in clips(G) for m in sys.argv[3:]]))
            print("EXPORT_DONE")
