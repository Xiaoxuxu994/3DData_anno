"""E13 focal sweep evaluation: gate-2 self-check + photometric warp error of each re-solved label (K scaled by s).
  cache/qc-venv/bin/python e13_fsweep_eval.py -> E13/raw/fsweep_signals.csv, E13/fsweep_curves.png
photo{g}: median |I_i - warp(I_j -> i)| (0-255, grayscale) on static pixels whose projection lands inside frame j,
using the label's depth_i, poses and K (gap g in 1, 5, 20). gt_hfov from the GT intrinsics (A/B) or none (C).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARGV = list(sys.argv)
sys.argv = sys.argv[:1]
import e13_signals as E  # noqa: E402

QS, QC = E.QS, E.QC


def bilinear(im, x, y):
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, im.shape[1] - 1), np.minimum(y0 + 1, im.shape[0] - 1)
    ax, ay = x - x0, y - y0
    return (im[y0, x0] * (1 - ax) * (1 - ay) + im[y0, x1] * ax * (1 - ay) + im[y1, x0] * (1 - ax) * ay + im[y1, x1] * ax * ay)


def photo(G, m, c, gaps=(1, 5, 20)):
    clip, n = c["clip"], c["n_frames"]
    hw = tuple(int(x) for x in np.load(E.LINKS / G / f"{clip}.npz")["hw"])
    c2w, K, D, _ = E.load_label(QC.BENCH[G], m, clip, n, *hw)
    D = np.where(np.isfinite(D) & (D > 0), D, np.nan).astype(np.float32)
    lab = QC.vipe_static_mask(QC.BENCH[G], clip, n, hw)
    h, w = hw
    out = {}
    gray = {}
    for g in gaps:
        errs = []
        for i in QC.anchors(n, g)[::2]:
            i, j = int(i), int(i + g)
            for k in (i, j):
                if k not in gray:
                    gray[k] = cv2.cvtColor(QC.rgb(G, clip, k, hw), cv2.COLOR_RGB2GRAY).astype(np.float32)
            sel = np.isfinite(D[i]) & ((lab[i] == 0) if lab is not None else True)
            if sel.sum() < 500:
                continue
            uv, zj, idx = QS.project(c2w, K, D, i, j, sel)
            ok = np.isfinite(uv).all(1) & (zj > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
            if ok.sum() < 500:
                continue
            Ij = bilinear(gray[j], uv[ok, 0], uv[ok, 1])
            errs.append(np.median(np.abs(gray[i].flat[idx[ok]] - Ij)))
        out[f"photo{g}"] = float(np.mean(errs)) if errs else np.nan
    return out


def run(job):
    G, m, c = job
    r = E.run(job)
    try:
        r.update(photo(G, m, c))
    except Exception as e:  # noqa: BLE001
        r["photo_error"] = str(e)[:200]
    return r


def gt_hfov(G, clip, w):
    if G == "C":
        return np.nan
    z = np.load(QC.BENCH[G] / "gt" / f"{clip}.npz", allow_pickle=True)
    W = z["depth"].shape[2]
    return float(np.degrees(2 * np.arctan(W / 2 / np.median(z["K"][:, 0, 0]))))


if __name__ == "__main__":
    jobs = []
    for clip in sorted(os.listdir(E.E13 / "fsweep")):
        G = "C" if clip.startswith("driving__") else ("B" if clip.startswith("vkitti") else "A")
        c = next(x for x in QC.clips(G) if x["clip"] == clip)
        c = dict(c, dataset=c.get("dataset", c.get("cat")))
        jobs += [(G, "fs" + s, c) for s in sorted(os.listdir(E.E13 / "fsweep" / clip))]
        jobs += [(G, m, c) for m in QC.METHODS]
    out = E.E13 / "raw/fsweep_signals.csv"
    if out.exists() and "--redo" not in ARGV:
        df = pd.read_csv(out)
    else:
        with ProcessPoolExecutor(24) as ex:
            df = pd.DataFrame(list(ex.map(run, jobs, chunksize=1)))
        df["gt_hfov"] = [gt_hfov(r.group, r.clip, 0) for r in df.itertuples()]
        df.to_csv(out, index=False, float_format="%.5f")
    sig = [("photo5", "photometric warp err, gap 5 (lower better)"), ("iflow5_epe", "induced flow EPE px, gap 5"),
           ("cons5_occ", "depth self-consistency gap 5, occl.-aware (higher better)"), ("cons20", "depth self-consistency gap 20"),
           ("fsv1", "free-space violation gap 1"), ("samp5_med", "Sampson px, gap 5")]
    clips = sorted(df["clip"].unique())
    fig, axes = plt.subplots(len(clips), len(sig), figsize=(3.2 * len(sig), 2.6 * len(clips)), squeeze=False)
    for a, clip in enumerate(clips):
        d = df[df["clip"] == clip]
        sw = d[d.method.str.startswith("fs")].sort_values("hfov")
        others = d[~d.method.str.startswith("fs")]
        for b, (s, t) in enumerate(sig):
            ax = axes[a, b]
            ax.plot(sw.hfov, sw[s], "-o", color="#2a78d6", ms=4, lw=2, label="v1.3 re-solved with K x s")
            ax.scatter(others.hfov, others[s], color="#eb6834", s=18, zorder=3, label="7 annotators (own K)")
            for r in others.itertuples():
                ax.annotate(r.method[:5], (r.hfov, getattr(r, s)), fontsize=6, color="#555")
            if np.isfinite(d.gt_hfov.iloc[0]):
                ax.axvline(d.gt_hfov.iloc[0], color="#1baf7a", lw=1.5, ls="--", label="GT HFOV")
            if a == 0:
                ax.set_title(t, fontsize=8)
            if b == 0:
                ax.set_ylabel(clip.split("__")[-1], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("HFOV (deg)", fontsize=8)
    fig.tight_layout()
    fig.savefig(E.E13 / "fsweep_curves.png", dpi=110)
    pd.set_option("display.width", 250)
    print(df[["clip", "method", "hfov", "gt_hfov", "photo1", "photo5", "photo20", "iflow5_epe", "iflow1_lt3", "cons5_occ", "cons20", "fsv1", "samp5_med", "samp20_med"]].round(3).to_string())
