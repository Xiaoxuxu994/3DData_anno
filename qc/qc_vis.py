"""Visual check sheets for C: per clip, 3 frames x (RGB | ViPE depth | ViPE pixel conf | Pi3X depth | Pi3X pixel conf),
title = QC risk / flags per method.   python qc_vis.py -> outputs/qc/vis/<rank>_<clip>.jpg (rank 00 = highest min-risk)
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/data/home/z00533370/code/bench")
from eval3d_bench import load_method  # noqa: E402
from qc_common import BENCH, QC, clips, qc_grid, rgb  # noqa: E402

C = pd.read_csv(QC / "qc_C.csv")
BY = pd.read_csv(QC / "qc_C_by_clip.csv", index_col=0)
CL = {c["clip"]: c for c in clips("C")}


def inv(d):
    v = np.isfinite(d) & (d > 0)
    x = np.where(v, 1 / np.where(v, d, 1), np.nan)
    lo, hi = (np.nanpercentile(x, 2), np.nanpercentile(x, 98)) if v.sum() > 10 else (0, 1)
    return x, lo, hi


def sheet(args):
    rank, clip = args
    c = CL[clip]
    n, hw = c["n_frames"], qc_grid("C", clip)
    ks = [k for k in (0, (n // 2) // 5 * 5, (n - 1) // 5 * 5)]
    cols = [("RGB", None)]
    data = {}
    for m in ("vipe", "pi3x"):
        _, _, D, _ = load_method(BENCH["C"], m, clip, n, *hw)
        z = np.load(QC / "pixconf/C" / m / f"{clip}.npz")
        conf = {int(f): cf for f, cf in zip(z["frames"], z["conf"])}
        data[m] = (D, conf)
        cols += [(f"{m} depth", m), (f"{m} conf", m)]
    fig, ax = plt.subplots(3, len(cols), figsize=(3.2 * len(cols) * hw[1] / hw[0] / 1.5, 6.6), squeeze=False)
    for r, k in enumerate(ks):
        ax[r, 0].imshow(rgb("C", clip, k, hw))
        ax[r, 0].set_ylabel(f"frame {k}")
        for j, (name, m) in enumerate(cols[1:], start=1):
            D, conf = data[m]
            if name.endswith("depth"):
                x, lo, hi = inv(D[k])
                ax[r, j].imshow(x, cmap="turbo", vmin=lo, vmax=hi)
            else:
                cf = conf.get(k)
                ax[r, j].imshow(np.where(cf == 255, np.nan, cf / 100.0), cmap="RdYlGn", vmin=0, vmax=1)
            if r == 0:
                ax[r, j].set_title(name, fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    g = C[C["clip"] == clip].set_index("method")
    risk = "  ".join(f"{m}:{g.loc[m, 'risk']:.2f}" for m in g.index)
    fl = g["clip_flags"].dropna().iloc[0] if g["clip_flags"].notna().any() else ""
    fig.suptitle(f"#{rank:02d} {clip} [{c['cat']}]{' HARD' if c.get('hard_case') else ''}  clip flags: {fl or '-'}\n"
                 f"geo-risk  {risk}", fontsize=8)
    fig.tight_layout()
    (QC / "vis").mkdir(exist_ok=True)
    fig.savefig(QC / "vis" / f"{rank:02d}_{clip}.jpg", dpi=90, pil_kwargs={"quality": 85})
    plt.close(fig)


if __name__ == "__main__":
    with ProcessPoolExecutor(16) as ex:
        list(ex.map(sheet, list(enumerate(BY.index))))
    print("VIS_DONE")
