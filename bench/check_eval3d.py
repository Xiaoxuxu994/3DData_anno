"""Reprojection sanity check of converted GT: depth_i + K + c2w -> frame j, compare with depth_j.

Usage: python check_eval3d.py DATASET [CLIP ...]   (default: first two clips)
Prints and appends to A_rank/<dataset>/CHECK.md: median relative depth error and fraction < 5% for gaps 1 and 5.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("/data/home/z00533370/data/eval3d/A_rank")


def load_depth(d, k, scale):
    return cv2.imread(str(d / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / scale


def check(ds, clip, gaps=(1, 5), n_pairs=8):
    d = OUT / ds / clip
    meta = json.loads((d / "meta.json").read_text())
    gt = np.load(d / "gt.npz")
    c2w = gt["c2w"]
    K = gt["K_depth"] if "K_depth" in gt.files else gt["K"]
    n = len(c2w)
    res = {}
    for gap in gaps:
        rels = []
        for i in np.linspace(0, n - 1 - gap, n_pairs).astype(int):
            j = i + gap
            di, dj = load_depth(d, i, meta["depth_scale"]), load_depth(d, j, meta["depth_scale"])
            H, W = di.shape
            v, u = np.mgrid[0:H, 0:W]
            m = di > 0
            z = di[m]
            Ki, Kj = K[i], K[j]
            X = np.stack([(u[m] - Ki[0, 2]) / Ki[0, 0] * z, (v[m] - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
            T = np.linalg.inv(c2w[j]) @ c2w[i]
            Xc = X @ T[:3, :3].T + T[:3, 3]
            zj = Xc[:, 2]
            ok = zj > 1e-3
            uj = np.round(Kj[0, 0] * Xc[ok, 0] / zj[ok] + Kj[0, 2]).astype(int)
            vj = np.round(Kj[1, 1] * Xc[ok, 1] / zj[ok] + Kj[1, 2]).astype(int)
            inb = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
            ref = dj[vj[inb], uj[inb]]
            g = ref > 0
            rels.append(np.abs(zj[ok][inb][g] - ref[g]) / ref[g])
        r = np.concatenate(rels)
        res[gap] = (float(np.median(r)), float((r < 0.05).mean()))
    return res


if __name__ == "__main__":
    ds = sys.argv[1]
    clips = sys.argv[2:] or [c["clip"] for c in json.loads((OUT / ds / "clips.json").read_text())][:2]
    lines = []
    for c in clips:
        r = check(ds, c)
        line = f"- {c}: " + "; ".join(f"gap {g}: median rel err {m:.4f}, <5% {f:.3f}" for g, (m, f) in r.items())
        print(line, flush=True)
        lines.append(line)
    with open(OUT / ds / "CHECK.md", "a") as f:
        f.write("\n".join(lines) + "\n")
