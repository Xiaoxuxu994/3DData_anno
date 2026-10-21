"""Recover the scale SpatialVID's camera_tracking.py divides out before tracking.

camera_tracking.main() aligns Depth-Anything disparity to UniDepth (metric) with one (scale, shift) and then
divides by normalize_scale = P98(aligned disparity) / 2, so every saved depth / translation equals
metric * normalize_scale. This re-runs exactly that arithmetic on the saved depth-anything / unidepth outputs
and writes <clip>/svid_norm_scale.json; eval3d_bench's "spatialvid_metric" divides by it.

  python svid_norm_scale.py BENCH_DIR [BENCH_DIR ...]
"""

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


def clip_scale(d):
    out = os.path.join(d, "svid_norm_scale.json")
    mono = sorted(glob.glob(os.path.join(d, "depth-anything", "*.npy")))
    met = sorted(glob.glob(os.path.join(d, "unidepth", "*.npz")))
    if not mono or not os.path.exists(os.path.join(d, "sgd_cvd_hr.npz")):
        return d, None
    scales, shifts, mono_list = [], [], []
    for mf, uf in zip(mono, met):  # same arithmetic as camera_tracking.main (final, unconditional branch)
        da = np.float32(np.load(mf))
        md = np.load(uf)["depth"]
        da = cv2.resize(da, (md.shape[1], md.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        mono_list.append(da)
        gt = 1.0 / (md + 1e-8)
        gt[(md < 2.0) & (da < 0.02)] = 1e-2
        s = np.median((gt - np.median(gt) + 1e-8) / (da - np.median(da) + 1e-8))
        scales.append(s)
        shifts.append(np.median(gt - s * da))
    ss = np.array(scales) * np.array(shifts)
    k = int(np.argmin(np.abs(ss - np.median(ss))))
    ns = float(np.percentile(scales[k] * np.array(mono_list) + shifts[k], 98) / 2.0)
    json.dump({"normalize_scale": ns, "align_scale": float(scales[k]), "align_shift": float(shifts[k])}, open(out, "w"))
    return d, ns


if __name__ == "__main__":
    dirs = [d for b in sys.argv[1:] for d in sorted(glob.glob(os.path.join(b, "spatialvid", "*"))) if os.path.isdir(d)]
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(clip_scale, dirs))
    ok = [r for r in res if r[1] is not None]
    print(len(ok), "clips; normalize_scale median", np.median([r[1] for r in ok]) if ok else None)
