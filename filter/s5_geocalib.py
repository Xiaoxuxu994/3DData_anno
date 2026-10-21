"""Stage calib: per-frame single-image calibration with GeoCalib (ViPE's copy, "distorted" weights, simple_radial).

Usage: CUDA_VISIBLE_DEVICES=4 code/vipe/.venv/bin/python s5_geocalib.py WORKER_ID N_WORKERS [POOL_DIR]
12 evenly spaced frames, each calibrated independently -> focal stability (zoom) and k1 (lens distortion).
Writes scores.json["calib"]: vfov_deg (per frame), focal_ratio_end (median f of last 3 / first 3),
focal_range (max/min f), focal_trend (Spearman rho of f vs time), focal_cv, k1_median, focal_unc_rel (median).
"""

import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from fcommon import StageLog, clips, even_idx, frame_paths, has_stage, load_frames, timer, write_scores
from vipe.priors.geocalib import GeoCalib

wid, nw = int(sys.argv[1]), int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else None
model = GeoCalib(weights="distorted").cuda().eval()
log = StageLog("calib", wid)

for d in clips(pool, wid, nw):
    if has_stage(d, "calib") or not (d / "meta.json").exists():
        continue
    t = timer()
    n = len(frame_paths(d))
    f_, v_, k_, u_ = [], [], [], []
    for im in load_frames(d, even_idx(n, 12)):
        x = torch.from_numpy(im).permute(2, 0, 1).float().cuda() / 255
        with torch.inference_mode():
            r = model.calibrate(x, camera_model="simple_radial")
        cam = r["camera"]
        f = float(cam.f.reshape(-1)[0])
        f_.append(f / im.shape[1])  # focal in units of image width
        v_.append(float(torch.rad2deg(cam.vfov).reshape(-1)[0]))
        k_.append(float(cam.k1.reshape(-1)[0]) if hasattr(cam, "k1") else 0.0)
        u = r.get("focal_uncertainty")
        u_.append(float(u.reshape(-1)[0]) / f if u is not None else np.nan)
    f_ = np.array(f_)
    s = {"vfov_deg": v_, "focal_w": f_, "focal_ratio_end": np.median(f_[-3:]) / np.median(f_[:3]),
         "focal_range": f_.max() / f_.min(), "focal_trend": spearmanr(np.arange(len(f_)), f_)[0],
         "focal_cv": f_.std() / f_.mean(), "k1_median": np.median(k_), "k1_abs_max": np.abs(k_).max(),
         "vfov_median": np.median(v_), "focal_unc_rel": np.nanmedian(u_)}
    write_scores(d, "calib", s)
    log(d, t(), **{k: v for k, v in s.items() if k not in ("vfov_deg", "focal_w")})
    print(d.name, {k: round(float(v), 3) for k, v in s.items() if k not in ("vfov_deg", "focal_w")}, f"{t():.1f}s", flush=True)
print("CALIB_DONE", wid, flush=True)
