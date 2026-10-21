"""Stage flow: RAFT-large optical flow -> camera-motion magnitude / smoothness / shake, homography-vs-epipolar parallax,
dynamic-area fraction, flow-based zoom (divergence) and cut spikes.

Usage: CUDA_VISIBLE_DEVICES=4 python s4_flow.py WORKER_ID N_WORKERS [POOL_DIR]
Flow at 512x288. Short pairs (i, i+3) = 0.2 s at 15 fps, long pairs (i, i+15) = 1 s.
Writes scores.json["flow"] (all lengths in pixels at 512 px width unless noted *_frac = fraction of width):
  mag_mean_frac / mag_median_frac  mean |flow| per 0.2 s step (DynPose uses mean flow vs. width)
  static_pair_frac                 short pairs with mean |flow| < 0.25 % width
  spike_ratio                      max step magnitude / (mean + 4 std of the others); > 1 = DynPose-style jump
  jerk                             median |v_t - v_t-1| / median |v_t| of the global (median) flow vector -> shake
  parallax_px                      long pairs: median homography transfer error of epipolar inliers (px)
  parallax_ratio                   parallax_px / median long-pair flow magnitude
  h_inlier                         long pairs: fraction of points within 1 px of the RANSAC homography
  dyn_frac                         long pairs: fraction of points with Sampson error > 1 px w.r.t. the RANSAC F
                                    (only pairs whose parallax makes F well-posed; else from the homography)
  zoom_total                       product of per-step similarity scales (flow divergence); 1 = no zoom
  rot_frac                         share of long pairs explained by a homography (parallax_px < 0.5)
"""

import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from fcommon import StageLog, clips, frame_paths, has_stage, load_frames, timer, write_scores

wid, nw = int(sys.argv[1]), int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else None
W, Hh = 512, 288
model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).cuda().eval()
log = StageLog("flow", wid)
gy, gx = np.mgrid[4:Hh:8, 4:W:8]
GRID = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


@torch.inference_mode()
def flows(fr, pairs, bs=16):
    out = []
    for k in range(0, len(pairs), bs):
        a = torch.stack([fr[i] for i, _ in pairs[k:k + bs]]).cuda()
        b = torch.stack([fr[j] for _, j in pairs[k:k + bs]]).cuda()
        out.append(model(a, b, num_flow_updates=12)[-1].float().cpu())
    return torch.cat(out).permute(0, 2, 3, 1).numpy()  # (P,H,W,2)


def geom(f):
    """Homography / fundamental analysis of one long-baseline flow field on an 8 px grid."""
    p0 = GRID
    p1 = p0 + f[GRID[:, 1].astype(int), GRID[:, 0].astype(int)]
    ok = (p1[:, 0] >= 0) & (p1[:, 0] < W) & (p1[:, 1] >= 0) & (p1[:, 1] < Hh)
    p0, p1 = p0[ok], p1[ok]
    mag = np.median(np.linalg.norm(p1 - p0, axis=1))
    if len(p0) < 50:
        return None
    Hm, hin = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 1.0)
    if Hm is None:
        return None
    ph = cv2.perspectiveTransform(p0[None], Hm)[0]
    herr = np.linalg.norm(ph - p1, axis=1)
    F, fin = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, 1.0, 0.999, 2000)
    if F is None or F.shape != (3, 3):
        par, dyn = 0.0, float((herr > 2.0).mean())
    else:
        x0 = np.c_[p0, np.ones(len(p0))]
        x1 = np.c_[p1, np.ones(len(p1))]
        Fx0, Ftx1 = x0 @ F.T, x1 @ F
        samp = (np.sum(x1 * Fx0, 1) ** 2) / (Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2 + 1e-12)
        inl = np.sqrt(samp) < 1.0
        par = float(np.median(herr[inl])) if inl.sum() > 20 else 0.0
        # F is only informative with real parallax; otherwise motion segmentation uses the homography
        dyn = float(1 - inl.mean()) if par > 0.5 else float((herr > 2.0).mean())
    return {"mag": mag, "par": par, "hin": float((herr < 1.0).mean()), "dyn": dyn}


def similarity_scale(f):
    p0 = GRID
    p1 = p0 + f[GRID[:, 1].astype(int), GRID[:, 0].astype(int)]
    A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    return 1.0 if A is None else float(np.sqrt(abs(np.linalg.det(A[:, :2]))))


for d in clips(pool, wid, nw):
    if has_stage(d, "flow") or not (d / "meta.json").exists():
        continue
    t = timer()
    n = len(frame_paths(d))
    raw = load_frames(d)
    fr = [TF.normalize(TF.to_tensor(cv2.resize(x, (W, Hh), interpolation=cv2.INTER_AREA)), [0.5] * 3, [0.5] * 3) for x in raw]
    short = [(i, i + 3) for i in range(0, n - 3, 3)]
    long_ = [(i, i + 15) for i in range(0, n - 15, 15)]
    fs = flows(fr, short)
    fl = flows(fr, long_)
    m = np.linalg.norm(fs, axis=-1).mean((1, 2))  # mean |flow| per short pair
    gv = np.median(fs.reshape(len(fs), -1, 2), axis=1)  # global (median) flow vector
    others = lambda k: np.delete(m, k)
    spike = max(m[k] / (others(k).mean() + 4 * others(k).std() + 1e-3) for k in range(len(m)))
    dv = np.linalg.norm(np.diff(gv, axis=0), axis=1)
    jerk = np.median(dv) / (np.median(np.linalg.norm(gv, axis=1)) + 0.05)
    def safe_geom(f):
        try:
            return geom(f)
        except cv2.error:  # degenerate correspondences (e.g. near-static CG frames)
            return None

    g = [r for r in (safe_geom(f) for f in fl) if r is not None]
    moving = [r for r in g if r["mag"] > 2.0] or g
    par = np.median([r["par"] for r in moving]) if moving else 0.0
    lmag = np.median([r["mag"] for r in g]) if g else 0.0
    zoom = float(np.prod([similarity_scale(f) for f in fs]))
    s = {"mag_mean_frac": m.mean() / W, "mag_median_frac": np.median(m) / W, "mag_max_frac": m.max() / W,
         "static_pair_frac": (m / W < 0.0025).mean(), "spike_ratio": spike, "jerk": jerk,
         "long_mag_px": lmag, "parallax_px": par, "parallax_ratio": par / (lmag + 1e-3),
         "h_inlier": np.median([r["hin"] for r in g]) if g else 1.0,
         "dyn_frac": np.median([r["dyn"] for r in g]) if g else 0.0,
         "dyn_frac_max": max([r["dyn"] for r in g], default=0.0),
         "rot_frac": np.mean([r["par"] < 0.5 for r in moving]) if moving else 1.0,
         "zoom_total": zoom, "per_step_mag_frac": (m / W).round(5)}
    write_scores(d, "flow", s)
    log(d, t(), **{k: v for k, v in s.items() if k != "per_step_mag_frac"})
    print(d.name, {k: round(float(v), 4) for k, v in s.items() if k != "per_step_mag_frac"}, f"{t():.1f}s", flush=True)
print("FLOW_DONE", wid, flush=True)
