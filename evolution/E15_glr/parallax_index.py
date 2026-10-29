"""Parallax index for a clip (python -I): how much of the inter-frame motion is NOT explained by a pure camera rotation.

usage: python -I parallax_index.py CLIP [CLIP ...]
For frame pairs 45 frames apart, match SIFT, keep fundamental-matrix inliers, fit a rotation-only model (f + 3-DOF rotation, robust)
and a free homography; report residual percentiles relative to the median flow.
Prints one JSON line per clip.
"""
import json
import math
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

cv2.setNumThreads(4)
SW = 960


def load(path, step=1):
    cap = cv2.VideoCapture(path)
    G = []
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % step == 0:
            h, w = fr.shape[:2]
            sh = int(round(h * SW / w))
            G.append(cv2.cvtColor(cv2.resize(fr, (SW, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        n += 1
    cap.release()
    return G


def analyse(path):
    G = load(path)
    N = len(G)
    SH = G[0].shape[0]
    cx, cy = SW / 2, SH / 2
    sift = cv2.SIFT_create(nfeatures=6000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def rot_h(f, rv):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        R, _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))
        return K @ R @ np.linalg.inv(K)

    def project(H, p):
        q = np.c_[p, np.ones(len(p))] @ H.T
        return q[:, :2] / q[:, 2:3]

    rows = []
    gap = 45
    starts = list(range(0, max(N - gap, 1), 30))[:5]
    for a in starts:
        b = min(a + gap, N - 1)
        kpa, da = sift.detectAndCompute(G[a], None)
        kpb, db = sift.detectAndCompute(G[b], None)
        if da is None or db is None or len(kpa) < 40 or len(kpb) < 40:
            continue
        mt = bf.knnMatch(da, db, k=2)
        good = [m for m, n in (x for x in mt if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 40:
            continue
        p0 = np.float32([kpa[m.queryIdx].pt for m in good])
        p1 = np.float32([kpb[m.trainIdx].pt for m in good])
        F, mf = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, 1.0, 0.999, 5000)
        if F is None:
            continue
        mf = mf.ravel().astype(bool)
        p0, p1 = p0[mf], p1[mf]
        if len(p0) < 30:
            continue
        flow = np.linalg.norm(p1 - p0, axis=1)
        fm = float(np.median(flow))
        if fm < 3:       # nearly static: parallax index undefined
            rows.append({"pair": [a, b], "n": int(len(p0)), "flow_med": round(fm, 1), "static": True})
            continue
        Hm, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 1.5)
        hres = np.linalg.norm(project(Hm, p0) - p1, axis=1) if Hm is not None else np.full(len(p0), np.nan)
        d = np.median(p1[:, 0] - p0[:, 0])
        best = None
        for sgn in (1, -1):
            x0 = np.array([900.0, 0, sgn * abs(d) / 900.0, 0])
            try:
                r = least_squares(lambda x: (project(rot_h(x[0], x[1:]), p0) - p1).ravel(), x0, loss="soft_l1", f_scale=1.0,
                                  bounds=([250, -1.5, -1.5, -1.5], [4000, 1.5, 1.5, 1.5]))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        rres = np.linalg.norm(project(rot_h(best.x[0], best.x[1:]), p0) - p1, axis=1) if best is not None else np.full(len(p0), np.nan)
        rows.append({"pair": [a, b], "n": int(len(p0)), "flow_med": round(fm, 1),
                     "f": round(float(best.x[0]), 0) if best is not None else None,
                     "rot_p50": round(float(np.nanmedian(rres)), 2), "rot_p90": round(float(np.nanpercentile(rres, 90)), 2),
                     "H_p90": round(float(np.nanpercentile(hres, 90)), 2),
                     "rot_p90_over_flow": round(float(np.nanpercentile(rres, 90) / fm), 3),
                     "H_p90_over_flow": round(float(np.nanpercentile(hres, 90) / fm), 3)})
    ok = [r for r in rows if not r.get("static") and "rot_p90_over_flow" in r]
    summ = {"clip": path.split("/")[-1], "frames": N, "wh": [SW, SH]}
    if ok:
        summ["flow_med_px_per_45f"] = round(float(np.median([r["flow_med"] for r in ok])), 1)
        summ["rot_p50"] = round(float(np.median([r["rot_p50"] for r in ok])), 2)
        summ["rot_p90_over_flow_med"] = round(float(np.median([r["rot_p90_over_flow"] for r in ok])), 3)
        summ["H_p90_over_flow_med"] = round(float(np.median([r["H_p90_over_flow"] for r in ok])), 3)
        summ["f_med"] = float(np.median([r["f"] for r in ok if r["f"]]))
    summ["n_pairs_used"] = len(ok)
    summ["n_pairs_static"] = sum(1 for r in rows if r.get("static"))
    return summ


for p in sys.argv[1:]:
    try:
        print(json.dumps(analyse(p), default=float), flush=True)
    except Exception as e:  # noqa
        print(json.dumps({"clip": p.split("/")[-1], "error": repr(e)}), flush=True)
