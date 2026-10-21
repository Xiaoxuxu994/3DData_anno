"""Per annotator x clip self-check signals (no GT used). Needs qc_cache.py output.

  python qc_signals.py GROUP [GROUP ...]   -> outputs/qc/signals_<G>.csv

Geometry (static, non-sky pixels from ViPE's GDINO+SAM masks; QC grid long side 512, px at that grid):
  cons{g}       |z_i->j - D_j| / D_j < 5 % on co-visible pixels, frame gap g in (1, 5, 20)
  cons{g}_occ   same, restricted to pixels whose RAFT forward-backward check passes (occlusion removed; g in 1, 5)
  fsv{g}        free-space violation rate: projected point lies > 5 % in front of the surface seen in frame j
                (cannot be explained by occlusion -> genuine inconsistency)
  iflow{g}_epe / _lt1 / _lt3 / _rel   induced flow (label depth+pose+K) vs RAFT flow: median EPE, <1px, <3px share,
                median EPE / max(|RAFT flow|, 1)
  samp{g}_med / samp{g}_excess / samp{g}_bad   Sampson distance (px) of LightGlue matches to the label's
                F = K_j^-T [t]x R K_i^-1; excess = minus the MAGSAC-F residual on the same matches; bad = share of
                pairs with median > 2 px
Motion / physical: norm_speed (step / median depth), speed_spike (max / median step), accel_spike, angvel_med/max
  (deg/frame), ang_spike, hfov, fov_out (outside 30-120 deg), focal_cv (per-frame focal variation), baseline_ratio
  (trajectory length / median depth), static_cam (focal unobservable), depth_med (label units)
Coverage: valid_frac (finite depth), dyn_frac / sky_frac (ViPE masks), fb_ok_frac (RAFT fb-consistent share),
  match_cnt (median LightGlue matches, gap 5)
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/bench")
from eval3d_bench import load_method  # noqa: E402
from qc_common import (BENCH, CACHE, CONS_GAPS, FLOW_GAPS, MATCH_GAPS, METHODS, QC, anchors, clips,  # noqa: E402
                       vipe_static_mask)


def project(c2w, K, D, i, j, sel):
    """Project pixels sel (bool h,w) of frame i into frame j -> (uv (m,2), zj (m,), idx flat indices)."""
    h, w = D.shape[1:]
    v, u = np.mgrid[0:h, 0:w]
    z = D[i][sel]
    X = np.stack([(u[sel] - K[i, 0, 2]) / K[i, 0, 0] * z, (v[sel] - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)
    T = np.linalg.inv(c2w[j]) @ c2w[i]
    Xc = X @ T[:3, :3].T + T[:3, 3]
    zj = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[j, 0, 0] * Xc[:, 0] / zj + K[j, 0, 2], K[j, 1, 1] * Xc[:, 1] / zj + K[j, 1, 2]], 1)
    return uv, zj, np.flatnonzero(sel)


def fb_ok(fw, bw):
    """RAFT forward-backward consistency (Sundaram et al.): True where pixel is not occluded / flow is reliable."""
    h, w = fw.shape[1:]
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = u + fw[0], v + fw[1]
    bwx = cv2.remap(bw[0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    bwy = cv2.remap(bw[1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    d2 = (fw[0] + bwx) ** 2 + (fw[1] + bwy) ** 2
    return d2 < 0.01 * (fw[0] ** 2 + fw[1] ** 2 + bwx ** 2 + bwy ** 2) + 0.5


def sampson(F, x1, x2):
    x1h = np.c_[x1, np.ones(len(x1))]
    x2h = np.c_[x2, np.ones(len(x2))]
    Fx1, Ftx2 = x1h @ F.T, x2h @ F
    num = np.sum(x2h * Fx1, 1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def signals(args):
    G, method, c = args
    clip, n = c["clip"], c["n_frames"]
    r = {"group": G, "method": method, "clip": clip, "dataset": c.get("dataset", c.get("cat")), "n_frames": n}
    try:
        cz = np.load(CACHE / G / f"{clip}.npz")
        hw = tuple(int(x) for x in cz["hw"])
        c2w, K, D, _ = load_method(BENCH[G], method, clip, n, *hw)
        if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
            r["nonfinite"] = 1.0
            return r
        r["nonfinite"] = 0.0
        D = np.where(np.isfinite(D) & (D > 0), D, np.nan).astype(np.float32)
        lab = vipe_static_mask(BENCH[G], clip, n, hw)
        if lab is None:
            lab = np.zeros((n, *hw), np.uint8)
        valid = np.isfinite(D)
        r["valid_frac"] = float(valid.mean())
        r["dyn_frac"] = float((lab == 1).mean())
        r["sky_frac"] = float((lab == 2).mean())
        static = (lab == 0) & valid
        dmed = float(np.nanmedian(D))
        r["depth_med"] = dmed
        h, w = hw
        # --- multi-gap self-consistency (+ occlusion-aware variants)
        fbm = {}
        for g in FLOW_GAPS:
            if f"flow{g}_i" in cz.files:
                fw, bw = cz[f"flow{g}_fw"].astype(np.float32), cz[f"flow{g}_bw"].astype(np.float32)
                fbm[g] = {int(i): (fw[k], fb_ok(fw[k], bw[k])) for k, i in enumerate(cz[f"flow{g}_i"])}
        fb_all = [m.mean() for d in fbm.values() for _, m in d.values()]
        r["fb_ok_frac"] = float(np.mean(fb_all)) if fb_all else np.nan
        for g in CONS_GAPS:
            cons, cons_occ, fsv = [], [], []
            epe, lt1, lt3, rel = [], [], [], []
            for i in anchors(n, g):
                i, j = int(i), int(i + g)
                sel = static[i]
                if sel.sum() < 200:
                    continue
                uv, zj, idx = project(c2w, K, D, i, j, sel)
                ok = np.isfinite(uv).all(1) & (zj > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
                if ok.sum() < 200:
                    continue
                ui, vi = np.round(uv[ok, 0]).astype(int), np.round(uv[ok, 1]).astype(int)
                dj = D[j][vi, ui]
                good = np.isfinite(dj) & (lab[j][vi, ui] == 0)
                if good.sum() < 100:
                    continue
                e = (zj[ok][good] - dj[good]) / dj[good]
                cons.append(np.mean(np.abs(e) < 0.05))
                fsv.append(np.mean(e < -0.05))
                if g in fbm and i in fbm[g]:
                    fw, fm = fbm[g][i]
                    idx_ok = idx[ok][good]
                    occ = fm.flat[idx_ok]
                    if occ.sum() > 100:
                        cons_occ.append(np.mean(np.abs(e[occ]) < 0.05))
                    # induced vs RAFT flow on static, fb-consistent pixels
                    yy, xx = np.unravel_index(idx[ok], (h, w))
                    ind = uv[ok] - np.stack([xx, yy], 1)
                    raft = np.stack([fw[0].flat[idx[ok]], fw[1].flat[idx[ok]]], 1)
                    m2 = fm.flat[idx[ok]]
                    if m2.sum() > 100:
                        d = np.linalg.norm(ind[m2] - raft[m2], axis=1)
                        epe.append(np.median(d))
                        lt1.append(np.mean(d < 1))
                        lt3.append(np.mean(d < 3))
                        rel.append(np.median(d / np.maximum(np.linalg.norm(raft[m2], axis=1), 1)))
            r[f"cons{g}"] = float(np.mean(cons)) if cons else np.nan
            r[f"fsv{g}"] = float(np.mean(fsv)) if fsv else np.nan
            if g in FLOW_GAPS:
                r[f"cons{g}_occ"] = float(np.mean(cons_occ)) if cons_occ else np.nan
                r[f"iflow{g}_epe"] = float(np.median(epe)) if epe else np.nan
                r[f"iflow{g}_lt1"] = float(np.mean(lt1)) if lt1 else np.nan
                r[f"iflow{g}_lt3"] = float(np.mean(lt3)) if lt3 else np.nan
                r[f"iflow{g}_rel"] = float(np.median(rel)) if rel else np.nan
        # --- Sampson epipolar error of LightGlue matches w.r.t. the label's F
        for g in MATCH_GAPS:
            if f"match{g}_i" not in cz.files:
                continue
            cnt = cz[f"match{g}_cnt"]
            off = np.r_[0, np.cumsum(cnt)]
            k0a, k1a = cz[f"match{g}_k0"], cz[f"match{g}_k1"]
            med, exc = [], []
            for k, i in enumerate(cz[f"match{g}_i"]):
                i, j = int(i), int(i + g)
                x1, x2 = k0a[off[k]:off[k + 1]].astype(np.float64), k1a[off[k]:off[k + 1]].astype(np.float64)
                if len(x1) < 30:
                    continue
                xi, yi = np.clip(np.round(x1).astype(int), 0, [w - 1, h - 1]).T
                st = lab[i][yi, xi] == 0
                x1, x2 = x1[st], x2[st]
                if len(x1) < 30:
                    continue
                T = np.linalg.inv(c2w[j]) @ c2w[i]
                t = T[:3, 3]
                if np.linalg.norm(t) < 1e-9:
                    t = np.array([0.0, 0.0, 1e-9])
                F = np.linalg.inv(K[j]).T @ skew(t / np.linalg.norm(t)) @ T[:3, :3] @ np.linalg.inv(K[i])
                s = np.median(sampson(F, x1, x2))
                med.append(s)
                try:
                    Fr, _ = cv2.findFundamentalMat(x1, x2, cv2.USAC_MAGSAC, 1.0, 0.999, 2000)
                except cv2.error:  # degenerate match set
                    Fr = None
                if Fr is not None and Fr.shape == (3, 3):
                    exc.append(s - np.median(sampson(Fr, x1, x2)))
            r[f"samp{g}_med"] = float(np.median(med)) if med else np.nan
            r[f"samp{g}_excess"] = float(np.median(exc)) if exc else np.nan
            r[f"samp{g}_bad"] = float(np.mean(np.array(med) > 2)) if med else np.nan
            if g == 5:
                r["match_cnt"] = float(np.median(cnt))
        # --- motion / physical plausibility
        p = c2w[:, :3, 3]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ang = np.array([rot_deg(c2w[k, :3, :3].T @ c2w[k + 1, :3, :3]) for k in range(n - 1)])
        vel = np.diff(p, axis=0)
        acc = np.linalg.norm(np.diff(vel, axis=0), axis=1)
        ms, ma = np.median(step) + 1e-9, np.median(ang) + 1e-6
        r["norm_speed"] = float(ms / dmed)
        r["speed_spike"] = float(step.max() / ms)
        r["accel_spike"] = float(acc.max() / ms) if len(acc) else np.nan
        r["angvel_med"] = float(np.median(ang))
        r["angvel_max"] = float(ang.max())
        r["ang_spike"] = float(ang.max() / ma)
        fx = K[:, 0, 0]
        hfov = np.degrees(2 * np.arctan(w / 2 / fx))
        r["hfov"] = float(hfov.mean())
        r["fov_out"] = float((hfov.mean() < 30) or (hfov.mean() > 120))
        r["focal_cv"] = float(fx.std() / fx.mean())
        r["baseline_ratio"] = float(step.sum() / dmed)
        r["rot_total"] = float(ang.sum())
        r["static_cam"] = float(r["baseline_ratio"] < 0.05)
    except Exception as e:  # noqa: BLE001
        r["error"] = (str(e) + traceback.format_exc()[-300:])[:500]
    return r


if __name__ == "__main__":
    groups = sys.argv[1:] or ["A", "B", "C"]
    for G in groups:
        jobs = [(G, m, c) for c in clips(G) for m in METHODS]
        with ProcessPoolExecutor(32) as ex:
            rows = list(ex.map(signals, jobs, chunksize=1))
        df = pd.DataFrame(rows)
        df.to_csv(QC / f"signals_{G}.csv", index=False, float_format="%.5f")
        err = df["error"].notna().sum() if "error" in df else 0
        print(G, len(df), "errors", err, flush=True)
