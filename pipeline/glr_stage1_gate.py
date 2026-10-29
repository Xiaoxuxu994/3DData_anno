"""Stage-1 exit gate for the GLRMap track: read a COLMAP/GLOMAP sparse model (+ database) and decide whether the
camera model is trustworthy enough to spend the expensive dense stages on.

GLRMap keeps the cameras of its first stage (COLMAP SIFT + sequential matching + global SfM) fixed for everything that
follows, and its own diagnostics are self-consistency numbers that stay "fine" when stage 1 is wrong (see
docs/GLR_INTEGRATION.md). This module only uses what stage 1 leaves on disk - the sparse model and database.db - and is
therefore independent of the GLRMap code; it needs pycolmap (a GLRMap requirement) for models, sqlite3 for the database.

Usage (library):  summarize_model(dir) / summarize_db(db) / gate(model, db, n_frames, cfg) / load_cameras(dir)
Usage (CLI):      python glr_stage1_gate.py MODEL_DIR DATABASE N_FRAMES [--cfg configs/pipeline_glr_v0.yaml] [--witness-hfov DEG]
                  prints the JSON verdict; exit code 0 = ok, 1 = suspect, 3 = stop (same code as the NPU camera_sanity gate)

Signals (all computed from stage-1 output only; thresholds live in configs/pipeline_glr_v0.yaml, provisional until the
Phase-0 replay in docs/GLR_INTEGRATION.md has been run on the stage-1 corpus):
  stop    : registered < 50 % of the frames, < 50 3D points per frame, focal outside the sane range, |fx/fy-1| > 0.1,
            >= 50 % of the verified image pairs are PLANAR_OR_PANORAMIC (no parallax / pure rotation)
  suspect : mean reprojection error > 1 px, adjacent-camera step max/median > 5 (pose outlier), focal/width within 0.06 of
            COLMAP's default 1.2 (focal probably never estimated), median triangulation angle < 2 deg (low parallax),
            |HFOV - witness HFOV| > 13.2 deg when a witness is given
"""

import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

DEFAULT_CFG = {
    "min_reg_frac": 0.5,
    "min_pts_per_img": 50,
    "hfov_range_deg": [14.3, 118.1],  # = focal 0.3 .. 4 x max(W, H), as in GLRMap's camera_sanity patch
    "max_fx_fy_dev": 0.1,
    "max_planar_frac": 0.5,
    "max_reproj_px": 1.0,
    "max_step_ratio": 5.0,
    "default_focal_w": 1.2,
    "default_focal_tol": 0.06,  # 0.03 in the evaluation sessions; the pycolmap replica of Ahlbeck ended at 1.16 vs 1.19 there
    "min_median_tri_angle_deg": 2.0,
    "max_witness_hfov_dis_deg": 13.2,
}
PAIR_BASE = 2147483647
CFG_PLANAR = 6  # COLMAP TwoViewGeometry::PLANAR_OR_PANORAMIC
CFG_VALID = (2, 3, 4, 5, 6, 8)  # CALIBRATED, UNCALIBRATED, PLANAR, PANORAMIC, PLANAR_OR_PANORAMIC, MULTIPLE


def _pycolmap():
    import pycolmap
    return pycolmap


def _cam_from_world(img):
    f = img.cam_from_world
    return f() if callable(f) else f


def _rt(img):
    """(R, t) with X_cam = R X_world + t."""
    cw = _cam_from_world(img)
    R = cw.rotation.matrix() if hasattr(cw.rotation, "matrix") else np.asarray(cw.rotation)
    return np.asarray(R, np.float64), np.asarray(cw.translation, np.float64).reshape(3)


def camera_K(cam):
    """3x3 intrinsics of a pycolmap camera (PINHOLE / SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL / OPENCV, distortion ignored)."""
    p = np.asarray(cam.params, np.float64)
    name = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
    if name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif name in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f"unsupported camera model {name}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), name


def load_cameras(model_dir):
    """-> dict(names, c2w (N,4,4) OpenCV camera-to-world in the model's (arbitrary) scale, K (N,3,3), W, H, model, order).
    Registered images only, sorted by name (= frame order for frames named 000001.jpg ...)."""
    rec = _pycolmap().Reconstruction(str(model_dir))
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    names, c2w, K = [], [], []
    for im in imgs:
        R, t = _rt(im)
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        k, name = camera_K(rec.cameras[im.camera_id])
        names.append(im.name)
        c2w.append(T)
        K.append(k)
    cam = rec.cameras[imgs[0].camera_id] if imgs else None
    return {"names": names, "c2w": np.array(c2w), "K": np.array(K), "W": int(cam.width) if cam else 0,
            "H": int(cam.height) if cam else 0, "model": name if imgs else None}


def tri_angles(rec, centers, max_points=4000, seed=0):
    """Max pairwise-ray angle (deg) of a random sample of 3D points with >= 2 observations."""
    rng = np.random.default_rng(seed)
    ids = [pid for pid, p in rec.points3D.items() if p.track.length() >= 2]
    if not ids:
        return np.array([])
    ids = rng.choice(ids, min(max_points, len(ids)), replace=False)
    out = []
    for pid in ids:
        p = rec.points3D[int(pid)]
        X = np.asarray(p.xyz, np.float64)
        rays = []
        for el in p.track.elements:
            c = centers.get(el.image_id)
            if c is not None:
                v = X - c
                n = np.linalg.norm(v)
                if n > 0:
                    rays.append(v / n)
        if len(rays) >= 2:
            r = np.array(rays)
            out.append(math.degrees(math.acos(np.clip((r @ r.T).min(), -1, 1))))
    return np.array(out)


def summarize_model(model_dir):
    rec = _pycolmap().Reconstruction(str(model_dir))
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    centers = {}
    for im in imgs:
        R, t = _rt(im)
        centers[im.id if hasattr(im, "id") else im.image_id] = -R.T @ t
    cams = {c.camera_id if hasattr(c, "camera_id") else k: c for k, c in rec.cameras.items()}
    cam = rec.cameras[imgs[0].camera_id] if imgs else None
    s = {"n_reg": len(imgs), "n_points": int(rec.num_points3D()), "n_cameras": len(rec.cameras)}
    try:
        s["reproj"] = float(rec.compute_mean_reprojection_error())
    except Exception:  # noqa: BLE001
        s["reproj"] = None
    try:
        s["mean_track_len"] = float(rec.compute_mean_track_length())
    except Exception:  # noqa: BLE001
        s["mean_track_len"] = None
    if cam is not None:
        K, name = camera_K(cam)
        W, H = int(cam.width), int(cam.height)
        s.update(model=name, W=W, H=H, fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                 focal_w=float(K[0, 0] / W), fx_fy=float(K[0, 0] / K[1, 1]),
                 hfov_deg=float(math.degrees(2 * math.atan(W / 2 / K[0, 0]))),
                 params=[float(x) for x in cam.params])
    if len(imgs) >= 3:
        C = np.array([centers[im.id if hasattr(im, "id") else im.image_id] for im in imgs])
        idx = np.array([int("".join(ch for ch in Path(im.name).stem if ch.isdigit()) or 0) for im in imgs])
        d = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.maximum(np.diff(idx), 1)
        med = float(np.median(d))
        s["step_med"] = med
        s["step_ratio"] = float(d.max() / med) if med > 0 else float("inf")
        s["path_len"] = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    ang = tri_angles(rec, centers)
    if len(ang):
        s["tri_angle_p50"], s["tri_angle_p90"] = float(np.percentile(ang, 50)), float(np.percentile(ang, 90))
    return s


def summarize_db(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    s = {}
    s["n_images"] = cur.execute("select count(*) from images").fetchone()[0]
    r = cur.execute("select avg(rows), min(rows), max(rows) from keypoints").fetchone()
    s["kp_mean"], s["kp_min"], s["kp_max"] = [float(x) if x is not None else None for x in r]
    cfg = {int(c): int(n) for c, n in cur.execute("select config, count(*) from two_view_geometries group by config")}
    s["pair_configs"] = cfg
    n_all = sum(cfg.values())
    n_valid = sum(n for c, n in cfg.items() if c in CFG_VALID)
    s["n_pairs"], s["n_pairs_valid"] = n_all, n_valid
    s["planar_frac"] = cfg.get(CFG_PLANAR, 0) / n_valid if n_valid else None
    s["planar_frac_all"] = cfg.get(CFG_PLANAR, 0) / n_all if n_all else None
    rows = [r[0] for r in cur.execute("select rows from two_view_geometries where config in (%s)" % ",".join(map(str, CFG_VALID)))]
    s["inliers_median"] = float(np.median(rows)) if rows else None
    con.close()
    return s


def gate(model, db, n_frames, cfg=None, witness_hfov=None):
    """-> dict(verdict 'ok'|'suspect'|'stop', stop=[...], suspect=[...], signals={...}). Missing signals never fire."""
    c = dict(DEFAULT_CFG, **(cfg or {}))
    stop, sus = [], []
    sig = {}
    if not model or not model.get("n_reg"):
        return {"verdict": "stop", "stop": ["no_model"], "suspect": [], "signals": {}}
    sig["reg_frac"] = model["n_reg"] / max(n_frames, 1)
    sig["pts_per_img"] = model["n_points"] / max(model["n_reg"], 1)
    sig["hfov_deg"] = model.get("hfov_deg")
    sig["fx_fy_dev"] = abs(model.get("fx_fy", 1.0) - 1.0)
    sig["planar_frac"] = (db or {}).get("planar_frac")
    sig["reproj"] = model.get("reproj")
    sig["step_ratio"] = model.get("step_ratio")
    sig["focal_w"] = model.get("focal_w")
    sig["tri_angle_p50"] = model.get("tri_angle_p50")
    if sig["reg_frac"] < c["min_reg_frac"]:
        stop.append("registered_lt_half")
    if sig["pts_per_img"] < c["min_pts_per_img"]:
        stop.append("points_per_image")
    lo, hi = c["hfov_range_deg"]
    if sig["hfov_deg"] is not None and not lo <= sig["hfov_deg"] <= hi:
        stop.append("hfov_out_of_range")
    if sig["fx_fy_dev"] > c["max_fx_fy_dev"]:
        stop.append("fx_fy_mismatch")
    if sig["planar_frac"] is not None and sig["planar_frac"] >= c["max_planar_frac"]:
        stop.append("planar_or_panoramic_pairs")
    if sig["reproj"] is not None and sig["reproj"] > c["max_reproj_px"]:
        sus.append("reproj_gt_1px")
    if sig["step_ratio"] is not None and sig["step_ratio"] > c["max_step_ratio"]:
        sus.append("pose_step_outlier")
    if sig["focal_w"] is not None and abs(sig["focal_w"] - c["default_focal_w"]) < c["default_focal_tol"]:
        sus.append("focal_at_colmap_default")
    if sig["tri_angle_p50"] is not None and sig["tri_angle_p50"] < c["min_median_tri_angle_deg"]:
        sus.append("low_parallax")
    if witness_hfov is not None and sig["hfov_deg"] is not None:
        sig["witness_hfov_dis"] = abs(sig["hfov_deg"] - witness_hfov)
        if sig["witness_hfov_dis"] > c["max_witness_hfov_dis_deg"]:
            sus.append("k_witness_disagree")
    return {"verdict": "stop" if stop else ("suspect" if sus else "ok"), "stop": stop, "suspect": sus, "signals": sig}


def _load_cfg(path):
    if not path:
        return {}
    import yaml  # optional dependency, only for the CLI
    d = yaml.safe_load(Path(path).read_text())
    return (d.get("stage1_gate") or {}) if isinstance(d, dict) else {}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("database")
    ap.add_argument("n_frames", type=int)
    ap.add_argument("--cfg")
    ap.add_argument("--witness-hfov", type=float)
    ap.add_argument("--job", help="job.json written by glr_prepare.py: thresholds (stage1_gate_cfg) and the K witness (k_prior)")
    a = ap.parse_args()
    cfg, wit = _load_cfg(a.cfg), a.witness_hfov
    if a.job:
        job = json.loads(Path(a.job).read_text())
        cfg = dict(job.get("stage1_gate_cfg") or {}, **cfg)
        wit = wit if wit is not None else (job.get("k_prior") or {}).get("hfov_deg")
    m, d = summarize_model(a.model_dir), (summarize_db(a.database) if a.database and Path(a.database).exists() else None)
    v = gate(m, d, a.n_frames, cfg, wit)
    print(json.dumps({"model": m, "db": d, "gate": v}, indent=1, default=float))
    sys.exit({"ok": 0, "suspect": 1, "stop": 3}[v["verdict"]])
