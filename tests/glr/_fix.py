"""Fixtures for the GLR-track tests: a synthetic scene, a COLMAP text model + database written from it, tiny job packages."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PIPE = Path(__file__).resolve().parents[2] / "pipeline"
sys.path.insert(0, str(PIPE))


def look_at_rig(n=20, step=0.2, curve=0.0):
    """c2w (n,4,4) of a camera translating along +x (optionally curving), looking along +z (OpenCV), metric units."""
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        T[i, 0, 3] = step * i
        T[i, 2, 3] = curve * (i * step) ** 2
    return T


def scene_points(n=3000, seed=0, depth=(3.0, 12.0)):
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-6, 10, n), rng.uniform(-3, 3, n), rng.uniform(*depth, n)]
    return X


def project(X, c2w, K):
    w2c = np.linalg.inv(c2w)
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def rot_to_qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def write_text_model(mdir, c2w, K, W, H, X, names=None, cam="PINHOLE", scale=1.0):
    """COLMAP text model (cameras/images/points3D .txt) of `X` seen by every camera that sees it in front within the image."""
    mdir = Path(mdir)
    mdir.mkdir(parents=True, exist_ok=True)
    n = len(c2w)
    names = names or [f"{i + 1:06d}.jpg" for i in range(n)]
    Xs = X * scale
    c2w_s = c2w.copy()
    c2w_s[:, :3, 3] *= scale
    if cam == "PINHOLE":
        (mdir / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    else:
        (mdir / "cameras.txt").write_text(f"1 SIMPLE_PINHOLE {W} {H} {K[0, 0]} {K[0, 2]} {K[1, 2]}\n")
    obs = [[] for _ in range(len(X))]
    lines = []
    for i in range(n):
        uv, z = project(Xs, c2w_s[i], K)
        w2c = np.linalg.inv(c2w_s[i])
        q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        pts = []
        for j in range(len(Xs)):
            if z[j] > 0.1 and 0 <= uv[j, 0] < W and 0 <= uv[j, 1] < H:
                obs[j].append((i + 1, len(pts)))
                pts.append(f"{uv[j, 0]:.3f} {uv[j, 1]:.3f} {j + 1}")
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {names[i]}")
        lines.append(" ".join(pts))
    (mdir / "images.txt").write_text("\n".join(lines) + "\n")
    p3 = []
    for j, o in enumerate(obs):
        if len(o) >= 2:
            p3.append(f"{j + 1} {Xs[j, 0]:.6f} {Xs[j, 1]:.6f} {Xs[j, 2]:.6f} 128 128 128 0.3 " + " ".join(f"{a} {b}" for a, b in o))
    (mdir / "points3D.txt").write_text("\n".join(p3) + "\n")


def write_db(path, n_images, planar_frac=0.0, n_pairs=40, kp=5000, seed=0):
    con = sqlite3.connect(str(path))
    c = con.cursor()
    c.execute("create table images (image_id integer primary key, name text, camera_id integer)")
    c.execute("create table keypoints (image_id integer primary key, rows integer, cols integer, data blob)")
    c.execute("create table two_view_geometries (pair_id integer primary key, rows integer, cols integer, data blob, config integer)")
    for i in range(1, n_images + 1):
        c.execute("insert into images values (?,?,1)", (i, f"{i:06d}.jpg"))
        c.execute("insert into keypoints values (?,?,6,NULL)", (i, kp))
    rng = np.random.default_rng(seed)
    for k in range(n_pairs):
        i = 1 + k % (n_images - 1)
        j = i + 1 + (k // (n_images - 1))
        if j > n_images:
            continue
        cfg = 6 if rng.random() < planar_frac else 2
        c.execute("insert or ignore into two_view_geometries values (?,?,2,NULL,?)", (i * 2147483647 + j, int(rng.integers(40, 400)), cfg))
    con.commit()
    con.close()


def make_job(out, clip, n_orig=180, stride=2, hw=(270, 480), k_prior=None, gate_cfg=None, with_images=True):
    jd = Path(out) / "glr_jobs" / clip
    (jd / "images").mkdir(parents=True)
    idx = list(range(0, n_orig, stride))
    names = [f"{k + 1:06d}.jpg" for k in range(len(idx))]
    if with_images:
        import cv2
        for nm in names[:3]:
            cv2.imwrite(str(jd / "images" / nm), np.full((hw[0], hw[1], 3), 90, np.uint8))
    job = {"clip": clip, "annotator": "glr", "version": "glr-v0", "n_orig": n_orig, "fps_orig": 15, "stride": stride, "frame_idx": idx,
           "image_names": names, "hw": list(hw), "route_reasons": "", "k_prior": k_prior, "stage1_gate_cfg": gate_cfg or {}}
    (jd / "job.json").write_text(json.dumps(job))
    return jd, job
