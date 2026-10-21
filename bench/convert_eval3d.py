"""Convert 3D evaluation datasets (ranking group) to one format.

Usage: python convert_eval3d.py <eth3d|tum|bonn|7scenes|sintel>
Raw:  /data/home/z00533370/data/eval3d_raw/<dataset>/   (archives already extracted next to them)
Out:  /data/home/z00533370/data/eval3d/A_rank/<dataset>/<clip>/{images/,depth/,gt.npz,meta.json} + clips.json

gt.npz: c2w (N,4,4) cam->world OpenCV (x right, y down, z forward); K (N,3,3) for the images.
depth png: uint16, metres = value / meta.depth_scale, 0 = invalid.
"""

import glob
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

RAW = Path("/data/home/z00533370/data/eval3d_raw")
OUT = Path("/data/home/z00533370/data/eval3d/A_rank")
MAXF = 200


def quat2mat(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_list(p):
    rows = []
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        rows.append(l.split())
    return rows


def tum_pose_list(p):
    out = []
    for r in read_list(p):
        T = np.eye(4)
        T[:3, :3] = quat2mat(*map(float, r[4:8]))
        T[:3, 3] = list(map(float, r[1:4]))
        out.append((float(r[0]), T))
    return out


def nearest(ts, arr_ts, max_dt=0.02):
    i = int(np.argmin(np.abs(arr_ts - ts)))
    return i if abs(arr_ts[i] - ts) <= max_dt else None


def associate_tum(seq_dir, max_dt=0.02):
    """RGB-major association with depth and groundtruth (nearest neighbour, |dt| <= max_dt)."""
    rgb = [(float(r[0]), r[1]) for r in read_list(seq_dir / "rgb.txt")]
    dep = [(float(r[0]), r[1]) for r in read_list(seq_dir / "depth.txt")]
    gt = tum_pose_list(seq_dir / "groundtruth.txt")
    dts = np.array([d[0] for d in dep])
    gts = np.array([g[0] for g in gt])
    triples = []
    for t, f in rgb:
        i, j = nearest(t, dts, max_dt), nearest(t, gts, max_dt)
        if i is None or j is None:
            continue
        triples.append((t, f, dep[i][1], gt[j][1]))
    return triples


def write_clip(ds, clip, frames, K, meta, depth_scale=None, depth_writer=None, image_ext=".png", K_depth=None):
    """frames: list of dict(img=path or array, depth=path or array, c2w=4x4, fid=int)."""
    d = OUT / ds / clip
    (d / "images").mkdir(parents=True, exist_ok=True)
    if depth_writer is not None:
        (d / "depth").mkdir(exist_ok=True)
    c2w = []
    for k, fr in enumerate(frames):
        dst = d / "images" / f"{k:06d}{image_ext}"
        if isinstance(fr["img"], (str, Path)) and str(fr["img"]).endswith(image_ext):
            shutil.copyfile(fr["img"], dst)
        else:
            im = fr["img"] if isinstance(fr["img"], np.ndarray) else cv2.imread(str(fr["img"]), cv2.IMREAD_COLOR)
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if depth_writer is not None:
            cv2.imwrite(str(d / "depth" / f"{k:06d}.png"), depth_writer(fr["depth"]))
        c2w.append(fr["c2w"])
    n = len(frames)
    Ks = np.repeat(np.asarray(K, float)[None], n, 0) if np.asarray(K).ndim == 2 else np.asarray(K, float)
    extra = {}
    if K_depth is not None:
        extra["K_depth"] = np.repeat(np.asarray(K_depth, float)[None], n, 0)
    np.savez(d / "gt.npz", c2w=np.stack(c2w).astype(np.float64), K=Ks, **extra)
    h, w = cv2.imread(str(d / "images" / f"000000{image_ext}")).shape[:2]
    meta.update({"dataset": ds, "clip": clip, "n_frames": n, "hw": [h, w], "depth_scale": depth_scale,
                 "frame_ids": [fr["fid"] for fr in frames]})
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"clip": clip, "n_frames": n, "source": meta["source"], "dynamic": meta["dynamic"]}


def copy_png16(p):
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def pick(n_total, stride, maxf=MAXF):
    return list(range(0, n_total, stride))[:maxf]


# ------------------------------------------------------------------ TUM-format datasets
TUM_K = {
    "freiburg1": [517.3, 516.5, 318.6, 255.3],
    "freiburg2": [520.9, 521.0, 325.1, 249.7],
    "freiburg3": [535.4, 539.2, 320.1, 247.6],
}
BONN_K = [542.822841, 542.576870, 315.593520, 237.756098]
T_ROS = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
T_M = np.array([[1.0157, 0.1828, -0.2389, 0.0113], [0.0009, -0.8431, -0.6413, -0.0098],
                [-0.3009, 0.6147, -0.8085, 0.0111], [0, 0, 0, 1.0]])


def Kmat(f):
    return np.array([[f[0], 0, f[2]], [0, f[1], f[3]], [0, 0, 1]], float)


def bonn_pose(T, mode):
    if mode == "raw":
        return T
    Tm = T_M.copy()
    s = np.cbrt(np.linalg.det(Tm[:3, :3]))
    Tm[:3, :3] /= s  # T_m carries a uniform scale (~1.06); keep only the rigid part
    return np.linalg.inv(T_ROS) @ T @ T_ROS @ Tm


def convert_tum_like(ds, seq_dirs, Kf, stride, dynamic_fn, license_, notes, pose_fn=lambda T: T):
    rows = []
    for seq in seq_dirs:
        tri = associate_tum(seq)
        idx = pick(len(tri), stride)
        frames = [{"img": seq / tri[i][1], "depth": seq / tri[i][2], "c2w": pose_fn(tri[i][3]), "fid": i} for i in idx]
        name = seq.name.replace("rgbd_dataset_", "").replace("rgbd_bonn_", "")
        meta = {"source": seq.name, "stride": stride, "fps_effective": 30.0 / stride, "depth_type": "dense",
                "metric": True, "dynamic": dynamic_fn(name), "license": license_, "notes": notes,
                "association": "rgb-major nearest depth & groundtruth, |dt|<=0.02s", "n_associated": len(tri)}
        rows.append(write_clip(ds, name, frames, Kmat(Kf(seq.name)), meta, depth_scale=5000.0, depth_writer=copy_png16))
        print(ds, name, len(frames), flush=True)
    return rows


def run_tum():
    seqs = sorted(p for p in (RAW / "tum").iterdir() if p.is_dir() and p.name.startswith("rgbd_dataset"))
    kf = lambda n: TUM_K[n.split("_")[2]]  # noqa: E731
    return convert_tum_like("tum", seqs, kf, 2, lambda n: "walking" in n or "sitting" in n, "CC BY 4.0",
                            "TUM RGB-D. groundtruth = colour optical centre cam->world (OpenCV axes). Depth registered to RGB "
                            "(OpenNI), scale 5000. fr1/fr2 images are NOT undistorted (small radial distortion), fr3 are.")


def run_bonn(mode=None):
    mode = mode or os.environ.get("BONN_MODE", "raw")
    seqs = sorted(p for p in (RAW / "bonn").iterdir() if p.is_dir() and p.name.startswith("rgbd_bonn_"))
    return convert_tum_like("bonn", seqs, lambda n: BONN_K, 2, lambda n: True, "see website (no explicit license)",
                            f"Bonn RGB-D Dynamic (TUM format). Depth registered to RGB, scale 5000. Pose mode: {mode} "
                            "(raw = groundtruth.txt as cam->world; corrected = T_ROS^-1 T T_ROS T_m with T_m scale removed). "
                            "Mode chosen by reprojection check (CHECK.md).", pose_fn=lambda T: bonn_pose(T, mode))


def run_eth3d():
    rows = []
    for seq in sorted(p for p in (RAW / "eth3d").iterdir() if p.is_dir()):
        cal = list(map(float, open(seq / "calibration.txt").read().split()))
        rgb = [(float(r[0]), r[1]) for r in read_list(seq / "rgb.txt")]
        dep = [(float(r[0]), r[1]) for r in read_list(seq / "depth.txt")]
        gt = tum_pose_list(seq / "groundtruth.txt")
        dts, gts = np.array([d[0] for d in dep]), np.array([g[0] for g in gt])
        tri = []
        for t, f in rgb:
            i, j = nearest(t, dts), nearest(t, gts)
            if i is not None and j is not None:
                tri.append((t, f, dep[i][1], gt[j][1]))
        stride = 2
        idx = pick(len(tri), stride)
        frames = [{"img": seq / tri[i][1], "depth": seq / tri[i][2], "c2w": tri[i][3], "fid": i} for i in idx]
        meta = {"source": seq.name, "stride": stride, "fps_effective": 27.0 / stride, "depth_type": "dense",
                "metric": True, "dynamic": False, "license": "CC BY-NC-SA 4.0 (ETH3D)",
                "notes": "ETH3D SLAM training (rgbd). groundtruth TUM format cam->world; depth registered to RGB, scale 5000.",
                "n_associated": len(tri)}
        rows.append(write_clip("eth3d", seq.name, frames, Kmat(cal[:4]), meta, depth_scale=5000.0, depth_writer=copy_png16))
        print("eth3d", seq.name, len(frames), flush=True)
    return rows


# ------------------------------------------------------------------ 7-Scenes
def run_7scenes():
    rows = []
    f_d = float(os.environ.get("SEVEN_FD", "585"))
    for scene in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]:
        sd = RAW / "7scenes" / scene
        test = [l.strip() for l in open(sd / "TestSplit.txt") if l.strip()]
        seq_num = int(test[0].replace("sequence", ""))
        seq = sd / f"seq-{seq_num:02d}"
        n = len(glob.glob(str(seq / "*.color.png")))
        stride = max(1, n // MAXF)
        idx = pick(n, stride)
        frames = []
        for i in idx:
            frames.append({"img": seq / f"frame-{i:06d}.color.png", "depth": seq / f"frame-{i:06d}.depth.png",
                           "c2w": np.loadtxt(seq / f"frame-{i:06d}.pose.txt"), "fid": i})

        def dw(p):
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.uint16)
            d[d == 65535] = 0
            return d

        K_rgb = Kmat([525.0, 525.0, 320.0, 240.0])
        K_dep = Kmat([f_d, f_d, 320.0, 240.0])
        meta = {"source": f"{scene}/seq-{seq_num:02d}", "stride": stride, "fps_effective": 30.0 / stride,
                "depth_type": "dense_unregistered", "metric": True, "dynamic": False, "license": "Microsoft Research, non-commercial",
                "notes": "7-Scenes TestSplit first test sequence. pose.txt = cam->world 4x4 (Kinect depth camera). Depth (mm, 65535->0) is "
                         "NOT registered to RGB; gt.npz K = commonly used RGB intrinsics (525,525,320,240), K_depth = depth intrinsics "
                         f"({f_d:g}). Use K_depth for depth-only geometry."}
        rows.append(write_clip("7scenes", f"{scene}_seq{seq_num:02d}", frames, K_rgb, meta, depth_scale=1000.0,
                               depth_writer=dw, K_depth=K_dep))
        print("7scenes", scene, len(frames), flush=True)
    return rows


# ------------------------------------------------------------------ Sintel
def read_dpt(p):
    with open(p, "rb") as f:
        tag, w, h = struct.unpack("fii", f.read(12))
        assert abs(tag - 202021.25) < 1e-3
        return np.fromfile(f, np.float32, w * h).reshape(h, w)


def read_cam(p):
    with open(p, "rb") as f:
        tag = struct.unpack("f", f.read(4))[0]
        assert abs(tag - 202021.25) < 1e-3
        M = np.fromfile(f, np.float64, 9).reshape(3, 3)
        N = np.fromfile(f, np.float64, 12).reshape(3, 4)
    return M, N


def run_sintel():
    root_img = next(Path(p) for p in glob.glob(str(RAW / "sintel" / "**" / "training" / "final"), recursive=True))
    root_dep = next(Path(p) for p in glob.glob(str(RAW / "sintel" / "**" / "training" / "depth"), recursive=True))
    root_cam = root_dep.parent / "camdata_left"
    rows = []
    scale = 500.0
    for scene in sorted(os.listdir(root_dep)):
        files = sorted(glob.glob(str(root_dep / scene / "*.dpt")))
        frames, Ks = [], []
        for i, fp in enumerate(files):
            stem = Path(fp).stem
            M, N = read_cam(root_cam / scene / f"{stem}.cam")
            w2c = np.eye(4)
            w2c[:3] = N
            frames.append({"img": root_img / scene / f"{stem}.png", "depth": fp, "c2w": np.linalg.inv(w2c), "fid": i})
            Ks.append(M)

        def dw(p):
            d = read_dpt(p)
            v = np.round(d * scale)
            v[(v > 65535) | ~np.isfinite(v) | (v <= 0)] = 0
            return v.astype(np.uint16)

        meta = {"source": scene, "stride": 1, "fps_effective": 24.0, "depth_type": "dense", "metric": False,
                "dynamic": True, "license": "MPI Sintel (research use)",
                "notes": "MPI Sintel training, final pass. .cam N = world->cam 3x4 (inverted to cam->world). depth .dpt in "
                         f"scene units (not metric), stored x{scale:g}; values > {65535 / scale:.0f} set invalid."}
        rows.append(write_clip("sintel", scene, frames, np.stack(Ks), meta, depth_scale=scale, depth_writer=dw,
                               image_ext=".png"))
        print("sintel", scene, len(frames), flush=True)
    return rows


# ------------------------------------------------------------------ ETH3D high-res multi-view (MVS), not a video
def run_eth3d_mvs():
    root = RAW / "eth3d_mvs" / "hf"
    rows = []
    scale = 1000.0
    for scene in sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        cal = root / scene / "dslr_calibration_undistorted"
        cams = {}
        for r in read_list(cal / "cameras.txt"):
            assert r[1] == "PINHOLE", r[1]
            cams[r[0]] = (int(r[2]), int(r[3]), Kmat(list(map(float, r[4:8]))))
        lines = [l for l in open(cal / "images.txt") if not l.startswith("#")]
        recs, cam_of = [], {}
        for k in range(0, len(lines), 2):
            r = lines[k].split()
            if len(r) < 10:
                continue
            qw, qx, qy, qz, tx, ty, tz = map(float, r[1:8])
            w2c = np.eye(4)
            w2c[:3, :3] = quat2mat(qx, qy, qz, qw)
            w2c[:3, 3] = [tx, ty, tz]
            recs.append((r[9], cams[r[8]], np.linalg.inv(w2c)))
            cam_of[r[9]] = r[8]
        recs.sort(key=lambda x: x[0])
        # GT depth is defined on the ORIGINAL distorted JPGs (THIN_PRISM_FISHEYE, dslr_calibration_jpg);
        # resample it onto the undistorted PINHOLE grid through the distortion model (z-depth is unchanged).
        dist = {}
        for r in read_list(root / scene / "dslr_calibration_jpg" / "cameras.txt"):
            assert r[1] == "THIN_PRISM_FISHEYE", r[1]
            dist[r[0]] = (int(r[2]), int(r[3]), np.array(list(map(float, r[4:16]))))
        maps = {}

        def undist_map(cid, wu, hu, Ku):
            if cid not in maps:
                wd, hd, prm = dist[cid]
                fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = prm
                v, u = np.mgrid[0:hu, 0:wu].astype(np.float64)
                x, y = (u - Ku[0, 2]) / Ku[0, 0], (v - Ku[1, 2]) / Ku[1, 1]
                r = np.sqrt(x * x + y * y)
                th = np.arctan(r)
                s = np.where(r > 1e-8, th / np.maximum(r, 1e-12), 1.0)
                uf, vf = x * s, y * s
                t2 = uf * uf + vf * vf
                rad = 1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4
                du = uf * rad + 2 * p1 * uf * vf + p2 * (t2 + 2 * uf * uf) + sx1 * t2
                dv = vf * rad + 2 * p2 * uf * vf + p1 * (t2 + 2 * vf * vf) + sy1 * t2
                maps[cid] = ((fx * du + cx).astype(np.float32), (fy * dv + cy).astype(np.float32), wd, hd)
            return maps[cid]

        frames, Ks = [], []
        for i, (name, (w, h, K), c2w) in enumerate(recs):
            base = Path(name).name
            frames.append({"img": root / scene / "images" / name,
                           "depth": (root / scene / "ground_truth_depth" / "dslr_images" / base, w, h, K, cam_of[name]),
                           "c2w": c2w, "fid": i})
            Ks.append(K)

        def dw(arg):
            p, wu, hu, Ku, cid = arg
            mx, my, wd, hd = undist_map(cid, wu, hu, Ku)
            d = np.fromfile(p, np.float32).reshape(hd, wd)
            d[~np.isfinite(d)] = 0
            du_ = cv2.remap(d, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            v = np.round(du_ * scale)
            v[(v > 65535) | (v <= 0)] = 0
            return v.astype(np.uint16)

        meta = {"source": scene, "stride": 1, "fps_effective": None, "depth_type": "dense_laser", "metric": True, "dynamic": False,
                "license": "CC BY-NC-SA 4.0 (ETH3D)",
                "notes": "ETH3D high-res multi-view training scene (DSLR, unordered multi-view set, NOT a video; sorted by image name). "
                         "Undistorted images + PINHOLE intrinsics; COLMAP images.txt world->cam inverted to cam->world. GT depth: raw float32 "
                         f"(inf = invalid), stored x{scale:g}. Downloaded from HF mirror PeterDAI/eth3d (ETH3D SLAM servers unreachable)."}
        rows.append(write_clip("eth3d_mvs", scene, frames, np.stack(Ks), meta, depth_scale=scale, depth_writer=dw, image_ext=".jpg"))
        print("eth3d_mvs", scene, len(frames), flush=True)
    return rows


if __name__ == "__main__":
    ds = sys.argv[1]
    rows = {"tum": run_tum, "bonn": run_bonn, "eth3d": run_eth3d, "7scenes": run_7scenes, "sintel": run_sintel,
            "eth3d_mvs": run_eth3d_mvs}[ds]()
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    (OUT / ds / "clips.json").write_text(json.dumps(rows, indent=1))
    idx_p = OUT / "index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    idx[ds] = {"n_clips": len(rows), "n_frames": sum(r["n_frames"] for r in rows), "clips": [r["clip"] for r in rows]}
    idx_p.write_text(json.dumps(idx, indent=1))
    print("DONE", ds, len(rows))
