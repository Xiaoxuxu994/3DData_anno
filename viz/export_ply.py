"""Export the 3D content behind each demo video as PLY files for interactive viewing (MeshLab / CloudCompare / Blender).

  OMP_NUM_THREADS=1 cache/qc-venv/bin/python code/viz/export_ply.py [--workers=4] [--only=id1,id2]
  -> outputs/viz/export/<video id>/{video.mp4, poster.jpg, *pointcloud.ply, *cameras.ply, README.txt}

Same labels / alignment / filters as render_demo.py (flying pixels, ViPE dynamic + sky masks, Pi3X conf < clip q25 for
the v1.3 single-label clouds), but denser: every 2nd frame, ~150k px per frame, voxel-downsampled (0.5 % of the clip's
median depth, clamped to 1-10 cm, coarsened until <= 1.5M points), no ceiling cut. Units: metres in the label's world frame (after v1.3 scale).
GT (A/B and ScanNet++): GT depth + GT poses/K, Sim3-aligned (camera centres) into the label frame.
cameras.ply: triangle mesh, a frustum pyramid every 10 frames (colour = time, purple -> yellow) + a thin tube along the
camera centres.
"""
import ast
import json
import shutil
import struct
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/data/home/z00533370/code/viz")
import render_demo as RD  # noqa: E402  (sets the 1-thread env vars)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = RD.ROOT
VID = ROOT / "outputs/viz/videos"
EXP = ROOT / "outputs/viz/export"
CAP = 1_500_000
PLASMA = RD.VIRID


# --------------------------------------------------------------------------------------------- PLY writers
def write_points(path, P, C):
    P = np.ascontiguousarray(P, np.float32)
    C = np.ascontiguousarray(C, np.uint8)
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
    rec["r"], rec["g"], rec["b"] = C[:, 0], C[:, 1], C[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(rec.tobytes())


def write_mesh(path, V, VC, F):
    V = np.asarray(V, np.float32)
    VC = np.asarray(VC, np.uint8)
    F = np.asarray(F, np.int32)
    rec = np.empty(len(V), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = V[:, 0], V[:, 1], V[:, 2]
    rec["r"], rec["g"], rec["b"] = VC[:, 0], VC[:, 1], VC[:, 2]
    fr = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    fr["n"], fr["i"] = 3, F
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(rec.tobytes())
        f.write(fr.tobytes())


def camera_mesh(c2w, K, W, H, sz, color=None, every=10):
    """Frustum pyramids every `every` frames + tube along the centres. color=None -> time colormap."""
    n = len(c2w)
    V, VC, F = [], [], []

    def col(i):
        return PLASMA[int(255 * i / max(n - 1, 1))] if color is None else np.array(color, np.uint8)

    def add(verts, faces, c):
        o = sum(len(v) for v in V)
        V.append(np.asarray(verts))
        VC.append(np.repeat(np.asarray(c)[None], len(verts), 0))
        F.append(np.asarray(faces) + o)

    idx = list(range(0, n, every))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    for i in idx:
        Kk = K[i]
        cr = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
        rays = np.stack([(cr[:, 0] - Kk[0, 2]) / Kk[0, 0], (cr[:, 1] - Kk[1, 2]) / Kk[1, 1], np.ones(4)], 1)
        rays *= sz / max(np.abs(rays[:, :2]).max(), 1.0)  # keep wide-FOV frusta the same size
        pts = np.concatenate([c2w[i, :3, 3][None], rays @ c2w[i, :3, :3].T + c2w[i, :3, 3]])
        # sides (both windings so they are visible from either side) + base
        faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
                 [1, 2, 3], [1, 3, 4], [1, 3, 2], [1, 4, 3]]
        add(pts, faces, col(i))
    # tube along the centres (square cross-section)
    r = sz * 0.06
    C = c2w[:, :3, 3]
    for j in range(n - 1):
        a, b = C[j], C[j + 1]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        d /= L
        u = np.cross(d, np.eye(3)[np.argmin(np.abs(d))])
        u /= np.linalg.norm(u)
        w = np.cross(d, u)
        ring = [u * r, w * r, -u * r, -w * r]
        verts = [a + q for q in ring] + [b + q for q in ring]
        faces = []
        for k in range(4):
            k2 = (k + 1) % 4
            faces += [[k, k2, 4 + k2], [k, 4 + k2, 4 + k]]
        add(verts, faces, col(j))
    return np.concatenate(V), np.concatenate(VC), np.concatenate(F)


# --------------------------------------------------------------------------------------------- dense fusion
def voxel(P, C, v):
    q = np.floor(P / v).astype(np.int64)
    q -= q.min(0)
    m = q.max(0) + 1
    key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
    _, idx = np.unique(key, return_index=True)
    return P[idx], C[idx]


def fuse(L, frames, mask, vox, use_conf=True, every=2, target=150000, dcap=None):
    D, K, c2w, conf = L["D"], L["K"], L["c2w"], L["conf"]
    n, H, W = D.shape
    thr = float(np.percentile(conf[::10], 25)) if (conf is not None and use_conf) else None
    if dcap is None:
        sm = D[::10]
        sm = sm[np.isfinite(sm) & (sm > 0)]
        dcap = float(min(np.percentile(sm, 97), 6 * np.median(sm)))
    if vox is None:  # ~0.5 % of the median depth, 1-10 cm
        sm = D[::10]
        sm = sm[np.isfinite(sm) & (sm > 0)]
        vox = float(np.clip(0.005 * np.median(sm), 0.01, 0.10))
    s = max(1, int(round(np.sqrt(H * W / target))))
    vv, uu = np.mgrid[s // 2:H:s, s // 2:W:s]
    accP, accC = [], []
    P0, C0 = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    for k in range(0, n, every):
        d = D[k]
        ok = np.isfinite(d) & (d > 0) & (d < dcap)
        dd = np.where(ok, d, 0)
        jump = np.zeros_like(dd)
        jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(np.diff(dd, axis=1)))
        jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(np.diff(dd, axis=1)))
        jump[1:] = np.maximum(jump[1:], np.abs(np.diff(dd, axis=0)))
        jump[:-1] = np.maximum(jump[:-1], np.abs(np.diff(dd, axis=0)))
        ok &= jump < 0.08 * np.maximum(dd, 1e-6)
        if thr is not None:
            ok &= conf[k] >= thr
        if mask is not None:
            ok &= mask[k] == 0
        m = ok[vv, uu]
        z = d[vv, uu][m]
        u, v = uu[m].astype(np.float64), vv[m].astype(np.float64)
        Kk = K[k]
        X = np.stack([(u - Kk[0, 2]) / Kk[0, 0] * z, (v - Kk[1, 2]) / Kk[1, 1] * z, z], 1)
        accP.append((X @ c2w[k, :3, :3].T + c2w[k, :3, 3]).astype(np.float32))
        accC.append(frames(k)[vv[m], uu[m]])
        if len(accP) >= 10:
            P0, C0 = voxel(np.concatenate([P0] + accP), np.concatenate([C0] + accC), vox)
            accP, accC = [], []
    P0, C0 = voxel(np.concatenate([P0] + accP), np.concatenate([C0] + accC), vox)
    while len(P0) > CAP:
        vox *= 1.4
        P0, C0 = voxel(P0, C0, vox)
    return P0, C0, vox, dcap


def gt_label(kind, c, H, W, L):
    """GT cameras + depth, Sim3-aligned into the label frame (camera centres)."""
    if kind in ("A", "B"):
        z = np.load(RD.BENCH[kind] / "gt" / f"{c['clip']}.npz", allow_pickle=True)
        G, Kg, GD = z["c2w"].astype(np.float64), z["K"].astype(np.float64), z["depth"].astype(np.float32)
        GD = np.where(GD < 80, GD, 0)
    else:
        meta = np.load(RD.SP_DATA / c["scene"] / "scene_metadata.npz")
        idx = {f: i for i, f in enumerate(meta["images"])}
        gi = [idx[f] for f in c["frames"]]
        G, Kg = meta["trajectories"][gi].astype(np.float64), meta["intrinsics"][gi].astype(np.float64)
        GD = np.stack([cv2.imread(str(RD.SP_DATA / c["scene"] / "depth" / f"{f}.png"), cv2.IMREAD_UNCHANGED)
                       .astype(np.float32)[:, :W] * 1e-3 for f in c["frames"]])
        GD = np.where(GD < 20, GD, 0)
    if Kg.ndim == 2:
        Kg = np.repeat(Kg[None], len(G), 0)
    assert GD.shape[1:] == (H, W), (GD.shape, H, W)
    s, R, t = RD.umeyama(G[:, :3, 3], L["c2w"][:, :3, 3], True)
    return RD.transform(dict(c2w=G, K=Kg, D=GD, conf=None, info={}), s, R, t), float(s)


def scene_size(P):
    c = np.median(P, 0)
    return float(np.percentile(np.linalg.norm(P - c, axis=1), 90))


def mb(p):
    return p.stat().st_size / 1e6


# --------------------------------------------------------------------------------------------- README
def load_desc():
    src = (ROOT / "code/viz/make_site.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DESC":
            d = ast.literal_eval(node.value)
            return {k: v.replace("&gt;", ">").replace("&lt;", "<") for k, v in d.items()}
    return {}


def numbers(m):
    out = []
    x = m.get("metrics")
    if isinstance(x, dict):
        f5 = "n/a（稀疏 GT）" if x["f5"] != x["f5"] else f"{x['f5']:.2f}"
        out.append(f"ATE（Sim3，占轨迹长度）{x['ate_pct']:.2f}%；AbsRel {x['absrel']:.3f}；F@5cm {f5}；Sim3 scale {x['scale']:.2f}")
    elif isinstance(x, list):
        for lab, y in zip(("ViPE", "v1.3"), x):
            out.append(f"{lab}：ATE {y['ate_pct']:.2f}%；AbsRel {y['absrel']:.3f}；F@5cm {y['f5']:.2f}")
    if "qc" in m:
        q = m["qc"]
        fl = f"（flags: {', '.join(q['flags'])}）" if q["flags"] else ""
        out.append(f"QC v1c {q['v1c']:.2f} → {'通过' if q['accept'] else '拒绝'}{fl}（>1 拒绝）")
    li = m.get("label_info")
    if li:
        out.append(f"HFOV {li['hfov']:.0f}°；尺度 k={li['k']:.3f}（k_depth={li['k_d']:.3f}，k_traj={li['k_t']:.3f}）")
    for li in m.get("labels", []):
        extra = f"，k={li['k']:.3f}" if "k" in li else ""
        out.append(f"{li['which']}：HFOV {li['hfov']:.0f}°{extra}")
    return out


# --------------------------------------------------------------------------------------------- per video
def export(spec):
    t0 = time.time()
    vid, kind, clip = spec["id"], spec["kind"], spec["clip"]
    meta = json.loads((VID / f"{vid}.json").read_text())
    out = EXP / vid
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VID / f"{vid}.mp4", out / "video.mp4")
    shutil.copy2(VID / f"{vid}.jpg", out / "poster.jpg")
    c = RD.clip_meta(kind, clip)
    H, W = RD.grid(kind, c)
    n = c["n_frames"]
    frames = RD.Frames(kind, clip, H, W)
    mask = RD.vipe_static_mask(RD.BENCH[kind], clip, n, (H, W))
    vox0 = None  # set from the label's median depth below
    files = []  # (name, description)
    if "compare" not in spec:
        L = RD.load_label(kind, c, H, W, "v1.3")
        P, C, vox, dcap = fuse(L, frames, mask, vox0)
        write_points(out / "pointcloud.ply", P, C)
        sz = 0.04 * scene_size(P)
        write_mesh(out / "cameras.ply", *camera_mesh(L["c2w"], L["K"], W, H, sz))
        files += [("pointcloud.ply", f"v1.3 融合彩色点云，{len(P) / 1e6:.2f}M 点，体素 {vox * 100:.1f} cm"),
                  ("cameras.ply", "v1.3 相机轨迹（三角网格）：每 10 帧一个 frustum，颜色 = 时间（紫 → 黄），细管连接相机中心")]
        if kind in ("A", "B", "scannetpp"):
            Lg, sg = gt_label(kind, c, H, W, L)
            Pg, Cg, voxg, _ = fuse(Lg, frames, mask, vox0, use_conf=False)
            write_points(out / "gt_pointcloud.ply", Pg, Cg)
            write_mesh(out / "gt_cameras.ply", *camera_mesh(Lg["c2w"], Lg["K"], W, H, sz, color=(30, 30, 30)))
            files += [("gt_pointcloud.ply", f"GT 深度 + GT 位姿/K 融合的点云，已 Sim3 对齐到 v1.3 坐标系（GT→标注尺度 s={sg:.3f}），"
                                             f"{len(Pg) / 1e6:.2f}M 点" + ("；KITTI GT 为稀疏 LiDAR，点较少" if "kitti" in clip else "")),
                      ("gt_cameras.ply", "GT 相机轨迹（黑色），同样 Sim3 对齐，可直接与 cameras.ply 叠加")]
        frame_txt = "v1.3 标注的世界坐标系（Pi3X 第一帧相机系，OpenCV 约定：x 右、y 下、z 前），单位为米（已乘 v1.3 尺度 k）"
    else:
        Ls = [RD.load_label(kind, c, H, W, w) for w in spec["compare"]]
        if spec["frame"] == "gt":
            G = RD.gt_c2w(kind, c)
            Ls = [RD.transform(L, *RD.umeyama(L["c2w"][:, :3, 3], G[:, :3, 3], True)) for L in Ls]
            frame_txt = "GT（ScanNet++）世界坐标系，单位为米；两组标注各自用相机中心 Sim3 对齐到 GT"
        else:
            Ls[1] = RD.transform(Ls[1], *RD.umeyama(Ls[1]["c2w"][:, :3, 3], Ls[0]["c2w"][:, :3, 3], True))
            frame_txt = "左侧标注（v1.3，HFOV 33°）的世界坐标系，单位为米；右侧（K×0.5，HFOV 61°）用相机中心 Sim3 对齐到左侧"
        prefixes = {"vipe": "vipe", "v1.3": "v13" if spec["frame"] == "gt" else "fov33", "fsweep0.50": "fov61"}
        names = {"vipe": "ViPE 单独", "v1.3": "v1.3", "fsweep0.50": "v1.3 以 K×0.5 重解（focal sweep，HFOV 61°）"}
        sm = Ls[0]["D"][::10]
        sm = sm[np.isfinite(sm) & (sm > 0)]
        dcap = float(min(np.percentile(sm, 97), 6 * np.median(sm)))
        sz = None
        for w, L in zip(spec["compare"], Ls):
            p = prefixes[w]
            P, C, vox, _ = fuse(L, frames, mask, vox0, use_conf=False, dcap=dcap)
            write_points(out / f"{p}_pointcloud.ply", P, C)
            sz = sz or 0.04 * scene_size(P)
            write_mesh(out / f"{p}_cameras.ply", *camera_mesh(L["c2w"], L["K"], W, H, sz))
            files += [(f"{p}_pointcloud.ply", f"{names[w]} 融合点云（无 conf 过滤，与视频一致），{len(P) / 1e6:.2f}M 点，体素 {vox * 100:.1f} cm"),
                      (f"{p}_cameras.ply", f"{names[w]} 相机轨迹网格（颜色 = 时间）")]
        if spec["frame"] == "gt":
            G = RD.gt_c2w(kind, c)
            Lg, _ = gt_label(kind, c, H, W, dict(c2w=G))
            Pg, Cg, _, _ = fuse(Lg, frames, mask, vox0, use_conf=False)
            write_points(out / "gt_pointcloud.ply", Pg, Cg)
            write_mesh(out / "gt_cameras.ply", *camera_mesh(Lg["c2w"], Lg["K"], W, H, sz, color=(30, 30, 30)))
            files += [("gt_pointcloud.ply", f"GT 点云（参考），{len(Pg) / 1e6:.2f}M 点"), ("gt_cameras.ply", "GT 相机轨迹（黑色）")]
    desc = load_desc().get(vid, "")
    ver = "对比：" + " vs ".join(spec["compare"]) if "compare" in spec else \
        "v1.3（Pi3X 以门控后的 ViPE K 为条件，尺度 k = sqrt(ViPE 深度比 × ViPE 轨迹比)）"
    lines = [f"{spec['name']}", f"类别：{spec['cat']}", f"片段：{clip}（{n} 帧）", f"标注版本：{ver}"]
    if spec.get("failure"):
        lines.append(f"失败样例：{spec.get('reason', '')}")
    lines += ["", "说明：" + desc, "", "关键数字："] + ["  - " + x for x in numbers(meta)]
    lines += ["", f"坐标系与单位：{frame_txt}。", "", "文件："]
    lines += ["  - video.mp4：演示视频（RGB / 深度 / 俯视轨迹 / 点云，1280×720，15 fps）",
              "  - poster.jpg：视频封面帧"] + [f"  - {a}：{b}" for a, b in files]
    lines += ["", "点云过滤：去飞点（相邻像素深度跳变 > 8%）、去 ViPE 动态物体和天空 mask"
              + ("、去 Pi3X conf 低于该片段第 25 百分位的像素" if "compare" not in spec else "")
              + "；每 2 帧取 1 帧融合，体素降采样，最多 150 万点；室内不去天花板。",
              "查看：把同一文件夹内的 .ply 一起拖进 MeshLab 或 CloudCompare，即可叠加显示（已在同一坐标系）。"]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sizes = {p.name: round(mb(p), 1) for p in sorted(out.iterdir())}
    return vid, round(time.time() - t0, 1), round(sum(sizes.values()), 1), sizes


def safe(spec):
    try:
        return export(spec)
    except Exception:  # noqa: BLE001
        return spec["id"], "ERROR", traceback.format_exc()[-1500:], {}


if __name__ == "__main__":
    workers = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--workers=")), 4))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    specs = [s for s in RD.SPECS if only is None or s["id"] in only]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(min(workers, 5, len(specs))) as ex:
        for r in ex.map(safe, specs):
            print(*r, flush=True)
            res.append(r)
    (EXP / "sizes.json").write_text(json.dumps({r[0]: r[3] for r in res}, indent=1, ensure_ascii=False))
    print(f"TOTAL {time.time() - t0:.0f} s", flush=True)
