"""Render per-stage example images + compact JSON for the ViPE flow doc."""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402

VIDEO, PROF = sys.argv[1], Path(sys.argv[2])
D = PROF / "_profile"
OUT = PROF / "doc_assets"
OUT.mkdir(exist_ok=True)
W = 480  # output width per tile
rec = json.loads((D / "profile.json").read_text())


def save(name, img_rgb, q=82):
    h, w = img_rgb.shape[:2]
    img = cv2.resize(img_rgb, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def colorize(depth, valid=None, lo=None, hi=None, cmap="turbo_r"):
    v = valid if valid is not None else np.isfinite(depth) & (depth > 0)
    lo = np.percentile(depth[v], 2) if lo is None else lo
    hi = np.percentile(depth[v], 98) if hi is None else hi
    x = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
    rgb[~v] = 0
    return rgb, float(lo), float(hi)


frames = read_frames(VIDEO)
H, Wd = frames[0].shape[:2]
meta = {"n_frames_decoded": len(frames), "hw": [H, Wd]}

# 0. raw frames
for i in (0, 60, 121):
    save(f"rgb_{i:04d}.jpg", frames[i])

# 1. GeoCalib sample frames
for i in rec["geocalib"]["sample_frame_inds"]:
    save(f"geocalib_{i:04d}.jpg", frames[i])

# 2a/2b. GroundingDINO boxes (keyframes at multiples of sam_run_gap)
gap = int(30000 / 1001 * 2)  # fps 29.97 * kf_gap_sec 2
det_frames = [0, gap, 2 * gap]
meta["detect_frames"] = det_frames
colors = {"animal": (255, 170, 0), "pet": (0, 200, 255)}
for k, call in zip(det_frames, rec["gdino"]):
    img = frames[k].copy()
    for j, ((x0, y0, x1, y1), ph) in enumerate(zip(call["boxes_xyxy"], call["phrases"])):
        c = colors.get(ph, (255, 60, 60))
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 4)
        org = (int(x0) + 8, int(y0) + 44 + j * 46)
        cv2.putText(img, f"#{j+1} {ph}", org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 9)
        cv2.putText(img, f"#{j+1} {ph}", org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 3)
    save(f"gdino_{k:04d}.jpg", img)

# 2c. instance overlay + SLAM mask
palette = {2: np.array([255, 170, 0]), 3: np.array([0, 200, 255])}
for i in (0, 60, 121):
    z = np.load(D / f"ta_{i:04d}.npz")
    rgb, inst, mask = z["rgb"], z["instance"], z["mask"]
    over = rgb.astype(np.float32)
    for iid in np.unique(inst):
        if iid == 0:
            continue
        col = palette.get(int(iid), np.array([255, 60, 60]))
        m = inst == iid
        over[m] = over[m] * 0.4 + col * 0.6
    save(f"instance_{i:04d}.jpg", over.astype(np.uint8))
    mk = rgb.copy()
    mk[~mask] = (mk[~mask] * 0.15 + np.array([200, 30, 30]) * 0.85).astype(np.uint8)
    save(f"slammask_{i:04d}.jpg", mk)
    meta[f"mask_invalid_frac_{i}"] = float((~mask).mean())
    meta[f"instance_ids_{i}"] = [int(x) for x in np.unique(inst)]

# 3. SLAM: trajectory + map (compact for SVG)
s = np.load(D / "slam.npz")
c2w, xyz, rgbp, kf = s["c2w"], s["xyz"], s["rgb"], s["kf"]
pos = c2w[:, :3, 3]
meta["traj_len_m"] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
meta["traj_extent_m"] = (pos.max(0) - pos.min(0)).round(3).tolist()
meta["traj_xyz"] = pos.round(4).tolist()
meta["traj_fwd"] = c2w[:, :3, 2].round(4).tolist()
sel = np.random.default_rng(1).choice(len(xyz), size=min(3000, len(xyz)), replace=False)
meta["map_xyz"] = xyz[sel].round(3).tolist()
meta["map_rgb"] = (rgbp[sel] * 255).astype(int).tolist()
meta["n_map_points"] = int(s["n_points"])
meta["keyframes"] = kf.tolist()
meta["motion"] = [round(float(x), 3) for x in s["motion"]]
meta["slam_hw"] = s["slam_hw"].tolist()
meta["pose0"] = c2w[0].round(4).tolist()
meta["pose60"] = c2w[60].round(4).tolist()

# 4. depth
pz = np.load(D / "priorda_0060.npz")
prompt, pout = pz["prompt_metric_depth"], pz["out_metric_depth"]
lo, hi = np.percentile(pout, 2), np.percentile(pout, 98)
dil = cv2.dilate((prompt > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
dprompt = cv2.dilate(prompt, np.ones((7, 7), np.uint8))
img, _, _ = colorize(dprompt, dil, lo, hi)
base = (frames[60] * 0.35).astype(np.uint8)
base[dil] = img[dil]
save("depth_prompt_0060.jpg", base)
meta["prompt_density"] = float((prompt > 0).mean())
img, _, _ = colorize(pout, None, lo, hi)
save("depth_priorda_0060.jpg", img)
vda = np.load(D / "vda_0060.npy")
img, vlo, vhi = colorize(vda, None, None, None, cmap="turbo")
save("depth_vda_0060.jpg", img)
meta["vda_range"] = [vlo, vhi]

sys.path.insert(0, "/data/home/z00533370/code/vipe")
from vipe.utils.io import read_depth_artifacts  # noqa: E402

name = Path(VIDEO).stem
final = None
for idx, d in read_depth_artifacts(PROF / "depth" / f"{name}.zip"):
    if idx == 60:
        final = d.numpy()
        break
img, _, _ = colorize(final, None, lo, hi)
save("depth_final_0060.jpg", img)
meta["depth_final_stats"] = {
    "min": float(np.nanmin(final)),
    "p50": float(np.nanmedian(final)),
    "max": float(np.nanmax(final)),
    "dtype": str(final.dtype),
    "shape": list(final.shape),
}
meta["depth_colormap_range_m"] = [float(lo), float(hi)]
meta["priorda_stats"] = {"min": float(pout.min()), "p50": float(np.median(pout)), "max": float(pout.max())}

# Output files listing
meta["files"] = {str(p.relative_to(PROF)): p.stat().st_size for p in sorted(PROF.rglob("*")) if p.is_file() and "_profile" not in p.parts and "doc_assets" not in p.parts}
(OUT / "meta.json").write_text(json.dumps(meta))
print("ok", len(list(OUT.iterdir())))
