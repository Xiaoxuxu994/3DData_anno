"""Assemble eval3d A/B groups into bench layout + GT caches (idempotent; re-run picks up new clips).

  python assemble_eval3d.py
-> outputs/bench/eval3d_{A,B}/frames/<dataset>__<clip>/000000.jpg ...
   outputs/bench/eval3d_{A,B}/gt/<dataset>__<clip>.npz   (c2w, K, depth[m, float16, 0=invalid], meta json)
   outputs/bench/eval3d_{A,B}/clips.json

Frame policy (all methods see exactly these frames):
- crop to even width/height from the right/bottom (ViPE's libx264 writer needs even dims); K unchanged
- downscale when the long side > 1280 (ETH3D DSLR 6221x4146 -> 1024 wide); K scaled per axis
- 7-Scenes depth is not registered to RGB; both cameras share the principal point, so the depth map is
  resampled from K_depth (f=585) to the RGB K (f=525) by a pure intrinsic warp (common practice).
B group skips scannetpp (already benchmarked under outputs/bench/scannetpp).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

E = Path("/data/home/z00533370/data/eval3d")
OUT = Path("/data/home/z00533370/outputs/bench")
GROUPS = {"A_rank": "eval3d_A", "B_diag": "eval3d_B"}
SKIP = {"scannetpp"}
MAX_SIDE = 1280


def target_size(h, w):
    s = min(1.0, MAX_SIDE / max(h, w))
    th, tw = int(round(h * s)), int(round(w * s))
    return th - th % 2, tw - tw % 2, s


def adjust_K(K, h, w, th, tw, s):
    K = K.copy().astype(np.float64)
    if s < 1.0:
        sx, sy = tw / w, th / h  # exact per-axis factors of the resize
        K[:, 0, :] *= sx
        K[:, 1, :] *= sy
        K[:, 0, 2] += 0.5 * sx - 0.5
        K[:, 1, 2] += 0.5 * sy - 0.5
    return K  # pure crop from right/bottom keeps K


def load_depth(path, scale, K=None, K_depth=None):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
    if K is not None and K_depth is not None:  # 7-Scenes intrinsic warp depth-cam -> rgb-cam
        h, w = d.shape
        v, u = np.mgrid[0:h, 0:w].astype(np.float32)
        mx = K_depth[0, 2] + (u - K[0, 2]) * K_depth[0, 0] / K[0, 0]
        my = K_depth[1, 2] + (v - K[1, 2]) * K_depth[1, 1] / K[1, 1]
        d = cv2.remap(d, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_NEAREST, borderValue=0)
    return d


def do_clip(args):
    group, ds, clip_dir, bench = args
    name = f"{ds}__{clip_dir.name}"
    fdir, gfile = bench / "frames" / name, bench / "gt" / f"{name}.npz"
    meta = json.loads((clip_dir / "meta.json").read_text())
    imgs = sorted(p for p in (clip_dir / "images").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    n = len(imgs)
    rec = {"clip": name, "dataset": ds, "group": group, "n_frames": n, "dynamic": meta.get("dynamic"),
           "depth_type": meta.get("depth_type"), "metric": meta.get("metric"), "source": meta.get("source")}
    if (gfile.exists() and fdir.exists() and len(list(fdir.glob("*.jpg"))) == n
            and gfile.stat().st_mtime > (clip_dir / "gt.npz").stat().st_mtime):  # source GT newer -> rebuild
        z = np.load(gfile, allow_pickle=True)
        rec["hw"] = list(z["depth"].shape[1:])
        return rec
    fdir.mkdir(parents=True, exist_ok=True)
    gfile.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    th, tw, s = target_size(h, w)
    gt = np.load(clip_dir / "gt.npz")
    K = adjust_K(gt["K"], h, w, th, tw, s)
    depths = []
    dfiles = sorted((clip_dir / "depth").iterdir()) if (clip_dir / "depth").exists() else []
    for j, p in enumerate(imgs):
        im = cv2.imread(str(p))
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA) if s < 1.0 else im[:th, :tw]
        cv2.imwrite(str(fdir / f"{j:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if dfiles:
            d = load_depth(dfiles[j], meta["depth_scale"], gt["K"][j] if "K_depth" in gt.files else None,
                           gt["K_depth"][j] if "K_depth" in gt.files else None)
            d = cv2.resize(d, (tw, th), interpolation=cv2.INTER_NEAREST) if s < 1.0 else d[:th, :tw]
            d[~np.isfinite(d)] = 0
            depths.append(d.astype(np.float16))
    depth = np.stack(depths) if depths else np.zeros((n, th, tw), np.float16)
    np.savez_compressed(gfile, c2w=gt["c2w"].astype(np.float64), K=K, depth=depth,
                        meta=json.dumps({**rec, "orig_hw": [h, w], "scale": s}))
    rec["hw"] = [th, tw]
    return rec


def main():
    for group, bname in GROUPS.items():
        bench = OUT / bname
        idx = json.loads((E / group / "index.json").read_text())
        jobs = []
        for ds in idx:
            if ds in SKIP:
                continue
            for cd in sorted((E / group / ds).iterdir()):
                if cd.is_dir() and (cd / "gt.npz").exists() and (cd / "images").exists():
                    jobs.append((group, ds, cd, bench))
        with ThreadPoolExecutor(16) as ex:
            recs = list(ex.map(do_clip, jobs))
        (bench / "clips.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
        by = {}
        for r in recs:
            by.setdefault(r["dataset"], []).append(r["n_frames"])
        print(bname, {k: (len(v), sum(v)) for k, v in by.items()})


if __name__ == "__main__":
    main()
