"""Shared helpers for the 3D evaluation sets: unified clip writer, index files, reprojection check.

Clip layout (<root>/<dataset>/<clip>/):
  images/000000.jpg ...   JPEG q95, original resolution
  depth/000000.png ...    uint16, value / depth_scale = metres, 0 = invalid
  gt.npz                  c2w (N,4,4) float64 OpenCV cam->world, K (N,3,3) at image resolution
  meta.json
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np


def write_clip(root, dataset, clip, images, depths, c2w, K, meta, depth_scale, symlink_images=None):
    """images: list of BGR uint8 arrays (or None when symlink_images is given: list of source paths).
    depths: list of float32 metre arrays (0/nan/inf = invalid) or uint16 arrays already scaled."""
    d = Path(root) / dataset / clip
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "depth").mkdir(parents=True, exist_ok=True)
    n = len(c2w)
    for i in range(n):
        dst = d / "images" / f"{i:06d}.jpg"
        if symlink_images is not None:
            if dst.is_symlink() or dst.exists():
                dst.rename(d / "images" / f".old_{i:06d}.jpg")
            os.symlink(symlink_images[i], dst)
        else:
            cv2.imwrite(str(dst), images[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        z = depths[i]
        if z.dtype != np.uint16:
            z = np.where(np.isfinite(z) & (z > 0), z, 0) * depth_scale
            z = np.clip(np.round(z), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(d / "depth" / f"{i:06d}.png"), z)
    np.savez(d / "gt.npz", c2w=np.asarray(c2w, np.float64), K=np.asarray(K, np.float64))
    meta = dict(meta, dataset=dataset, clip=clip, n_frames=n, depth_scale=depth_scale)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return d


def write_clips_json(root, dataset):
    d = Path(root) / dataset
    clips = []
    for c in sorted(p for p in d.iterdir() if (p / "meta.json").exists()):
        m = json.loads((c / "meta.json").read_text())
        clips.append({"clip": c.name, "n_frames": m["n_frames"], "hw": m["hw"], "depth_type": m["depth_type"],
                      "metric": m["metric"], "dynamic": m["dynamic"]})
    (d / "clips.json").write_text(json.dumps(clips, indent=1))
    return clips


def merge_index(index_path, dataset, entry):
    """Read-merge-write so entries written by other workers are kept."""
    p = Path(index_path)
    idx = json.loads(p.read_text()) if p.exists() else {}
    idx[dataset] = entry
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=1, ensure_ascii=False))
    tmp.replace(p)


def load_clip(d):
    d = Path(d)
    m = json.loads((d / "meta.json").read_text())
    g = np.load(d / "gt.npz")
    return m, g["c2w"], g["K"]


def read_depth(d, i, scale):
    z = cv2.imread(str(Path(d) / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
    return z


def reproj_check(d, gaps=(1, 5), n_pairs=8, max_depth=None):
    """Project valid depth of frame i into frame i+gap with GT poses/K; compare to depth of frame i+gap."""
    m, c2w, K = load_clip(d)
    s = m["depth_scale"]
    n = len(c2w)
    out = {}
    for gap in gaps:
        rels = []
        for i in np.linspace(0, n - 1 - gap, n_pairs).astype(int):
            j = i + gap
            zi, zj = read_depth(d, i, s), read_depth(d, j, s)
            H, W = zi.shape
            v, u = np.nonzero(zi > 0)
            z = zi[v, u]
            if max_depth:
                keep = z < max_depth
                v, u, z = v[keep], u[keep], z[keep]
            X = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * z, (v - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)
            T = np.linalg.inv(c2w[j]) @ c2w[i]
            Xj = X @ T[:3, :3].T + T[:3, 3]
            zz = Xj[:, 2]
            ok = zz > 0.05
            uj = np.round(K[j, 0, 0] * Xj[ok, 0] / zz[ok] + K[j, 0, 2]).astype(int)
            vj = np.round(K[j, 1, 1] * Xj[ok, 1] / zz[ok] + K[j, 1, 2]).astype(int)
            inb = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
            ref = zj[vj[inb], uj[inb]]
            pr = zz[ok][inb]
            g = ref > 0
            if g.sum() > 50:
                rels.append(np.abs(pr[g] - ref[g]) / ref[g])
        r = np.concatenate(rels) if rels else np.array([np.nan])
        out[f"gap{gap}"] = {"median_relerr": float(np.median(r)), "frac_lt5pct": float(np.mean(r < 0.05)),
                            "n_px": int(r.size)}
    return out


def write_check(dataset_dir, results, notes=""):
    lines = ["# Reprojection check", "", "Depth of frame i projected into frame i+gap with GT pose and K, "
             "compared with GT depth of frame i+gap (only pixels valid in both).", ""]
    lines += ["| clip | gap | median rel. err | <5% px | px |", "| --- | --- | --- | --- | --- |"]
    for clip, res in results.items():
        for gap, v in res.items():
            lines.append(f"| {clip} | {gap} | {v['median_relerr']:.4f} | {v['frac_lt5pct']:.3f} | {v['n_px']} |")
    if notes:
        lines += ["", notes]
    (Path(dataset_dir) / "CHECK.md").write_text("\n".join(lines) + "\n")
