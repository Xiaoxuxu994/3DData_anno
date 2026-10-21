"""Split each ScanNet++ scene into continuous segments and build ViPE frame dirs (symlinks).

Frames are ordered by DSC number (same as SceneGeneration's ScannetppDataset). A segment is
broken wherever consecutive GT poses jump by > MAX_T metres or > MAX_R degrees; segments
shorter than MIN_LEN frames are dropped.

Frames are written as JPEGs with the last column cropped (1035 -> 1034 px wide), because
ViPE's libx264 rgb export needs even dimensions. Dropping the right-most column keeps K unchanged.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

DATA = Path("/data/home/z00533370/data/scannetpp_sample30")
OUT = Path("/data/home/z00533370/outputs/vipe/scannetpp_sample30")
MAX_T, MAX_R, MIN_LEN = 0.5, 30.0, 48


def rot_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


segments = []
for scene_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    z = np.load(scene_dir / "scene_metadata.npz")
    names = list(z["images"])
    have = {p.stem for p in (scene_dir / "images").glob("*.jpg")} & {p.stem for p in (scene_dir / "depth").glob("*.png")}
    idx = sorted((i for i, n in enumerate(names) if n in have), key=lambda i: int(names[i][-5:]))
    T = z["trajectories"][idx]
    brk = [
        np.linalg.norm(T[k + 1, :3, 3] - T[k, :3, 3]) > MAX_T or rot_deg(T[k, :3, :3], T[k + 1, :3, :3]) > MAX_R
        for k in range(len(idx) - 1)
    ]
    cuts = [0] + [k + 1 for k, b in enumerate(brk) if b] + [len(idx)]
    sid = 0
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s < MIN_LEN:
            continue
        seg_name = f"{scene_dir.name}_s{sid:02d}"
        frames = [names[i] for i in idx[s:e]]
        fdir = OUT / "frames" / seg_name
        fdir.mkdir(parents=True, exist_ok=True)
        def write(jn, scene_dir=scene_dir, fdir=fdir):
            j, n = jn
            dst = fdir / f"{j:06d}.jpg"
            if dst.is_symlink():
                dst.unlink()
            if not dst.exists():
                img = cv2.imread(str(scene_dir / "images" / f"{n}.jpg"))
                w = img.shape[1] // 2 * 2
                cv2.imwrite(str(dst), img[:, :w], [cv2.IMWRITE_JPEG_QUALITY, 97])

        with ThreadPoolExecutor(32) as ex:
            list(ex.map(write, enumerate(frames)))
        segments.append({"segment": seg_name, "scene": scene_dir.name, "n_frames": len(frames), "frames": frames})
        sid += 1

(OUT / "segments.json").write_text(json.dumps(segments, indent=1))
n = sum(s["n_frames"] for s in segments)
print(f"{len(segments)} segments, {n} frames, scenes covered {len({s['scene'] for s in segments})}")
print("lengths:", sorted((s["n_frames"] for s in segments), reverse=True))
