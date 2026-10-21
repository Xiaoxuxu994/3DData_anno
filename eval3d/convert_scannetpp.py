"""ScanNet++ benchmark (diagnostic group): the 28 clips already used in the annotator comparison.

Images: symlinks to the existing 1034x690 frames (right-most column cropped). Depth: rendered laser-mesh
z-depth (mm), cropped the same way. Poses: scene_metadata.npz trajectories (cam->world OpenCV), K per frame.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval3d_common import merge_index, reproj_check, write_check, write_clip, write_clips_json  # noqa: E402

DATA = Path("/data/home/z00533370/data/scannetpp_sample30")
BENCH = Path("/data/home/z00533370/outputs/bench/scannetpp")
OUT = Path("/data/home/z00533370/data/eval3d/B_diag")
W = 1034

for c in json.loads((BENCH / "clips.json").read_text()):
    scene, frames, n = c["scene"], c["frames"], c["n_frames"]
    meta_npz = np.load(DATA / scene / "scene_metadata.npz")
    idx = {f: i for i, f in enumerate(meta_npz["images"])}
    gi = [idx[f] for f in frames]
    c2w, K = meta_npz["trajectories"][gi], meta_npz["intrinsics"][gi]
    srcs = [os.path.realpath(BENCH / "frames" / c["clip"] / f"{j:06d}.jpg") for j in range(n)]
    deps = [cv2.imread(str(DATA / scene / "depth" / f"{f}.png"), cv2.IMREAD_UNCHANGED)[:, :W].astype(np.uint16)
            for f in frames]
    meta = {
        "source": f"ScanNet++ v2 DSLR (undistorted) scene {scene}, segment {c['segment']} first {n} frames",
        "frame_ids": frames, "stride": 1, "fps_effective": None, "hw": [690, W], "depth_type": "dense",
        "metric": True, "dynamic": False, "camera_model": "pinhole (undistorted fisheye DSLR, HFOV ~109/124 deg)",
        "license": "ScanNet++ Terms of Use (non-commercial research)",
        "notes": "Right-most image column cropped (1035->1034); K unchanged. Depth rendered from the laser mesh (mm). "
                 "See SPLIT.md for the official split of these scenes.",
    }
    write_clip(OUT, "scannetpp", c["clip"], None, deps, c2w, K, meta, depth_scale=1000.0, symlink_images=srcs)

clips = write_clips_json(OUT, "scannetpp")
chk = {c: reproj_check(OUT / "scannetpp" / c, max_depth=20) for c in ["00dd871005_b", "2a1a3afad9_b"]}
write_check(OUT / "scannetpp", chk)
merge_index(OUT / "index.json", "scannetpp", {"group": "B_diag", "n_clips": len(clips),
                                              "n_frames": sum(c["n_frames"] for c in clips), "path": str(OUT / "scannetpp")})
print(len(clips), json.dumps(chk, indent=1))
