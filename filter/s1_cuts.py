"""Stage cuts: TransNetV2 (PyTorch port, transnetv2-pytorch 1.0.5) + ffmpeg scene score on the 15 fps frames.

Usage: CUDA_VISIBLE_DEVICES=4 python s1_cuts.py WORKER_ID N_WORKERS [POOL_DIR]
Writes scores.json["cuts"]: tn_max (max per-frame transition prob), tn_cuts (frames with prob > 0.5, NMS 5 frames),
tn_max_many (max of the many-hot head), ff_max (max ffmpeg scene score), ff_cuts (frames with scene > 0.3).
"""

import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from transnetv2_pytorch import TransNetV2

from fcommon import StageLog, clips, frame_paths, has_stage, timer, write_scores

wid, nw = int(sys.argv[1]), int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else None
FFMPEG = os.environ.get("FFMPEG", "/data/home/z00533370/cache/bin/ffmpeg")
model = TransNetV2(device="cuda").eval()
log = StageLog("cuts", wid)


def ff_scene(d):
    r = subprocess.run([FFMPEG, "-hide_banner", "-framerate", "15", "-i", str(d / "frames/%06d.jpg"),
                        "-vf", "scale=480:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score", "-f", "null", "-"],
                       capture_output=True, text=True)
    return np.array([float(l.split("=")[1]) for l in r.stderr.splitlines() if "lavfi.scene_score=" in l] or [0.0])


for d in clips(pool, wid, nw):
    if has_stage(d, "cuts") or not (d / "meta.json").exists():
        continue
    t = timer()
    small = np.stack([cv2.resize(cv2.imread(str(p))[..., ::-1], (48, 27), interpolation=cv2.INTER_AREA) for p in frame_paths(d)])
    with torch.inference_mode():
        single, many = model.predict_frames(torch.from_numpy(small).cuda(), quiet=True)
    single, many = single.float().numpy(), many.float().numpy()
    peaks = [i for i in range(len(single)) if single[i] > 0.5 and single[i] == single[max(0, i - 5):i + 6].max()]
    ff = ff_scene(d)
    s = {"tn_max": single.max(), "tn_cuts": peaks, "tn_n": len(peaks), "tn_max_many": many.max(),
         "ff_max": ff.max(), "ff_cuts": np.nonzero(ff > 0.3)[0].tolist(), "ff_n": int((ff > 0.3).sum())}
    write_scores(d, "cuts", s)
    log(d, t(), **s)
    print(d.name, {k: (round(float(v), 3) if np.isscalar(v) else v) for k, v in s.items()}, f"{t():.1f}s", flush=True)
print("CUTS_DONE", wid, flush=True)
