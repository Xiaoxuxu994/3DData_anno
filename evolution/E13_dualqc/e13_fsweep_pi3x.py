"""E13 focal sweep: re-solve the v1.3 label (Pi3X conditioned on K) with the gated K scaled by s (fx, fy * s).
Gate-2 self-consistency is then computed per s by e13_fsweep_eval.py; if it is clearly best near one focal length
that is model-independent evidence of the true FOV. Controls with GT K: KITTI / vKITTI2 (forward driving, where the
feed-forward models under-estimate HFOV by 10-35 deg).

  CUDA_VISIBLE_DEVICES=7 code/Pi3/.venv/bin/python e13_fsweep_pi3x.py
  -> outputs/evolution/E13/fsweep/<clip>/s<s>/pred.npz  (same keys as pi3x_cond.py)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor

ROOT = Path("/data/home/z00533370")
OUT = ROOT / "outputs/evolution/E13/fsweep"
CLIPS = [("wild", "driving__youtube_cc__BTbkSEotmX0"), ("eval3d_A", "kitti__kitti_00"), ("eval3d_A", "kitti__kitti_05"),
         ("eval3d_B", "vkitti2__vkitti2_Scene01_clone"), ("eval3d_B", "vkitti2__vkitti2_Scene06_clone")]
S = [0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]
dev = torch.device("cuda")
model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
for bench, clip in CLIPS:
    B = ROOT / "outputs/bench" / bench
    base = np.load(B / "pi3x_kgate" / clip / "pred.npz")
    K0 = base["K_cond"].astype(np.float64)
    imgs = load_images_as_tensor(str(B / "frames" / clip), interval=1, verbose=False).to(dev)
    h, w = imgs.shape[-2:]
    for s in S:
        od = OUT / clip / f"s{s:.2f}"
        if (od / "pred.npz").exists():
            continue
        od.mkdir(parents=True, exist_ok=True)
        K = K0.copy()
        K[:, 0, 0] *= s
        K[:, 1, 1] *= s
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            res = model(imgs[None], intrinsics=torch.from_numpy(K).float().to(dev)[None])
        torch.cuda.synchronize()
        lp = res["local_points"][0].float()
        np.savez(od / "pred.npz", c2w=res["camera_poses"][0].float().cpu().numpy().astype(np.float64), K=K, K_cond=K,
                 depth=lp[..., 2].cpu().numpy().astype(np.float32), sec=time.time() - t0)
        hf = float(np.degrees(2 * np.arctan(w / 2 / np.median(K[:, 0, 0]))))
        print(json.dumps({"clip": clip, "s": s, "hfov": round(hf, 2), "sec": round(time.time() - t0, 1)}), flush=True)
print("ALL_DONE", flush=True)
os._exit(0)
