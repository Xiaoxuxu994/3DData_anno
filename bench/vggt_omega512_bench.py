"""VGGT-Omega (512 checkpoint) on benchmark clips, one forward pass per clip.

Usage: python vggt_omega_bench.py WORKER_TAG BENCH_DIR [BENCH_DIR ...]
Workers claim clips with an O_EXCL .claim file (same scheme as voyager_bench.py).
Writes BENCH_DIR/vggt_omega/<clip>/pred.npz: c2w (N,4,4, OpenCV cam->world), K (N,3,3 at depth res),
depth (N,h,w float32, model scale), conf (N,h,w), sec, load_sec, peak_gb.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

CKPT = "/data/home/z00533370/cache/modelscope/VGGT-Omega/vggt_omega_1b_512.pt"
RES = 512  # official default resolution of the 512 checkpoint
WID, BENCHES = sys.argv[1], [Path(p) for p in sys.argv[2:]]

t0 = time.time()
model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
load_sec = time.time() - t0
dtype = torch.bfloat16


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


for B in BENCHES:
    OV = B / "vggt_omega512"
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        names = [str(B / "frames" / name / f"{j:06d}.jpg") for j in range(n)]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        images = load_and_preprocess_images(names, image_resolution=RES).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            pred = model(images)
        w2c, K = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
        torch.cuda.synchronize()
        sec = time.time() - t1
        w2c = w2c[0].float().cpu().numpy()
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, :3, :4] = w2c
        c2w = np.linalg.inv(T)
        depth = pred["depth"][0, ..., 0].float().cpu().numpy()
        conf = pred["depth_conf"][0].float().cpu().numpy()
        np.savez(OV / name / "pred.npz", c2w=c2w, K=K[0].float().cpu().numpy(), depth=depth.astype(np.float32),
                 conf=conf.astype(np.float32), sec=sec, load_sec=load_sec,
                 peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        rec = {"clip": name, "n_frames": n, "worker": WID, "sec": round(sec, 2), "hw": list(depth.shape[1:]),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", flush=True)
os._exit(0)
