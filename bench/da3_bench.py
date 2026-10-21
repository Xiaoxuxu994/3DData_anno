"""Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1) on benchmark clips, one inference call per clip.

Usage: python da3_bench.py WORKER_TAG BENCH_DIR [BENCH_DIR ...]
Workers claim clips with an O_EXCL .claim file (same scheme as vggt_omega_bench.py).
Writes BENCH_DIR/da3/<clip>/pred.npz: c2w (N,4,4, OpenCV cam->world), K (N,3,3 at depth res),
depth (N,h,w float32, metric from the nested model), conf (N,h,w), sec, load_sec, peak_gb.

Official defaults: process_res=504, process_res_method="upper_bound_resize" (longest side -> 504,
then resized to a patch multiple; no crop), ref_view_strategy="saddle_balanced".
DA3 extrinsics are OpenCV world->camera (3x4); inverted to cam->world here.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

MODEL = os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
WID, BENCHES = sys.argv[1], [Path(p) for p in sys.argv[2:]]

t0 = time.time()
model = DepthAnything3.from_pretrained(MODEL).to(device=torch.device("cuda"))
model.eval()
load_sec = time.time() - t0


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


for B in BENCHES:
    OV = B / "da3"
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        names = [str(B / "frames" / name / f"{j:06d}.jpg") for j in range(n)]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        try:
            pred = model.inference(names)
            torch.cuda.synchronize()
            sec = time.time() - t1
            w2c = np.asarray(pred.extrinsics, dtype=np.float64)
            T = np.tile(np.eye(4), (n, 1, 1))
            T[:, :3, :4] = w2c[:, :3, :4]
            c2w = np.linalg.inv(T)
            depth = np.asarray(pred.depth, dtype=np.float32)
            np.savez(OV / name / "pred.npz", c2w=c2w, K=np.asarray(pred.intrinsics, dtype=np.float64), depth=depth,
                     conf=np.asarray(pred.conf, dtype=np.float32), sec=sec, load_sec=load_sec,
                     peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": True, "sec": round(sec, 2), "hw": list(depth.shape[1:]),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": False, "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", flush=True)
os._exit(0)
