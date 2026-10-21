"""E1: DA3 (DA3NESTED-GIANT-LARGE-1.1) conditioned on ViPE cameras.

DA3 only injects a camera token when extrinsics are given (model/da3.py: `if extrinsics is not None`); an
intrinsics-only call is silently ignored, so the only conditioned variant is poses + intrinsics.
Usage: CUDA_VISIBLE_DEVICES=g code/depth-anything-3/.venv/bin/python da3_cond.py WORKER BENCH [BENCH ...]
Writes BENCH/da3_pkvipe/<clip>/pred.npz (same keys as bench/da3_bench.py). With align_to_input_ext_scale=True
(default) DA3 returns depth/extrinsics in the scale and frame of the input (ViPE) cameras.
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
    OV = B / "da3_pkvipe"
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if B.name == "eval3d_A" and c.get("dataset") == "eth3d_mvs":
            continue
        if (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        names = [str(B / "frames" / name / f"{j:06d}.jpg") for j in range(n)]
        torch.cuda.reset_peak_memory_stats()
        try:
            pz, iz = np.load(B / "vipe" / "pose" / f"{name}.npz"), np.load(B / "vipe" / "intrinsics" / f"{name}.npz")
            c2w = pz["data"][np.argsort(pz["inds"])].astype(np.float64)
            f = iz["data"][np.argsort(iz["inds"])]
            K = np.zeros((n, 3, 3), np.float32)
            K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], 1
            w2c = np.linalg.inv(c2w).astype(np.float32)
            torch.cuda.synchronize()
            t1 = time.time()
            pred = model.inference(names, extrinsics=w2c, intrinsics=K)
            torch.cuda.synchronize()
            sec = time.time() - t1
            e = np.asarray(pred.extrinsics, dtype=np.float64)
            T = np.tile(np.eye(4), (n, 1, 1))
            T[:, :3, :4] = e[:, :3, :4]
            depth = np.asarray(pred.depth, dtype=np.float32)
            Kp = np.asarray(pred.intrinsics, dtype=np.float64)
            if Kp.shape[-2:] == (3, 3) and Kp[0, 0, 2] > depth.shape[2] * 0.75:  # returned at input resolution
                W0 = 2 * Kp[0, 0, 2]
                Kp = Kp.copy()
                Kp[:, :2, :] *= depth.shape[2] / W0
            np.savez(OV / name / "pred.npz", c2w=np.linalg.inv(T), K=Kp, depth=depth,
                     conf=np.asarray(pred.conf, dtype=np.float32), sec=sec, load_sec=load_sec,
                     peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": True, "sec": round(sec, 2), "hw": list(depth.shape[1:]),
                   "K00": float(Kp[0, 0, 0]), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": False, "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE da3_pkvipe", flush=True)
os._exit(0)
