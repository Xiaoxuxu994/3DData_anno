"""Pi3X (yyfz233/Pi3X) on benchmark clips, image-only (no conditioning), one forward pass per clip.

Usage: python pi3x_bench.py WORKER_TAG BENCH_DIR [BENCH_DIR ...]
Writes BENCH_DIR/pi3x/<clip>/pred.npz: c2w (N,4,4, OpenCV cam->world, as output by Pi3X),
K (N,3,3 at depth res), depth (N,h,w float32 = local_points z, Pi3X approx. metric scale),
conf (N,h,w sigmoid prob), metric (scalar scale factor the model applied), sec, load_sec, peak_gb.

Preprocessing = official load_images_as_tensor: uniform resize to <= 255,000 pixels with both sides
multiples of 14 (no crop). Intrinsics = official recover_intrinsic_from_rays_d on normalized
local_points with force_center_principal_point=True (as in example_mm.py).
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
from pi3.utils.geometry import recover_intrinsic_from_rays_d

WID, BENCHES = sys.argv[1], [Path(p) for p in sys.argv[2:]]
dev = torch.device("cuda")
dtype = torch.bfloat16

t0 = time.time()
model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
load_sec = time.time() - t0


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


for B in BENCHES:
    OV = B / "pi3x"
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        torch.cuda.reset_peak_memory_stats()
        try:
            imgs = load_images_as_tensor(str(B / "frames" / name), interval=1, verbose=False).to(dev)
            assert imgs.shape[0] == n, f"loaded {imgs.shape[0]} frames, expected {n}"
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None])
            lp = res["local_points"][0].float()
            K = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(lp, dim=-1), force_center_principal_point=True)
            torch.cuda.synchronize()
            sec = time.time() - t1
            depth = lp[..., 2].cpu().numpy().astype(np.float32)
            np.savez(OV / name / "pred.npz", c2w=res["camera_poses"][0].float().cpu().numpy().astype(np.float64),
                     K=K.float().cpu().numpy().astype(np.float64), depth=depth,
                     conf=torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy().astype(np.float32),
                     metric=float(res["metric"][0]), sec=sec, load_sec=load_sec,
                     peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": True, "sec": round(sec, 2), "hw": list(depth.shape[1:]),
                   "metric": round(float(res["metric"][0]), 4), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": False, "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", flush=True)
os._exit(0)
