"""E10: MapAnything (Apache-2.0 weights, facebook/map-anything-apache) as a commercially usable annotator.

Usage: CUDA_VISIBLE_DEVICES=g cache/ma-venv/bin/python ma_cond.py WORKER VARIANT BENCH [BENCH ...]
VARIANT: img   images only
         kgate intrinsics = the gated K used for pi3x_kgate (E1a gate; read from BENCH/pi3x_kgate/<clip>/pred.npz)
Frames are resized (no crop) to ~255k px with both sides multiples of 14 (same budget as Pi3X) and passed with
resize_mode="fixed_size" so MapAnything's own pre-processing does not crop. Writes BENCH/ma_<VARIANT>/<clip>/pred.npz
(c2w OpenCV cam->world, K at depth res, depth = depth_z (metric), conf, sec, peak_gb) like the other benches.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

W_DIR = "/data/home/z00533370/cache/modelscope/map-anything-apache"
WID, VAR, BENCHES = sys.argv[1], sys.argv[2], [Path(p) for p in sys.argv[3:]]
t0 = time.time()
model = MapAnything.from_pretrained(W_DIR).to("cuda").eval()
load_sec = time.time() - t0


def target(W0, H0, limit=255000):
    s = math.sqrt(limit / (W0 * H0))
    k, m = round(W0 * s / 14), round(H0 * s / 14)
    while (k * 14) * (m * 14) > limit:
        if k / m > W0 / H0:
            k -= 1
        else:
            m -= 1
    return k * 14, m * 14


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


for B in BENCHES:
    OV = B / f"ma_{VAR}"
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if c.get("dataset") == "eth3d_mvs" or (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        torch.cuda.reset_peak_memory_stats()
        try:
            paths = [B / "frames" / name / f"{j:06d}.jpg" for j in range(n)]
            W0, H0 = Image.open(paths[0]).size
            w, h = target(W0, H0)
            Kc = None
            if VAR == "kgate":
                z = np.load(B / "pi3x_kgate" / name / "pred.npz")
                Kp, dh, dw = z["K"], z["depth"].shape[1], z["depth"].shape[2]
                Kc = Kp.copy()
                Kc[:, 0, :] *= w / dw
                Kc[:, 1, :] *= h / dh
            views = []
            for j, p in enumerate(paths):
                v = {"img": Image.open(p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)}
                if Kc is not None:
                    v["intrinsics"] = torch.from_numpy(Kc[j]).float()
                views.append(v)
            views = preprocess_inputs(views, resize_mode="fixed_size", size=(w, h))
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad():
                out = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                  apply_mask=False, mask_edges=False)
            torch.cuda.synchronize()
            sec = time.time() - t1
            c2w = np.stack([o["camera_poses"][0].float().cpu().numpy() for o in out]).astype(np.float64)
            K = np.stack([o["intrinsics"][0].float().cpu().numpy() for o in out]).astype(np.float64)
            depth = np.stack([o["depth_z"][0, ..., 0].float().cpu().numpy() for o in out]).astype(np.float32)
            conf = np.stack([o["conf"][0].float().cpu().numpy().reshape(depth.shape[1:]) for o in out]).astype(np.float32) \
                if "conf" in out[0] else np.ones_like(depth)
            np.savez(OV / name / "pred.npz", c2w=c2w, K=K, depth=depth, conf=conf, sec=sec, load_sec=load_sec,
                     peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            fov = lambda fx: round(float(np.degrees(2 * np.arctan(w / 2 / fx))), 2)  # noqa: E731
            rec = {"clip": name, "n_frames": n, "worker": WID, "variant": VAR, "ok": True, "sec": round(sec, 2),
                   "hw": [h, w], "hfov_out": fov(np.median(K[:, 0, 0])),
                   "hfov_in": fov(np.median(Kc[:, 0, 0])) if Kc is not None else None,
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "variant": VAR, "ok": False, "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", VAR, flush=True)
os._exit(0)
