"""E12: MoGe-2 / MoGe-3 single-image metric geometry on every frame of the eval benches.

Usage: CUDA_VISIBLE_DEVICES=g cache/moge-venv/bin/python moge_run.py WORKER MODEL MODE BENCH [BENCH ...]
MODEL: m2l = Ruicheng/moge-2-vitl-normal (v2) | m3l = Ruicheng/moge-3-vitl (v3) | m3g = Ruicheng/moge-3-vitg (v3)
MODE : free  MoGe estimates its own FOV
       kgate fov_x = HFOV of the gated K used by v1.3 (BENCH/pi3x_kgate/<clip>/pred.npz "K" = K_cond, E1a gate)
Frames are resized (no crop) to the pi3x_kgate depth grid (h, w), so K/depth live on the same grid as the v1.3 label.
Official defaults otherwise: resolution_level 9, fp16, apply_mask (masked pixels -> NaN), v3 refine_steps 3.
Writes BENCH/moge_<MODEL>_<MODE>/<clip>/pred.npz with
  depth (N,h,w) float32 metric (NaN = MoGe mask), K (N,3,3) MoGe's own (free) / the conditioning K (kgate) at depth res,
  c2w = pi3x_kgate cameras copied unchanged (MoGe has no poses; only so the bench loaders work - pose metrics of this
  method are Pi3X's), mask_frac, sec (pure inference time of all frames), peak_gb, load_sec.
Claim-file pattern as pi3x_cond.py / ma_cond.py: several workers may share one (MODEL, MODE) queue.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOS = {"m2l": ("v2", "Ruicheng/moge-2-vitl-normal"), "m3l": ("v3", "Ruicheng/moge-3-vitl"),
         "m3g": ("v3", "Ruicheng/moge-3-vitg")}
WID, MODEL, MODE, BENCHES = sys.argv[1], sys.argv[2], sys.argv[3], [Path(p) for p in sys.argv[4:]]
BS = int(os.environ.get("MOGE_BS", "16"))
ver, repo = REPOS[MODEL]
if ver == "v2":
    from moge.model.v2 import MoGeModel
else:
    from moge.model.v3 import MoGeModel
t0 = time.time()
model = MoGeModel.from_pretrained(repo).to("cuda").eval()
load_sec = time.time() - t0
POOL = ThreadPoolExecutor(16)


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def load_img(p, w, h):
    im = Image.open(p).convert("RGB")
    if im.size != (w, h):
        im = im.resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(im)


hfov = lambda fx, w: float(np.degrees(2 * np.arctan(w / 2 / fx)))  # noqa: E731

for B in BENCHES:
    OV = B / f"moge_{MODEL}_{MODE}"
    clips = json.loads((B / "clips.json").read_text())
    for c in sorted(clips, key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if c.get("dataset") == "eth3d_mvs" or (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        torch.cuda.reset_peak_memory_stats()
        try:
            z = np.load(B / "pi3x_kgate" / name / "pred.npz")
            Kg, c2w = z["K"].astype(np.float64), z["c2w"].astype(np.float64)
            h, w = z["depth"].shape[1:]
            paths = [B / "frames" / name / f"{j:06d}.jpg" for j in range(n)]
            imgs = list(POOL.map(lambda p: load_img(p, w, h), paths))
            fov_in = np.array([hfov(Kg[j, 0, 0], w) for j in range(n)], np.float32)
            D = np.empty((n, h, w), np.float32)
            K = np.zeros((n, 3, 3))
            sec = 0.0
            for s in range(0, n, BS):
                x = torch.from_numpy(np.stack(imgs[s:s + BS])).cuda().permute(0, 3, 1, 2).float() / 255
                kw = {"fov_x": torch.from_numpy(fov_in[s:s + BS]).cuda()} if MODE == "kgate" else {}
                torch.cuda.synchronize()
                t1 = time.time()
                with torch.no_grad():
                    o = model.infer(x, use_fp16=True, **kw)
                torch.cuda.synchronize()
                sec += time.time() - t1
                d = o["depth"].float().cpu().numpy()
                D[s:s + BS] = np.where(np.isfinite(d) & (d > 0), d, np.nan)
                Kn = o["intrinsics"].float().cpu().numpy()
                K[s:s + BS] = Kn
            K[:, 0, :] *= w
            K[:, 1, :] *= h
            if MODE == "kgate":  # report MoGe's realised focal but ship the conditioning K (as pi3x_kgate does)
                K_out, K = K, Kg.copy()
            else:
                K_out = K
            mask_frac = float(np.isfinite(D).mean())
            peak = torch.cuda.max_memory_allocated() / 1e9
            np.savez(OV / name / "pred.npz", c2w=c2w, K=K, K_moge=K_out, depth=D, sec=sec, load_sec=load_sec,
                     peak_gb=peak, mask_frac=mask_frac)
            rec = {"clip": name, "n_frames": n, "worker": WID, "model": MODEL, "mode": MODE, "ok": True,
                   "sec": round(sec, 2), "hw": [h, w], "hfov_out": round(hfov(np.median(K_out[:, 0, 0]), w), 2),
                   "hfov_in": round(float(np.median(fov_in)), 2), "mask_frac": round(mask_frac, 3),
                   "peak_gb": round(peak, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "model": MODEL, "mode": MODE, "ok": False,
                   "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", MODEL, MODE, flush=True)
os._exit(0)
