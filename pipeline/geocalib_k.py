"""Single-image GeoCalib intrinsics per clip (one of the K-gate fallback sources).

Usage: CUDA_VISIBLE_DEVICES=g PYTHONPATH=code/vipe code/vipe/.venv/bin/python geocalib_k.py BENCH
Same computation as code/evolution/geocalib_bench.py (12 evenly spaced frames, ViPE's GeoCalib copy, "distorted"
weights, simple_radial; clip focal = median over frames); only the output location differs:
BENCH/geocalib/<clip>.json : {f_px, f_frames, vfov, hfov, f_cv, unc_rel, W, H, sec}
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vipe.priors.geocalib import GeoCalib

torch.set_num_threads(4)
B = Path(sys.argv[1])
od = B / "geocalib"
od.mkdir(parents=True, exist_ok=True)
model = GeoCalib(weights="distorted").cuda().eval()
for c in json.loads((B / "clips.json").read_text()):
    clip, n = c["clip"], c["n_frames"]
    f = od / f"{clip}.json"
    if f.exists():
        continue
    t0 = time.time()
    fs, us = [], []
    for k in np.linspace(0, n - 1, 12).round().astype(int):
        im = cv2.imread(str(B / "frames" / clip / f"{k:06d}.jpg"))[..., ::-1].copy()
        H, W = im.shape[:2]
        x = torch.from_numpy(im).permute(2, 0, 1).float().cuda() / 255
        with torch.inference_mode():
            r = model.calibrate(x, camera_model="simple_radial")
        fp = float(r["camera"].f.reshape(-1)[0])
        fs.append(fp)
        u = r.get("focal_uncertainty")
        us.append(float(u.reshape(-1)[0]) / fp if u is not None else np.nan)
    fm = float(np.median(fs))
    rec = {"clip": clip, "f_px": fm, "f_frames": fs, "W": W, "H": H,
           "hfov": float(np.degrees(2 * np.arctan(W / 2 / fm))), "vfov": float(np.degrees(2 * np.arctan(H / 2 / fm))),
           "f_cv": float(np.std(fs) / np.mean(fs)), "unc_rel": float(np.nanmedian(us)), "sec": time.time() - t0}
    f.write_text(json.dumps(rec))
    print(clip, round(rec["hfov"], 1), round(rec["sec"], 1), flush=True)
print("GEOCALIB_DONE", flush=True)
