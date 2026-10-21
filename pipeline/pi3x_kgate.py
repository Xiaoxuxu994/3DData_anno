"""v1.3 annotator: K gate + Pi3X conditioned on the gated intrinsics (poses + depth from one forward pass).

Usage: CUDA_VISIBLE_DEVICES=g code/Pi3/.venv/bin/python pi3x_kgate.py BENCH
Needs, per clip: BENCH/vipe/{pose,intrinsics} (ViPE), BENCH/geocalib/<clip>.json (geocalib_k.py),
BENCH/pi3x/<clip>/pred.npz (image-only Pi3X, code/bench/pi3x_bench.py), BENCH/da3/<clip>/pred.npz (image-only DA3,
code/bench/da3_bench.py).

K gate (E1a, code/evolution/k_select.py "gate" + fov_sources.py):
  hfov_vipe = HFOV of the median ViPE focal at the frame width; others = median(HFOV GeoCalib, Pi3X, DA3)
  ViPE HFOV inside [25, 130] deg -> condition on ViPE's per-frame intrinsics (identical to pi3x_cond.py "kvipe")
  else                           -> constant K with HFOV = others, principal point at the image centre ("kgate")
  kdis = |hfov_vipe - others| is the QC K-disagreement signal.
  -> BENCH/gate/<clip>.json
Pi3X forward: the code of code/evolution/pi3x_cond.py (same preprocessing, bf16 autocast, K rescaled to the network
resolution, K_rec from the predicted rays) -> BENCH/pi3x_kgate/<clip>/pred.npz with the same keys
(c2w, K = K_cond, K_rec, depth, conf, metric, sec, load_sec, peak_gb). Depth / translation are in Pi3X's own scale;
the metric scale k is applied later (v13_label_qc.py).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import recover_intrinsic_from_rays_d

torch.set_num_threads(4)
B = Path(sys.argv[1])
LO, HI = 25.0, 130.0
dev, dtype = torch.device("cuda"), torch.bfloat16


def hfov(fx, W):
    return float(np.degrees(2 * np.arctan(W / 2 / fx)))


def vipe_cams(clip):
    pz, iz = np.load(B / "vipe" / "pose" / f"{clip}.npz"), np.load(B / "vipe" / "intrinsics" / f"{clip}.npz")
    c2w = pz["data"][np.argsort(pz["inds"])].astype(np.float64)
    f = iz["data"][np.argsort(iz["inds"])]
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], 1
    return c2w, K


def gate(clip):
    g = json.loads((B / "geocalib" / f"{clip}.json").read_text())
    W = g["W"]
    r = {"clip": clip, "W": W, "hfov_geo": g["hfov"]}
    iz = np.load(B / "vipe" / "intrinsics" / f"{clip}.npz")["data"]
    r["hfov_vipe"] = hfov(float(np.median(iz[:, 0])), W)
    for m in ("pi3x", "da3"):
        p = B / m / clip / "pred.npz"
        r[f"hfov_{m}"] = np.nan
        if p.exists():
            z = np.load(p)
            r[f"hfov_{m}"] = hfov(float(np.median(z["K"][:, 0, 0])), z["depth"].shape[2])
    others = float(np.nanmedian([r["hfov_geo"], r["hfov_pi3x"], r["hfov_da3"]]))
    fired = not (LO <= r["hfov_vipe"] <= HI)  # NaN / negative focal -> fired
    r.update(others_med=others, kdis=abs(r["hfov_vipe"] - others), fired=bool(fired),
             hfov_label=others if fired else r["hfov_vipe"], k_source="median(GeoCalib,Pi3X,DA3)" if fired else "ViPE")
    (B / "gate").mkdir(exist_ok=True)
    (B / "gate" / f"{clip}.json").write_text(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                                         for k, v in r.items()}, indent=1))
    return r


t0 = time.time()
model = Pi3X.from_pretrained("yyfz233/Pi3X").to(dev).eval()
load_sec = time.time() - t0
for c in json.loads((B / "clips.json").read_text()):
    name, n = c["clip"], c["n_frames"]
    OV = B / "pi3x_kgate" / name
    if (OV / "pred.npz").exists():
        continue
    OV.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        gr = gate(name)
        imgs = load_images_as_tensor(str(B / "frames" / name), interval=1, verbose=False).to(dev)
        assert imgs.shape[0] == n, f"loaded {imgs.shape[0]} frames, expected {n}"
        W0, H0 = Image.open(B / "frames" / name / "000000.jpg").size
        h, w = imgs.shape[-2:]
        if gr["fired"]:
            fx = W0 / 2 / np.tan(np.radians(gr["others_med"]) / 2)
            K0 = np.repeat(np.array([[fx, 0, W0 / 2], [0, fx, H0 / 2], [0, 0, 1]], np.float64)[None], n, 0)
        else:
            K0 = vipe_cams(name)[1]
        Kc = K0.copy()
        Kc[:, 0, :] *= w / W0
        Kc[:, 1, :] *= h / H0
        kw = {"intrinsics": torch.from_numpy(Kc).float().to(dev)[None]}
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs[None], **kw)
        lp = res["local_points"][0].float()
        K_rec = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(lp, dim=-1), force_center_principal_point=True)
        torch.cuda.synchronize()
        sec = time.time() - t1
        depth = lp[..., 2].cpu().numpy().astype(np.float32)
        np.savez(OV / "pred.npz", c2w=res["camera_poses"][0].float().cpu().numpy().astype(np.float64),
                 K=Kc, K_cond=Kc, K_rec=K_rec.float().cpu().numpy().astype(np.float64), depth=depth,
                 conf=torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy().astype(np.float32),
                 metric=float(res["metric"][0]), sec=sec, load_sec=load_sec,
                 peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        rec = {"clip": name, "n_frames": n, "ok": True, "gate_fired": gr["fired"], "sec": round(sec, 2), "hw": [h, w],
               "hfov_cond": round(hfov(float(np.median(Kc[:, 0, 0])), w), 2),
               "hfov_rec": round(hfov(float(K_rec[:, 0, 0].median()), w), 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    except Exception as e:  # noqa: BLE001
        rec = {"clip": name, "n_frames": n, "ok": False, "error": repr(e)[:300]}
        torch.cuda.empty_cache()
    with open(B / "pi3x_kgate" / "status.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("ALL_DONE pi3x_kgate", flush=True)
os._exit(0)
