"""E1: Pi3X with geometric conditioning (intrinsics, optionally poses) from an external camera source.

Usage: CUDA_VISIBLE_DEVICES=g code/Pi3/.venv/bin/python pi3x_cond.py WORKER VARIANT BENCH [BENCH ...]
VARIANT : kgt    GT intrinsics (oracle upper bound; A/B/scannetpp only)
          kvipe  ViPE intrinsics
          kgeo   GeoCalib intrinsics (outputs/evolution/E1/geocalib, run geocalib_bench.py first)
          pkvipe ViPE poses + ViPE intrinsics
          kgate  ViPE intrinsics unless ViPE's HFOV is implausible (outside [25, 130] deg or invalid) -> median HFOV
                 of GeoCalib / Pi3X / DA3 (E1 k_select.py "gate" rule). Clips that keep ViPE K are symlinked to
                 pi3x_kvipe (identical input), only switched clips are re-run.
Writes BENCH/pi3x_<VARIANT>/<clip>/pred.npz with the same keys as bench/pi3x_bench.py plus
  K_cond (conditioning intrinsics at depth res; also stored as K, i.e. the label we would ship) and
  K_rec  (intrinsics recovered from the predicted rays, as in the unconditioned bench -> does Pi3X obey K?).
Preprocessing identical to the unconditioned run (load_images_as_tensor, <= 255k px, multiples of 14).
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

ROOT = Path("/data/home/z00533370")
GEO = ROOT / "outputs/evolution/E1/geocalib"
SPP = ROOT / "data/scannetpp_sample30"
WID, VAR, BENCHES = sys.argv[1], sys.argv[2], [Path(p) for p in sys.argv[3:]]
dev, dtype = torch.device("cuda"), torch.bfloat16

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


def vipe_cams(B, clip):
    pz, iz = np.load(B / "vipe" / "pose" / f"{clip}.npz"), np.load(B / "vipe" / "intrinsics" / f"{clip}.npz")
    c2w = pz["data"][np.argsort(pz["inds"])].astype(np.float64)
    f = iz["data"][np.argsort(iz["inds"])]
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], 1
    return c2w, K


GATE = {}
if VAR == "kgate":
    import csv
    for r in csv.DictReader(open(ROOT / "outputs/evolution/E1/fov_sources.csv")):
        num = lambda k: float(r[k]) if r.get(k) not in (None, "") else np.nan  # noqa: E731
        o = np.nanmedian([num("hfov_geo"), num("hfov_pi3x"), num("hfov_da3")])
        bad = not (25 <= num("hfov_vipe") <= 130)
        GATE[(r["bench"], r["clip"])] = (bad, o)


def source_K(B, c, W0, H0):
    """(N,3,3) intrinsics at the stored frame resolution (corner convention: centred cx = W/2), poses or None."""
    clip, n = c["clip"], c["n_frames"]
    if VAR in ("kvipe", "pkvipe"):
        c2w, K = vipe_cams(B, clip)
        return K, (c2w if VAR == "pkvipe" else None)
    if VAR == "kgeo":
        g = json.loads((GEO / B.name / f"{clip}.json").read_text())
        K = np.array([[g["f_px"], 0, W0 / 2], [0, g["f_px"], H0 / 2], [0, 0, 1]], np.float64)
        return np.repeat(K[None], n, 0), None
    if VAR == "kgate":
        fx = W0 / 2 / np.tan(np.radians(GATE[(B.name, clip)][1]) / 2)
        K = np.array([[fx, 0, W0 / 2], [0, fx, H0 / 2], [0, 0, 1]], np.float64)
        return np.repeat(K[None], n, 0), None
    if VAR == "kgt":
        if B.name == "scannetpp":
            meta = np.load(SPP / c["scene"] / "scene_metadata.npz")
            idx = {f: i for i, f in enumerate(meta["images"])}
            return meta["intrinsics"][[idx[f] for f in c["frames"]]].astype(np.float64), None
        z = np.load(B / "gt" / f"{clip}.npz", allow_pickle=True)
        return z["K"].astype(np.float64), None
    raise ValueError(VAR)


for B in BENCHES:
    OV = B / f"pi3x_{VAR}"
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if B.name == "eval3d_A" and c.get("dataset") == "eth3d_mvs":
            continue
        if VAR == "kgt" and B.name == "wild":
            continue
        if (OV / name / "pred.npz").exists():
            continue
        if VAR == "kgate" and not GATE.get((B.name, name), (True, 0))[0]:
            src = B / "pi3x_kvipe" / name / "pred.npz"
            if src.exists():
                (OV / name).mkdir(parents=True, exist_ok=True)
                (OV / name / "pred.npz").symlink_to(src)
                continue
        if not claim(OV / name):
            continue
        torch.cuda.reset_peak_memory_stats()
        try:
            imgs = load_images_as_tensor(str(B / "frames" / name), interval=1, verbose=False).to(dev)
            assert imgs.shape[0] == n, f"loaded {imgs.shape[0]} frames, expected {n}"
            from PIL import Image
            W0, H0 = Image.open(B / "frames" / name / "000000.jpg").size
            h, w = imgs.shape[-2:]
            K0, poses = source_K(B, c, W0, H0)
            Kc = K0.copy()
            Kc[:, 0, :] *= w / W0
            Kc[:, 1, :] *= h / H0
            kw = {"intrinsics": torch.from_numpy(Kc).float().to(dev)[None]}
            if poses is not None:
                kw["poses"] = torch.from_numpy(poses).float().to(dev)[None]
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None], **kw)
            lp = res["local_points"][0].float()
            K_rec = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(lp, dim=-1), force_center_principal_point=True)
            torch.cuda.synchronize()
            sec = time.time() - t1
            depth = lp[..., 2].cpu().numpy().astype(np.float32)
            np.savez(OV / name / "pred.npz", c2w=res["camera_poses"][0].float().cpu().numpy().astype(np.float64),
                     K=Kc, K_cond=Kc, K_rec=K_rec.float().cpu().numpy().astype(np.float64), depth=depth,
                     conf=torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy().astype(np.float32),
                     metric=float(res["metric"][0]), sec=sec, load_sec=load_sec,
                     peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            fov = lambda fx: float(np.degrees(2 * np.arctan(w / 2 / fx)))  # noqa: E731
            rec = {"clip": name, "n_frames": n, "worker": WID, "variant": VAR, "ok": True, "sec": round(sec, 2),
                   "hw": [h, w], "metric": round(float(res["metric"][0]), 4),
                   "hfov_cond": round(fov(np.median(Kc[:, 0, 0])), 2),
                   "hfov_rec": round(fov(float(K_rec[:, 0, 0].median())), 2),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "variant": VAR, "ok": False, "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", VAR, flush=True)
os._exit(0)
