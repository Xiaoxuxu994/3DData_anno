"""PURE vs. base models on benchmark clips, one forward pass per clip, paired preprocessing.

Usage: python pure_bench.py METHOD WORKER_TAG BENCH_DIR [BENCH_DIR ...]

METHOD          venv          weights                               preprocessing
pi3             Pi3/.venv     yyfz233/Pi3 (official Pi3 class)      official load_images_as_tensor (<=255k px, x14, no crop)
pure_pi3        PURE/.venv    cyun9286/PURE pi3                     same as pi3
vggt_b          PURE/.venv    facebook/VGGT-1B in PURE VGGT adapter VGGT 'crop' mode: width 518, height x14, center-crop to <=518
pure_vggt       PURE/.venv    cyun9286/PURE vggt                    same as vggt_b
da3g            PURE/.venv    DA3-GIANT (== DA3-GIANT-1.1) in PURE DA3  longest side 504 (x14 rounding), no crop
pure_da3        PURE/.venv    cyun9286/PURE da3                     same as da3g

Base and PURE models of a family share the forward code (VGGT, DA3: PURE adapter) or at least the
input tensor and post-processing (Pi3: official model vs. PURE adapter with replaced head), so the
only difference inside a pair is the weights.
Writes BENCH_DIR/<METHOD>/<clip>/pred.npz with c2w (N,4,4 OpenCV cam->world), K (N,3,3 at depth res),
depth (N,h,w float32, model scale), sec, load_sec, peak_gb (same contract as the other *_bench.py).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

METHOD, WID, BENCHES = sys.argv[1], sys.argv[2], [Path(p) for p in sys.argv[3:]]
H = "/data/home/z00533370"
RELAY = f"{H}/cache/relay"  # weights relayed from HF (server egress is slow); see relay.sh
sys.path.insert(0, f"{H}/code/Pi3")  # pi3.utils (preprocessing, intrinsics recovery) for both Pi3 runs
from pi3.utils.basic import load_images_as_tensor  # noqa: E402
from pi3.utils.geometry import recover_intrinsic_from_rays_d  # noqa: E402

dev = torch.device("cuda")
dtype = torch.bfloat16


def load_model():
    if METHOD == "pi3":
        from pi3.models.pi3 import Pi3
        return Pi3.from_pretrained(f"{RELAY}/yyfz233/Pi3")
    from pure import PURE
    if METHOD.startswith("pure_"):
        return PURE.from_pretrained(f"{RELAY}/cyun9286/PURE", subfolder=METHOD[5:])
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    if METHOD == "vggt_b":
        from pure.hub import PUREConfig
        model = PURE(PUREConfig(backbone="vggt"))
        root = Path(snapshot_download("facebook/VGGT-1B", local_files_only=True))
        f = root / "model.safetensors"
        sd = load_file(str(f)) if f.exists() else torch.load(root / "model.pt", map_location="cpu")
    elif METHOD == "da3g":
        from pure.hub import PUREConfig
        model = PURE(PUREConfig(backbone="da3"))
        root = Path(f"{RELAY}/depth-anything/DA3-GIANT")  # byte-identical to DA3-GIANT-1.1 (same LFS sha256)
        sd = load_file(str(root / "model.safetensors"))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    else:
        raise ValueError(METHOD)
    own = model.backbone.state_dict()
    sd = {k: v for k, v in sd.items() if k in own}
    missing = [k for k in own if k not in sd]
    # PURE's recovered DualDPT has a LayerNorm in aux (ray) levels 1-3 that official DA3 lacks. The ray branch
    # is not used for depth/pose here (forward only reads depth + cam_dec), so keep their default init.
    allowed = [k for k in missing if METHOD == "da3g" and "output_conv2_aux" in k]
    missing = [k for k in missing if k not in allowed]
    assert not missing, f"{METHOD}: {len(missing)} adapter keys missing from base checkpoint, e.g. {missing[:5]}"
    model.backbone.load_state_dict(sd, strict=not allowed)
    if allowed:
        print(json.dumps({"default_init_keys": allowed}), flush=True)
    return model


def preprocess(frame_dir, n):
    if METHOD in ("pi3", "pure_pi3"):
        return load_images_as_tensor(str(frame_dir), interval=1, verbose=False)
    names = [frame_dir / f"{j:06d}.jpg" for j in range(n)]
    W0, H0 = Image.open(names[0]).size
    if METHOD in ("vggt_b", "pure_vggt"):  # official VGGT load_and_preprocess_images(mode="crop")
        w, h = 518, round(H0 * 518 / W0 / 14) * 14
        crop = max(h - 518, 0)
    else:  # DA3 upper_bound_resize to 504 then patch multiple
        s = 504 / max(W0, H0)
        w, h = max(14, round(W0 * s / 14) * 14), max(14, round(H0 * s / 14) * 14)
        crop = 0
    out = []
    for p in names:
        im = Image.open(p).convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
        a = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
        if crop:
            a = a[:, crop // 2: crop // 2 + 518]
        out.append(a)
    return torch.stack(out)


def claim(d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


t0 = time.time()
model = load_model().to(dev).eval()
load_sec = time.time() - t0
print(json.dumps({"method": METHOD, "worker": WID, "load_sec": round(load_sec, 1)}), flush=True)

for B in BENCHES:
    OV = B / METHOD
    for c in sorted(json.loads((B / "clips.json").read_text()), key=lambda c: -c["n_frames"]):
        name, n = c["clip"], c["n_frames"]
        if (OV / name / "pred.npz").exists() or not claim(OV / name):
            continue
        torch.cuda.reset_peak_memory_stats()
        try:
            imgs = preprocess(B / "frames" / name, n).to(dev)
            assert imgs.shape[0] == n, f"loaded {imgs.shape[0]} frames, expected {n}"
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                res = model(imgs[None])
            if METHOD in ("pi3", "pure_pi3"):
                lp = res["local_points"][0].float()
                K = recover_intrinsic_from_rays_d(F.normalize(lp, dim=-1), force_center_principal_point=True)
                depth = lp[..., 2]
            else:
                K = res["intrinsics"][0].float()
                depth = res["depth"][0, ..., 0].float()
            c2w = res["camera_poses"][0].float()
            torch.cuda.synchronize()
            sec = time.time() - t1
            depth = depth.cpu().numpy().astype(np.float32)
            np.savez(OV / name / "pred.npz", c2w=c2w.cpu().numpy().astype(np.float64),
                     K=K.cpu().numpy().astype(np.float64), depth=depth, sec=sec, load_sec=load_sec,
                     in_hw=np.array(imgs.shape[-2:]), peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": True, "sec": round(sec, 2),
                   "in_hw": list(imgs.shape[-2:]), "hw": list(depth.shape[1:]),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"clip": name, "n_frames": n, "worker": WID, "ok": False, "error": repr(e)[:300]}
            torch.cuda.empty_cache()
        with open(OV / "status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("ALL_DONE", flush=True)
os._exit(0)
