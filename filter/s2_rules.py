"""Stage rules: luminance, Laplacian sharpness, LAION aesthetic (CLIP ViT-L/14 + improved-aesthetic MLP, as SpatialVID).

Usage: CUDA_VISIBLE_DEVICES=4 python s2_rules.py WORKER_ID N_WORKERS [POOL_DIR]
Writes scores.json["rules"]: lum_mean/min/max (0-255, 16 frames), lap_var_median/min (gray, 960 px wide),
aesthetic (mean over 3 frames, 1-10 scale).
"""

import sys

import cv2
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image

from fcommon import H, StageLog, clips, even_idx, frame_paths, has_stage, load_frames, timer, write_scores

wid, nw = int(sys.argv[1]), int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else None
AES_CKPT = H / "code/SpatialVID/checkpoints/aesthetic.pth"


class MLP(nn.Module):  # identical layout to SpatialVID scoring/aesthetic/inference.py
    def __init__(self, n=768):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                                    nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device="cuda")
clip_model.eval()
mlp = MLP().cuda().eval()
mlp.load_state_dict(torch.load(AES_CKPT, map_location="cpu"))
log = StageLog("rules", wid)

for d in clips(pool, wid, nw):
    if has_stage(d, "rules") or not (d / "meta.json").exists():
        continue
    t = timer()
    n = len(frame_paths(d))
    fr = load_frames(d, even_idx(n, 16))
    fr = [cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA) if f.shape[1] != 960 else f
          for f in fr]
    lum = np.array([(0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]).mean() for f in fr])
    lap = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in fr])
    with torch.inference_mode():
        x = torch.stack([preprocess(Image.fromarray(fr[i])) for i in (0, 8, 15)]).cuda()
        e = clip_model.encode_image(x)
        aes = mlp(torch.nn.functional.normalize(e, dim=-1).float()).squeeze(-1).cpu().numpy()
    s = {"lum_mean": lum.mean(), "lum_min": lum.min(), "lum_max": lum.max(), "lap_var_median": np.median(lap),
         "lap_var_min": lap.min(), "aesthetic": aes.mean(), "aesthetic_min": aes.min()}
    write_scores(d, "rules", s)
    log(d, t(), **s)
    print(d.name, {k: round(float(v), 2) for k, v in s.items()}, flush=True)
print("RULES_DONE", wid, flush=True)
