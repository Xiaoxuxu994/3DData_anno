"""Qualitative grid for the PURE report: middle-frame inverse depth per method (rows) x clip (columns).

  python pure_render.py OUT_DIR
Each tile is normalized by its own 2-98 percentile (shape comparison, not scale).
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

O = Path("/data/home/z00533370/outputs/bench")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CLIPS = [("eval3d_A", "sintel__market_5"), ("eval3d_A", "bonn__crowd"),
         ("wild", "outdoor_dynamic__youtube_cc__DPumdBzLxqc"), ("wild", "indoor_dynamic__yt__UNn-xlbWc9M"),
         ("wild", "embodied__droid__ep025_exterior")]
METHODS = ["pi3", "pure_pi3", "vggt_b", "pure_vggt", "da3g", "pure_da3", "vggt_omega", "vggt_omega512"]
TW, TH = 288, 162


def inv_tile(d):
    inv = np.where(np.isfinite(d) & (d > 0), 1.0 / np.maximum(d, 1e-6), np.nan)
    v = np.isfinite(inv)
    lo, hi = np.percentile(inv[v], 2), np.percentile(inv[v], 98)
    x = np.clip((np.nan_to_num(inv, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (matplotlib.colormaps["magma"](x)[..., :3] * 255).astype(np.uint8)
    rgb[~v] = 0
    return rgb


def fit(img):
    h, w = img.shape[:2]
    s = min(TW / w, TH / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((TH, TW, 3), 255, np.uint8)
    y0, x0 = (TH - r.shape[0]) // 2, (TW - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


cols = []
for bench, clip in CLIPS:
    z = np.load(O / bench / METHODS[0] / clip / "pred.npz")
    k = len(z["depth"]) // 2
    rgb = cv2.cvtColor(cv2.imread(str(O / bench / "frames" / clip / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
    col = [fit(rgb)]
    for m in METHODS:
        p = O / bench / m / clip / "pred.npz"
        col.append(fit(inv_tile(np.load(p)["depth"][k].astype(np.float32))) if p.exists() else np.full((TH, TW, 3), 230, np.uint8))
    cols.append(np.concatenate(col, 0))
grid = np.concatenate([np.pad(c, ((0, 0), (0, 6), (0, 0)), constant_values=255) for c in cols], 1)
cv2.imwrite(str(OUT / "qual_grid.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
print("rows:", ["rgb"] + METHODS, "cols:", [c for _, c in CLIPS], grid.shape)
