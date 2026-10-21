"""Overview mosaics for the visual check: 40 random passed and 40 random rejected contact sheets.

Usage: python make_mosaic.py decisions.csv POOL_DIR OUT_DIR [N=40] [SEED=0]
Each tile = the clip's 6-frame sheet.jpg (2x3), captioned with clip name and reject reason.
Writes OUT_DIR/mosaic_pass.jpg, mosaic_reject.jpg and the sampled lists mosaic_{pass,reject}.csv.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

csv, pool, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv)
TW, COLS = 480, 5
for kind in ("pass", "reject"):
    sub = df[df.decision == kind]
    sub = sub.sample(min(N, len(sub)), random_state=seed)
    sub[["clip", "decision", "reason", "url"]].to_csv(out / f"mosaic_{kind}.csv", index=False)
    tiles = []
    for k, (_, r) in enumerate(sub.iterrows()):
        im = cv2.imread(str(pool / r["clip"] / "sheet.jpg"))
        im = cv2.resize(im, (TW, int(TW * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
        cap = np.zeros((44, TW, 3), np.uint8)
        cv2.putText(cap, f"{k + 1:02d} {r["clip"][:52]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        reason = "" if pd.isna(r.reason) else str(r.reason)
        cv2.putText(cap, reason[:70], (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (80, 220, 80) if kind == "pass" else (80, 80, 255), 1)
        tiles.append(np.vstack([cap, im]))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % COLS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(str(out / f"mosaic_{kind}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(kind, len(sub), "->", out / f"mosaic_{kind}.jpg")
