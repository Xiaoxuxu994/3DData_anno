"""ocr_recheck.py POOL_DIR OUT_JSON - recompute the text-box areas of filter/s3_ocr.py with float arithmetic, for every clip of a filter pool.

Why: with the locked paddleocr / paddlepaddle versions the detector returns int16 polygons and `(x1 - x0) * (y1 - y0)` overflows for boxes larger
than ~181 x 181 px (docs/KNOWN_ISSUES.md item 9), so `ocr.text_area_max / text_area_mean / bottom_band_area / corner_area` under-report large
captions. This script does NOT change the stage: it reads the frames of POOL_DIR/<clip>/frames, runs the same detector on the same three frames,
and reports the corrected `text_area_max` next to the stored one, and the static-text area recomputed from the fillPoly masks (that one was fine).
Needs the OCR environment (envs/ocr-venv.txt, CPU is fine). OUT_JSON must not exist.
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextDetection

pool, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    sys.exit(f"{out} exists (results are never overwritten)")
det = TextDetection(model_name="PP-OCRv5_mobile_det", device="cpu", enable_mkldnn=False)
res = {}
for d in sorted(p for p in pool.iterdir() if p.is_dir()):
    ps = sorted(glob.glob(str(d / "frames" / "*.jpg")))
    if not ps or not (d / "scores.json").exists() or "ocr" not in json.loads((d / "scores.json").read_text()):
        continue
    areas, masks, nb = [], [], 0
    for i in np.linspace(0, len(ps) - 1, 3).round().astype(int):
        im = cv2.imread(ps[i])
        h, w = im.shape[:2]
        r = det.predict(im)[0]
        polys = np.asarray(r["dt_polys"]) if len(r["dt_polys"]) else np.zeros((0, 4, 2))
        m, a = np.zeros((h, w), np.uint8), 0.0
        for p in polys:
            pf = p.astype(np.float64)
            (x0, y0), (x1, y1) = pf.min(0), pf.max(0)
            a += (x1 - x0) * (y1 - y0) / (h * w)
            cv2.fillPoly(m, [p.astype(np.int32)], 1)
        nb += len(polys)
        areas.append(a)
        masks.append(m)
    s = json.loads((d / "scores.json").read_text())["ocr"]
    res[d.name] = {"as_run_text_area_max": s["text_area_max"], "corrected_text_area_max": max(areas), "static_as_run": s["static_text_area"],
                   "static_recomputed": float(np.logical_and.reduce(masks).mean()), "n_boxes": nb}
    print(d.name, {k: round(v, 4) for k, v in res[d.name].items()}, flush=True)
out.write_text(json.dumps(res, indent=1))
