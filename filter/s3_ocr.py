"""Stage OCR: PaddleOCR text detection (PP-OCRv5 mobile det) on 3 frames, as SpatialVID's ocr score.

Usage: python s3_ocr.py WORKER_ID N_WORKERS [POOL_DIR]   (ocr-venv; CPU)
Writes scores.json["ocr"]: text_area_max / text_area_mean (box area / frame area, SpatialVID ocr score = max),
bottom_band_area (text boxes centred in the lowest 25 %: subtitles), corner_area (boxes inside a corner 20 % x 20 %:
watermarks / logos), static_text_area (box area present at the same place in all 3 frames: burned-in overlays), n_boxes.
"""

import sys

import cv2
import numpy as np
from paddleocr import TextDetection

from fcommon import StageLog, clips, even_idx, frame_paths, has_stage, timer, write_scores

wid, nw = int(sys.argv[1]), int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else None
det = TextDetection(model_name="PP-OCRv5_mobile_det", device="cpu", enable_mkldnn=False)
log = StageLog("ocr", wid)

for d in clips(pool, wid, nw):
    if has_stage(d, "ocr") or not (d / "meta.json").exists():
        continue
    t = timer()
    ps = frame_paths(d)
    imgs = [cv2.imread(str(ps[i])) for i in even_idx(len(ps), 3)]
    h, w = imgs[0].shape[:2]
    areas, bottom, corner, masks, nb = [], [], [], [], 0
    for im in imgs:
        res = det.predict(im)[0]
        polys = np.asarray(res["dt_polys"]) if len(res["dt_polys"]) else np.zeros((0, 4, 2))
        m = np.zeros((h, w), np.uint8)
        a = b = c = 0.0
        for p in polys:
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            ar = (x1 - x0) * (y1 - y0) / (h * w)
            a += ar
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            b += ar * (cy > 0.75)
            c += ar * ((cx < 0.2 or cx > 0.8) and (cy < 0.2 or cy > 0.8))
            cv2.fillPoly(m, [p.astype(np.int32)], 1)
        nb += len(polys)
        areas.append(a), bottom.append(b), corner.append(c), masks.append(m)
    static = float(np.logical_and.reduce(masks).mean())
    s = {"text_area_max": max(areas), "text_area_mean": np.mean(areas), "bottom_band_area": np.mean(bottom),
         "corner_area": np.mean(corner), "static_text_area": static, "n_boxes": nb}
    write_scores(d, "ocr", s)
    log(d, t(), **s)
    print(d.name, {k: round(float(v), 4) for k, v in s.items()}, flush=True)
print("OCR_DONE", wid, flush=True)
