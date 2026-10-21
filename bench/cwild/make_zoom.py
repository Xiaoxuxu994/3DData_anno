"""Hard cases: synthetic digital zoom (1.0x -> 2.0x centre crop over the clip) made from real C_wild clips."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

C = Path("/data/home/z00533370/data/eval3d/C_wild/clips")
for src in ["outdoor_static__sekai__b06ThXzoqRI_10369", "outdoor_static__sekai__uO99uLuqisc_1261"]:
    s = C / src
    m = json.loads((s / "meta.json").read_text())
    name = f"hard__zoom__{src.split('__')[-1]}"
    d = C / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    fs = sorted((s / "frames").glob("*.jpg"))
    n = len(fs)
    out = []
    for i, f in enumerate(fs):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        z = 1.0 + i / max(n - 1, 1)  # 1.0x -> 2.0x
        cw, ch = int(round(w / z)), int(round(h / z))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        im = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(im)
    sel = np.linspace(0, n - 1, 6).astype(int)
    tiles = [cv2.resize(out[i], (480, 270)) for i in sel]
    cv2.imwrite(str(d / "sheet.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    meta = {"clip": name, "cat": "hard", "source": "synthetic_zoom", "n_frames": n, "hw": [h, w], "fps_out": 15,
            "license": m.get("license"), "url_or_ref": f"derived from {src} ({m.get('url_or_ref')})", "title": m.get("title"),
            "hard_case": True, "notes": "难例：数字变焦，焦距在 12 s 内线性放大到 2 倍（由真实视频中心裁剪合成，主点不变）",
            "zoom_factor": [1.0, 2.0], "cuts": []}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(name, n)
