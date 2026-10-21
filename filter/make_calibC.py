"""Mirror group C (72 clips) into a pool-style dir so every filter stage can run on it for calibration.
Usage: python make_calibC.py   -> data/filter_pool/_calibC/<clip>/{frames -> C frames, sheet.jpg, meta.json}
"""
import json
from pathlib import Path

H = Path("/data/home/z00533370")
C = H / "data/eval3d/C_wild"
OUT = H / "data/filter_pool/_calibC"
rows = json.loads((C / "clips.json").read_text())
for r in rows:
    d = OUT / r["clip"]
    d.mkdir(parents=True, exist_ok=True)
    src = C / "clips" / r["clip"]
    for n in ("frames", "sheet.jpg"):
        if not (d / n).exists():
            (d / n).symlink_to((src / n).resolve())
    (d / "meta.json").write_text(json.dumps({**r, "cat_hint": r["cat"]}, indent=1, ensure_ascii=False))
(OUT / "pool.json").write_text(json.dumps([{**r, "cat_hint": r["cat"]} for r in rows], indent=1, ensure_ascii=False))
print(len(rows), "clips ->", OUT)
