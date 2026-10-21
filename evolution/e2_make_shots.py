"""E2: split cut-rejected pool clips into single-shot sub-clips at TransNetV2 boundaries.

python e2_make_shots.py [--min_frames=60] [--guard=3]
For every clip whose decision reason contains "cut": boundaries = scores.json["cuts"]["tn_cuts"] (frames with transition
prob > 0.5 after NMS); each shot drops `guard` frames on both sides of a boundary (dissolves span several frames) and is
kept when it has >= min_frames frames (60 = 4 s at 15 fps). Shots are written to data/filter_pool_shots/<clip>__s<k>/
with frames/ as symlinks to the parent's frames (renumbered from 000000.jpg) and a meta.json copied from the parent
plus {parent, shot_start, shot_end, n_frames}. -> outputs/evolution/E2/shots.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

H = Path("/data/home/z00533370")
POOL, OUT = H / "data/filter_pool", H / "data/filter_pool_shots"
MIN = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--min_frames=")), 60))
GUARD = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--guard=")), 3))

dec = pd.read_csv(POOL / "decisions.csv")
cut = dec[dec["reason"].fillna("").str.split(";").apply(lambda r: "cut" in r)]
rows = []
for _, r in cut.iterrows():
    d = POOL / r["clip"]
    meta = json.loads((d / "meta.json").read_text())
    n = meta["n_frames"]
    cuts = sorted(json.loads((d / "scores.json").read_text())["cuts"]["tn_cuts"])
    bounds = [0] + cuts + [n]
    frames = sorted((d / "frames").glob("*.jpg"))
    for k in range(len(bounds) - 1):
        a = bounds[k] + (GUARD if k > 0 else 0)
        e = bounds[k + 1] - (GUARD if k + 1 < len(bounds) - 1 else 0)
        rec = {"parent": r["clip"], "shot": k, "start": a, "end": e, "n_frames": e - a, "n_cuts": len(cuts),
               "parent_reason": r["reason"], "cat_hint": meta.get("cat_hint")}
        rec["kept"] = e - a >= MIN
        if rec["kept"]:
            name = f"{r['clip']}__s{k}"
            sd = OUT / name
            (sd / "frames").mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(frames[a:e]):
                dst = sd / "frames" / f"{i:06d}.jpg"
                if not dst.exists():
                    dst.symlink_to(src)
            m = dict(meta, clip=name, parent=r["clip"], shot_start=a, shot_end=e, n_frames=e - a,
                     start=meta.get("start", 0) + a / meta.get("fps_out", 15), len=(e - a) / meta.get("fps_out", 15))
            (sd / "meta.json").write_text(json.dumps(m, ensure_ascii=False))
            rec["clip"] = name
        rows.append(rec)
df = pd.DataFrame(rows)
(H / "outputs/evolution/E2").mkdir(parents=True, exist_ok=True)
df.to_csv(H / "outputs/evolution/E2/shots.csv", index=False)
k = df[df["kept"]]
print(f"cut-rejected parents {len(cut)}, shots total {len(df)}, kept >= {MIN} frames: {len(k)} "
      f"from {k['parent'].nunique()} parents; kept frames {k['n_frames'].sum()} / parent frames {len(cut) * 180}")
