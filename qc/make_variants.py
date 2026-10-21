"""Stability variants of the eval benches (symlinked frames, no copies).

  python make_variants.py  -> outputs/qc/variants/{A,B,C}_{rev,sub2}/{frames/<clip>/%06d.jpg, clips.json}
rev  = frame order reversed (ViPE paper "Shuttle" check); sub2 = every 2nd frame (half frame rate).
"""
import json
import os
from pathlib import Path

O = Path("/data/home/z00533370/outputs")
GROUPS = {"A": O / "bench/eval3d_A", "B": O / "bench/eval3d_B", "C": O / "bench/wild"}
for g, src in GROUPS.items():
    clips = json.loads((src / "clips.json").read_text())
    for var in ("rev", "sub2"):
        dst = O / "qc/variants" / f"{g}_{var}"
        out = []
        for c in clips:
            n = c["n_frames"]
            idx = list(range(n - 1, -1, -1)) if var == "rev" else list(range(0, n, 2))
            d = dst / "frames" / c["clip"]
            d.mkdir(parents=True, exist_ok=True)
            for new, old in enumerate(idx):
                p = d / f"{new:06d}.jpg"
                if not p.exists():
                    os.symlink(src / "frames" / c["clip"] / f"{old:06d}.jpg", p)
            out.append(dict(c, n_frames=len(idx), variant=var, src_index=idx))
        (dst / "clips.json").write_text(json.dumps(out))
        print(dst, len(out), sum(c["n_frames"] for c in out))
