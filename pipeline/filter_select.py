"""Filter bookkeeping between the filter stages and annotation.

Usage: python filter_select.py OUT split  [--min_frames=90] [--guard=3]
       python filter_select.py OUT select [--force]
split : E2 shot splitting (code/evolution/e2_make_shots.py logic). Every clip whose reject reasons include "cut" is split at its
        TransNetV2 boundaries (scores.json cuts.tn_cuts), `guard` frames dropped on both sides of each boundary, shots
        with >= min_frames frames (90 = 6 s at 15 fps, the v1.3 setting) become new clips <clip>__s<k> in OUT/bench and
        OUT/pool (frames are symlinks). Prints the number of new shots; the filter stages then run again on them.
select: OUT/bench/clips_all.json = every clip; OUT/bench/clips.json = clips that passed the filter (all with --force)
        -> these are the clips the annotation steps process. Writes OUT/filter/selection.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd

OUT, MODE = Path(sys.argv[1]), sys.argv[2]
opt = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[3:] if a.startswith("--"))
B, P = OUT / "bench", OUT / "pool"
dec = pd.read_csv(OUT / "filter" / "decisions.csv")
allc = json.loads((B / "clips_all.json").read_text()) if (B / "clips_all.json").exists() else \
    json.loads((B / "clips.json").read_text())
byname = {c["clip"]: c for c in allc}

if MODE == "split":
    MIN, GUARD = int(opt.get("min_frames", 90)), int(opt.get("guard", 3))
    new = 0
    for _, r in dec.iterrows():
        reasons = set(str(r["reason"]).split(";")) if isinstance(r["reason"], str) else set()
        if r["decision"] != "reject" or "cut" not in reasons or "__s" in r["clip"]:
            continue
        c = byname[r["clip"]]
        n = c["n_frames"]
        cuts = sorted(json.loads((P / r["clip"] / "scores.json").read_text())["cuts"]["tn_cuts"])
        bounds = [0] + cuts + [n]
        frames = sorted((B / "frames" / r["clip"]).glob("*.jpg"))
        for k in range(len(bounds) - 1):
            a = bounds[k] + (GUARD if k > 0 else 0)
            e = bounds[k + 1] - (GUARD if k + 1 < len(bounds) - 1 else 0)
            name = f"{r['clip']}__s{k}"
            if e - a < MIN or name in byname:
                continue
            fd = B / "frames" / name
            fd.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(frames[a:e]):
                (fd / f"{i:06d}.jpg").symlink_to(src.resolve())
            pd_ = P / name
            pd_.mkdir(parents=True, exist_ok=True)
            (pd_ / "frames").symlink_to(fd.resolve())
            meta = json.loads((P / r["clip"] / "meta.json").read_text())
            (pd_ / "meta.json").write_text(json.dumps(dict(meta, clip=name, parent=r["clip"], shot_start=a, shot_end=e,
                                                           n_frames=e - a), ensure_ascii=False))
            byname[name] = dict(c, clip=name, n_frames=e - a, parent=r["clip"], shot_start=a, shot_end=e)
            allc.append(byname[name])
            new += 1
    (B / "clips_all.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1))
    (B / "clips.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1))
    print(new)
elif MODE == "select":
    passed = set(dec.loc[dec["decision"] == "pass", "clip"])
    keep = allc if "force" in opt else [c for c in allc if c["clip"] in passed]
    (B / "clips_all.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1))
    (B / "clips.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1))
    sel = {"n_all": len(allc), "n_pass": len(passed), "annotate": [c["clip"] for c in keep], "forced": "force" in opt,
           "rejected": {r["clip"]: r["reason"] for _, r in dec.iterrows() if r["decision"] != "pass"}}
    (OUT / "filter" / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    print(len(keep))
