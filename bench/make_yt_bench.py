"""Lay out downloaded YouTube clips like the ScanNet++ bench: frames/<clip>/ + clips.json (no GT)."""
import json
import os
from pathlib import Path

SRC = Path("/data/home/z00533370/data/youtube_samples/clips")
B = Path("/data/home/z00533370/outputs/bench/youtube")
clips = []
for d in sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "meta.json").exists()):
    meta = json.loads((d / "meta.json").read_text())
    fr = sorted((d / "frames").glob("*.jpg"))
    dst = B / "frames" / d.name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(d / "frames", dst)
    clips.append({"clip": d.name, "cat": meta["cat"], "n_frames": len(fr), "hw": meta["hw"], "title": meta["title"],
                  "url": meta["webpage_url"], "license": meta["license"], "channel": meta["channel"], "cuts": meta["cuts"]})
(B / "clips.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False))
for c in clips:
    print(c["clip"], c["n_frames"], c["hw"], "cuts", c["cuts"])
