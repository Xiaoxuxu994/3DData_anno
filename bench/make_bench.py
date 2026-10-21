"""Benchmark subset for cross-pipeline comparison: per scene the longest segment, first 200 frames."""
import json
import os
from pathlib import Path

ROOT = Path("/data/home/z00533370/outputs/vipe/scannetpp_sample30")
B = Path("/data/home/z00533370/outputs/bench/scannetpp")
N = 200
segs = json.loads((ROOT / "segments.json").read_text())
best = {}
for s in segs:
    if s["scene"] not in best or s["n_frames"] > best[s["scene"]]["n_frames"]:
        best[s["scene"]] = s
clips = []
for scene, s in sorted(best.items()):
    name = f"{scene}_b"
    d = B / "frames" / name
    d.mkdir(parents=True, exist_ok=True)
    n = min(N, s["n_frames"])
    for j in range(n):
        dst = d / f"{j:06d}.jpg"
        if not dst.exists():
            os.symlink(ROOT / "frames" / s["segment"] / f"{j:06d}.jpg", dst)
    clips.append({"clip": name, "scene": scene, "segment": s["segment"], "n_frames": n, "frames": s["frames"][:n]})
(B / "clips.json").write_text(json.dumps(clips, indent=1))
print(len(clips), "clips,", sum(c["n_frames"] for c in clips), "frames")
