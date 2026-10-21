"""Prepare a batch directory for the v1.3 pipeline from videos or frame directories.

Usage: python prep.py OUT INPUT [INPUT ...] [--fps=15] [--max_frames=180] [--long=1920] [--gt_bench=DIR]
INPUT: an .mp4 (or any ffmpeg-readable video) or a directory of frames named 000000.jpg, 000001.jpg, ...
       Optional "NAME=INPUT" sets the clip name (default: file / directory name).
Creates (same layout as outputs/bench/<bench>, so the experiment scripts run on it unchanged):
  OUT/bench/clips.json              [{clip, n_frames, hw, cat, ...}]
  OUT/bench/frames/<clip>/%06d.jpg  video: decoded at --fps, first --max_frames frames, long side <= --long
                                    frame dir: symlinks to all given frames (renumbered from 000000.jpg; --max_frames limits)
  OUT/pool/<clip>/{frames, meta.json}  filter-stage view of the same frames (FILTER_POOL layout of code/filter)
--gt_bench: also link OUT/bench/gt/<clip>.npz from DIR/gt and copy the clip's entry from DIR/clips.json
            (dataset, group, metric, ...) so that eval3d_bench.eval_clip can score the labels (tests only).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
FPS, MAXF, LONG = int(opt.get("fps", 15)), int(opt.get("max_frames", 180)), int(opt.get("long", 1920))
GT = Path(opt["gt_bench"]) if "gt_bench" in opt else None
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
OUT = Path(args[0])
B = OUT / "bench"
(B / "frames").mkdir(parents=True, exist_ok=True)
gt_clips = {c["clip"]: c for c in json.loads((GT / "clips.json").read_text())} if GT else {}

clips = json.loads((B / "clips.json").read_text()) if (B / "clips.json").exists() else []
have = {c["clip"] for c in clips}
for spec in args[1:]:
    name, src = spec.split("=", 1) if "=" in spec and not Path(spec).exists() else (None, spec)
    raw = Path(src.rstrip("/"))
    src = raw.resolve()
    name = name or (raw.stem if src.is_file() else raw.name)  # name from the given path, not the symlink target
    fd = B / "frames" / name
    if name in have:
        print("exists, skipped:", name)
        continue
    if src.is_dir():
        frames = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        frames = frames[:MAXF] if "max_frames" in opt else frames  # frame dirs: all frames unless limited
        fd.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(frames):
            dst = fd / f"{i:06d}.jpg"
            if p.suffix.lower() == ".png":
                cv2.imwrite(str(dst), cv2.imread(str(p)), [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif not dst.exists():
                dst.symlink_to(p)
    else:
        fd.mkdir(parents=True, exist_ok=True)
        vf = f"fps={FPS},scale='if(gt(iw,ih),min({LONG},iw),-2)':'if(gt(iw,ih),-2,min({LONG},ih))'"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", vf,
                        "-frames:v", str(MAXF), "-q:v", "2", "-start_number", "0", str(fd / "%06d.jpg")], check=True)
    n = len(list(fd.glob("*.jpg")))
    h, w = cv2.imread(str(fd / "000000.jpg")).shape[:2]
    c = {"clip": name, "n_frames": n, "hw": [h, w], "cat": opt.get("cat", "unknown"), "source": str(src), "fps_out": FPS}
    if name in gt_clips:
        c = dict(gt_clips[name], source=str(src))
        assert c["n_frames"] == n, f"{name}: {n} frames vs {c['n_frames']} in the GT bench"
        (B / "gt").mkdir(exist_ok=True)
        g = B / "gt" / f"{name}.npz"
        if not g.exists():
            g.symlink_to((GT / "gt" / f"{name}.npz").resolve())
    pd_ = OUT / "pool" / name
    pd_.mkdir(parents=True, exist_ok=True)
    if not (pd_ / "frames").exists():
        (pd_ / "frames").symlink_to(fd.resolve())
    (pd_ / "meta.json").write_text(json.dumps({"clip": name, "n_frames": n, "hw": [h, w], "fps_out": FPS,
                                                "cat_hint": c.get("cat", c.get("dataset"))}, ensure_ascii=False))
    clips.append(c)
    print(name, n, (h, w))
(B / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1))
