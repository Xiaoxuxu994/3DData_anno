"""Download a short section of selected YouTube videos, check for shot cuts, extract frames, make contact sheets.

Usage: python yt_clip.py picks.json OUT_DIR
picks.json: [{"cat": "...", "id": "...", "start": seconds, "len": seconds}, ...]
Output per clip OUT_DIR/<cat>_<id>/: clip.mp4, frames/000000.jpg ..., sheet.jpg, meta.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yt_dlp

picks = json.load(open(sys.argv[1]))
OUT = Path(sys.argv[2])
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FPS = 15  # frames fed to the annotators


def cuts(path):
    """Shot-cut timestamps via ffmpeg scene score > 0.35."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path), "-vf", "select='gt(scene,0.35)',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return [float(l.split("pts_time:")[1].split()[0]) for l in r.stderr.splitlines() if "pts_time:" in l]


for p in picks:
    d = OUT / f"{p['cat']}_{p['id']}"
    (d / "frames").mkdir(parents=True, exist_ok=True)
    clip = d / "clip.mp4"
    if not clip.exists():
        base = {"format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "quiet": True, "noprogress": True,
                "writeinfojson": True, "outtmpl": str(d / "raw.%(ext)s")}
        sec = dict(base, download_ranges=yt_dlp.utils.download_range_func(None, [(p["start"], p["start"] + p["len"])]),
                   force_keyframes_at_cuts=True)
        with yt_dlp.YoutubeDL(sec) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={p['id']}", download=True)
        raw = next(d.glob("raw.*[!json]"))
        ss = []
        if raw.stat().st_size < 1_000_000:  # section download failed -> full download, then trim
            raw.rename(d / "raw_failed_section")
            with yt_dlp.YoutubeDL(dict(base, outtmpl=str(d / "full.%(ext)s"))) as y:
                y.extract_info(f"https://www.youtube.com/watch?v={p['id']}", download=True)
            raw = next(d.glob("full.*[!json]"))
            ss = ["-ss", str(p["start"]), "-t", str(p["len"])]
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *ss, "-i", str(raw), "-an", "-vf", f"fps={FPS}",
                        *(["-t", str(p["keep"])] if "keep" in p else []),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(clip)], check=True)
        meta = {k: info.get(k) for k in ("id", "title", "channel", "license", "duration", "height", "fps", "webpage_url")}
    else:
        meta = json.loads((d / "meta.json").read_text())
    meta.update({"cat": p["cat"], "start": p["start"], "len": p["len"], "fps_out": FPS, "cuts": cuts(clip)})
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta["n_frames"], meta["hw"] = len(frames), list(frames[0].shape[:2])
    sel = np.linspace(0, len(frames) - 1, 6).astype(int)
    tiles = [cv2.resize(frames[i], (480, int(480 * frames[0].shape[0] / frames[0].shape[1]))) for i in sel]
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(str(d / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(p["cat"], p["id"], meta["n_frames"], meta["hw"], "cuts:", meta["cuts"], "|", meta["title"][:60], flush=True)
