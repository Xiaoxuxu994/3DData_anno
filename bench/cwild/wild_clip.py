"""Download/cut YouTube sections into C_wild sample dirs (12 s, 15 fps, <=1920 wide, no audio).

Usage: python wild_clip.py picks.json
picks: [{"name": "<cat>__<source>__<id>", "cat", "source", "id", "start", "len", "license"?, "notes"?, "extra"?}]
Writes C_wild/clips/<name>/{clip.mp4, frames/, sheet.jpg, meta.json}; raw downloads in C_wild/_raw/yt/<id>_<start>/.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yt_dlp

C = Path("/data/home/z00533370/data/eval3d/C_wild")
FF = os.environ.get("FFMPEG", "ffmpeg")
FPS, MAXW = 15, 1920


def cuts(path):
    r = subprocess.run([FF, "-hide_banner", "-i", str(path), "-vf", "select='gt(scene,0.3)',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return [round(float(l.split("pts_time:")[1].split()[0]), 2) for l in r.stderr.splitlines() if "pts_time:" in l]


def finish(d, clip, meta):
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    (d / "frames").mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    sel = np.linspace(0, len(frames) - 1, 6).astype(int)
    h0, w0 = frames[0].shape[:2]
    tiles = [cv2.resize(frames[i], (480, int(480 * h0 / w0))) for i in sel]
    cv2.imwrite(str(d / "sheet.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    meta.update({"n_frames": len(frames), "hw": [h0, w0], "fps_out": FPS, "cuts": cuts(clip)})
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def main():
    picks = json.load(open(sys.argv[1]))
    vf = f"fps={FPS},scale='min({MAXW},iw)':-2"
    for p in picks:
        d = C / "clips" / p["name"]
        if (d / "meta.json").exists():
            print("skip", p["name"], flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        raw_dir = C / "_raw" / "yt" / f"{p['id']}_{p['start']}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://www.youtube.com/watch?v={p['id']}"
        base = {"format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "quiet": True, "noprogress": True,
                "writeinfojson": True, "outtmpl": str(raw_dir / "raw.%(ext)s")}
        try:
            with yt_dlp.YoutubeDL(dict(base, download_ranges=yt_dlp.utils.download_range_func(None, [(p["start"], p["start"] + p["len"])]),
                                       force_keyframes_at_cuts=True)) as y:
                info = y.extract_info(url, download=True)
            raws = [f for f in raw_dir.glob("raw.*") if not f.name.endswith(".json") and not f.name.endswith(".part")]
            raw, ss = (raws[0] if raws else None), []
            if raw is None or raw.stat().st_size < 500_000:
                with yt_dlp.YoutubeDL(dict(base, outtmpl=str(raw_dir / "full.%(ext)s"))) as y:
                    y.extract_info(url, download=True)
                raw = [f for f in raw_dir.glob("full.*") if not f.name.endswith(".json")][0]
                ss = ["-ss", str(p["start"]), "-t", str(p["len"])]
            clip = d / "clip.mp4"
            subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *ss, "-i", str(raw), "-an", "-vf", vf,
                            "-t", str(p["len"]), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(clip)], check=True)
        except Exception as e:  # noqa: BLE001
            (d / "FAILED").write_text(str(e)[:2000])
            print("FAIL", p["name"], str(e)[:200], flush=True)
            continue
        meta = {"clip": p["name"], "cat": p["cat"], "source": p["source"], "url_or_ref": url + f"&t={p['start']}",
                "title": info.get("title"), "channel": info.get("channel"),
                "license": p.get("license") or info.get("license") or "YouTube standard license",
                "start": p["start"], "len": p["len"], "hard_case": p.get("hard_case", False), "notes": p.get("notes", ""),
                **p.get("extra", {})}
        m = finish(d, clip, meta)
        print("ok", p["name"], m["n_frames"], m["hw"], "cuts", m["cuts"], "|", (m["title"] or "")[:50], flush=True)


if __name__ == "__main__":
    main()
