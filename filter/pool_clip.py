"""Pick ~400 CC-BY videos from candidates.json (balanced by cat_hint, capped per channel) and cut 3-5 x 12 s clips each.

Usage:
  python pool_clip.py select candidates.json picks.json [N_VIDEOS=400]
  python pool_clip.py cut picks.json POOL_DIR WORKER_ID N_WORKERS
Per clip POOL_DIR/<clip>/: clip.mp4 (<=1080p, 15 fps), frames/000000.jpg (960 px wide), sheet.jpg, meta.json.
Index: POOL_DIR/pool.json (written by `index`): python pool_clip.py index POOL_DIR [picks.json]
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

H = Path("/data/home/z00533370")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FPS, LEN, FW = 15, 12, 960


def excluded_ids():
    ids = set()
    for f in [H / "data/eval3d/C_wild/clips.json", H / "outputs/bench/youtube/clips.json"]:
        for c in json.loads(f.read_text()):
            m = re.search(r"v=([\w-]{11})", c.get("url_or_ref") or c.get("url") or "")
            if m:
                ids.add(m.group(1))
    return ids


def select(cand_path, out_path, n_videos=400, n_cats=12):
    """Per category (search order, seeded shuffle) take up to n_videos // n_cats videos, max 3 videos per channel.
    Stable when more categories are appended later, so cutting can start before the search has finished."""
    rows = [r for r in json.loads(Path(cand_path).read_text()) if r["ok"]]
    excl = excluded_ids()
    seen, by_cat = set(), defaultdict(list)
    for r in rows:
        if r["id"] in seen or r["id"] in excl:
            continue
        seen.add(r["id"])
        by_cat[r["cat_hint"]].append(r)
    quota = -(-n_videos // n_cats)
    per_chan, picks, spare = Counter(), [], []
    for cat, v in by_cat.items():
        random.Random(cat).shuffle(v)
        got = 0
        for r in v:
            if got < quota and per_chan[r["channel_id"]] < 3:
                per_chan[r["channel_id"]] += 1
                picks.append(r)
                got += 1
            else:
                spare.append(r)
    if len(by_cat) >= n_cats:  # all categories searched: top up round-robin over categories from the spare pool
        spare_by = defaultdict(list)
        for r in spare:
            spare_by[r["cat_hint"]].append(r)
        while len(picks) < n_videos and any(spare_by.values()):
            for cat in list(spare_by):
                while spare_by[cat] and len(picks) < n_videos:
                    r = spare_by[cat].pop(0)
                    if per_chan[r["channel_id"]] < 3:
                        per_chan[r["channel_id"]] += 1
                        picks.append(r)
                        break
    for r in picks:
        d = r["duration"]
        k = 5 if d >= 600 else 4 if d >= 300 else 3
        lo, hi = 0.1 * d, 0.9 * d - LEN
        r["starts"] = [int(lo + (hi - lo) * (i + 0.5) / k) for i in range(k)]
    Path(out_path).write_text(json.dumps(picks, indent=1, ensure_ascii=False))
    print(len(picks), "videos", sum(len(r["starts"]) for r in picks), "clips", Counter(r["cat_hint"] for r in picks))


def cut(picks_path, pool, wid, nw):
    """One yt-dlp call per video (all 3-5 sections at once -> one metadata request per video)."""
    import cv2
    import numpy as np
    pool = Path(pool)
    picks = json.loads(Path(picks_path).read_text())[wid::nw]
    raw_dir = pool / "_raw"
    for r in picks:
        todo = [s for s in r["starts"] if not (pool / f"{r['cat_hint']}__{r['id']}__{s:05d}" / "meta.json").exists()]
        if not todo or (raw_dir / r["id"] / "FAILED").exists():
            continue
        vd = raw_dir / r["id"]
        vd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        # yt-dlp CLI in its own process group with a hard timeout (in-process downloads hung for hours on stalled
        # ffmpeg section fetches)
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-progress", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
               "--force-keyframes-at-cuts", "--socket-timeout", "30", "--retries", "3",
               "-o", str(vd / "sec_%(section_start)d.%(ext)s")]
        for s in todo:
            cmd += ["--download-sections", f"*{s}-{s + LEN}"]
        cmd.append(r["url"])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            outp, _ = p.communicate(timeout=180 + 120 * len(todo))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            p.communicate()
            outp = "TIMEOUT"
        if p.returncode != 0 and not any(vd.glob("sec_*.mp4")):
            print("SKIP", r["id"], outp.strip()[-160:].replace("\n", " "), flush=True)
            (vd / "FAILED").write_text(outp[-2000:])
            if "not a bot" in outp or "unavailable" in outp.lower():
                time.sleep(60)
            continue
        for s in todo:
            name = f"{r['cat_hint']}__{r['id']}__{s:05d}"
            d = pool / name
            raws = [p for p in vd.glob(f"sec_{s}.*") if p.suffix != ".json"]
            if not raws or raws[0].stat().st_size < 300_000:
                print("SKIP_SMALL", name, flush=True)
                continue
            (d / "frames").mkdir(parents=True, exist_ok=True)
            clip = d / "clip.mp4"
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raws[0]), "-an", "-t", str(LEN),
                            "-vf", f"fps={FPS}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip)], check=False)
            cap, frames = cv2.VideoCapture(str(clip)), []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            if len(frames) < FPS * LEN * 0.9:
                print("SKIP_SHORT", name, len(frames), flush=True)
                if clip.exists():
                    clip.rename(d / "clip_short.mp4")
                continue
            h0, w0 = frames[0].shape[:2]
            fh = int(round(FW * h0 / w0 / 2) * 2)
            for i, f in enumerate(frames):
                cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), cv2.resize(f, (FW, fh), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            sel = np.linspace(0, len(frames) - 1, 6).astype(int)
            tiles = [cv2.resize(frames[i], (480, int(480 * h0 / w0))) for i in sel]
            cv2.imwrite(str(d / "sheet.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            meta = {"clip": name, "id": r["id"], "title": r["title"], "channel": r["channel"], "channel_id": r["channel_id"],
                    "license": r["license"], "url": f"{r['url']}&t={s}", "duration": r["duration"], "src_height": r["height"],
                    "src_fps": r["fps"], "query": r["query"], "cat_hint": r["cat_hint"], "start": s, "len": LEN,
                    "fps_out": FPS, "n_frames": len(frames), "hw_clip": [h0, w0], "hw_frames": [fh, FW]}
            (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            print("OK", name, len(frames), [h0, w0], flush=True)
        print("VIDEO", r["id"], len(todo), f"{time.time() - t0:.0f}s", flush=True)
        time.sleep(random.uniform(5.0, 12.0))
    print("CUT_DONE", wid, flush=True)


def index(pool, picks_path=None):
    pool = Path(pool)
    metas = [json.loads(m.read_text()) for m in sorted(pool.glob("*/meta.json"))]
    if picks_path:  # keep only clips of the final selection
        keep = {(r["id"], s) for r in json.loads(Path(picks_path).read_text()) for s in r["starts"]}
        metas = [m for m in metas if (m["id"], m["start"]) in keep]
    (pool / "pool.json").write_text(json.dumps(metas, indent=1, ensure_ascii=False))
    print(len(metas), "clips", len({m["id"] for m in metas}), "videos", Counter(m["cat_hint"] for m in metas))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:]))
    elif cmd == "cut":
        cut(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "index":
        index(sys.argv[2], *sys.argv[3:])
