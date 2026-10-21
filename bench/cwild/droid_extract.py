"""Cut 10 DROID-100 episodes (LeRobot v3 mirror) into C_wild samples: 6 wrist-camera + 4 exterior (fixed) clips."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/bench/cwild")
C = Path("/data/home/z00533370/data/eval3d/C_wild")
R = C / "_raw" / "droid_100"
FF = "/data/home/z00533370/cache/bin/ffmpeg"
e = pd.read_parquet(R / "meta/episodes/chunk-000/file-000.parquet")
e = e[e.length >= 180].copy()
e["task"] = e.tasks.map(lambda t: t[0] if len(t) else "")
e = e.drop_duplicates("task").sample(frac=1, random_state=3)
plan = [("wrist_image_left", "腕部相机，随机械臂运动")] * 6 + [("exterior_image_1_left", "外部相机，固定机位")] * 4
for (_, r), (cam, note) in zip(e.iterrows(), plan):
    key = f"observation.images.{cam}"
    t0 = float(r[f"videos/{key}/from_timestamp"])
    src = R / f"videos/{key}/chunk-000/file-000.mp4"
    name = f"embodied__droid__ep{int(r.episode_index):03d}_{cam.split('_')[0]}"
    d = C / "clips" / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    clip = d / "clip.mp4"
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t0:.4f}", "-i", str(src), "-t", "12", "-an",
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(clip)], check=True)
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    frames = frames[:180]
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / "frames" / f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    sel = np.linspace(0, len(frames) - 1, 6).astype(int)
    tiles = [cv2.resize(frames[i], (480, 270)) for i in sel]
    cv2.imwrite(str(d / "sheet.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    meta = {"clip": name, "cat": "embodied", "source": "droid_100", "n_frames": len(frames), "hw": list(frames[0].shape[:2]),
            "fps_out": 15, "license": "CC-BY 4.0 (DROID); LeRobot mirror lerobot/droid_100",
            "url_or_ref": f"lerobot/droid_100 episode {int(r.episode_index)} {key} t={t0:.2f}s",
            "title": r.task, "hard_case": False, "camera": cam,
            "notes": f"{note}；任务：{r.task}；DROID 公开版分辨率仅 320×180", "cuts": []}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(name, len(frames), meta["hw"], r.task, flush=True)
