"""Stage track: CoTracker3 (offline) track-loss statistics to catch cuts that shot detectors miss.

Usage: CUDA_VISIBLE_DEVICES=4 python s6_track.py WORKER_ID N_WORKERS [POOL_DIR]
Video at 512x288, every 2nd frame (90 frames, 7.5 fps). Queries: 16x9 grid at t = 0, 30, 60.
Writes scores.json["track"]:
  vis_drop_max      max over t of the share of points visible at t-1 that are invisible at t (cut -> ~1)
  vis_drop_t        frame (original 15 fps index) of that drop
  loss_rate         mean share of frames after its query frame where a track is invisible
  survive_1s        share of t=0 queries still visible 1 s later
  jump_max          max over t of median |track displacement| t-1 -> t, in fraction of width (teleport -> cut)
"""

import sys

import cv2
import numpy as np
import torch

from fcommon import StageLog, clips, frame_paths, has_stage, load_frames, timer, write_scores

wid, nw = int(sys.argv[1]), int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else None
W, Hh, STEP = 512, 288, 2
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True, skip_validation=True).cuda().eval()
log = StageLog("track", wid)
gy, gx = np.mgrid[0:9, 0:16]
G = np.stack([(gx.ravel() + 0.5) * W / 16, (gy.ravel() + 0.5) * Hh / 9], 1)

for d in clips(pool, wid, nw):
    if has_stage(d, "track") or not (d / "meta.json").exists():
        continue
    t = timer()
    n = len(frame_paths(d))
    idx = list(range(0, n, STEP))
    v = np.stack([cv2.resize(x, (W, Hh), interpolation=cv2.INTER_AREA) for x in load_frames(d, idx)])
    T = len(v)
    qt = [0, T // 3, 2 * T // 3]
    q = np.concatenate([np.c_[np.full(len(G), k), G] for k in qt]).astype(np.float32)
    video = torch.from_numpy(v).permute(0, 3, 1, 2)[None].float().cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tracks, vis = model(video, queries=torch.from_numpy(q)[None].cuda())
    tracks, vis = tracks[0].float().cpu().numpy(), vis[0].float().cpu().numpy() > 0.5  # (T,N,2), (T,N)
    active = np.arange(T)[:, None] >= q[None, :, 0]
    drop = []
    jump = []
    for k in range(1, T):
        a = active[k] & active[k - 1] & vis[k - 1]
        drop.append((~vis[k][a]).mean() if a.sum() > 10 else 0.0)
        b = a & vis[k]
        jump.append(np.median(np.linalg.norm(tracks[k][b] - tracks[k - 1][b], axis=1)) / W if b.sum() > 10 else 0.0)
    after = active & (np.arange(T)[:, None] > q[None, :, 0])
    k1 = min(int(round(15 / STEP)), T - 1)
    s = {"vis_drop_max": max(drop), "vis_drop_t": idx[int(np.argmax(drop)) + 1], "loss_rate": (~vis[after]).mean(),
         "survive_1s": vis[k1, :len(G)].mean(), "jump_max": max(jump), "jump_median": np.median(jump)}
    write_scores(d, "track", s)
    log(d, t(), **s)
    print(d.name, {k: round(float(v), 4) for k, v in s.items()}, f"{t():.1f}s", flush=True)
print("TRACK_DONE", wid, flush=True)
