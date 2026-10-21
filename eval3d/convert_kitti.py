"""KITTI (rank group): 5 odometry sequences -> raw drives, 200 consecutive frames each (odometry frames 5..204).

Images: raw sync image_02 (rectified left colour). Depth: official data_depth_annotated (sparse, uint16/256 m).
Poses: odometry ground-truth poses (cam0 rectified frame), moved to cam2 with the P_rect_02 baseline.
Only the needed members are used. They are read from RAW/files/<member path> when present (fetched on a
machine with a fast route to S3 by fetch_kitti_mac.py), otherwise from the remote zips via HTTP ranges.
"""

import io
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).parent))
from eval3d_common import merge_index, reproj_check, write_check, write_clip, write_clips_json  # noqa: E402

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
RAW = Path("/data/home/z00533370/data/eval3d_raw/kitti")
OUT = Path("/data/home/z00533370/data/eval3d/A_rank")
# odometry sequence -> (raw drive, start frame of the odometry sequence inside the raw drive)
SEQS = {
    "00": ("2011_10_03_drive_0027", 0),
    "02": ("2011_10_03_drive_0034", 0),
    "05": ("2011_09_30_drive_0018", 0),
    "07": ("2011_09_30_drive_0027", 0),
    "08": ("2011_09_30_drive_0028", 1100),
}
FIRST, N = 5, 200  # annotated depth skips the first/last 5 raw frames
RAW.mkdir(parents=True, exist_ok=True)


def fetch(url, dst):
    if not dst.exists():
        dst.write_bytes(requests.get(url, timeout=300).content)
    return dst


poses_zip = zipfile.ZipFile(fetch(f"{S3}/data_odometry_poses.zip", RAW / "data_odometry_poses.zip"))
calib = {}
for date in {d[0][:10] for d in SEQS.values()}:
    z = zipfile.ZipFile(fetch(f"{S3}/raw_data/{date}_calib.zip", RAW / f"{date}_calib.zip"))
    txt = z.read(f"{date}/calib_cam_to_cam.txt").decode()
    P = np.array([float(x) for x in next(l for l in txt.splitlines() if l.startswith("P_rect_02:")).split()[1:]]).reshape(3, 4)
    calib[date] = P

depth_zip_url = f"{S3}/data_depth_annotated.zip"
_tl = threading.local()


def rz(url):
    """One RemoteZip per thread and URL (the central directory is read once)."""
    cache = getattr(_tl, "z", None)
    if cache is None:
        cache = _tl.z = {}
    if url not in cache:
        cache[url] = RemoteZip(url)
    return cache[url]


results = {}
for seq, (drive, start) in SEQS.items():
    date = drive[:10]
    P2 = calib[date]
    K = P2[:, :3]
    t02 = np.linalg.inv(K) @ P2[:, 3]  # cam2 = cam0 + t02 (rectified frames)
    T_cam0_cam2 = np.eye(4)
    T_cam0_cam2[:3, 3] = -t02
    poses = np.loadtxt(io.StringIO(poses_zip.read(f"dataset/poses/{seq}.txt").decode())).reshape(-1, 3, 4)
    frames = list(range(FIRST, FIRST + N))
    raw_ids = [start + j for j in frames]
    c2w = []
    for j in frames:
        T = np.eye(4)
        T[:3, :4] = poses[j]
        c2w.append(T @ T_cam0_cam2)

    def local(name):
        f = RAW / "files" / name
        return f.read_bytes() if f.exists() else None

    def get_img(r):
        name = f"{date}/{drive}_sync/image_02/data/{r:010d}.png"
        b = local(name) or rz(f"{S3}/raw_data/{drive}/{drive}_sync.zip").read(f"{date}/{drive}_sync/image_02/data/{r:010d}.png")
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    def get_depth(r):
        name = f"train/{drive}_sync/proj_depth/groundtruth/image_02/{r:010d}.png"
        b = local(name) or rz(depth_zip_url).read(f"train/{drive}_sync/proj_depth/groundtruth/image_02/{r:010d}.png")
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)

    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(get_img, raw_ids))
        deps = list(ex.map(get_depth, raw_ids))
    assert all(d.dtype == np.uint16 for d in deps)
    h, w = imgs[0].shape[:2]
    clip = f"kitti_{seq}"
    meta = {
        "source": f"KITTI odometry seq {seq} = raw {drive}_sync frames {raw_ids[0]}..{raw_ids[-1]}; image_02",
        "frame_ids": raw_ids, "stride": 1, "fps_effective": 10.0, "hw": [h, w], "depth_type": "sparse",
        "metric": True, "dynamic": "partial (moving cars/cyclists in some frames)",
        "license": "CC BY-NC-SA 3.0 (KITTI)",
        "notes": "Poses: odometry GT (cam0) shifted to cam2 by P_rect_02 baseline. Depth: data_depth_annotated "
                 "(accumulated LiDAR, KITTI depth *train* split; some monocular depth models train on it).",
    }
    write_clip(OUT, "kitti", clip, imgs, deps, c2w, np.repeat(K[None], N, 0), meta, depth_scale=256.0)
    results[clip] = None
    print(clip, h, w, flush=True)

clips = write_clips_json(OUT, "kitti")
chk = {c: reproj_check(OUT / "kitti" / c, max_depth=80) for c in ["kitti_00", "kitti_07"]}
write_check(OUT / "kitti", chk, "Sparse LiDAR: only pixels with depth in both frames; moving objects inflate the error.")
merge_index(OUT / "index.json", "kitti", {"group": "A_rank", "n_clips": len(clips),
                                          "n_frames": sum(c["n_frames"] for c in clips), "path": str(OUT / "kitti")})
print(json.dumps(chk, indent=1))
