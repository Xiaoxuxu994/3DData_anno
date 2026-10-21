"""TartanAir v2 (diagnostic group): 6 environments, Data_easy/P000, lcam_front pinhole, 200 frames (stride 2).

Files are read straight from the remote zips on Hugging Face (HTTP range requests), only P000 frames.
Pose files are NED (x fwd, y right, z down) body->world; converted to OpenCV cam->world by right-multiplying
the axis permutation. Depth png = RGBA bytes of float32 metres. Pinhole 640x640, FOV 90 deg -> f = 320.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from remotezip import RemoteZip
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from eval3d_common import merge_index, reproj_check, write_check, write_clip, write_clips_json  # noqa: E402

HF = "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main"
OUT = Path("/data/home/z00533370/data/eval3d/B_diag")
ENVS = {  # env -> scene type
    "Office": "indoor", "House": "indoor",
    "ModernCityDowntown": "outdoor_urban", "JapaneseAlley": "outdoor_urban",
    "SeasonalForestAutumn": "nature", "Gascola": "nature",
}
N, STRIDE = 200, 2
# columns = OpenCV camera axes (right, down, forward) expressed in the NED body frame (fwd, right, down)
M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
K = np.array([[320.0, 0, 320.0], [0, 320.0, 320.0], [0, 0, 1]])
_tl = threading.local()


def rz(url):
    cache = getattr(_tl, "z", None)
    if cache is None:
        cache = _tl.z = {}
    if url not in cache:
        cache[url] = RemoteZip(url)
    return cache[url]


for env, kind in ENVS.items():
    img_url = f"{HF}/{env}/Data_easy/image_lcam_front.zip"
    dep_url = f"{HF}/{env}/Data_easy/depth_lcam_front.zip"
    zi = rz(img_url)
    names = zi.namelist()
    pose_lines = zi.read(f"{env}/Data_easy/P000/pose_lcam_front.txt").decode().split("\n")
    pose = np.array([[float(x) for x in l.split()] for l in pose_lines if l.strip()])
    n_all = len(pose)
    stride = STRIDE if n_all >= N * STRIDE else max(1, n_all // N)
    ids = list(range(0, min(n_all, N * stride), stride))[:N]

    def get_img(i, env=env, img_url=img_url):
        b = rz(img_url).read(f"{env}/Data_easy/P000/image_lcam_front/{i:06d}_lcam_front.png")
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    def get_dep(i, env=env, dep_url=dep_url):
        b = rz(dep_url).read(f"{env}/Data_easy/P000/depth_lcam_front/{i:06d}_lcam_front_depth.png")
        a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        return a.view("<f4").squeeze().astype(np.float32)

    with ThreadPoolExecutor(12) as ex:
        imgs = list(ex.map(get_img, ids))
        deps = list(ex.map(get_dep, ids))
    c2w = []
    for i in ids:
        t, q = pose[i, :3], pose[i, 3:7]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q).as_matrix() @ M
        T[:3, 3] = t
        c2w.append(T)
    # depth beyond 80 m (sky / far terrain) is kept up to the uint16 limit at scale 500 (131 m); farther -> invalid
    deps = [np.where(d < 131.0, d, 0).astype(np.float32) for d in deps]
    meta = {
        "source": f"TartanAir v2 {env}/Data_easy/P000 lcam_front", "scene_type": kind,
        "frame_ids": ids, "stride": stride, "fps_effective": 10.0 / stride, "hw": [640, 640],
        "depth_type": "dense", "metric": True, "dynamic": False, "camera_model": "pinhole, FOV 90",
        "license": "CC BY 4.0 (TartanAir)",
        "notes": "Pose NED body->world converted to OpenCV cam->world (world frame stays NED). Depth > 131 m set "
                 "invalid (uint16 at scale 500). TartanAir is in the training sets of several feed-forward models.",
    }
    write_clip(OUT, "tartanair", f"tartanair_{env}", imgs, deps, c2w, np.repeat(K[None], len(ids), 0), meta,
               depth_scale=500.0)
    print(env, len(ids), stride, flush=True)

clips = write_clips_json(OUT, "tartanair")
chk = {c: reproj_check(OUT / "tartanair" / c, max_depth=80) for c in ["tartanair_Office", "tartanair_ModernCityDowntown"]}
write_check(OUT / "tartanair", chk)
merge_index(OUT / "index.json", "tartanair", {"group": "B_diag", "n_clips": len(clips),
                                              "n_frames": sum(c["n_frames"] for c in clips), "path": str(OUT / "tartanair")})
print(json.dumps(chk, indent=1))
