"""Virtual KITTI 2 (diagnostic group) without downloading the 15.7 GB tars.

The official tars are mirrored unchanged on Hugging Face (kronecker0122/vkitti2, same byte sizes), which
serves HTTP range requests. Uncompressed tar = 512-byte headers followed by file data, so:
  1. probe the tar every ~20 MB, find headers in each probe (ustar magic + checksum) -> coarse offset->path map
  2. from the last probe before the wanted directory, walk headers one by one (512-byte reads)
  3. download only the wanted members' byte ranges
Poses/intrinsics come from textgt.tar.gz (small, downloaded whole). extrinsic.txt is world->camera.
Depth png: uint16 centimetres (65535 = sky / far).
"""

import io
import json
import re
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
from eval3d_common import merge_index, reproj_check, write_check, write_clip, write_clips_json  # noqa: E402

HF = "https://huggingface.co/datasets/kronecker0122/vkitti2/resolve/main"
RAW = Path("/data/home/z00533370/data/eval3d_raw/vkitti2")
OUT = Path("/data/home/z00533370/data/eval3d/B_diag")
CLIPS = [("Scene01", "clone"), ("Scene02", "clone"), ("Scene06", "clone"), ("Scene18", "clone"),
         ("Scene20", "clone"), ("Scene01", "fog"), ("Scene20", "rain")]
N = 200
sess = requests.Session()


def rng(url, off, n):
    for _ in range(6):
        try:
            r = sess.get(url, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=120)
            if r.status_code in (200, 206):
                return r.content
        except requests.RequestException:
            pass
    raise RuntimeError(f"range failed {url} {off}")


def size_of(url):
    r = sess.head(url, allow_redirects=True, timeout=60)
    return int(r.headers["Content-Length"])


def parse_header(b):
    if len(b) < 512 or b[257:262] != b"ustar":
        return None
    chk = int(b[148:156].split(b"\0")[0].strip() or b"0", 8)
    if chk != sum(b[:148]) + 8 * 32 + sum(b[156:512]):
        return None
    name = b[0:100].split(b"\0")[0].decode()
    prefix = b[345:500].split(b"\0")[0].decode()
    size = int(b[124:136].split(b"\0")[0].strip() or b"0", 8)
    return (prefix + "/" + name if prefix else name), size


def headers_in(chunk, base):
    """Return [(abs_offset, path, size)] of headers found in a chunk starting at absolute offset base."""
    out = []
    start = (-base) % 512
    for o in range(start, len(chunk) - 511, 512):
        h = parse_header(chunk[o:o + 512])
        if h:
            out.append((base + o, h[0], h[1]))
    return out


def coarse_map(url, total, step=20_000_000, probe=600_000):
    offs = list(range(0, total, step))

    def one(o):
        hs = headers_in(rng(url, o, min(probe, total - o)), o)
        return hs[0] if hs else None

    with ThreadPoolExecutor(8) as ex:
        res = [r for r in ex.map(one, offs) if r]
    return sorted(res)


def walk(url, off, total, stop, chunk=4_000_000):
    """Yield (offset, path, size) from the header at off onwards until stop(path); reads 4 MB chunks so one
    request covers many members instead of one 512-byte request per member."""
    buf, base = b"", off
    while off < total:
        if off + 512 > base + len(buf):
            base = off
            buf = rng(url, off, min(chunk, total - off))
        h = parse_header(buf[off - base:off - base + 512])
        if h is None:
            break
        path, size = h
        if stop(path):
            break
        yield off, path, size
        off += 512 + ((size + 511) // 512) * 512


def locate(url, total, cmap, dirpath):
    """Find header offsets of all files directly in dirpath (directory assumed contiguous in the tar)."""
    before = [c for c in cmap if c[1] < dirpath]  # tar written in sorted order? verified by caller
    start = before[-1][0] if before else 0
    files, seen = {}, False
    for off, path, size in walk(url, start, total, lambda p, d=dirpath: False):
        if path.startswith(dirpath):
            seen = True
            files[path] = (off, size)
        elif seen:
            break
    return files


def fetch_files(url, items):
    def one(it):
        path, (off, size) = it
        return path, rng(url, off + 512, size)

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, items))


RAW.mkdir(parents=True, exist_ok=True)
tg = RAW / "vkitti_2.0.3_textgt.tar.gz"
if not tg.exists():
    tg.write_bytes(sess.get(f"{HF}/vkitti_2.0.3_textgt.tar.gz", timeout=600).content)
textgt = tarfile.open(tg)

urls = {k: f"{HF}/vkitti_2.0.3_{k}.tar" for k in ("rgb", "depth")}
totals = {k: size_of(u) for k, u in urls.items()}
cmaps = {}
for k in urls:
    cache = RAW / f"coarse_{k}.json"
    if cache.exists():
        cmaps[k] = [tuple(x) for x in json.loads(cache.read_text())]
    else:
        cmaps[k] = coarse_map(urls[k], totals[k])
        cache.write_text(json.dumps(cmaps[k]))
    paths = [c[1] for c in cmaps[k]]
    print(k, totals[k], len(cmaps[k]), "sorted:", paths == sorted(paths), paths[:2], flush=True)

for scene, var in CLIPS:
    ext = np.loadtxt(io.StringIO(textgt.extractfile(f"{scene}/{var}/extrinsic.txt").read().decode()), skiprows=1)
    intr = np.loadtxt(io.StringIO(textgt.extractfile(f"{scene}/{var}/intrinsic.txt").read().decode()), skiprows=1)
    ext, intr = ext[ext[:, 1] == 0], intr[intr[:, 1] == 0]  # Camera_0
    n = min(N, len(ext))
    prefix = {k: next(c[1] for c in cmaps[k] if "/" in c[1]).split("Scene")[0] for k in urls}
    rgb_dir = f"{prefix['rgb']}{scene}/{var}/frames/rgb/Camera_0/"
    dep_dir = f"{prefix['depth']}{scene}/{var}/frames/depth/Camera_0/"
    rgb_files = locate(urls["rgb"], totals["rgb"], cmaps["rgb"], rgb_dir)
    dep_files = locate(urls["depth"], totals["depth"], cmaps["depth"], dep_dir)
    want_r = {f"{rgb_dir}rgb_{i:05d}.jpg": rgb_files[f"{rgb_dir}rgb_{i:05d}.jpg"] for i in range(n)}
    want_d = {f"{dep_dir}depth_{i:05d}.png": dep_files[f"{dep_dir}depth_{i:05d}.png"] for i in range(n)}
    rb = fetch_files(urls["rgb"], list(want_r.items()))
    db = fetch_files(urls["depth"], list(want_d.items()))
    imgs = [cv2.imdecode(np.frombuffer(rb[f"{rgb_dir}rgb_{i:05d}.jpg"], np.uint8), cv2.IMREAD_COLOR) for i in range(n)]
    deps = []
    for i in range(n):
        d = cv2.imdecode(np.frombuffer(db[f"{dep_dir}depth_{i:05d}.png"], np.uint8), cv2.IMREAD_UNCHANGED)
        deps.append(np.where(d >= 65535, 0, d).astype(np.uint16))  # cm; 65535 = sky/far -> invalid
    c2w, Ks = [], []
    for i in range(n):
        w2c = ext[i, 2:].reshape(4, 4)
        c2w.append(np.linalg.inv(w2c))
        fx, fy, cx, cy = intr[i, 2:6]
        Ks.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    h, w = imgs[0].shape[:2]
    meta = {
        "source": f"Virtual KITTI 2.0.3 {scene}/{var} Camera_0 (via HF mirror kronecker0122/vkitti2, byte-identical tars)",
        "frame_ids": list(range(n)), "stride": 1, "fps_effective": 10.0, "hw": [h, w], "depth_type": "dense",
        "metric": True, "dynamic": "yes (moving cars)", "variation": var,
        "license": "CC BY-NC-SA 3.0 (Virtual KITTI 2)",
        "notes": "extrinsic.txt is world->camera (inverted to cam->world). Depth uint16 cm; 65535 (sky) set to 0. "
                 "vKITTI2 is in the training sets of several feed-forward models (VGGT, CUT3R, ...).",
    }
    write_clip(OUT, "vkitti2", f"vkitti2_{scene}_{var}", imgs, deps, c2w, Ks, meta, depth_scale=100.0)
    print(scene, var, n, h, w, flush=True)

clips = write_clips_json(OUT, "vkitti2")
chk = {c: reproj_check(OUT / "vkitti2" / c, max_depth=80) for c in ["vkitti2_Scene01_clone", "vkitti2_Scene20_clone"]}
write_check(OUT / "vkitti2", chk, "Moving cars are included in the check and inflate the error slightly.")
merge_index(OUT / "index.json", "vkitti2", {"group": "B_diag", "n_clips": len(clips),
                                            "n_frames": sum(c["n_frames"] for c in clips), "path": str(OUT / "vkitti2")})
print(json.dumps(chk, indent=1))
