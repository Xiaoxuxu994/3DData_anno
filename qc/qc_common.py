"""Shared helpers for the line-G label self-check (QC) signals."""
import json
from pathlib import Path

import cv2
import numpy as np

H0 = Path("/data/home/z00533370")
BENCH = {"A": H0 / "outputs/bench/eval3d_A", "B": H0 / "outputs/bench/eval3d_B", "C": H0 / "outputs/bench/wild"}
VAR = H0 / "outputs/qc/variants"
QC = H0 / "outputs/qc"
CACHE = QC / "cache"
METHODS = ["vipe", "spatialvid_metric", "voyager", "vggt", "vggt_omega", "da3", "pi3x"]
BA_CLASS = {"vipe", "spatialvid_metric"}  # optimisation (bundle adjustment) family
METRIC = {"vipe", "spatialvid_metric", "voyager", "da3", "pi3x"}  # methods that claim metric scale
LONG = 512  # QC grid: long side 512, both sides multiples of 8 (RAFT)
FLOW_GAPS = (1, 5)
MATCH_GAPS = (5, 20)
CONS_GAPS = (1, 5, 20)
N_ANCHORS = 24


def clips(g):
    return json.loads((BENCH[g] / "clips.json").read_text())


def frame_hw(g, clip):
    im = cv2.imread(str(BENCH[g] / "frames" / clip / "000000.jpg"))
    return im.shape[:2]


def qc_grid(g, clip):
    h, w = frame_hw(g, clip)
    s = LONG / max(h, w)
    return max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)


def rgb(g, clip, k, hw):
    im = cv2.imread(str(BENCH[g] / "frames" / clip / f"{k:06d}.jpg"))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def anchors(n, gap):
    """Anchor frames i (with i + gap < n) shared by all signals so that flow / matches are cached once per clip."""
    if n <= gap:
        return np.array([], int)
    return np.unique(np.linspace(0, n - 1 - gap, min(N_ANCHORS, n - gap)).astype(int))


def vipe_static_mask(bench_dir, clip, n, hw):
    """(n,h,w) uint8: 0 background/static, 1 dynamic instance, 2 sky (ViPE GDINO+SAM masks); None if missing."""
    import zipfile
    zp, tp = bench_dir / "vipe/mask" / f"{clip}.zip", bench_dir / "vipe/mask" / f"{clip}.txt"
    if not zp.exists():
        return None
    names = {}
    for line in tp.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            names[int(k)] = v.strip()
    out = np.zeros((n, *hw), np.uint8)
    with zipfile.ZipFile(zp) as z:
        for f in z.namelist():
            k = int(f.split(".")[0])
            if k >= n:
                continue
            m = cv2.imdecode(np.frombuffer(z.read(f), np.uint8), cv2.IMREAD_UNCHANGED)
            m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            lab = np.zeros_like(out[k])
            for i in np.unique(m):
                if i == 0:
                    continue
                lab[m == i] = 2 if names.get(int(i), "") == "sky" else 1
            out[k] = lab
    return out
