"""dense_standin.py S1_DIR OUT_DIR - STAND-IN for GLRMap's dense stage, used only to exercise the GLR track's plumbing on a machine without
GPU and without the GLRMap code (E15 sandbox demo). Nothing here is RoMa, Pi3X or PDA.

For every registered image of the sparse model in S1_DIR/sparse/0 it projects the model's 3D points into the image and interpolates their depth
(linear inside the convex hull of the points, nearest outside) on an (H/8, W/8) grid -> OUT_DIR/pda/fused/<image number>.npz with key `fine`
(and `coarse` = the same), the file layout glr_collect.read_depth_maps expects. Images with fewer than 20 points are skipped (PDA skips frames
with fewer than 5 anchors the same way). The depth is in the SfM scale of the model, like GLRMap's.
Usage (as GLR_DENSE_CMD):   {py} evolution/E15_glr/dense_standin.py {s1} {out}/dense
"""
import sys
from pathlib import Path

import numpy as np
import pycolmap
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import glr_stage1_gate as G  # noqa: E402


def depth_map(rec, im, grid=8, min_pts=20):
    """-> (h, w) float32 depth map of one registered image from the model's 3D points, or None."""
    cam = rec.cameras[im.camera_id]
    W, H = int(cam.width), int(cam.height)
    R, t = G._rt(im)
    uv, z = [], []
    for p2 in im.points2D:
        if p2.has_point3D():
            X = np.asarray(rec.points3D[p2.point3D_id].xyz, np.float64)
            zc = float((R @ X + t)[2])
            if zc > 0:
                uv.append(np.asarray(p2.xy, np.float64) / grid)
                z.append(zc)
    if len(z) < min_pts:
        return None
    uv, z = np.array(uv), np.array(z)
    h, w = max(H // grid, 2), max(W // grid, 2)
    gx, gy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    lin = LinearNDInterpolator(uv, z)(gx, gy)
    near = NearestNDInterpolator(uv, z)(gx, gy)
    return np.where(np.isfinite(lin), lin, near).astype(np.float32)


def main(s1, out):
    model = sorted(Path(s1, "sparse").glob("*/cameras.*"), key=lambda p: (len(str(p)), str(p)))[0].parent
    rec = pycolmap.Reconstruction(str(model))
    fused = Path(out) / "pda" / "fused"
    fused.mkdir(parents=True)
    n = 0
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        d = depth_map(rec, im)
        if d is None:
            continue
        digits = "".join(ch for ch in Path(im.name).stem if ch.isdigit())
        np.savez(fused / f"{digits}.npz", fine=d, coarse=d)
        n += 1
    print(f"dense stand-in: {n}/{len(rec.images)} depth maps -> {fused}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
