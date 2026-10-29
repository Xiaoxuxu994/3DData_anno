"""witness_standin.py S1_ROOT POOL_DIR JOBS_DIR OUT_DIR CLIP [CLIP ...] - STAND-IN witnesses for glr_collect.py --witness-dir (E15 sandbox demo).

The real witnesses are ViPE (metric trajectory + depth), DA3 conditioned on ViPE's cameras and the image-only K sources, produced by
run_batch.sh on a GPU host. Here, from the silver reference reconstruction S1_ROOT/<clip>/ref (exhaustive matching, SIMPLE_RADIAL, incremental
mapper, every 6th frame of the 15 fps clip) and the filter's GeoCalib scores in POOL_DIR/<clip>/scores.json:
  c2w, K   reference cameras interpolated to every frame of the clip (SLERP + linear), K scaled to the witness grid
  D        per frame the median depth of the reference points seen from the interpolated camera, constant over a 36 x 64 grid (the "ViPE" depth)
  da3_D    per pixel depth from the reference points projected into the interpolated camera (linear / nearest interpolation), NaN on frames the
           job does not use (the "DA3" depth)
  others_med   GeoCalib's median HFOV over 12 frames (one K source instead of the median of three)
  vipe_focal_cv = 0
The reference is a second SfM reconstruction of the SAME frames, not an independent annotator and not metric: it demonstrates the code path
on real clips and measures how well two differently configured SfM runs agree, it does not validate the QC.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import glr_common as C  # noqa: E402
import glr_stage1_gate as G  # noqa: E402

GRID = (36, 64)


def ref_frames(model, stride):
    cams = G.load_cameras(model)
    f = np.array([(int("".join(ch for ch in Path(n).stem if ch.isdigit())) - 1) * stride for n in cams["names"]])
    return cams, f


def interp_poses(c2w, frames, n):
    """c2w (m,4,4) at integer frames -> (n,4,4) at 0..n-1 (SLERP + linear, clamped outside the covered range)."""
    allf = np.arange(n)
    fq = np.clip(allf, frames[0], frames[-1])
    R = Slerp(frames, Rotation.from_matrix(c2w[:, :3, :3]))(fq).as_matrix()
    T = np.stack([np.interp(fq, frames, c2w[:, i, 3]) for i in range(3)], 1)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3], out[:, :3, 3] = R, T
    return out


def depth_from_points(P, c2w, K, hw, W, H):
    """Depth of 3D points P (m,3) seen by camera (c2w, K of a W x H image) on a grid hw; None when fewer than 20 points are in front and inside."""
    w2c = np.linalg.inv(c2w)
    Xc = P @ w2c[:3, :3].T + w2c[:3, 3]
    ok = Xc[:, 2] > 0.01
    uv = (Xc[ok] @ K.T)[:, :2] / Xc[ok][:, 2:3]
    z = Xc[ok][:, 2]
    ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, z = uv[ins], z[ins]
    if len(z) < 20:
        return None
    uv = uv / np.array([W / hw[1], H / hw[0]])
    gx, gy = np.meshgrid(np.arange(hw[1]) + 0.5, np.arange(hw[0]) + 0.5)
    lin = LinearNDInterpolator(uv, z)(gx, gy)
    return np.where(np.isfinite(lin), lin, NearestNDInterpolator(uv, z)(gx, gy)).astype(np.float32)


def main(s1_root, pool, jobs, out, clips):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        job = json.loads((Path(jobs) / clip / "job.json").read_text())
        n = job["n_orig"]
        ref_dir = Path(s1_root) / clip / "ref"
        ref_res = json.loads((ref_dir / "stage1.json").read_text())
        stride = int(ref_res["args"]["stride"])
        mdir = Path(ref_res.get("model_dir") or ref_dir / "sparse" / "0")     # the largest model of the reference run (not always sparse/0)
        if not mdir.is_dir():
            mdir = ref_dir / "sparse" / mdir.name
        cams, fr = ref_frames(mdir, stride)
        rec = pycolmap.Reconstruction(str(mdir))
        P = np.array([p.xyz for p in rec.points3D.values()])
        c2w = interp_poses(cams["c2w"], fr, n)
        W, H = cams["W"], cams["H"]
        K0 = np.median(cams["K"], axis=0)
        Kg = K0.copy()
        Kg[0, :] *= GRID[1] / W
        Kg[1, :] *= GRID[0] / H
        D = np.full((n, *GRID), np.nan, np.float32)
        da3 = np.full((n, *GRID), np.nan, np.float32)
        used = set(job["frame_idx"])
        for f in range(n):
            d = depth_from_points(P, c2w[f], K0, GRID, W, H)
            if d is None:
                continue
            D[f] = np.nanmedian(d)
            if f in used:
                da3[f] = d
        s = json.loads((Path(pool) / clip / "scores.json").read_text())
        vf = float(np.median(s["calib"]["vfov_deg"]))
        hfov = float(np.degrees(2 * np.arctan(np.tan(np.radians(vf) / 2) * W / H)))
        np.savez(out / f"{clip}.npz", c2w=c2w, K=np.tile(Kg, (n, 1, 1)), D=D, da3_D=da3, others_med=hfov, vipe_focal_cv=0.0)
        print(f"{clip}: ref frames {len(fr)} (stride {stride}), points {len(P)}, GeoCalib HFOV {hfov:.1f}, ref HFOV {C.hfov_deg(float(K0[0, 0]), W):.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
