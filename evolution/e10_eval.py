"""E10 evaluation: MapAnything (Apache-2.0 weights) image-only and K-gated, alone and with the v1.1 ViPE scale rule,
against the v1.x label (pi3x_kgate~vipe) and v0.

  python e10_eval.py A|B|scannetpp|wild -> outputs/evolution/E10/raw/<kind>.csv
Methods: vipe (default), vipe_kfda3 (slam.keyframe_depth=dav3, artifacts in BENCH/vipe_kfda3),
  pi3x_kgate~<v>: pi3x_kgate scaled by sqrt(k_depth * k_traj) w.r.t. ViPE variant <v> (v1.1 rule, E3b).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import run_hybrid as RH  # noqa: E402
from eval_vipe import umeyama  # noqa: E402

KIND = RH.KIND
M = RH.M
_orig = RH.ORIG


def load_vipe_dir(B, d, clip, n, H, W):
    o = B / d
    pz, iz = np.load(o / "pose" / f"{clip}.npz"), np.load(o / "intrinsics" / f"{clip}.npz")
    c2w = pz["data"][np.argsort(pz["inds"])]
    f = iz["data"][np.argsort(iz["inds"])]
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], 1
    dz = M.exr_zip(o / "depth" / f"{clip}.zip")
    D = np.stack([dz[k] for k in sorted(dz)])
    D, K = M.to_grid(D, K, H, W)
    return c2w, K, D, np.nan


def scale_to(c2w, D, cv, Dv):
    ks = np.linspace(0, len(D) - 1, 10).astype(int)
    a, b = Dv[ks], D[ks]
    mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    kd = float(np.median(a[mk]) / np.median(b[mk])) if mk.sum() > 1000 else 1.0
    pv, pm = cv[:, :3, 3], c2w[:, :3, 3]
    Lv = np.linalg.norm(np.diff(pv, axis=0), axis=1).sum()
    s, R, t = umeyama(pm, pv, True)
    res = np.sqrt(((s * pm @ R.T + t - pv) ** 2).sum(1).mean()) / max(Lv, 1e-9)
    dv = float(np.median(a[mk])) if mk.sum() > 1000 else np.inf
    kt = s if (Lv > 0.02 * dv and res < 0.2 and s > 0) else kd
    k = float(np.sqrt(kd * kt))
    c2w = c2w.copy()
    c2w[:, :3, 3] *= k
    return c2w, D * k


def orig(*a):
    if KIND == "scannetpp":  # eval_bench loader signature (m, clip, n)
        m, clip, n = a
        if "~" in m:
            base, src = m.split("~")
            c2w, K, D, sec = orig(base, clip, n)
            cv, _, Dv, _ = orig(src, clip, n)
            c2w, D = scale_to(c2w, D, cv, Dv)
            return c2w, K, D, sec
        return _orig(m, clip, n)
    return orig_b(*a)


def orig_b(B, m, clip, n, H, W):
    if m == "vipe_kfda3":
        return load_vipe_dir(B, m, clip, n, H, W)
    if "~" in m:
        base, src = m.split("~")
        c2w, K, D, sec = orig_b(B, base, clip, n, H, W)
        cv, _, Dv, _ = orig_b(B, src, clip, n, H, W)
        c2w, D = scale_to(c2w, D, cv, Dv)
        return c2w, K, D, sec
    return _orig(B, m, clip, n, H, W)


RH.ORIG = orig
RH.SPECS = ["vipe", "H:vipe+pi3x+med", "pi3x_kgate~vipe", "ma_img", "ma_kgate", "ma_img~vipe", "ma_kgate~vipe"]
RH.RES = Path("/data/home/z00533370/outputs/evolution/E10/raw")

if __name__ == "__main__":
    RH.main()
