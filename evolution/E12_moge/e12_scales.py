"""E12: per-clip metric-scale sources for the v1.3 base (pi3x_kgate), incl. MoGe-2 / MoGe-3 (no GT used to build them).

  python e12_scales.py A|B|scannetpp|wild  -> outputs/evolution/E12/scales_<kind>.csv
k_<src> multiplies pi3x_kgate translation and depth (same convention as E3):
  k_vd, k_vt, k_v13 : ViPE depth ratio, ViPE trajectory Sim3 scale, v1.3 rule sqrt(k_vd * k_vt) (copy of
                      e10_eval.scale_to: k_vt falls back to k_vd when ViPE's path is < 2% of depth or residual > 20%)
  k_da3             : median(D_da3) / median(D_pi3x) over co-valid pixels of 10 frames (E3 r_da3)
  k_<moge>          : median over <= 40 frames of the per-frame ratio median(D_moge) / median(D_pi3x)
  cv_<moge>         : std of log per-frame ratios (temporal scale flicker of single-image MoGe)
  vm_<moge>         : same per-frame median ratio but against ViPE depth (MoGe applied directly to ViPE)
  hfov_<moge>       : MoGe's own median HFOV (free mode)
GT columns (A / B / scannetpp): metric, s_pi3x / s_vipe / s_da3 = Sim3 scale of raw cameras onto GT
  (GT ~ s * pred, so scaling pred by k gives s/k), gtd = median over 10 frames of median(GT) / median(D_pi3x).
"""

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
KIND = sys.argv[1]
sys.argv = sys.argv[:2]
import run_hybrid as RH  # noqa: E402
from eval_vipe import umeyama  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("/data/home/z00533370/outputs/evolution/E12")
BASE = "pi3x_kgate"
MOGE = [f"moge_{m}_{v}" for m in ("m2l", "m3l", "m3g") for v in ("free", "kgate")]


def loader(c):
    clip, n = c["clip"], c["n_frames"]
    if KIND in ("A", "B"):
        g = np.load(RH.BENCH / "gt" / f"{clip}.npz", allow_pickle=True)["depth"]
        H, W = g.shape[1:]
        return lambda m: RH.ORIG(RH.BENCH, m, clip, n, H, W)
    if KIND == "scannetpp":
        return lambda m: RH.ORIG(m, clip, n)
    H, W = RH.M.grid(c)
    return lambda m: RH.ORIG(RH.M.B, m, clip, n, H, W)


def ratio(a, b):
    mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return (float(np.median(a[mk]) / np.median(b[mk])) if mk.sum() > 1000 else np.nan), mk


def frame_ratios(Da, Db, ks):
    r = np.array([ratio(Da[k], Db[k])[0] for k in ks])
    return r[np.isfinite(r) & (r > 0)]


def one(c):
    load = loader(c)
    clip, n = c["clip"], c["n_frames"]
    r = {"clip": clip, "dataset": c.get("dataset", c.get("cat", c.get("scene", "")))}
    if KIND == "scannetpp":
        r["scene"] = c["scene"]
    try:
        cb, _, Db, _ = load(BASE)
        cv, _, Dv, _ = load("vipe")
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:150]
        return r
    ks10 = np.linspace(0, n - 1, 10).astype(int)
    ks40 = np.unique(np.linspace(0, n - 1, min(40, n)).astype(int))
    # v1.3 rule (identical to e10_eval.scale_to)
    a, b = Dv[ks10], Db[ks10]
    kd, mk = ratio(a, b)
    kd = kd if np.isfinite(kd) else 1.0
    pv, pm = cv[:, :3, 3], cb[:, :3, 3]
    Lv = np.linalg.norm(np.diff(pv, axis=0), axis=1).sum()
    s, R, t = umeyama(pm, pv, True)
    res = np.sqrt(((s * pm @ R.T + t - pv) ** 2).sum(1).mean()) / max(Lv, 1e-9)
    dv = float(np.median(a[mk])) if mk.sum() > 1000 else np.inf
    kt = s if (Lv > 0.02 * dv and res < 0.2 and s > 0) else kd
    vt_ok = bool(Lv > 0.02 * dv and res < 0.2 and s > 0)
    r.update(k_vd=kd, k_vt=kt, k_v13=float(np.sqrt(kd * kt)), vt_valid=float(vt_ok))
    try:
        cd, _, Dd, _ = load("da3")
        r["k_da3"] = ratio(Dd[ks10], Db[ks10])[0]
    except Exception:  # noqa: BLE001
        cd = None
        r["k_da3"] = np.nan
    W = Db.shape[2]
    for m in MOGE:
        try:
            _, Km, Dm, sec = load(m)
        except Exception:  # noqa: BLE001
            continue
        fr = frame_ratios(Dm, Db, ks40)
        fv = frame_ratios(Dm, Dv, ks40)
        r[f"k_{m}"] = float(np.median(fr)) if len(fr) else np.nan
        r[f"cv_{m}"] = float(np.std(np.log(fr))) if len(fr) > 2 else np.nan
        r[f"vm_{m}"] = float(np.median(fv)) if len(fv) else np.nan
        r[f"hfov_{m}"] = float(np.degrees(2 * np.arctan(W / 2 / np.median(Km[:, 0, 0]))))
        r[f"sec_{m}"] = sec
    if KIND in ("A", "B", "scannetpp"):
        T_gt, _, GD, metric, _ = RH.gt_eval3d(c) if KIND != "scannetpp" else RH.gt_scannetpp(c)
        r["metric"] = float(metric)
        g = T_gt[:, :3, 3]
        r["traj_len_gt"] = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
        r["s_pi3x"] = float(umeyama(pm, g, True)[0])
        r["s_vipe"] = float(umeyama(pv, g, True)[0])
        if cd is not None:
            r["s_da3"] = float(umeyama(cd[:, :3, 3], g, True)[0])
        GDn = np.where((GD > 0) & (GD < RH.MAX_DEPTH), GD, np.nan)
        fr = frame_ratios(GDn, Db, ks10)
        r["gtd"] = float(np.median(fr)) if len(fr) else np.nan
    return r


def main():
    clips = json.loads((RH.BENCH / "clips.json").read_text())
    clips = [c for c in clips if c.get("dataset") != "eth3d_mvs"]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(one, clips))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"scales_{KIND}.csv", index=False, float_format="%.5g")
    print(KIND, len(df), df.filter(like="k_").notna().sum().to_dict(), flush=True)


if __name__ == "__main__":
    main()
