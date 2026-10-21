"""E1b: use the K-conditioned Pi3X as the main annotator (its own poses + depth) and borrow only ViPE's metric scale.

  python e1b_eval.py A|B|scannetpp|wild -> outputs/evolution/E1b/raw/<kind>.csv
Pseudo-methods (built on the fly from existing outputs, no GPU):
  <m>~vipe   : m's cameras and depth, translation and depth multiplied by s = Sim(3) scale mapping m's camera centres
               onto ViPE's (ViPE is metric); falls back to median depth ratio when ViPE barely moves (< 2% of its
               median depth) or the trajectories disagree (aligned residual > 20% of ViPE's path).
  route_h    : ViPE K implausible (E1a gate) -> pi3x_kgate (own metric); else H:vipe+pi3x_kvipe+med
  route_s    : ViPE K implausible -> pi3x_kgate; else pi3x_kvipe~vipe
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import hybrid as HY  # noqa: E402
import run_hybrid as RH  # noqa: E402
from eval_vipe import umeyama  # noqa: E402

KIND = RH.KIND
ROOT = Path("/data/home/z00533370")
BENCHNAME = {"A": "eval3d_A", "B": "eval3d_B", "scannetpp": "scannetpp", "wild": "wild"}[KIND]
SWITCH = {}
for r in csv.DictReader(open(ROOT / "outputs/evolution/E1/fov_sources.csv")):
    if r["bench"] == BENCHNAME:
        v = float(r["hfov_vipe"]) if r["hfov_vipe"] else np.nan
        SWITCH[r["clip"]] = not (25 <= v <= 130)
_orig = RH.ORIG
POS = (1, 2) if KIND in ("A", "B", "wild") else (0, 1)  # (method arg index, clip arg index)


def with_method(a, m):
    b = list(a)
    b[POS[0]] = m
    return b


def to_vipe_scale(c2w, D, cv, Dv):
    pv, pm = cv[:, :3, 3], c2w[:, :3, 3]
    Lv = np.linalg.norm(np.diff(pv, axis=0), axis=1).sum()
    ks = np.linspace(0, len(D) - 1, 10).astype(int)
    a, b = Dv[ks], D[ks]
    mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    r_dep = float(np.median(a[mk]) / np.median(b[mk])) if mk.sum() > 1000 else 1.0
    s, R, t = umeyama(pm, pv, True)
    res = np.sqrt(((s * pm @ R.T + t - pv) ** 2).sum(1).mean()) / max(Lv, 1e-9)
    med_depth_v = float(np.median(a[mk])) if mk.sum() > 1000 else np.inf
    use_traj = Lv > 0.02 * med_depth_v and res < 0.2 and np.isfinite(s) and s > 0
    k = s if use_traj else r_dep
    c2w = c2w.copy()
    c2w[:, :3, 3] *= k
    return c2w, D * k


def orig(*a):
    m, clip = a[POS[0]], a[POS[1]]
    if m.endswith("~vipe"):
        c2w, K, D, sec = _orig(*with_method(a, m[:-5]))
        cv, _, Dv, _ = _orig(*with_method(a, "vipe"))
        c2w, D = to_vipe_scale(c2w, D, cv, Dv)
        return c2w, K, D, sec
    if m in ("route_h", "route_s"):
        if SWITCH.get(clip, False):
            return _orig(*with_method(a, "pi3x_kgate"))
        if m == "route_s":
            return orig(*with_method(a, "pi3x_kvipe~vipe"))
        base = lambda mm: orig(*with_method(a, mm))  # noqa: E731
        return HY.compose("H:vipe+pi3x_kvipe+med", base)[:4]
    return _orig(*a)


RH.ORIG = orig
RH.SPECS = ["vipe", "pi3x", "pi3x_kvipe", "pi3x_kgate", "H:vipe+pi3x+med", "H:vipe+pi3x_kvipe+med",
            "pi3x_kvipe~vipe", "pi3x_kgate~vipe", "pi3x_pkvipe~vipe", "da3_pkvipe", "route_h", "route_s"]
RH.RES = ROOT / "outputs/evolution/E1b/raw"

if __name__ == "__main__":
    RH.main()
