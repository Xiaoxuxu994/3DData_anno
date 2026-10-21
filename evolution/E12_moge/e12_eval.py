"""E12 evaluation: MoGe-2 / MoGe-3 as depth model, as metric-scale source and as depth source in v1.3.

  python e12_eval.py A|B|scannetpp|wild  -> outputs/evolution/E12/raw/<kind>.csv   (needs scales_<kind>.csv)
Methods (run_hybrid machinery, same metrics / failure flags as eval3d_bench and E3b / E10):
  vipe, pi3x_kgate (raw), da3 (raw)                          single annotators
  pi3x_kgate@v13                                             v1.3 label (= E10 pi3x_kgate~vipe)
  moge_<m>_<mode>                                            raw MoGe depth / K; cameras are Pi3X's (pose cols meaningless)
  pi3x_kgate@<src>                                           v1.3 base with metric scale k_<src> from scales_<kind>.csv:
        moge_*   MoGe alone (median per-frame depth ratio)
        da3      DA3 depth ratio
        med3_<m> median(k_v13, k_<m>, k_da3)
        gm_<m>   sqrt(k_v13 * k_<m>)
  D:<m>   v1.3 cameras + gated K, depth = MoGe depth x one per-clip factor (median per-frame ratio to v1.3 depth)
  Df:<m>  same, per-frame factor (removes MoGe's frame-to-frame scale flicker)
  VD:<m>  ViPE cameras + gated K, depth = MoGe depth x per-clip factor to ViPE depth (Pi3X-free, licence-clean path)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import run_hybrid as RH  # noqa: E402

KIND = RH.KIND
E12 = Path("/data/home/z00533370/outputs/evolution/E12")
SC = pd.read_csv(E12 / f"scales_{KIND}.csv").set_index("clip")
_orig = RH.ORIG
BASE = "pi3x_kgate"
MOGE = [f"moge_{m}_{v}" for m in ("m2l", "m3l", "m3g") for v in ("free", "kgate")]
VOTERS = ["moge_m3l_kgate", "moge_m3g_kgate"]
DSRC = ["moge_m2l_kgate", "moge_m3l_kgate", "moge_m3g_kgate"]


def kval(clip, src):
    if clip not in SC.index:
        return 1.0
    row = SC.loc[clip]
    g = lambda c: float(row.get(f"k_{c}", np.nan))  # noqa: E731
    if src.startswith("med3_"):
        v = np.nanmedian([g("v13"), g(src[5:]), g("da3")])
    elif src.startswith("gm_"):
        v = np.sqrt(g("v13") * g(src[3:]))
    else:
        v = g(src)
    return float(v) if np.isfinite(v) and v > 0 else (g("v13") if np.isfinite(g("v13")) else 1.0)


def frame_ratio(Da, Db, ks):
    out = []
    for k in ks:
        a, b = Da[k], Db[k]
        mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        out.append(np.median(a[mk]) / np.median(b[mk]) if mk.sum() > 1000 else np.nan)
    return np.array(out)


def resolve(m, clip, load):
    if m.startswith(f"{BASE}@"):
        c2w, K, D, sec = load(BASE)
        k = kval(clip, m.split("@")[1])
        c2w = c2w.copy()
        c2w[:, :3, 3] *= k
        return c2w, K, D * k, sec
    if m.startswith(("D:", "Df:", "VD:")):
        tag, src = m.split(":")
        _, Kg, _, _ = load(BASE)
        if tag == "VD":
            c2w, _, Dref, _ = load("vipe")
        else:
            c2w, _, Dref, _ = resolve(f"{BASE}@v13", clip, load)
        _, _, Dm, sec = load(src)
        n = len(Dm)
        if tag == "Df":
            a = frame_ratio(Dref, Dm, range(n))
            a = np.where(np.isfinite(a), a, np.nanmedian(a))
            D = Dm * a[:, None, None].astype(np.float32)
        else:
            ks = np.unique(np.linspace(0, n - 1, min(40, n)).astype(int))
            D = Dm * np.float32(np.nanmedian(frame_ratio(Dref, Dm, ks)))
        return c2w, Kg, D, sec
    return load(m)


_CACHE = {}


def cached(clip, fn):
    """Per-process cache of raw loads for the current clip (each spec re-uses pi3x_kgate / vipe / MoGe arrays)."""
    if _CACHE.get("clip") != clip:
        _CACHE.clear()
        _CACHE["clip"] = clip

    def load(x):
        if x not in _CACHE:
            _CACHE[x] = fn(x)
        return tuple(v.copy() if isinstance(v, np.ndarray) else v for v in _CACHE[x])
    return load


def orig(*a):
    if KIND == "scannetpp":
        m, clip, n = a
        return resolve(m, clip, cached(clip, lambda x: _orig(x, clip, n)))
    B, m, clip, n, H, W = a
    return resolve(m, clip, cached(clip, lambda x: _orig(B, x, clip, n, H, W)))


RH.ORIG = orig
RH.SPECS = (["vipe", BASE, "da3", f"{BASE}@v13", f"{BASE}@da3"] + MOGE + [f"{BASE}@{m}" for m in MOGE]
            + [f"{BASE}@{v}_{m}" for v in ("med3", "gm") for m in VOTERS]
            + [f"{t}:{m}" for t in ("D", "Df", "VD") for m in DSRC])
RH.RES = E12 / "raw"

if __name__ == "__main__":
    RH.main()
