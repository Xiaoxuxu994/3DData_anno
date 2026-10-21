"""Cross-model agreement and re-run stability (no GT).

  python qc_cross.py GROUP [GROUP ...] -> outputs/qc/cross_<G>.csv (method x clip), stab_<G>.csv (vipe / vggt_omega x clip)

Cross-model, per method m and clip, median over the partner set P (all others / BA family / feed-forward family):
  x{P}_traj   symmetric Sim(3) trajectory ATE as % of trajectory length
  x{P}_fov    |HFOV difference| (deg)
  x{P}_depth  depth AbsRel after one per-clip median scale (8 frames)
  xmetric_scale  median |log depth-scale ratio| vs the other metric methods (only for metric methods)
  clip-level: fov_spread (std of HFOV over methods), traj_spread_ff (median pairwise traj % within feed-forward)
Stability (ViPE "Shuttle"-style; vipe and vggt_omega re-run on outputs/qc/variants):
  rev_traj / sub_traj  Sim(3) ATE % between original and reversed / half-frame-rate run (frames re-indexed)
  rev_fov / sub_fov    |HFOV difference|; rev_scale / sub_scale  |log metric depth-scale ratio|; *_depth aligned AbsRel
"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/bench")
sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
from eval3d_bench import load_method  # noqa: E402
from eval_vipe import umeyama  # noqa: E402
from qc_common import BA_CLASS, BENCH, CACHE, METHODS, METRIC, QC, VAR, clips, vipe_static_mask  # noqa: E402


def traj_pct(pa, pb):
    La, Lb = np.linalg.norm(np.diff(pa, axis=0), axis=1).sum(), np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()
    if min(La, Lb) < 1e-9 or not (np.isfinite(pa).all() and np.isfinite(pb).all()):
        return np.nan
    out = []
    for x, y, L in ((pa, pb, Lb), (pb, pa, La)):
        s, R, t = umeyama(x, y, True)
        out.append(100 * np.sqrt(((s * x @ R.T + t - y) ** 2).sum(1).mean()) / L)
    return float(np.mean(out))


def hfov(K, w):
    return float(np.degrees(2 * np.arctan(w / 2 / K[:, 0, 0])).mean())


def depth_cmp(Da, Db, static=None):
    m = np.isfinite(Da) & np.isfinite(Db) & (Da > 0) & (Db > 0)
    if static is not None:  # drop sky / dynamic pixels (ViPE masks): sky depth is arbitrary per method
        m &= static
    if m.sum() < 1000:
        return np.nan, np.nan
    ratio = np.median(Db[m]) / np.median(Da[m])
    return float(np.mean(np.abs(Da[m] * ratio - Db[m]) / Db[m])), float(abs(np.log(ratio)))


def cross_clip(args):
    G, c = args
    clip, n = c["clip"], c["n_frames"]
    hw = tuple(int(x) for x in np.load(CACHE / G / f"{clip}.npz")["hw"])
    ks = np.linspace(0, n - 1, 8).astype(int)
    lab = vipe_static_mask(BENCH[G], clip, n, hw)
    static = None if lab is None else lab[ks] == 0
    data = {}
    for m in METHODS:
        try:
            T, K, D, _ = load_method(BENCH[G], m, clip, n, *hw)
            if np.isfinite(T).all() and np.isfinite(K).all():
                data[m] = (T[:, :3, 3], hfov(K, hw[1]), D[ks])
        except Exception:  # noqa: BLE001
            pass
    pair = {}
    for a, b in itertools.combinations(sorted(data), 2):
        d, s = depth_cmp(data[a][2], data[b][2], static)
        pair[(a, b)] = pair[(b, a)] = {"traj": traj_pct(data[a][0], data[b][0]), "fov": abs(data[a][1] - data[b][1]),
                                       "depth": d, "scale": s}
    ff = [m for m in data if m not in BA_CLASS]
    clip_row = {"fov_spread": float(np.std([v[1] for v in data.values()])) if data else np.nan,
                "traj_spread_ff": float(np.nanmedian([pair[(a, b)]["traj"] for a, b in itertools.combinations(ff, 2)]))
                if len(ff) > 1 else np.nan}
    rows = []
    for m in data:
        r = {"group": G, "method": m, "clip": clip, **clip_row}
        for tag, P in (("all", [o for o in data if o != m]), ("ba", [o for o in data if o != m and o in BA_CLASS]),
                       ("ff", [o for o in data if o != m and o not in BA_CLASS])):
            for k in ("traj", "fov", "depth"):
                v = [pair[(m, o)][k] for o in P]
                r[f"x{tag}_{k}"] = float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else np.nan
        if m in METRIC:
            v = [pair[(m, o)]["scale"] for o in data if o != m and o in METRIC]
            r["xmetric_scale"] = float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else np.nan
        rows.append(r)
    return rows


def stab_clip(args):
    G, c, m = args
    clip, n = c["clip"], c["n_frames"]
    hw = tuple(int(x) for x in np.load(CACHE / G / f"{clip}.npz")["hw"])
    r = {"group": G, "method": m, "clip": clip}
    try:
        T, K, D, _ = load_method(BENCH[G], m, clip, n, *hw)
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
        return r
    for var, idx in (("rev", np.arange(n)[::-1]), ("sub", np.arange(0, n, 2))):
        try:
            Tv, Kv, Dv, _ = load_method(VAR / f"{G}_{var}2" if var == "sub" else VAR / f"{G}_{var}", m, clip, len(idx), *hw)
        except Exception as e:  # noqa: BLE001
            r[f"{var}_error"] = str(e)[:200]
            continue
        # variant frame k corresponds to original frame idx[k]; re-order to ascending original index
        order = np.argsort(idx)
        Tv, Kv, Dv, oi = Tv[order], Kv[order], Dv[order], idx[order]
        r[f"{var}_traj"] = traj_pct(T[oi, :3, 3], Tv[:, :3, 3])
        r[f"{var}_fov"] = abs(hfov(K[oi], hw[1]) - hfov(Kv, hw[1]))
        ks = np.linspace(0, len(oi) - 1, 8).astype(int)
        d, s = depth_cmp(D[oi][ks], Dv[ks])
        r[f"{var}_depth"], r[f"{var}_scale"] = d, s
    return r


if __name__ == "__main__":
    for G in sys.argv[1:] or ["A", "B", "C"]:
        cl = clips(G)
        with ProcessPoolExecutor(24) as ex:
            rows = [r for rs in ex.map(cross_clip, [(G, c) for c in cl]) for r in rs]
            pd.DataFrame(rows).to_csv(QC / f"cross_{G}.csv", index=False, float_format="%.5f")
            if "--stab" in sys.argv or True:
                st = list(ex.map(stab_clip, [(G, c, m) for c in cl for m in ("vipe", "vggt_omega")]))
                pd.DataFrame(st).to_csv(QC / f"stab_{G}.csv", index=False, float_format="%.5f")
        print(G, len(rows), flush=True)
