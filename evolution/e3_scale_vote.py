"""E3: metric-scale voting for the K-conditioned Pi3X annotation (no GT used).

python e3_scale_vote.py A|B|scannetpp|wild [BASE] -> outputs/evolution/E3/scales_<kind>.csv   (BASE default pi3x_kgate)
Per clip, scale factors that would bring BASE (cameras + depth) to another metric annotator m:
  s_<m> : trajectory scale, Sim(3) Umeyama of BASE camera centres onto m's (x_m ~ s R x_base + t); NaN when m's path
          is < 2% of the scene depth or the aligned residual is > 20% of m's path (no usable trajectory evidence)
  r_<m> : depth scale, median(D_m) / median(D_base) over co-valid pixels of 10 frames
Votes (multiply BASE translation and depth by k):
  own=1 | vipe_t=s_vipe | vipe_d=r_vipe | da3_d=r_da3 | med3_t=median(1, s_vipe, s_da3) | med3_d=median(1, r_vipe, r_da3)
  med5=median(1, s_vipe, r_vipe, s_da3, r_da3) | md_vipe_da3=median(r_vipe, r_da3, s_vipe)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
BASE = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "pi3x_kgate"
sys.argv = sys.argv[:2]
import run_hybrid as RH  # noqa: E402

warnings.filterwarnings("ignore")
from eval_vipe import umeyama  # noqa: E402

KIND = RH.KIND
OUT = Path("/data/home/z00533370/outputs/evolution/E3")
OTHERS = ["vipe", "da3", "pi3x"]


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


def one(c):
    load = loader(c)
    r = {"clip": c["clip"], "dataset": c.get("dataset", c.get("cat", "scannetpp"))}
    try:
        cb, _, Db, _ = load(BASE)
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:150]
        return r
    ks = np.linspace(0, len(Db) - 1, 10).astype(int)
    pb = cb[:, :3, 3]
    for m in OTHERS:
        try:
            cm, _, Dm, _ = load(m)
        except Exception:  # noqa: BLE001
            continue
        a, b = Dm[ks], Db[ks]
        mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        r[f"r_{m}"] = float(np.median(a[mk]) / np.median(b[mk])) if mk.sum() > 1000 else np.nan
        pm = cm[:, :3, 3]
        Lm = np.linalg.norm(np.diff(pm, axis=0), axis=1).sum()
        dm = float(np.median(a[mk])) if mk.sum() > 1000 else np.inf
        s, R, t = umeyama(pb, pm, True)
        res = np.sqrt(((s * pb @ R.T + t - pm) ** 2).sum(1).mean()) / max(Lm, 1e-9)
        r[f"res_{m}"], r[f"move_{m}"] = res, Lm / dm
        r[f"s_{m}"] = s if (Lm > 0.02 * dm and res < 0.2 and s > 0) else np.nan
    return r


def votes(df):
    def med(cols, one=True):
        X = [df[c] if c in df else pd.Series(np.nan, index=df.index) for c in cols]
        if one:
            X = [pd.Series(1.0, index=df.index)] + X
        v = np.nanmedian(np.column_stack(X), 1)
        return np.where(np.isfinite(v), v, 1.0)
    df["k_own"] = 1.0
    df["k_vipe_t"] = df["s_vipe"].fillna(df["r_vipe"]).fillna(1.0)
    df["k_vipe_d"] = df["r_vipe"].fillna(1.0)
    df["k_da3_d"] = df["r_da3"].fillna(1.0)
    df["k_med3_t"] = med(["s_vipe", "s_da3"])
    df["k_med3_d"] = med(["r_vipe", "r_da3"])
    df["k_med5"] = med(["s_vipe", "r_vipe", "s_da3", "r_da3"])
    df["k_md_vipe_da3"] = med(["r_vipe", "r_da3", "s_vipe"], one=False)
    df["k_vipe_td"] = np.sqrt(df["k_vipe_t"] * df["k_vipe_d"])  # E3b: compromise between ViPE depth and trajectory scale
    return df


if __name__ == "__main__":
    import json
    from concurrent.futures import ProcessPoolExecutor
    clips = json.loads((RH.BENCH / "clips.json").read_text())
    clips = [c for c in clips if c.get("dataset") != "eth3d_mvs"]
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(one, clips))
    df = votes(pd.DataFrame(rows))
    df["base"] = BASE
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"scales_{KIND}.csv", index=False, float_format="%.5g")
    print(KIND, len(df))
    print(df.filter(like="k_").describe().round(3).to_string())
