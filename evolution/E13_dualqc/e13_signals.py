"""E13 gate 2: model-independent geometric self-check of a label (its own pose, K, depth + the video only).
Re-uses code/qc/qc_signals.signals (Sampson vs LightGlue, induced flow vs RAFT, multi-gap self-consistency with
fb-occlusion check, free-space violation) with a loader that also builds the composite labels:
  v0  = H:vipe+pi3x+med (ViPE pose/K + Pi3X depth, median scale)
  v13 = pi3x_kgate scaled by k_vipe_td (Pi3X conditioned on gated K, sqrt(ViPE depth x traj ratio) scale)
  fs<s> = focal-sweep re-solve of v13 (e13_fsweep_pi3x.py), K scaled by s
Groups A, B, C (line-G cache) and S = ScanNet++ (cache from e13_cache.py).

  cache/qc-venv/bin/python e13_signals.py composites|fsweep -> outputs/evolution/E13/raw/gate2_{composites,fsweep}.csv
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

H0 = Path("/data/home/z00533370")
for p in ("code/qc", "code/hybrid", "code/bench"):
    sys.path.insert(0, str(H0 / p))
import qc_common as QC  # noqa: E402
import qc_signals as QS  # noqa: E402
import hybrid as HY  # noqa: E402
import eval3d_bench as E3D  # noqa: E402

E13 = H0 / "outputs/evolution/E13"
QC.BENCH["S"] = H0 / "outputs/bench/scannetpp"
LINKS = E13 / "cache_links"  # symlinks: A/B/C -> outputs/qc/cache/<G>, S -> E13/cache/S
LINKS.mkdir(exist_ok=True)
for g, tgt in (("A", QC.CACHE / "A"), ("B", QC.CACHE / "B"), ("C", QC.CACHE / "C"), ("S", E13 / "cache/S")):
    if not (LINKS / g).exists():
        (LINKS / g).symlink_to(tgt)
QS.CACHE = LINKS
SC = {g: pd.read_csv(H0 / "outputs/evolution/E3" / f"scales_{k}.csv").set_index("clip")
      for g, k in (("A", "A"), ("B", "B"), ("C", "wild"), ("S", "scannetpp"))}
GOF = {str(v): g for g, v in QC.BENCH.items()}
_spp = None


def orig(B, m, clip, n, H, W):
    G = GOF[str(B)]
    if G != "S":
        return E3D.load_method(B, m, clip, n, H, W)
    global _spp
    if _spp is None:
        import eval_bench as _spp_mod
        _spp = _spp_mod
    if m == "spatialvid_metric":
        m = "spatialvid"
    c2w, K, D, sec = _spp.load_method(m, clip, n)  # on the 690x1034 GT grid
    D2 = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in D])
    K = K.copy()
    K[:, 0, :] *= W / D.shape[2]
    K[:, 1, :] *= H / D.shape[1]
    return c2w, K, D2, sec


def load_label(B, m, clip, n, H, W):
    G = GOF[str(B)]
    base = lambda x: orig(B, x, clip, n, H, W)  # noqa: E731
    if m == "v0":
        return HY.compose("H:vipe+pi3x+med", base)[:4]
    if m == "v13":
        c2w, K, D, s = base("pi3x_kgate")
        k = float(SC[G].loc[clip, "k_vipe_td"]) if clip in SC[G].index else 1.0
        k = k if np.isfinite(k) and k > 0 else 1.0
        c2w = c2w.copy()
        c2w[:, :3, 3] *= k
        return c2w, K, D * k, s
    if m.startswith("fs"):
        z = np.load(E13 / "fsweep" / clip / m[2:] / "pred.npz")
        D, K = E3D.to_grid(z["depth"].astype(np.float32), z["K"].astype(np.float64), H, W)
        return z["c2w"].astype(np.float64), K, D, float(z["sec"])
    return orig(B, m, clip, n, H, W)


QS.load_method = load_label


def run(job):
    t0, c0 = time.time(), time.process_time()
    r = QS.signals(job)
    r["wall_s"], r["cpu_s"] = time.time() - t0, time.process_time() - c0
    return r


if __name__ == "__main__":
    mode = sys.argv[1]
    jobs = []
    if mode == "composites":
        for G in ("A", "B", "S", "C"):
            for c in QC.clips(G):
                if c.get("dataset") == "eth3d_mvs":
                    continue
                c = dict(c, dataset=c.get("dataset", c.get("cat", "scannetpp")))
                jobs += [(G, m, c) for m in ("v0", "v13")]
    else:
        gmap = {"wild": "C", "eval3d_A": "A", "eval3d_B": "B"}
        for clip in sorted(os.listdir(E13 / "fsweep")):
            G = "C" if clip.startswith("driving__") else ("B" if clip.startswith("vkitti") else "A")
            c = next(x for x in QC.clips(G) if x["clip"] == clip)
            c = dict(c, dataset=c.get("dataset", c.get("cat")))
            for s in sorted(os.listdir(E13 / "fsweep" / clip)):
                if (E13 / "fsweep" / clip / s / "pred.npz").exists():
                    jobs.append((G, "fs" + s, c))
            jobs += [(G, "v13", c), (G, "vipe", c)]
    with ProcessPoolExecutor(int(os.environ.get("NW", 24))) as ex:
        df = pd.DataFrame(list(ex.map(run, jobs, chunksize=1)))
    df.to_csv(E13 / "raw" / f"gate2_{mode}.csv", index=False, float_format="%.5f")
    print(mode, len(df), "errors", int(df["error"].notna().sum()) if "error" in df else 0)
