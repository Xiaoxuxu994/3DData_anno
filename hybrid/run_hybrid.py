"""Evaluate hybrid annotations (hybrid.py specs + the single annotators) with the existing bench metrics.

  python run_hybrid.py A|B|scannetpp|wild [--workers N] [--limit K]  -> outputs/hybrid/raw/<kind>.csv

A / B   : eval3d_bench.eval_clip (same metrics + failure flags as the eval3d report), A without eth3d_mvs
scannetpp: eval_bench.eval_clip (ScanNet++ 28 clips; unseen-3 split in the report)
wild    : wild_eval.per_clip (self-consistency / photometric) + depth agreement with the 7 single annotators,
          metric-scale deviation and physical plausibility (camera speed, median depth per category)
Extras on GT benches: pc_traj_* (fused cloud aligned by the camera-centre Sim3 -> tests whether depth and
translation share one scale) and confidence-retention curves (vote spread, Pi3X / DA3 own conf).
"""

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
sys.path.insert(0, "/data/home/z00533370/code/bench")
sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
import hybrid as HY  # noqa: E402
import geo_vipe as G  # noqa: E402
from eval_vipe import umeyama  # noqa: E402

warnings.filterwarnings("ignore")
cv2.setNumThreads(2)  # shared machine: cap OpenCV's thread pool per worker
OUT = Path("/data/home/z00533370/outputs/bench")
RES = Path("/data/home/z00533370/outputs/hybrid/raw")
KIND = sys.argv[1]
WORKERS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--workers=")), 32))
LIMIT = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--limit=")), 0))
SPECS = HY.SINGLES + HY.specs()
KEEP = (1.0, 0.8, 0.6, 0.4, 0.2)
MAX_DEPTH = 80.0
# camera speed (m/s) and median depth (m) ranges considered physically plausible per wild category
PLAUS = {"driving": ((0.5, 40), (3, 80)), "embodied": ((0, 1.5), (0.15, 3)), "indoor_static": ((0, 3), (0.4, 10)),
         "indoor_dynamic": ((0, 3), (0.4, 10)), "outdoor_static": ((0, 30), (1, 300)), "outdoor_dynamic": ((0, 30), (1, 300))}

if KIND in ("A", "B"):
    import eval3d_bench as M
    BENCH = OUT / f"eval3d_{KIND}"
elif KIND == "scannetpp":
    import eval_bench as M
    BENCH = M.B
else:
    import wild_eval as M
    BENCH = M.B
ORIG = M.load_method


def conf_map(method, clip, H, W):
    """Own confidence of Pi3X / DA3 on the eval grid (higher = more confident)."""
    c = np.load(BENCH / method / clip / "pred.npz")["conf"].astype(np.float32)
    return np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in c])


def retention(D, unc, GD, metric):
    """Pooled depth error on the most-confident fraction of GT-valid pixels (unc: lower = more confident)."""
    m = (GD > 0) & (GD < MAX_DEPTH) & np.isfinite(D) & (D > 0)
    p, g, u = D[m], GD[m], np.nan_to_num(unc[m], nan=np.inf, posinf=np.inf)
    if len(p) < 1000:
        return {}
    s = np.median(g) / np.median(p)
    order = np.argsort(u, kind="stable")
    r = {}
    for k in KEEP:
        idx = order[: max(1, int(k * len(order)))]
        pk, gk = p[idx], g[idx]
        r[f"ret{int(k * 100)}_seq_absrel"] = float(np.mean(np.abs(pk * s - gk) / gk))
        r[f"ret{int(k * 100)}_seq_delta1"] = float(np.mean(np.maximum(pk * s / gk, gk / (pk * s)) < 1.25))
        if metric:
            r[f"ret{int(k * 100)}_metric_absrel"] = float(np.mean(np.abs(pk - gk) / gk))
    return r


def capped_cloud_metrics(Gc, Pc, cap=1.0):
    """Like geo_vipe.cloud_metrics but with a distance cap: far-off (mis-scaled) clouds make plain cKDTree queries crawl."""
    from scipy.spatial import cKDTree
    dp = np.minimum(cKDTree(Gc).query(Pc, distance_upper_bound=cap, workers=2)[0], cap)
    dg = np.minimum(cKDTree(Pc).query(Gc, distance_upper_bound=cap, workers=2)[0], cap)
    a, b = (dp < 0.05).mean(), (dg < 0.05).mean()
    return {"pc_traj_acc_m": float(np.median(dp)), "pc_traj_comp_m": float(np.median(dg)),
            "pc_traj_fscore_5cm": float(2 * a * b / max(a + b, 1e-9))}


def gt_eval3d(c):
    z = np.load(BENCH / "gt" / f"{c['clip']}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return z["c2w"], z["K"], z["depth"].astype(np.float32), bool(meta.get("metric")), \
        str(meta.get("depth_type", "")).startswith("dense")


def gt_scannetpp(c):
    meta = np.load(M.DATA / c["scene"] / "scene_metadata.npz")
    idx = {f: i for i, f in enumerate(meta["images"])}
    gi = [idx[f] for f in c["frames"]]
    GD = np.stack([cv2.imread(str(M.DATA / c["scene"] / "depth" / f"{f}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :M.W] * 1e-3
                   for f in c["frames"]])
    return meta["trajectories"][gi], meta["intrinsics"][gi], GD, True, True


def job(c):
    f = RES / KIND / f"{c['clip']}.csv"
    if f.exists():  # resumable: one CSV per clip
        return pd.read_csv(f).to_dict("records")
    try:
        rows = _job(c)
    except Exception as e:  # noqa: BLE001
        rows = [{"method": "*", "clip": c["clip"], "error": f"job: {str(e)[:200]}"}]
    pd.DataFrame(rows).to_csv(f, index=False, float_format="%.5g")
    return rows


def _job(c):
    cache, memo, infos, last = {}, {}, {}, {}
    clip, n = c["clip"], c["n_frames"]

    def make_base(call):
        def base(m):
            if m not in cache:
                if m == "spatialvid_metric" and KIND == "scannetpp":
                    c2w, K, D, sec = call("spatialvid")
                    ns = json.loads((BENCH / "spatialvid" / clip / "svid_norm_scale.json").read_text())["normalize_scale"]
                    c2w = c2w.copy()
                    c2w[:, :3, 3] /= ns
                    cache[m] = (c2w, K, D / ns, sec)
                else:
                    cache[m] = call(m)
            return cache[m]
        return base

    if KIND == "scannetpp":
        base = make_base(lambda m: ORIG(m, clip, n))

        def patched(spec, clip_, n_):
            if last.get("spec") != spec:  # eval + extras call the same spec back to back
                last.update(spec=spec, r=HY.compose(spec, base, memo))
            infos[spec] = last["r"][4]
            return last["r"][:4]
    else:
        holder = {}

        def patched(B, spec, clip_, n_, H, W):
            if "base" not in holder:
                holder["base"] = make_base(lambda m: ORIG(B, m, clip_, n_, H, W))
            if last.get("spec") != spec:
                last.update(spec=spec, r=HY.compose(spec, holder["base"], memo))
            infos[spec] = last["r"][4]
            return last["r"][:4]
    M.load_method = patched
    if KIND == "wild":
        return job_wild(c, infos)
    T_gt, K_gt, GD, metric, dense = gt_eval3d(c) if KIND in ("A", "B") else gt_scannetpp(c)
    rows = []
    for spec in SPECS:
        r = M.eval_clip((BENCH, spec, c)) if KIND in ("A", "B") else M.eval_clip((spec, c))
        r["method"] = spec
        info = infos.get(spec, {})
        r.update({k: v for k, v in info.items() if k != "spread"})
        if "error" not in r:
            try:
                c2w, K, D, _ = patched(BENCH, spec, clip, n, *GD.shape[1:]) if KIND != "scannetpp" else patched(spec, clip, n)
                if dense and n >= 8:  # cloud aligned by camera centres: depth must live in the trajectory's scale
                    d = {"seg": {"scene": c.get("scene", c.get("dataset")), "segment": c.get("segment", clip), "frames": c.get("frames", [str(i) for i in range(n)])},
                         "T_gt": T_gt, "K_gt": K_gt, "T_p": c2w, "K_p": K, "D_p": {k: D[k] for k in range(n)}, "GD": GD}
                    d["sim3"] = umeyama(c2w[:, :3, 3], T_gt[:, :3, 3], True)
                    L_gt = np.linalg.norm(np.diff(T_gt[:, :3, 3], axis=0), axis=1).sum()
                    # metric GT only (5 cm is meaningless in Sintel units); degenerate Sim3 -> skip
                    if metric and L_gt > 0.05 and 1e-2 < d["sim3"][0] < 1e2:
                        Gc, Pc, _, _ = G.fused_clouds(d, n_frames=min(40, n), align="traj")
                        if np.ptp(Pc, 0).max() > 1e-2:
                            r.update(capped_cloud_metrics(Gc, Pc))
                if "spread" in info:
                    r.update({f"vote_{k}": v for k, v in retention(D, info["spread"], GD, metric).items()})
                    sp = info["spread"][np.isfinite(info["spread"])]
                    r["vote_spread_med"] = float(np.median(sp)) if len(sp) else np.nan
                if spec in ("pi3x", "da3"):
                    cm = conf_map(spec, clip, *GD.shape[1:])
                    r.update({f"own_{k}": v for k, v in retention(D, -cm, GD, metric).items()})
            except Exception as e:  # noqa: BLE001
                r["extra_error"] = str(e)[:200]
        rows.append(r)
    return rows


def job_wild(c, infos):
    rows = []
    H, W = M.grid(c)
    ks = np.linspace(0, c["n_frames"] - 1, 8).astype(int)
    singles = {}
    for m in M.METHODS:
        try:
            singles[m] = ORIG(M.B, m, c["clip"], c["n_frames"], H, W)[2][ks]
        except Exception:  # noqa: BLE001
            pass
    metric_singles = ["vipe", "spatialvid_metric", "voyager", "da3", "pi3x"]
    fps = c.get("fps_out", 15)
    for spec in SPECS:
        r = M.per_clip((spec, c))
        r["method"] = spec
        info = infos.get(spec, {})
        r.update({k: v for k, v in info.items() if k != "spread"})
        if "error" in r and isinstance(r["error"], str):
            rows.append(r)
            continue
        c2w, K, D, _ = M.load(spec, c)
        if spec.startswith("H:"):
            p, d, _a = HY.parse(spec)
            involved = {p, *HY.VOTE.get(d, [p if d == "own" else d])}
        else:
            involved = {spec}
        x = D[ks]
        ab, lr = [], []
        for m, y in singles.items():
            if m in involved:
                continue
            mk = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if mk.sum() < 1000:
                continue
            ratio = np.median(y[mk]) / np.median(x[mk])
            ab.append(np.mean(np.abs(x[mk] * ratio - y[mk]) / y[mk]))
            if m in metric_singles:
                lr.append(np.log(ratio))
        r["depth_agree_absrel"] = float(np.median(ab)) if ab else np.nan
        r["metric_scale_dev"] = float(np.exp(np.abs(np.median(lr)))) if lr else np.nan  # x vs median of other metric methods
        if "spread" in info:
            sp = info["spread"][np.isfinite(info["spread"])]
            r["vote_spread_med"] = float(np.median(sp)) if len(sp) else np.nan
            r["vote_conf_frac5"] = float(np.mean(sp < 0.05)) if len(sp) else np.nan
        dur = c["n_frames"] / fps
        r["speed_mps"] = r.get("traj_len", np.nan) / dur
        if c["cat"] in PLAUS:
            (s0, s1), (d0, d1) = PLAUS[c["cat"]]
            r["implaus_speed"] = float(not (s0 <= r["speed_mps"] <= s1))
            r["implaus_depth"] = float(not (d0 <= r["depth_p50"] <= d1))
        rows.append(r)
    return rows


def main():
    clips = json.loads((BENCH / "clips.json").read_text())
    if KIND == "A":
        clips = [c for c in clips if c["dataset"] != "eth3d_mvs"]
    if LIMIT:
        clips = clips[:LIMIT]
    (RES / KIND).mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, rs in enumerate(ex.map(job, sorted(clips, key=lambda c: -c["n_frames"]))):
            rows += rs
            print(f"{KIND} {i + 1}/{len(clips)}", flush=True)
    tag = f"_limit{LIMIT}" if LIMIT else ""
    pd.DataFrame(rows).to_csv(RES / f"{KIND}{tag}.csv", index=False, float_format="%.5g")
    print("DONE", KIND, len(rows), flush=True)


if __name__ == "__main__":
    main()
