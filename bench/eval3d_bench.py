"""Unified 3D evaluation on the eval3d A/B benches (GT caches written by assemble_eval3d.py).

  python eval3d_bench.py BENCH_DIR [METHOD ...]   -> BENCH_DIR/eval/{per_clip.csv, summary.json}
Methods default: vipe spatialvid voyager vggt vggt_omega da3 pi3x.

Metrics (same definitions as eval_bench.py / the ViPE report appendix):
  pose ATE Sim3 / SE3 (cm), Sim3 scale, RTE / RRE, FOV error; depth metric / seq / frame AbsRel, delta1, LogRMSE
  (sparse GT -> valid pixels only); point cloud Acc / Comp / F@5cm and reprojection EPE + self-consistency
  (dense GT only). Non-metric GT (Sintel) -> SE3 ATE, scale and metric depth reported as NaN.
Failure flags: ATE_Sim3 > 5% of trajectory length (scale-free, used for ranking), ATE_SE3 > 5% of trajectory length, |scale-1| > 10%, metric AbsRel > 0.2, FOV error > 5 deg.
"""

import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
import geo_vipe as G  # noqa: E402
from eval_vipe import depth_errors, pose_metrics, umeyama  # noqa: E402

METHODS = ["vipe", "spatialvid", "spatialvid_metric", "voyager", "vggt", "vggt_omega", "da3", "pi3x"]
MAX_DEPTH = 80.0


def exr_zip(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            with z.open(n) as f:
                e = OpenEXR.InputFile(f)
                dw = e.header()["dataWindow"]
                w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
                out[int(n.split(".")[0])] = np.frombuffer(e.channels(["Z"])[0], np.float16).reshape(h, w).astype(np.float32)
    return out


def to_grid(depth, K, H, W):
    n, h, w = depth.shape
    sx, sy = W / w, H / h
    D = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth])
    K = K.copy()
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    K[:, 0, 2] += 0.5 * sx - 0.5
    K[:, 1, 2] += 0.5 * sy - 0.5
    return D, K


def load_method(B, method, clip, n, H, W):
    if method == "vipe":
        o = B / "vipe"
        pz, iz = np.load(o / "pose" / f"{clip}.npz"), np.load(o / "intrinsics" / f"{clip}.npz")
        c2w = pz["data"][np.argsort(pz["inds"])]
        f = iz["data"][np.argsort(iz["inds"])]
        K = np.zeros((len(f), 3, 3))
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], 1
        dz = exr_zip(o / "depth" / f"{clip}.zip")
        D = np.stack([dz[k] for k in sorted(dz)])
        sec = np.nan
        st = o / "logs" / "status.jsonl"
        if st.exists():
            for line in st.read_text().splitlines():
                r = json.loads(line)
                if r.get("clip") == clip:
                    sec = r.get("sec", np.nan)
    elif method.startswith("spatialvid"):
        z = np.load(B / "spatialvid" / clip / "sgd_cvd_hr.npz")
        c2w, D = z["cam_c2w"].astype(np.float64), z["depths"].astype(np.float32)
        if method == "spatialvid_metric":  # undo camera_tracking's disparity normalisation (svid_norm_scale.py)
            ns = json.loads((B / "spatialvid" / clip / "svid_norm_scale.json").read_text())["normalize_scale"]
            c2w[:, :3, 3] /= ns
            D = D / ns
        K = np.repeat(z["intrinsic"][None].astype(np.float64), len(c2w), 0)
        sec = np.nan
    else:
        z = np.load(B / method / clip / "pred.npz")
        c2w, K, D = z["c2w"].astype(np.float64), z["K"].astype(np.float64), z["depth"].astype(np.float32)
        sec = float(z["sec"]) if "sec" in z.files else np.nan
    if method == "voyager":  # sky pixels are filled with one far constant -> invalid
        far = np.nanmax(D)
        D = np.where(D >= far * 0.999, np.nan, D).astype(np.float32)
    assert len(c2w) == n, f"{method}/{clip}: {len(c2w)} poses vs {n} frames"
    if method == "vggt_omega":  # official preprocessing center-crops the aspect ratio into [0.5, 2.0]
        ch, cw = min(H, int(round(W / 0.5))), min(W, int(round(H / 0.5)))
        top, left = max((H - ch) // 2, 0), max((W - cw) // 2, 0)
        Dc, K = to_grid(D, K, ch, cw)
        D = np.full((n, H, W), np.nan, np.float32)
        D[:, top:top + ch, left:left + cw] = Dc
        K[:, 0, 2] += left
        K[:, 1, 2] += top
        return c2w, K, D, sec
    D, K = to_grid(D, K, H, W)
    return c2w, K, D, sec


def eval_clip(args):
    B, method, c = args
    clip, n = c["clip"], c["n_frames"]
    r = {"method": method, "clip": clip, "dataset": c["dataset"], "group": c["group"], "n_frames": n}
    z = np.load(B / "gt" / f"{clip}.npz", allow_pickle=True)
    T_gt, K_gt = z["c2w"], z["K"]
    GD = z["depth"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    H, W = GD.shape[1:]
    metric = bool(meta.get("metric"))
    dense = str(meta.get("depth_type", "")).startswith("dense")
    try:
        c2w, K, D, sec = load_method(B, method, clip, n, H, W)
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
        return r
    r["sec"] = sec
    pm = pose_metrics(c2w, T_gt)
    r.update(pm)
    if not metric:
        r["ate_se3_cm"] = r["sim3_scale"] = np.nan
    fov = lambda fx: np.degrees(2 * np.arctan(W / 2 / fx))  # noqa: E731
    r["fov_err_deg"] = float(np.abs(fov(K[:, 0, 0]) - fov(K_gt[:, 0, 0])).mean())
    # depth
    acc = {f"{t}_{k}": [] for t in ("metric", "seq", "frame") for k in ("absrel", "delta1", "logrmse")}
    P, Gs = [], []
    for k in range(n):
        g, p = GD[k], D[k]
        m = (g > 0) & (g < MAX_DEPTH) & np.isfinite(p) & (p > 0)
        if m.sum() < 50:
            P.append(None); Gs.append(None)
            continue
        P.append(p[m]); Gs.append(g[m])
    valid = [i for i in range(n) if P[i] is not None]
    if valid:
        s_seq = np.median(np.concatenate([Gs[i] for i in valid])) / np.median(np.concatenate([P[i] for i in valid]))
        for i in valid:
            pv, gv = P[i], Gs[i]
            for t, sc in (("metric", 1.0), ("seq", s_seq), ("frame", np.median(gv) / np.median(pv))):
                for kk, v in depth_errors(pv * sc, gv).items():
                    acc[f"{t}_{kk}"].append(v)
        r.update({f"depth_{k}": float(np.mean(v)) if v else np.nan for k, v in acc.items()})
        r["depth_seq_scale"] = float(s_seq)
        if not metric:
            for kk in ("absrel", "delta1", "logrmse"):
                r[f"depth_metric_{kk}"] = np.nan
    # point cloud + reprojection on dense GT (geo_vipe functions with patched loaders)
    if dense and n >= 8:
        G.H, G.W = H, W
        frames_dir = B / "frames" / clip
        G.gt_depth = lambda d, k: np.where(d["GD"][k] < MAX_DEPTH, d["GD"][k], 0)
        G.rgb = lambda d, k: cv2.cvtColor(cv2.imread(str(frames_dir / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
        d = {"seg": {"scene": c["dataset"], "segment": clip, "frames": [str(i) for i in range(n)]},
             "T_gt": T_gt, "K_gt": K_gt, "T_p": c2w, "K_p": K, "D_p": {k: D[k] for k in range(n)}, "GD": GD}
        d["sim3"] = umeyama(c2w[:, :3, 3], T_gt[:, :3, 3], True)
        try:
            Gc, Pc, _, (s_pair, _, _) = G.fused_clouds(d, n_frames=min(40, n), align="pairs")
            r.update(G.cloud_metrics(Gc, Pc))
            r["pc_pair_scale"] = float(s_pair)
            for gap in (5, 20):
                if n > gap + 10:
                    r.update(G.reproj_metrics(d, gap, every=max(1, n // 20)))
        except Exception as e:  # noqa: BLE001
            r["geo_error"] = str(e)[:200]
    # failure flags
    L = pm.get("traj_len_m", np.nan)
    r["fail_ate_sim3"] = float(np.isfinite(r["ate_sim3_cm"]) and r["ate_sim3_cm"] / 100 > 0.05 * max(L, 1e-6))
    r["fail_ate"] = float(metric and np.isfinite(r["ate_se3_cm"]) and r["ate_se3_cm"] / 100 > 0.05 * max(L, 1e-6))
    r["fail_scale"] = float(metric and abs(r["sim3_scale"] - 1) > 0.10)
    r["fail_depth"] = float(metric and np.isfinite(r.get("depth_metric_absrel", np.nan)) and r["depth_metric_absrel"] > 0.2)
    r["fail_fov"] = float(r["fov_err_deg"] > 5)
    if not metric:
        r["fail_ate"] = r["fail_scale"] = r["fail_depth"] = np.nan
    return r


KEYS = ["ate_sim3_cm", "ate_se3_cm", "sim3_scale", "rte_cm", "rre_deg", "fov_err_deg",
        "depth_metric_absrel", "depth_metric_delta1", "depth_seq_absrel", "depth_seq_delta1", "depth_frame_absrel",
        "pc_acc_m", "pc_comp_m", "pc_fscore_5cm", "g5_epe_med_px", "g20_epe_med_px", "g5_vipe_selfcons_5pct", "sec"]
FAILS = ["fail_ate_sim3", "fail_ate", "fail_scale", "fail_depth", "fail_fov"]


def main():
    B = Path(sys.argv[1])
    methods = [m for m in sys.argv[2:] if not m.startswith("--")] or METHODS
    tag = next((a[6:] for a in sys.argv[2:] if a.startswith("--tag=")), "")
    clips = json.loads((B / "clips.json").read_text())
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(eval_clip, [(B, m, c) for m in methods for c in clips]))
    df = pd.DataFrame(rows)
    out = B / "eval"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"per_clip{tag}.csv", index=False, float_format="%.4f")
    ok = df[df["error"].isna()] if "error" in df else df
    summ = {"n_clips": int(len(clips)), "by_method_dataset": {}, "errors": []}
    if "error" in df:
        summ["errors"] = df.loc[df["error"].notna(), ["method", "clip", "error"]].to_dict("records")
    for (m, ds), g in ok.groupby(["method", "dataset"]):
        summ["by_method_dataset"].setdefault(m, {})[ds] = {
            "n_ok": int(len(g)),
            "median": {k: (round(float(g[k].median()), 4) if k in g and g[k].notna().any() else None) for k in KEYS},
            "fail_rate": {k: (round(float(g[k].mean()), 3) if k in g and g[k].notna().any() else None) for k in FAILS}}
    (out / f"summary{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({m: {ds: v["n_ok"] for ds, v in d.items()} for m, d in summ["by_method_dataset"].items()}))


if __name__ == "__main__":
    main()
