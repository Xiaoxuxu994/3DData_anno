"""Cross-pipeline evaluation on the ScanNet++ benchmark clips (same frames for every method).

Every method is converted to the GT image grid (690x1034): c2w (N,4,4), K (N,3,3), depth (N,690,1034).
Metrics reuse eval_vipe.py / geo_vipe.py definitions (see the report appendix).

  python eval_bench.py METHOD [METHOD ...]   -> bench/scannetpp/eval/<method>.csv + summary.json
Methods: vipe | spatialvid | <name> (generic: <name>/<clip>/pred.npz with c2w, K, depth[, sec])
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
from eval_vipe import DATA, MAX_DEPTH, depth_errors, pose_metrics, umeyama  # noqa: E402

B = Path("/data/home/z00533370/outputs/bench/scannetpp")
H, W = 690, 1034


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


def to_grid(depth, K):
    """Resize depth maps (N,h,w) to the GT grid and scale K accordingly (per-axis)."""
    n, h, w = depth.shape
    sx, sy = W / w, H / h
    D = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth])
    K = K.copy()
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    # pixel-centre convention shift for the resize: x' = sx*(x+0.5)-0.5
    K[:, 0, 2] += 0.5 * sx - 0.5
    K[:, 1, 2] += 0.5 * sy - 0.5
    return D, K


def load_method(method, clip, n):
    if method.startswith("vipe"):
        o = B / method
        pz, iz = np.load(o / "pose" / f"{clip}.npz"), np.load(o / "intrinsics" / f"{clip}.npz")
        c2w = pz["data"][np.argsort(pz["inds"])]
        f = iz["data"][np.argsort(iz["inds"])]
        K = np.zeros((len(f), 3, 3))
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], 1
        dz = exr_zip(o / "depth" / f"{clip}.zip")
        D = np.stack([dz[k] for k in sorted(dz)])
        sec = next((json.loads(l)["sec"] for l in (o / "logs" / "status.jsonl").read_text().splitlines()
                    if json.loads(l)["clip"] == clip), np.nan)
    elif method == "spatialvid":
        z = np.load(B / "spatialvid" / clip / "sgd_cvd_hr.npz")
        c2w, D = z["cam_c2w"].astype(np.float64), z["depths"].astype(np.float32)
        K = np.repeat(z["intrinsic"][None].astype(np.float64), len(c2w), 0)
        sec = np.nan
    else:
        z = np.load(B / method / clip / "pred.npz")
        c2w, K, D = z["c2w"].astype(np.float64), z["K"].astype(np.float64), z["depth"].astype(np.float32)
        sec = float(z["sec"]) if "sec" in z.files else np.nan
    assert len(c2w) == n, f"{method}/{clip}: {len(c2w)} poses vs {n} frames"
    if method == "voyager":  # sky pixels are filled with one far constant -> invalid
        far = np.nanmax(D)
        D = np.where(D >= far * 0.999, np.nan, D).astype(np.float32)
    D, K = to_grid(D, K)
    return c2w, K, D, sec


def eval_clip(args):
    method, c = args
    clip, n = c["clip"], c["n_frames"]
    r = {"method": method, "clip": clip, "scene": c["scene"], "n_frames": n}
    try:
        c2w, K, D, sec = load_method(method, clip, n)
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
        return r
    meta = np.load(DATA / c["scene"] / "scene_metadata.npz")
    idx = {f: i for i, f in enumerate(meta["images"])}
    gi = [idx[f] for f in c["frames"]]
    T_gt, K_gt = meta["trajectories"][gi], meta["intrinsics"][gi]
    r["sec"] = sec
    r.update(pose_metrics(c2w, T_gt))
    fov = lambda fx: np.degrees(2 * np.arctan(W / 2 / fx))  # noqa: E731
    r["fov_err_deg"] = float(np.abs(fov(K[:, 0, 0]) - fov(K_gt[:, 0, 0])).mean())
    # depth (all frames)
    acc = {f"{t}_{k}": [] for t in ("metric", "seq", "frame") for k in ("absrel", "delta1", "logrmse")}
    P, Gs = [], []
    for k in range(n):
        g = cv2.imread(str(DATA / c["scene"] / "depth" / f"{c['frames'][k]}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :W] * 1e-3
        p = D[k]
        m = (g > 0) & (g < MAX_DEPTH) & np.isfinite(p) & (p > 0)
        if m.sum() < 100:
            continue
        P.append(p[m]); Gs.append(g[m])
    s_seq = np.median(np.concatenate(Gs)) / np.median(np.concatenate(P))
    for pv, gv in zip(P, Gs):
        for t, sc in (("metric", 1.0), ("seq", s_seq), ("frame", np.median(gv) / np.median(pv))):
            for kk, v in depth_errors(pv * sc, gv).items():
                acc[f"{t}_{kk}"].append(v)
    r.update({f"depth_{k}": float(np.mean(v)) for k, v in acc.items()})
    r["depth_seq_scale"] = float(s_seq)
    # point cloud + reprojection (geo_vipe on the converted geometry)
    d = {"seg": {"scene": c["scene"], "segment": c["segment"], "frames": c["frames"]},
         "T_gt": T_gt, "K_gt": K_gt, "T_p": c2w, "K_p": K, "D_p": {k: D[k] for k in range(n)}}
    d["sim3"] = umeyama(c2w[:, :3, 3], T_gt[:, :3, 3], True)
    Gc, Pc, _, (s_pair, _, _) = G.fused_clouds(d, align="pairs")
    r.update(G.cloud_metrics(Gc, Pc))
    r["pc_pair_scale"] = float(s_pair)
    for gap in (5, 20):
        if n > gap + 1:
            r.update(G.reproj_metrics(d, gap))
    return r


def main():
    clips = json.loads((B / "clips.json").read_text())
    jobs = [(m, c) for m in sys.argv[1:] for c in clips]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(eval_clip, jobs))
    df = pd.DataFrame(rows)
    out = B / "eval"
    out.mkdir(exist_ok=True)
    for m, g in df.groupby("method"):
        g.to_csv(out / f"{m}.csv", index=False, float_format="%.4f")
    cols = ["ate_sim3_cm", "ate_se3_cm", "sim3_scale", "rte_cm", "rre_deg", "fov_err_deg", "depth_metric_absrel",
            "depth_metric_delta1", "depth_seq_absrel", "depth_seq_delta1", "depth_frame_absrel", "depth_frame_delta1",
            "pc_acc_m", "pc_comp_m", "pc_fscore_5cm", "pc_pair_scale", "g5_epe_med_px", "g20_epe_med_px",
            "g5_vipe_selfcons_5pct", "sec"]
    summ = {}
    for m, g in df.groupby("method"):
        ok = g[g.get("error").isna()] if "error" in g else g
        summ[m] = {"n_ok": int(len(ok)), "n_fail": int(len(g) - len(ok)),
                   "failed": g.loc[g["error"].notna(), ["clip", "error"]].to_dict("records") if "error" in g else [],
                   "median": ok[[c for c in cols if c in ok]].median().round(4).to_dict(),
                   "mean": ok[[c for c in cols if c in ok]].mean().round(4).to_dict(),
                   "fps": float(ok.n_frames.sum() / ok.sec.sum()) if ok.sec.notna().all() else None}
    prev = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    prev.update(summ)
    (out / "summary.json").write_text(json.dumps(prev, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
