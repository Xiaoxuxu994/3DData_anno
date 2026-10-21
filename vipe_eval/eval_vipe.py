"""Evaluate ViPE pseudo-labels against ScanNet++ GT (pose, intrinsics, depth).

Pose:   ATE (RMSE, cm) after Sim(3) and after SE(3) Umeyama alignment, Sim(3) scale,
        RTE (cm) / RRE (deg) on consecutive-frame relative motion (after Sim(3) scale).
Intr.:  horizontal FOV error (deg) and relative fx error, averaged over frames.
Depth:  AbsRel, delta<1.25, LogRMSE under three alignments:
          metric  - raw ViPE depth in metres
          seq     - one median scale per segment
          frame   - per-frame median scale (monocular-style upper bound)

Usage: python eval_vipe.py   (writes eval/per_segment.csv, eval/summary.json)
"""

import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import pandas as pd

DATA = Path("/data/home/z00533370/data/scannetpp_sample30")
ROOT = Path("/data/home/z00533370/outputs/vipe/scannetpp_sample30")
PRED = ROOT / "vipe"
MAX_DEPTH = 20.0  # metres, ignore GT beyond this
DEPTH_STRIDE = 1  # evaluate every k-th frame for depth


def umeyama(src, dst, with_scale):
    """Find s, R, t minimising ||dst - (s R src + t)||. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def pose_metrics(T_pred, T_gt):
    p, g = T_pred[:, :3, 3], T_gt[:, :3, 3]
    out = {}
    s, R, t = umeyama(p, g, True)
    out["ate_sim3_cm"] = 100 * np.sqrt(((s * p @ R.T + t - g) ** 2).sum(1).mean())
    out["sim3_scale"] = s
    _, R1, t1 = umeyama(p, g, False)
    out["ate_se3_cm"] = 100 * np.sqrt(((p @ R1.T + t1 - g) ** 2).sum(1).mean())
    # relative motion between consecutive frames
    rte, rre = [], []
    for k in range(len(p) - 1):
        dp = np.linalg.inv(T_pred[k]) @ T_pred[k + 1]
        dg = np.linalg.inv(T_gt[k]) @ T_gt[k + 1]
        rte.append(np.linalg.norm(s * dp[:3, 3] - dg[:3, 3]))
        rre.append(rot_angle_deg(dp[:3, :3].T @ dg[:3, :3]))
    out["rte_cm"] = 100 * float(np.mean(rte))
    out["rre_deg"] = float(np.mean(rre))
    out["traj_len_m"] = float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum())
    return out


def read_exr_zip(path):
    res = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                exr = OpenEXR.InputFile(f)
                dw = exr.header()["dataWindow"]
                w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
                res[int(name.split(".")[0])] = np.frombuffer(exr.channels(["Z"])[0], dtype=np.float16).reshape(h, w).astype(np.float32)
    return res


def depth_errors(pred, gt):
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "delta1": float(np.mean(ratio < 1.25)),
        "logrmse": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
    }


def eval_segment(seg):
    name, scene, frames = seg["segment"], seg["scene"], seg["frames"]
    meta = np.load(DATA / scene / "scene_metadata.npz")
    idx = {n: i for i, n in enumerate(meta["images"])}
    gi = [idx[f] for f in frames]
    T_gt, K_gt = meta["trajectories"][gi], meta["intrinsics"][gi]

    pz = np.load(PRED / "pose" / f"{name}.npz")
    iz = np.load(PRED / "intrinsics" / f"{name}.npz")
    inds = pz["inds"]
    r = {"segment": name, "scene": scene, "n_frames": len(frames), "n_pred_pose": len(inds)}
    r.update(pose_metrics(pz["data"][np.argsort(inds)], T_gt[np.sort(inds)]))

    W = 1034  # cropped width fed to ViPE
    fx_p = iz["data"][:, 0]
    fx_g = K_gt[iz["inds"], 0, 0]
    fov = lambda fx: np.degrees(2 * np.arctan(W / 2 / fx))
    r["fx_gt"], r["fx_pred"] = float(fx_g.mean()), float(fx_p.mean())
    r["hfov_gt_deg"], r["hfov_pred_deg"] = float(fov(fx_g).mean()), float(fov(fx_p).mean())
    r["fov_err_deg"] = float(np.abs(fov(fx_p) - fov(fx_g)).mean())
    r["fx_relerr"] = float(np.mean(np.abs(fx_p - fx_g) / fx_g))

    # depth
    dz = read_exr_zip(PRED / "depth" / f"{name}.zip")
    preds, gts, per_frame = [], [], {k: [] for k in ("absrel", "delta1", "logrmse")}
    for k in sorted(dz)[::DEPTH_STRIDE]:
        g = cv2.imread(str(DATA / scene / "depth" / f"{frames[k]}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
        g = g[:, :W]
        p = dz[k]
        if p.shape != g.shape:
            p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (g > 0) & (g < MAX_DEPTH) & np.isfinite(p) & (p > 0)
        if m.sum() < 100:
            continue
        pv, gv = p[m], g[m]
        preds.append(pv)
        gts.append(gv)
        e = depth_errors(pv * np.median(gv) / np.median(pv), gv)
        for kk in per_frame:
            per_frame[kk].append(e[kk])
    P, G = np.concatenate(preds), np.concatenate(gts)
    # per-frame means for metric / seq-scale so each frame weighs equally
    s_seq = np.median(G) / np.median(P)
    for tag, sc in (("metric", 1.0), ("seq", s_seq)):
        acc = {k: [] for k in ("absrel", "delta1", "logrmse")}
        for pv, gv in zip(preds, gts):
            for kk, v in depth_errors(pv * sc, gv).items():
                acc[kk].append(v)
        for kk, v in acc.items():
            r[f"depth_{tag}_{kk}"] = float(np.mean(v))
    for kk, v in per_frame.items():
        r[f"depth_frame_{kk}"] = float(np.mean(v))
    r["depth_seq_scale"] = float(s_seq)
    r["depth_valid_px_frac"] = float(sum(len(x) for x in gts) / (len(gts) * 690 * W))

    # mask: ViPE instance ids, phrases in <seg>.txt ("1: sky" etc.). Scenes are static, so any
    # non-background pixel is a false positive (sky on white walls / ceilings, objects, people).
    phrases = dict(l.split(": ", 1) for l in (PRED / "mask" / f"{name}.txt").read_text().splitlines() if ": " in l)
    sky_ids = [int(k) for k, v in phrases.items() if v.strip() == "sky"]
    r["mask_phrases"] = ";".join(sorted({v.strip() for k, v in phrases.items() if k != "0"}))
    sky, dyn = [], []
    with zipfile.ZipFile(PRED / "mask" / f"{name}.zip") as z:
        for n in sorted(z.namelist())[::5]:
            m = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
            s_ = np.isin(m, sky_ids)
            sky.append(s_.mean())
            dyn.append(((m > 0) & ~s_).mean())
    r["sky_frac"], r["dyn_frac"] = float(np.mean(sky)), float(np.mean(dyn))
    return r


def main():
    segs = json.loads((ROOT / "segments.json").read_text())
    status = {}
    for line in (ROOT / "logs" / "status.jsonl").read_text().splitlines():
        rec = json.loads(line)
        status[rec["segment"]] = rec
    done = [s for s in segs if (PRED / "depth" / f"{s['segment']}.zip").exists()]
    failed = [s["segment"] for s in segs if s not in done]
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(eval_segment, done))
    df = pd.DataFrame(rows)
    df["run_sec"] = df.segment.map(lambda s: status.get(s, {}).get("sec", np.nan))
    df["fps"] = df.n_frames / df.run_sec
    df["fx_group"] = np.where(df.fx_gt < 320, "fx~274 (HFOV~124deg)", "fx~368 (HFOV~109deg)")
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    df.sort_values("segment").to_csv(out / "per_segment.csv", index=False, float_format="%.4f")

    cols = [
        "ate_sim3_cm", "ate_se3_cm", "sim3_scale", "rte_cm", "rre_deg", "fov_err_deg", "fx_relerr",
        "depth_metric_absrel", "depth_metric_delta1", "depth_metric_logrmse",
        "depth_seq_absrel", "depth_seq_delta1", "depth_seq_logrmse",
        "depth_frame_absrel", "depth_frame_delta1", "depth_frame_logrmse", "depth_seq_scale", "sky_frac", "dyn_frac",
    ]
    w = df.n_frames / df.n_frames.sum()
    summary = {
        "n_segments": len(df), "n_frames": int(df.n_frames.sum()), "failed_segments": failed,
        "mean_over_segments": df[cols].mean().round(4).to_dict(),
        "median_over_segments": df[cols].median().round(4).to_dict(),
        "frame_weighted_mean": {c: round(float((df[c] * w).sum()), 4) for c in cols},
        "by_fx_group": df.groupby("fx_group")[cols].mean().round(4).to_dict(orient="index"),
        "by_length": df.groupby(pd.cut(df.n_frames, [0, 100, 300, 1000, 10000]), observed=True)[cols].mean().round(4).to_dict(orient="index"),
        "runtime": {"total_gpu_sec": float(df.run_sec.sum()), "mean_fps_per_gpu": float(df.n_frames.sum() / df.run_sec.sum())},
    }
    summary["by_length"] = {str(k): v for k, v in summary["by_length"].items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
