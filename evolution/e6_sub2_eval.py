"""E6a: does running ViPE at half frame rate (every 2nd frame) lose accuracy? Zero-GPU: re-uses outputs/qc/variants.

python e6_sub2_eval.py -> outputs/evolution/E6/vipe_sub2.csv (+ printed summary)
For every A (no ETH3D) / B clip: ViPE on all frames (bench) vs ViPE on every 2nd frame (variants/*_sub2),
both evaluated on the SAME frame subset (src_index) against GT: ATE Sim3 (% of path), Sim3 scale, FOV error,
metric depth AbsRel (10 frames), failure flags (scale > 10%, FOV > 5 deg, ATE Sim3 > 5%), wall time.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/bench")
sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
import eval3d_bench as M  # noqa: E402
from eval_vipe import depth_errors, pose_metrics  # noqa: E402

ROOT = Path("/data/home/z00533370")
VAR = ROOT / "outputs/qc/variants"


def vipe(o, clip):
    pz, iz = np.load(o / "pose" / f"{clip}.npz"), np.load(o / "intrinsics" / f"{clip}.npz")
    c2w = pz["data"][np.argsort(pz["inds"])]
    f = iz["data"][np.argsort(iz["inds"])]
    dz = M.exr_zip(o / "depth" / f"{clip}.zip")
    sec = next((json.loads(l)["sec"] for l in (o / "logs" / "status.jsonl").read_text().splitlines()
                if json.loads(l)["clip"] == clip), np.nan)
    return c2w, f[:, 0], np.stack([dz[k] for k in sorted(dz)]), sec


def metrics(c2w, fx, D, T, Kg, GD, W, metric):
    r = pose_metrics(c2w, T)
    L = max(r["traj_len_m"], 1e-6)
    r["ate_pct"] = r["ate_sim3_cm"] / 100 / L * 100
    fov = lambda x: np.degrees(2 * np.arctan(W / 2 / x))  # noqa: E731
    r["fov_err"] = float(np.abs(fov(fx) - fov(Kg[:, 0, 0])).mean())
    ab = []
    for k in np.linspace(0, len(D) - 1, 10).astype(int):
        p = cv2.resize(D[k], (GD.shape[2], GD.shape[1]), interpolation=cv2.INTER_NEAREST)
        m = (GD[k] > 0) & (GD[k] < 80) & np.isfinite(p) & (p > 0)
        if m.sum() > 50:
            ab.append(depth_errors(p[m], GD[k][m])["absrel"])
    r["depth_absrel"] = float(np.mean(ab)) if ab and metric else np.nan
    r["fail_scale"] = float(metric and abs(r["sim3_scale"] - 1) > 0.1)
    r["fail_fov"] = float(r["fov_err"] > 5)
    r["fail_ate"] = float(r["ate_pct"] > 5)
    return r


rows = []
for g in ("A", "B"):
    B = ROOT / f"outputs/bench/eval3d_{g}"
    sub = {c["clip"]: c for c in json.loads((VAR / f"{g}_sub2" / "clips.json").read_text())}
    for c in json.loads((B / "clips.json").read_text()):
        if c["dataset"] == "eth3d_mvs" or c["clip"] not in sub:
            continue
        clip, idx = c["clip"], np.array(sub[c["clip"]]["src_index"])
        z = np.load(B / "gt" / f"{clip}.npz", allow_pickle=True)
        metric = bool(json.loads(str(z["meta"])).get("metric"))
        T, Kg, GD = z["c2w"][idx], z["K"][idx], z["depth"][idx].astype(np.float32)
        W = GD.shape[2]
        try:
            full = vipe(B / "vipe", clip)
            half = vipe(VAR / f"{g}_sub2" / "vipe", clip)
        except Exception as e:  # noqa: BLE001
            rows.append({"clip": clip, "error": str(e)[:150]})
            continue
        for tag, (c2w, fx, D, sec) in (("full", (full[0][idx], full[1][idx], full[2][idx], full[3])), ("sub2", half)):
            r = metrics(c2w, fx, D, T, Kg, GD, W, metric)
            r.update(clip=clip, dataset=c["dataset"], group=g, run=tag, sec=sec, metric=metric)
            rows.append(r)
df = pd.DataFrame(rows)
out = ROOT / "outputs/evolution/E6"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "vipe_sub2.csv", index=False, float_format="%.5g")
ok = df[df.get("error", pd.Series(np.nan, index=df.index)).isna()]
cols = ["ate_pct", "sim3_scale", "fov_err", "depth_absrel", "fail_ate", "fail_scale", "fail_fov", "sec"]
print(ok.groupby(["group", "run"])[cols].agg(["median", "mean"]).round(3).T.to_string())
print(ok.groupby(["dataset", "run"])[["ate_pct", "fov_err", "fail_scale", "fail_fov", "sec"]].median().round(3).to_string())
