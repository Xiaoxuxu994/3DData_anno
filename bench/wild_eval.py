"""No-GT evaluation of the 7 annotators on the C group (wild clips), based on yt_eval.py.

Per method x clip: multi-view self-consistency and photometric warp error (gap 5 / 20), HFOV, depth median,
runtime. Across methods (cross-model agreement):
  - pairwise Sim(3) trajectory ATE as % of trajectory length, pairwise HFOV difference;
  - pairwise depth agreement: AbsRel after one per-clip median scale, and the metric scale ratio itself;
  - per method "consensus deviation" = median over the other methods of the pairwise numbers.
Renders outputs/bench/wild/eval/vis/<clip>.jpg: RGB + every method's inverse depth at 3 frames + trajectories.

  python wild_eval.py [METHOD ...]  -> bench/wild/eval/{per_clip.csv, agreement.csv, consensus.csv, summary.json, vis/}
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/data/home/z00533370/code/bench")
from eval3d_bench import load_method  # noqa: E402

# spatialvid_metric = spatialvid with camera_tracking's normalisation undone (same geometry, metric scale)
METHODS = ["vipe", "spatialvid_metric", "voyager", "vggt", "vggt_omega", "da3", "pi3x"]

sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
from eval_vipe import umeyama  # noqa: E402

B = Path("/data/home/z00533370/outputs/bench/wild")
MAX_SIDE = 960  # evaluation grid: long side <= 960


def grid(c):
    h, w = c["hw"]
    s = min(1.0, MAX_SIDE / max(h, w))
    return int(round(h * s)) // 2 * 2, int(round(w * s)) // 2 * 2


def load(method, c):
    H, W = grid(c)
    return load_method(B, method, c["clip"], c["n_frames"], H, W)


def rgb(c, k):
    H, W = grid(c)
    im = cv2.imread(str(B / "frames" / c["clip"] / f"{k:06d}.jpg"))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)


def warp_stats(c2w, K, D, c, gap, every=6):
    H, W = D.shape[1:]
    v, u = np.mgrid[0:H, 0:W]
    cons, photo = [], []
    for i in range(0, len(D) - gap, every):
        j = i + gap
        z = D[i]
        m = np.isfinite(z) & (z > 0)
        X = np.stack([(u[m] - K[i, 0, 2]) / K[i, 0, 0] * z[m], (v[m] - K[i, 1, 2]) / K[i, 1, 1] * z[m], z[m]], 1)
        Tij = np.linalg.inv(c2w[j]) @ c2w[i]
        Xc = X @ Tij[:3, :3].T + Tij[:3, 3]
        zj = Xc[:, 2]
        uv = np.stack([K[j, 0, 0] * Xc[:, 0] / zj + K[j, 0, 2], K[j, 1, 1] * Xc[:, 1] / zj + K[j, 1, 2]], 1)
        ok = (zj > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        if ok.sum() < 500:
            continue
        mx = np.full((H, W), -1, np.float32)
        my = np.full((H, W), -1, np.float32)
        idx = np.flatnonzero(m)[ok]
        mx.flat[idx], my.flat[idx] = uv[ok, 0], uv[ok, 1]
        dj = np.nan_to_num(cv2.remap(D[j], mx, my, cv2.INTER_NEAREST, borderValue=0), nan=0).flat[idx]
        good = dj > 0
        if good.sum() < 100:
            continue
        cons.append(np.mean(np.abs(zj[ok][good] - dj[good]) / dj[good] < 0.05))
        Ii, Ij = rgb(c, i).astype(np.float32), rgb(c, j).astype(np.float32)
        wj = cv2.remap(Ij, mx, my, cv2.INTER_LINEAR, borderValue=0)
        photo.append(np.abs(wj - Ii).mean(-1).flat[idx].mean())
    return (float(np.mean(cons)) if cons else np.nan), (float(np.mean(photo)) if photo else np.nan)


def per_clip(args):
    method, c = args
    r = {"method": method, "clip": c["clip"], "cat": c["cat"], "n_frames": c["n_frames"],
         "hard_case": c.get("hard_case")}
    try:
        c2w, K, D, sec = load(method, c)
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
        return r
    W = D.shape[2]
    p = c2w[:, :3, 3]
    r["sec"] = sec
    r["traj_len"] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    r["rot_total_deg"] = float(sum(np.degrees(np.arccos(np.clip((np.trace(c2w[k, :3, :3].T @ c2w[k + 1, :3, :3]) - 1) / 2, -1, 1)))
                                   for k in range(len(c2w) - 1)))
    r["hfov_deg"] = float(np.degrees(2 * np.arctan(W / 2 / K[:, 0, 0])).mean())
    r["depth_p50"] = float(np.nanmedian(D[D > 0]))
    r["nonfinite"] = float(not (np.isfinite(c2w).all() and np.isfinite(K).all()))
    for gap in (5, 20):
        r[f"g{gap}_selfcons"], r[f"g{gap}_photo"] = warp_stats(c2w, K, D, c, gap)
    return r


def agree_clip(c, methods=METHODS):
    data = {}
    for m in methods:
        try:
            data[m] = load(m, c)
        except Exception:  # noqa: BLE001
            pass
    ks = np.linspace(0, c["n_frames"] - 1, 8).astype(int)
    rows = []
    for a, b in itertools.combinations(sorted(data), 2):
        (Ta, Ka, Da, _), (Tb, Kb, Db, _) = data[a], data[b]
        pa, pb = Ta[:, :3, 3], Tb[:, :3, 3]
        r = {"clip": c["clip"], "cat": c["cat"], "a": a, "b": b}
        La, Lb = np.linalg.norm(np.diff(pa, axis=0), axis=1).sum(), np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()
        if min(La, Lb) > 1e-6 and np.isfinite(pa).all() and np.isfinite(pb).all():
            # symmetric: average of a->b and b->a, each as % of the target trajectory length
            ab = []
            for (x, y, L) in ((pa, pb, Lb), (pb, pa, La)):
                s, R, t = umeyama(x, y, True)
                ab.append(100 * np.sqrt(((s * x @ R.T + t - y) ** 2).sum(1).mean()) / L)
            r["traj_ate_pct"] = float(np.mean(ab))
        W = Da.shape[2]
        r["hfov_diff_deg"] = float(abs(np.degrees(2 * np.arctan(W / 2 / Ka[:, 0, 0])).mean()
                                       - np.degrees(2 * np.arctan(W / 2 / Kb[:, 0, 0])).mean()))
        x, y = Da[ks], Db[ks]
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.sum() > 1000:
            ratio = np.median(y[m]) / np.median(x[m])
            r["depth_absrel_aligned"] = float(np.mean(np.abs(x[m] * ratio - y[m]) / y[m]))
            r["depth_log_scale_ratio"] = float(np.log(ratio))  # b relative to a, 0 = same metric scale
        rows.append(r)
    return rows


def render(c, methods=METHODS):
    out = B / "eval" / "vis"
    out.mkdir(parents=True, exist_ok=True)
    n = c["n_frames"]
    ks = [0, n // 2, n - 1]
    data = {}
    for m in methods:
        try:
            data[m] = load(m, c)
        except Exception:  # noqa: BLE001
            pass
    H, W = grid(c)
    cols = 1 + len(data) + 1
    fig, ax = plt.subplots(3, cols, figsize=(2.6 * cols * W / H / 1.6, 6.0), dpi=100, squeeze=False)
    for r_, k in enumerate(ks):
        ax[r_, 0].imshow(rgb(c, k))
        ax[r_, 0].set_title(f"frame {k}", fontsize=9)
        for j, (m, (c2w, K, D, _)) in enumerate(data.items(), start=1):
            d = D[k]
            v = np.isfinite(d) & (d > 0)
            lo, hi = (np.percentile(1 / d[v], 2), np.percentile(1 / d[v], 98)) if v.sum() > 10 else (0, 1)
            ax[r_, j].imshow(np.where(v, 1 / np.maximum(np.nan_to_num(d, nan=1), 1e-6), np.nan), cmap="turbo", vmin=lo, vmax=hi)
            if r_ == 0:
                ax[r_, j].set_title(f"{m}  med {np.nanmedian(D[D > 0]):.2g}", fontsize=9)
    gs = ax[0, -1].get_gridspec()
    for a in ax[:, -1]:
        a.remove()
    at = fig.add_subplot(gs[:, -1])
    ref = data["vipe"][0][:, :3, 3] if "vipe" in data else (next(iter(data.values()))[0][:, :3, 3] if data else None)
    for m, (c2w, *_rest) in data.items():
        p = c2w[:, :3, 3]
        if not np.isfinite(p).all():
            continue
        s, R, t = umeyama(p, ref, True)
        q = s * p @ R.T + t
        at.plot(q[:, 0], q[:, 2], label=m, lw=1.3)
        at.plot(q[:1, 0], q[:1, 2], "k.", ms=4)
    at.set_aspect("equal", adjustable="datalim")
    at.legend(fontsize=7)
    at.set_title("trajectories x-z (Sim3 to vipe)", fontsize=9)
    for a in ax[:, :-1].ravel():
        a.axis("off")
    fig.suptitle(f"{c['clip']}  [{c['cat']}]  {n} frames  {c.get('hard_case') or ''}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{c['clip']}.jpg", pil_kwargs={"quality": 85})
    plt.close(fig)


def main():
    methods = sys.argv[1:] or METHODS
    clips = json.loads((B / "clips.json").read_text())
    out = B / "eval"
    out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(per_clip, [(m, c) for m in methods for c in clips]))
        ag = [r for rs in ex.map(agree_clip, clips, [methods] * len(clips)) for r in rs]
        list(ex.map(render, clips, [methods] * len(clips)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_clip.csv", index=False, float_format="%.4f")
    ag = pd.DataFrame(ag)
    ag.to_csv(out / "agreement.csv", index=False, float_format="%.4f")
    # consensus deviation per method x clip: median over partners
    both = pd.concat([ag.rename(columns={"a": "method", "b": "other"}),
                      ag.rename(columns={"b": "method", "a": "other"}).assign(
                          depth_log_scale_ratio=lambda d: -d["depth_log_scale_ratio"])])
    cons = both.groupby(["method", "clip", "cat"])[["traj_ate_pct", "hfov_diff_deg", "depth_absrel_aligned"]].median().reset_index()
    cons["outlier_traj"] = (cons["traj_ate_pct"] > 5).astype(float)
    cons["outlier_fov"] = (cons["hfov_diff_deg"] > 10).astype(float)
    cons.to_csv(out / "consensus.csv", index=False, float_format="%.4f")
    metric_m = ["vipe", "spatialvid_metric", "voyager", "da3", "pi3x"]
    mm = ag[ag["a"].isin(metric_m) & ag["b"].isin(metric_m)]
    ok = df[df["error"].isna()] if "error" in df else df
    cols = ["g5_selfcons", "g20_selfcons", "g5_photo", "g20_photo", "hfov_deg", "depth_p50", "sec"]
    summ = {
        "n_clips": len(clips),
        "n_ok": ok.groupby("method").size().to_dict(),
        "by_method": ok.groupby("method")[cols].median().round(4).to_dict(orient="index"),
        "consensus_by_method": cons.groupby("method")[["traj_ate_pct", "hfov_diff_deg", "depth_absrel_aligned",
                                                       "outlier_traj", "outlier_fov"]].median().round(3).to_dict(orient="index"),
        "consensus_outlier_rate": cons.groupby("method")[["outlier_traj", "outlier_fov"]].mean().round(3).to_dict(orient="index"),
        "selfcons_by_cat": ok.pivot_table(index="cat", columns="method", values="g5_selfcons", aggfunc="median").round(3).to_dict(),
        "traj_agreement_by_pair": ag.assign(pair=ag["a"] + "~" + ag["b"]).groupby("pair").traj_ate_pct.median().round(3).to_dict(),
        "metric_scale_spread_by_pair": mm.assign(pair=mm["a"] + "~" + mm["b"]).groupby("pair").depth_log_scale_ratio.apply(
            lambda x: float(np.exp(np.median(np.abs(x))))).round(3).to_dict(),
        "errors": df.loc[df["error"].notna(), ["method", "clip", "error"]].to_dict("records") if "error" in df else [],
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("n_ok", "consensus_outlier_rate")}, indent=1, default=str))


if __name__ == "__main__":
    main()
