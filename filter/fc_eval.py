"""No-GT QC of annotators on the filter end-to-end check (copy of code/bench/yt_eval.py with B from $FC_BENCH).

Per method x clip: multi-view self-consistency and photometric warp error (gap 5 / 20), HFOV, depth range,
runtime. Across methods: Sim(3) trajectory agreement (ATE as % of trajectory length) and HFOV difference.
Also renders a panel per clip: RGB + each method's depth at 3 frames + top-down trajectories.

  python yt_eval.py METHOD [METHOD ...]  -> bench/youtube/eval/{per_clip.csv, agreement.csv, summary.json, vis/*.jpg}
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
from eval_bench import exr_zip  # noqa: E402

sys.path.insert(0, "/data/home/z00533370/code/vipe_eval")
from eval_vipe import umeyama  # noqa: E402

import os
B = Path(os.environ.get("FC_BENCH", "/data/home/z00533370/outputs/bench/filter_check"))
H, W = 540, 960  # evaluation grid (half of 1080p)


def load(method, clip, n):
    if method == "vipe":
        o = B / "vipe"
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
    assert len(c2w) == n, f"{method}/{clip}: {len(c2w)} vs {n}"
    if method == "voyager":  # sky pixels are filled with one far constant -> invalid
        far = np.nanmax(D)
        D = np.where(D >= far * 0.999, np.nan, D).astype(np.float32)
    h, w = D.shape[1:]
    sx, sy = W / w, H / h
    D = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in D])
    K = K.copy()
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    K[:, 0, 2] += 0.5 * sx - 0.5
    K[:, 1, 2] += 0.5 * sy - 0.5
    return c2w, K, D, sec


def rgb(clip, k):
    im = cv2.imread(str(B / "frames" / clip / f"{k:06d}.jpg"))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)


def warp_stats(c2w, K, D, clip, gap, every=6):
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
        if ok.sum() < 1000:
            continue
        mx = np.full((H, W), -1, np.float32)
        my = np.full((H, W), -1, np.float32)
        idx = np.flatnonzero(m)[ok]
        mx.flat[idx], my.flat[idx] = uv[ok, 0], uv[ok, 1]
        dj = cv2.remap(D[j], mx, my, cv2.INTER_NEAREST, borderValue=0).flat[idx]
        good = dj > 0
        cons.append(np.mean(np.abs(zj[ok][good] - dj[good]) / dj[good] < 0.05))
        Ii, Ij = rgb(clip, i).astype(np.float32), rgb(clip, j).astype(np.float32)
        wj = cv2.remap(Ij, mx, my, cv2.INTER_LINEAR, borderValue=0)
        photo.append(np.abs(wj - Ii).mean(-1).flat[idx].mean())
    return (float(np.mean(cons)) if cons else np.nan), (float(np.mean(photo)) if photo else np.nan)


def per_clip(args):
    method, c = args
    r = {"method": method, "clip": c["clip"], "cat": c["cat"], "n_frames": c["n_frames"]}
    try:
        c2w, K, D, sec = load(method, c["clip"], c["n_frames"])
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:200]
        return r
    p = c2w[:, :3, 3]
    r["sec"] = sec
    r["traj_len"] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    r["rot_total_deg"] = float(sum(np.degrees(np.arccos(np.clip((np.trace(c2w[k, :3, :3].T @ c2w[k + 1, :3, :3]) - 1) / 2, -1, 1)))
                                   for k in range(len(c2w) - 1)))
    r["hfov_deg"] = float(np.degrees(2 * np.arctan(W / 2 / K[:, 0, 0])).mean())
    r["depth_p50"] = float(np.nanmedian(D[D > 0]))
    for gap in (5, 20):
        r[f"g{gap}_selfcons"], r[f"g{gap}_photo"] = warp_stats(c2w, K, D, c["clip"], gap)
    return r


def agreement(clips, methods):
    rows = []
    for c in clips:
        tr = {}
        for m in methods:
            try:
                tr[m] = load(m, c["clip"], c["n_frames"])[0]
            except Exception:  # noqa: BLE001
                pass
        for a, b in itertools.combinations(sorted(tr), 2):
            pa, pb = tr[a][:, :3, 3], tr[b][:, :3, 3]
            s, R, t = umeyama(pa, pb, True)
            L = np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()
            ate = np.sqrt(((s * pa @ R.T + t - pb) ** 2).sum(1).mean())
            rows.append({"clip": c["clip"], "cat": c["cat"], "pair": f"{a}~{b}", "ate_pct_of_len": 100 * ate / max(L, 1e-9)})
    return pd.DataFrame(rows)


def render(c, methods):
    out = B / "eval" / "vis"
    out.mkdir(parents=True, exist_ok=True)
    n = c["n_frames"]
    ks = [0, n // 2, n - 1]
    data = {}
    for m in methods:
        try:
            data[m] = load(m, c["clip"], n)
        except Exception:  # noqa: BLE001
            pass
    cols = 1 + len(data) + 1
    fig, ax = plt.subplots(3, cols, figsize=(3.2 * cols, 6.2), dpi=100)
    for r_, k in enumerate(ks):
        ax[r_, 0].imshow(rgb(c["clip"], k)); ax[r_, 0].set_title(f"frame {k}", fontsize=9)
        for j, (m, (c2w, K, D, _)) in enumerate(data.items(), start=1):
            d = D[k]
            v = np.isfinite(d) & (d > 0)
            lo, hi = (np.percentile(1 / d[v], 2), np.percentile(1 / d[v], 98)) if v.any() else (0, 1)
            ax[r_, j].imshow(np.where(v, 1 / np.maximum(d, 1e-6), np.nan), cmap="turbo", vmin=lo, vmax=hi)
            ax[r_, j].set_title(f"{m} (inverse depth)", fontsize=9)
    gs = ax[0, -1].get_gridspec()
    for a in ax[:, -1]:
        a.remove()
    at = fig.add_subplot(gs[:, -1])
    ref = next(iter(data.values()))[0][:, :3, 3] if data else None
    for m, (c2w, *_rest) in data.items():
        p = c2w[:, :3, 3]
        s, R, t = umeyama(p, ref, True)
        q = s * p @ R.T + t
        at.plot(q[:, 0], q[:, 2], label=m, lw=1.4)
    at.set_aspect("equal"); at.legend(fontsize=8); at.set_title("trajectories (x-z, Sim3 to first)", fontsize=9)
    for a in ax[:, :-1].ravel():
        a.axis("off")
    fig.suptitle(f"{c['clip']}  ({c['cat']}, {n} frames @15fps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{c['clip']}.jpg", pil_kwargs={"quality": 85})
    plt.close(fig)


def main():
    methods = sys.argv[1:]
    clips = json.loads((B / "clips.json").read_text())
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(per_clip, [(m, c) for m in methods for c in clips]))
        list(ex.map(render, clips, [methods] * len(clips)))
    df = pd.DataFrame(rows)
    (B / "eval").mkdir(exist_ok=True)
    df.to_csv(B / "eval" / "per_clip.csv", index=False, float_format="%.4f")
    ag = agreement(clips, methods)
    ag.to_csv(B / "eval" / "agreement.csv", index=False, float_format="%.3f")
    cols = ["g5_selfcons", "g20_selfcons", "g5_photo", "g20_photo", "hfov_deg", "sec"]
    summ = {"by_method": df.groupby("method")[[c for c in cols if c in df]].median().round(4).to_dict(orient="index"),
            "by_cat_method_selfcons": df.pivot_table(index="cat", columns="method", values="g5_selfcons").round(3).to_dict(),
            "agreement_median_by_pair": ag.groupby("pair").ate_pct_of_len.median().round(3).to_dict() if len(ag) else {},
            "errors": df[df.get("error").notna()][["method", "clip", "error"]].to_dict("records") if "error" in df else []}
    (B / "eval" / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
