"""Point-cloud and reprojection evaluation of ViPE pseudo-labels on ScanNet++.

  python geo_vipe.py metrics            -> eval/geo_per_segment.csv (all finished segments)
  python geo_vipe.py render SEG [SEG..] -> eval/vis/<seg>_*.jpg|png + eval/vis/<seg>_cloud.json

Point clouds: fuse every k-th frame (GT depth + GT pose + GT K) vs (ViPE depth + ViPE pose + ViPE K,
then the trajectory Sim(3) alignment to GT). Accuracy = ViPE->GT NN distance, Completeness =
GT->ViPE NN distance, Chamfer = mean of the two (all in metres, GT frame).

Reprojection, frame pairs (i, i+gap): project pixels of frame i into frame j using
  GT geometry   (GT depth_i, GT poses, GT K)
  ViPE geometry (ViPE depth_i, ViPE poses, ViPE K)   -- no alignment needed, scale cancels
and report the end-point error of the ViPE correspondences w.r.t. the GT ones (px),
the photometric error of warping image j into i (GT vs ViPE geometry), and ViPE's
own multi-view depth consistency (projected depth vs ViPE depth_j, |rel| < 5%).
"""

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
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_vipe import DATA, MAX_DEPTH, PRED, ROOT, read_exr_zip, umeyama  # noqa: E402

W, H = 1034, 690
VIS = ROOT / "eval" / "vis"
GAPS = (5, 20)


def load_segment(name):
    seg = next(s for s in json.loads((ROOT / "segments.json").read_text()) if s["segment"] == name)
    meta = np.load(DATA / seg["scene"] / "scene_metadata.npz")
    idx = {n: i for i, n in enumerate(meta["images"])}
    gi = [idx[f] for f in seg["frames"]]
    pz, iz = np.load(PRED / "pose" / f"{name}.npz"), np.load(PRED / "intrinsics" / f"{name}.npz")
    Kp = np.zeros((len(iz["inds"]), 3, 3))
    Kp[:, 0, 0], Kp[:, 1, 1], Kp[:, 0, 2], Kp[:, 1, 2], Kp[:, 2, 2] = *iz["data"].T, 1
    d = dict(
        seg=seg, T_gt=meta["trajectories"][gi], K_gt=meta["intrinsics"][gi],
        T_p=pz["data"][np.argsort(pz["inds"])], K_p=Kp[np.argsort(iz["inds"])],
        D_p=read_exr_zip(PRED / "depth" / f"{name}.zip"),
    )
    s, R, t = umeyama(d["T_p"][:, :3, 3], d["T_gt"][:, :3, 3], True)
    d["sim3"] = (s, R, t)
    return d


def gt_depth(d, k):
    g = cv2.imread(str(DATA / d["seg"]["scene"] / "depth" / f"{d['seg']['frames'][k]}.png"), cv2.IMREAD_UNCHANGED)
    g = g.astype(np.float32)[:, :W] * 1e-3
    g[g > MAX_DEPTH] = 0
    return g


def rgb(d, k):
    return cv2.cvtColor(cv2.imread(str(ROOT / "frames" / d["seg"]["segment"] / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)


def unproject(depth, K, T, stride=1, mask=None):
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    m = (z > 0) & np.isfinite(z)
    if mask is not None:
        m &= mask[::stride, ::stride]
    X = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
    return X @ T[:3, :3].T + T[:3, 3], m


def project(Xw, K, T):
    Ti = np.linalg.inv(T)
    Xc = Xw @ Ti[:3, :3].T + Ti[:3, 3]
    z = Xc[:, 2]
    uv = np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)
    return uv, z


# ------------------------------------------------------------------ point clouds
def fused_clouds(d, n_frames=40, stride=6, align="pairs"):
    """Fused GT / ViPE clouds over the same pixels, ViPE mapped into the GT frame.

    align="pairs": Sim(3) fitted on the pixel-paired 3D points (pure geometry quality);
    align="traj":  Sim(3) fitted on camera centres (what a trajectory-only alignment gives).
    """
    N = len(d["T_gt"])
    ks = np.linspace(0, N - 1, min(n_frames, N)).astype(int)
    G, P, C = [], [], []
    for k in ks:
        g, p, img = gt_depth(d, k), d["D_p"][k], rgb(d, k)
        common = (g > 0) & np.isfinite(p) & (p > 0)
        Xg, m = unproject(g, d["K_gt"][k], d["T_gt"][k], stride, mask=common)
        Xp, _ = unproject(p, d["K_p"][k], d["T_p"][k], stride, mask=common)
        G.append(Xg)
        P.append(Xp)
        C.append(img[::stride, ::stride][m])
    G, P, C = np.concatenate(G), np.concatenate(P), np.concatenate(C)
    if align == "pairs":
        s, R, t = umeyama(P, G, True)
    else:
        s, R, t = d["sim3"]
    return G, s * P @ R.T + t, C, (s, R, t)


def cloud_metrics(G, P):
    dp = cKDTree(G).query(P, workers=4)[0]
    dg = cKDTree(P).query(G, workers=4)[0]
    return {
        "pc_acc_m": float(np.median(dp)), "pc_comp_m": float(np.median(dg)),
        "pc_chamfer_m": float((dp.mean() + dg.mean()) / 2),
        "pc_acc_5cm": float((dp < 0.05).mean()), "pc_comp_5cm": float((dg < 0.05).mean()),
        "pc_fscore_5cm": float(2 * (dp < 0.05).mean() * (dg < 0.05).mean() / max((dp < 0.05).mean() + (dg < 0.05).mean(), 1e-9)),
    }


# ------------------------------------------------------------------ reprojection
def warp_pair(d, i, j, which):
    """Return (uv in frame j for every pixel of i, valid mask HxW, projected z)."""
    if which == "gt":
        dep, Ki, Kj, Ti, Tj = gt_depth(d, i), d["K_gt"][i], d["K_gt"][j], d["T_gt"][i], d["T_gt"][j]
    else:
        dep, Ki, Kj, Ti, Tj = d["D_p"][i], d["K_p"][i], d["K_p"][j], d["T_p"][i], d["T_p"][j]
    Xw, m = unproject(dep, Ki, Ti)
    uv, z = project(Xw, Kj, Tj)
    full = np.full((H, W, 2), np.nan, np.float32)
    zz = np.full((H, W), np.nan, np.float32)
    full[m], zz[m] = uv, z
    ok = m.copy()
    ok[m] = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    return full, ok, zz


def remap(img, uv):
    return cv2.remap(img, np.nan_to_num(uv[..., 0], nan=-1), np.nan_to_num(uv[..., 1], nan=-1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def reproj_metrics(d, gap, every=10):
    N = len(d["T_gt"])
    rows = []
    for i in range(0, N - gap, every):
        j = i + gap
        uv_g, ok_g, _ = warp_pair(d, i, j, "gt")
        uv_p, ok_p, z_p = warp_pair(d, i, j, "vipe")
        both = ok_g & ok_p
        if both.sum() < 1000:
            continue
        epe = np.linalg.norm(uv_p[both] - uv_g[both], axis=1)
        Ii, Ij = rgb(d, i).astype(np.float32), rgb(d, j).astype(np.float32)
        ph_g = np.abs(remap(Ij, uv_g) - Ii).mean(-1)[both].mean()
        ph_p = np.abs(remap(Ij, uv_p) - Ii).mean(-1)[both].mean()
        dj = remap(d["D_p"][j], uv_p)[both]
        cons = (np.abs(z_p[both] - dj) / np.maximum(dj, 1e-6) < 0.05).mean()
        rows.append((np.mean(epe), np.median(epe), (epe < 1).mean(), (epe < 3).mean(), ph_g, ph_p, cons))
    if not rows:
        return {}
    a = np.array(rows).mean(0)
    keys = ["epe_mean_px", "epe_med_px", "epe_lt1px", "epe_lt3px", "photo_gt", "photo_vipe", "vipe_selfcons_5pct"]
    return {f"g{gap}_{k}": float(v) for k, v in zip(keys, a)}


def segment_metrics(name):
    d = load_segment(name)
    r = {"segment": name, "n_frames": len(d["T_gt"])}
    G, P, _, (s_pair, _, _) = fused_clouds(d, align="pairs")
    r.update(cloud_metrics(G, P))
    r["pc_pair_scale"] = float(s_pair)
    G, P, _, _ = fused_clouds(d, align="traj")
    r.update({k.replace("pc_", "pc_traj_"): v for k, v in cloud_metrics(G, P).items()})
    for gap in GAPS:
        if len(d["T_gt"]) > gap + 1:
            r.update(reproj_metrics(d, gap))
    return r


# ------------------------------------------------------------------ rendering
def depth_color(x, lo, hi):
    c = (matplotlib.colormaps["turbo_r"](np.clip((x - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
    c[~(np.isfinite(x) & (x > 0))] = 0
    return c


def render(name):
    VIS.mkdir(parents=True, exist_ok=True)
    d = load_segment(name)
    N = len(d["T_gt"])

    # 1) fused clouds: top-down + oblique, GT / ViPE / overlay
    G, P, Cg, (s_pair, R, t) = fused_clouds(d, n_frames=60, stride=5, align="pairs")
    Cp = Cg
    rng = np.random.default_rng(0)
    tg = d["T_gt"][:, :3, 3]
    tp = s_pair * d["T_p"][:, :3, 3] @ R.T + t  # camera centres under the same point-based Sim(3)
    zlo, zhi = np.percentile(G[:, 2], 5) + 0.3, np.percentile(G[:, 2], 95) - 0.3
    def pick(X, n, top):
        idx = np.arange(len(X))
        if top:
            idx = idx[(X[:, 2] > zlo) & (X[:, 2] < zhi)]
        return rng.choice(idx, min(len(idx), n), replace=False)
    lo, hi = np.percentile(G, 1, 0), np.percentile(G, 99, 0)
    fig, ax = plt.subplots(2, 3, figsize=(18, 11), dpi=110)
    for row, (a_, b_, lab) in enumerate(((0, 1, "top-down (x-y, floor/ceiling removed)"), (0, 2, "side (x-z)"))):
        sg, sp = pick(G, 120000, row == 0), pick(P, 120000, row == 0)
        for col, (X, C, title) in enumerate(((G[sg], Cg[sg] / 255, "GT"), (P[sp], Cp[sp] / 255, "ViPE"), (None, None, "overlay"))):
            axx = ax[row, col]
            if X is not None:
                axx.scatter(X[:, a_], X[:, b_], c=C, s=0.3, linewidths=0)
            else:
                axx.scatter(G[sg][:, a_], G[sg][:, b_], c="#5f6368", s=0.3, linewidths=0, alpha=0.6, label="GT")
                axx.scatter(P[sp][:, a_], P[sp][:, b_], c="#e8453c", s=0.3, linewidths=0, alpha=0.35, label="ViPE")
            axx.plot(tg[:, a_], tg[:, b_], c="k", lw=1.2, label="GT traj")
            axx.plot(tp[:, a_], tp[:, b_], c="#1a73e8", lw=1.2, ls="--", label="ViPE traj")
            axx.set_xlim(lo[a_], hi[a_]); axx.set_ylim(lo[b_], hi[b_]); axx.set_aspect("equal")
            axx.set_title(f"{title} - {lab}", fontsize=11)
            if row == 0 and col == 2:
                axx.legend(loc="upper right", markerscale=12, fontsize=8)
    fig.suptitle(f"{name}  ({N} frames)  fused point clouds from 60 frames, ViPE Sim(3)-aligned on paired points (scale {s_pair:.3f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(VIS / f"{name}_cloud.png")
    plt.close(fig)

    # interactive viewer payload (GT frame, metres)
    def pack(X, C, n):
        k = rng.choice(len(X), min(len(X), n), replace=False)
        return {"xyz": np.round(X[k], 3).ravel().tolist(), "rgb": C[k].ravel().tolist()}

    (VIS / f"{name}_cloud.json").write_text(json.dumps({
        "segment": name, "gt": pack(G, Cg, 30000), "vipe": pack(P, Cp, 30000),
        "traj_gt": np.round(tg, 3).tolist(), "traj_vipe": np.round(tp, 3).tolist(),
    }))

    # 2) depth panel: RGB / GT / ViPE (metric) / AbsRel for 3 frames
    ks = [0, N // 2, N - 1]
    fig, ax = plt.subplots(3, 4, figsize=(20, 10), dpi=100)
    for r_, k in enumerate(ks):
        g, p = gt_depth(d, k), d["D_p"][k]
        v = g > 0
        lo_, hi_ = np.percentile(g[v], 2), np.percentile(g[v], 98)
        err = np.where(v, np.abs(p - g) / np.where(v, g, 1), np.nan)
        for c_, (im, tt) in enumerate(((rgb(d, k), f"frame {k} RGB"), (depth_color(g, lo_, hi_), "GT depth"),
                                       (depth_color(p, lo_, hi_), "ViPE depth (metric, same colormap)"))):
            ax[r_, c_].imshow(im); ax[r_, c_].set_title(tt, fontsize=10); ax[r_, c_].axis("off")
        h = ax[r_, 3].imshow(err, cmap="magma", vmin=0, vmax=0.3)
        ax[r_, 3].set_title(f"AbsRel  mean={np.nanmean(err):.3f}", fontsize=10); ax[r_, 3].axis("off")
        fig.colorbar(h, ax=ax[r_, 3], fraction=0.03)
    fig.tight_layout()
    fig.savefig(VIS / f"{name}_depth.jpg", pil_kwargs={"quality": 85})
    plt.close(fig)

    # 3) reprojection panel for a pair in the middle (gap 20)
    i = max(0, N // 2 - 10)
    j = min(N - 1, i + 20)
    Ii, Ij = rgb(d, i).astype(np.float32), rgb(d, j).astype(np.float32)
    uv_g, ok_g, _ = warp_pair(d, i, j, "gt")
    uv_p, ok_p, _ = warp_pair(d, i, j, "vipe")
    both = ok_g & ok_p
    wg, wp = remap(Ij, uv_g), remap(Ij, uv_p)
    eg = np.where(ok_g, np.abs(wg - Ii).mean(-1), np.nan)
    ep = np.where(ok_p, np.abs(wp - Ii).mean(-1), np.nan)
    epe = np.where(both, np.linalg.norm(np.nan_to_num(uv_p - uv_g), axis=-1), np.nan)
    fig, ax = plt.subplots(2, 4, figsize=(20, 7.2), dpi=100)
    panels = [
        (Ii / 255, f"target frame i={i}"), (Ij / 255, f"source frame j={j}"),
        (np.where(ok_g[..., None], wg, 0) / 255, "j warped to i (GT geometry)"),
        (np.where(ok_p[..., None], wp, 0) / 255, "j warped to i (ViPE geometry)"),
    ]
    for c_, (im, tt) in enumerate(panels):
        ax[0, c_].imshow(np.clip(im, 0, 1)); ax[0, c_].set_title(tt, fontsize=10); ax[0, c_].axis("off")
    for c_, (im, tt, vmax, cm) in enumerate((
        (eg, f"photometric |I_i - warp| GT  mean={np.nanmean(eg[both]):.1f}", 60, "inferno"),
        (ep, f"photometric |I_i - warp| ViPE  mean={np.nanmean(ep[both]):.1f}", 60, "inferno"),
        (epe, f"EPE ViPE vs GT corr.  median={np.nanmedian(epe):.2f}px", 5, "viridis"),
    )):
        h = ax[1, c_].imshow(im, cmap=cm, vmin=0, vmax=vmax); ax[1, c_].set_title(tt, fontsize=10); ax[1, c_].axis("off")
        fig.colorbar(h, ax=ax[1, c_], fraction=0.03)
    ax[1, 3].hist(epe[both], bins=np.linspace(0, 10, 51), color="#1a73e8")
    ax[1, 3].set_title("EPE histogram (px)", fontsize=10)
    fig.tight_layout()
    fig.savefig(VIS / f"{name}_reproj.jpg", pil_kwargs={"quality": 85})
    plt.close(fig)
    print("rendered", name)


if __name__ == "__main__":
    if sys.argv[1] == "metrics":
        done = [s["segment"] for s in json.loads((ROOT / "segments.json").read_text())
                if (PRED / "depth" / f"{s['segment']}.zip").exists()]
        with ProcessPoolExecutor(16) as ex:
            rows = list(ex.map(segment_metrics, done))
        df = pd.DataFrame(rows).sort_values("segment")
        df.to_csv(ROOT / "eval" / "geo_per_segment.csv", index=False, float_format="%.4f")
        print(df.drop(columns="segment").mean().round(4).to_string())
    else:
        for n in sys.argv[2:]:
            render(n)
