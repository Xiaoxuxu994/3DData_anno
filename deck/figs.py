"""Deck figures from real results (CPU only). Output: /data/home/z00533370/outputs/deck_figs/*.png"""
import json
import sys
import traceback
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
cv2.setNumThreads(2)
H0 = Path("/data/home/z00533370")
OUT = H0 / "outputs/deck_figs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(H0 / "code/hybrid"))
sys.path.insert(0, str(H0 / "code/bench"))
sys.path.insert(0, str(H0 / "code/vipe_eval"))
from eval_vipe import umeyama  # noqa: E402

INK, MUTED, GRID = "#1B2430", "#5B6675", "#D9DEE5"
COL = {"GT": "#1B2430", "vipe": "#2F6DB5", "spatialvid": "#7FB0E0", "spatialvid_metric": "#7FB0E0", "voyager": "#9AA3AE",
       "vggt": "#C49BD6", "vggt_omega": "#7B4FA0", "da3": "#3E9C6A", "pi3x": "#E07B39", "hybrid": "#C8364A"}
NAME = {"vipe": "ViPE", "spatialvid": "SpatialVID", "spatialvid_metric": "SpatialVID", "voyager": "Voyager", "vggt": "VGGT",
        "vggt_omega": "VGGT-Ω", "da3": "DA3", "pi3x": "Pi3X", "hybrid": "ViPE+Pi3X"}
plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 110})
M7 = ["vipe", "spatialvid", "voyager", "vggt", "vggt_omega", "da3", "pi3x"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name, flush=True)


def inv_show(ax, d, vmax, title, sub=None):
    with np.errstate(all="ignore"):
        inv = np.where((d > 0) & np.isfinite(d), 1.0 / d, np.nan)
    ax.imshow(inv, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=17, color=INK, pad=6)
    if sub:
        ax.text(0.02, 0.04, sub, transform=ax.transAxes, fontsize=14, color="white",
                bbox=dict(facecolor="#1B2430", alpha=0.75, pad=3, edgecolor="none"))
    ax.axis("off")


def frame_align(p, g):
    m = (g > 0) & np.isfinite(p) & (p > 0)
    s = np.median(g[m]) / np.median(p[m])
    return p * s, float(np.mean(np.abs(p[m] * s - g[m]) / g[m]))


# ---------------------------------------------------------------- 1. ScanNet++ unseen depth + error + cloud
def fig_scannetpp():
    import eval_bench as EB
    import geo_vipe as G
    import hybrid as HY
    clips = json.loads((EB.B / "clips.json").read_text())
    c = next(x for x in clips if x["clip"] == "1c7a683c92_b")
    clip, n = c["clip"], c["n_frames"]
    meta = np.load(EB.DATA / c["scene"] / "scene_metadata.npz")
    idx = {f: i for i, f in enumerate(meta["images"])}
    gi = [idx[f] for f in c["frames"]]
    T_gt, K_gt = meta["trajectories"][gi], meta["intrinsics"][gi]
    k = n // 2
    gd = cv2.imread(str(EB.DATA / c["scene"] / "depth" / f"{c['frames'][k]}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :EB.W] * 1e-3
    img = cv2.cvtColor(cv2.imread(str(EB.B / "frames" / clip / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
    cache = {}

    def base(m):
        if m not in cache:
            cache[m] = EB.load_method(m, clip, n)
        return cache[m]
    preds = {m: base(m) for m in M7}
    hyb = HY.compose("H:vipe+pi3x+geo", base, {})
    preds["hybrid"] = hyb[:4]
    vmax = np.nanpercentile(1 / gd[gd > 0], 98)
    order = ["vipe", "spatialvid", "voyager", "vggt", "vggt_omega", "da3", "pi3x", "hybrid"]
    fig, axs = plt.subplots(2, 5, figsize=(22, 8.2))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("RGB", fontsize=17)
    axs[0, 0].axis("off")
    inv_show(axs[0, 1], gd, vmax, "GT depth")
    errs = {}
    for ax, m in zip(list(axs[0, 2:]) + list(axs[1]), order):
        p = preds[m][2][k]
        pa, e = frame_align(p, gd)
        errs[m] = (pa, e)
        inv_show(ax, pa, vmax, NAME[m], f"AbsRel {e:.3f}")
    save(fig, "f01_depth_scannetpp_unseen.png")
    # error maps
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.4))
    for ax, m in zip(axs.flat, order):
        pa, e = errs[m]
        with np.errstate(all="ignore"):
            er = np.where(gd > 0, np.abs(pa - gd) / gd, np.nan)
        ax.imshow(er, cmap="inferno", vmin=0, vmax=0.3)
        ax.set_title(f"{NAME[m]}  AbsRel {e:.3f}", fontsize=16)
        ax.axis("off")
    save(fig, "f02_depth_error_scannetpp_unseen.png")
    # fused point clouds, aligned by camera centres (tests depth/trajectory scale consistency)
    pc = pd.read_csv(H0 / "outputs/hybrid/scannetpp_per_clip.csv")
    pc = pc[pc["clip"] == clip].set_index("method")
    show = [("vipe", "vipe"), ("pi3x", "pi3x"), ("H:vipe+pi3x+geo", "hybrid")]
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.6))
    for j, (spec, key) in enumerate(show):
        c2w, K, D, _ = preds[key]
        d = {"seg": {"scene": c["scene"], "segment": c["segment"], "frames": c["frames"]},
             "T_gt": T_gt, "K_gt": K_gt, "T_p": c2w, "K_p": K, "D_p": {i: D[i] for i in range(n)}}
        d["sim3"] = umeyama(c2w[:, :3, 3], T_gt[:, :3, 3], True)
        Gc, Pc, C, _ = G.fused_clouds(d, n_frames=40, stride=8, align="traj")
        if j == 0:
            sel = np.random.default_rng(0).choice(len(Gc), min(60000, len(Gc)), replace=False)
            up = np.argmin(np.abs(np.linalg.svd(Gc[sel] - Gc[sel].mean(0))[2]), axis=None)
            ax0 = axs[0]
            ax0.scatter(Gc[sel, 0], Gc[sel, 1], s=0.3, c=C[sel] / 255.0)
            ax0.set_title("GT (fused, top view)", fontsize=17)
            lim = (np.percentile(Gc[sel, 0], [1, 99]), np.percentile(Gc[sel, 1], [1, 99]))
        sel2 = np.random.default_rng(1).choice(len(Pc), min(60000, len(Pc)), replace=False)
        ax = axs[j + 1]
        ax.scatter(Gc[sel, 0], Gc[sel, 1], s=0.3, c="#C9CED6")
        ax.scatter(Pc[sel2, 0], Pc[sel2, 1], s=0.3, c=COL[key], alpha=0.6)
        f = pc.loc[spec, "pc_traj_fscore_5cm"] if spec in pc.index else np.nan
        ax.set_title(f"{NAME[key]}   traj-aligned F@5cm {f:.2f}", fontsize=17)
    for ax in axs:
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "f03_cloud_scannetpp_unseen.png")


# ---------------------------------------------------------------- 2. A-group depth (TUM dynamic) + trajectories
def fig_A():
    import eval3d_bench as E
    B = H0 / "outputs/bench/eval3d_A"
    clips = {x["clip"]: x for x in json.loads((B / "clips.json").read_text())}

    def load(clip):
        c = clips[clip]
        z = np.load(B / "gt" / f"{clip}.npz", allow_pickle=True)
        GD = z["depth"].astype(np.float32)
        H, W = GD.shape[1:]
        P = {}
        for m in M7:
            try:
                P[m] = E.load_method(B, m, clip, c["n_frames"], H, W)
            except Exception as e:  # noqa: BLE001
                print("skip", m, clip, e)
        return c, z["c2w"], GD, P

    # trajectories: TUM fr1_room + KITTI 07
    fig, axs = plt.subplots(1, 2, figsize=(20, 8.5))
    for ax, clip in zip(axs, ["tum__freiburg1_room", "kitti__kitti_07"]):
        c, Tg, GD, P = load(clip)
        pg = Tg[:, :3, 3]
        _, _, Vt = np.linalg.svd(pg - pg.mean(0))
        A2 = Vt[:2].T
        q = (pg - pg.mean(0)) @ A2
        ax.plot(q[:, 0], q[:, 1], color=COL["GT"], lw=4, label="GT", zorder=10)
        for m in M7:
            if m not in P:
                continue
            pp = P[m][0][:, :3, 3]
            s, R, t = umeyama(pp, pg, True)
            al = s * pp @ R.T + t
            qq = (al - pg.mean(0)) @ A2
            ate = np.sqrt(np.mean(np.sum((al - pg) ** 2, 1)))
            L = np.linalg.norm(np.diff(pg, axis=0), axis=1).sum()
            ax.plot(qq[:, 0], qq[:, 1], color=COL[m], lw=2.2, label=f"{NAME[m]}  {100 * ate / L:.2f}%")
        ax.set_aspect("equal")
        ax.set_title(f"{clip.split('__')[1]}  (Sim3-aligned, ATE % of path)", fontsize=17)
        ax.legend(fontsize=13, frameon=False, loc="best")
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "f04_traj_A.png")

    # depth grid: Bonn (dynamic) + 7-Scenes
    for clip, name in [("tum__freiburg3_walking_xyz", "f05_depth_tum_walking.png")]:
        c, Tg, GD, P = load(clip)
        k = c["n_frames"] // 2
        gd = GD[k]
        img = cv2.cvtColor(cv2.imread(str(B / "frames" / clip / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
        vmax = np.nanpercentile(1 / gd[gd > 0], 98)
        fig, axs = plt.subplots(2, 5, figsize=(22, 7.4))
        axs[0, 0].imshow(img)
        axs[0, 0].set_title("RGB", fontsize=17)
        axs[0, 0].axis("off")
        inv_show(axs[0, 1], gd, vmax, "GT depth")
        for ax, m in zip(list(axs[0, 2:]) + list(axs[1]), M7):
            pa, e = frame_align(P[m][2][k], gd)
            inv_show(ax, pa, vmax, NAME[m], f"AbsRel {e:.3f}")
        axs[1, 4].axis("off")
        save(fig, name)


# ---------------------------------------------------------------- 3. FOV disagreement in the wild
def fig_fov():
    w = pd.read_csv(H0 / "outputs/bench/eval3d_report/wild_per_clip.csv")
    clip = "driving__youtube_cc__BTbkSEotmX0"
    d = w[w["clip"] == clip].set_index("method")
    ms = ["vipe", "spatialvid_metric", "voyager", "vggt", "vggt_omega", "da3", "pi3x"]
    img = cv2.cvtColor(cv2.imread(str(H0 / "outputs/bench/wild/frames" / clip / "000090.jpg")), cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6.5), gridspec_kw={"width_ratios": [1.25, 1]})
    axs[0].imshow(img)
    axs[0].axis("off")
    axs[0].set_title("BTbkSEotmX0 (driving, CC-BY)", fontsize=17)
    vals = [d.loc[m, "hfov_deg"] for m in ms]
    y = np.arange(len(ms))
    axs[1].barh(y, vals, color=[COL[m] for m in ms], height=0.62)
    for i, v in enumerate(vals):
        axs[1].text(v + 0.8, i, f"{v:.1f}°", va="center", fontsize=15)
    axs[1].set_yticks(y, [NAME[m] for m in ms])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("estimated horizontal FOV (deg)")
    axs[1].axvspan(min(vals[:2]), max(vals[:2]), color=COL["vipe"], alpha=0.08)
    axs[1].set_xlim(0, max(vals) * 1.18)
    axs[1].set_title("BA methods ≈ 33–35°, feed-forward 55–63°", fontsize=17)
    save(fig, "f06_fov_wild.png")


# ---------------------------------------------------------------- 4. A ranking heatmap
def fig_rank():
    r = pd.read_csv(H0 / "outputs/bench/eval3d_report/ranking_A.csv", index_col=0)
    cols = ["pose_rank_mean", "depth_rank_mean", "metric_rank_mean", "geom_rank_mean"]
    lab = ["Pose", "Depth / cloud", "Metric scale", "Overall geom"]
    rows = ["vggt_omega", "da3", "pi3x", "spatialvid", "vipe", "vggt", "voyager"]
    r = r.loc[[x for x in rows if x in r.index], cols]
    if pd.isna(r.loc["spatialvid", "metric_rank_mean"]):
        rr = pd.read_csv(H0 / "outputs/bench/eval3d_report/ranking_A.csv", index_col=0)
        r.loc["spatialvid", "metric_rank_mean"] = rr.loc["spatialvid_metric", "metric_rank_mean"]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    im = ax.imshow(r.values, cmap="Blues_r", vmin=1, vmax=7.5, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=17,
                    color="white" if v < 3.2 else INK, fontweight="bold" if v == np.nanmin(r.values[:, j]) else "normal")
    ax.set_xticks(range(len(cols)), lab, fontsize=16)
    ax.set_yticks(range(r.shape[0]), [NAME[x] for x in r.index], fontsize=16)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A group mean rank (1 = best; 7-Scenes / TUM / Bonn / Sintel / KITTI)", fontsize=16, pad=12)
    save(fig, "f07_rank_heatmap_A.png")


# ---------------------------------------------------------------- 5. hybrid vs single (A group medians / failure)
def fig_hybrid():
    rows = ["ViPE", "Pi3X", "DA3", "ViPE+Pi3X (med)", "ViPE+Pi3X (geo)"]
    data = {  # outputs/hybrid/README.md, A group (no ETH3D)
        "Metric AbsRel ↓": [0.114, 0.075, 0.076, 0.061, 0.067],
        "Point cloud F@5cm ↑": [0.49, 0.84, 0.70, 0.81, 0.80],
        "Depth failure ↓": [14.3, 10.7, 10.7, 7.1, 3.6],
        "FOV failure ↓": [19.6, 45.1, 35.3, 19.6, 19.6],
    }
    cols = [COL["vipe"], COL["pi3x"], COL["da3"], "#E58A9A", COL["hybrid"]]
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.2))
    for ax, (t, v) in zip(axs, data.items()):
        ax.bar(range(5), v, color=cols, width=0.7)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:g}" + ("%" if "failure" in t else ""), ha="center", va="bottom", fontsize=14)
        ax.set_title(t, fontsize=18)
        ax.set_xticks(range(5), ["ViPE", "Pi3X", "DA3", "+med", "+geo"], fontsize=14)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.text(0.5, -0.02, "+med / +geo = ViPE pose & K + Pi3X depth, scale aligned to ViPE (median ratio / multi-view geometry)",
             ha="center", fontsize=15, color=MUTED)
    save(fig, "f08_hybrid_A.png")


# ---------------------------------------------------------------- 6. filtering
def fig_filter():
    rules = [("cut (TransNetV2)", 31.3), ("static camera", 16.7), ("too dynamic", 6.6), ("text overlay", 6.1),
             ("fisheye", 2.2), ("zoom (GeoCalib)", 2.1), ("dark", 1.6), ("blur", 1.6), ("overexposed", 0.7)]
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(rules))
    ax.barh(y, [r[1] for r in rules], color=COL["vipe"], height=0.62)
    for i, r in enumerate(rules):
        ax.text(r[1] + 0.4, i, f"{r[1]}%", va="center", fontsize=15)
    ax.set_yticks(y, [r[0] for r in rules], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("share of 1,835 pool clips rejected by the rule")
    ax.set_title("Any rule: 49.8%  →  with Qwen3-VL-32B: 64% rejected, 668 pass", fontsize=16)
    save(fig, "f09_filter_rules.png")

    vl = [("Qwen3-VL-8B", 0.56, 0.21, 17000), ("Qwen3-VL-32B", 0.67, 0.58, 6300), ("Qwen3.5-27B", 0.61, 0.58, 8700),
          ("InternVL3.5-8B", 0.50, 0.04, 6700), ("Rules only", 0.95, 0.79, None)]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    for nm, p, r, tp in vl:
        cc = COL["hybrid"] if "32B" in nm else (INK if tp is None else COL["vggt_omega"])
        ax.scatter(r, p, s=(tp or 6000) / 18, color=cc, alpha=0.85, marker="s" if tp is None else "o")
        ax.annotate(nm + (f"\n{tp:,} clips/GPU·h" if tp else ""), (r, p), xytext=(12, -6), textcoords="offset points", fontsize=14)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("recall on group C (hard cases)")
    ax.set_ylabel("precision")
    ax.grid(color=GRID, lw=0.8)
    ax.set_title("VLM alone vs rules (group C, revised labels; size = throughput)", fontsize=16)
    save(fig, "f10_vlm_pr.png")

    reasons = [("too dynamic", 95), ("VLM: inaccurate", 56), ("text overlay", 50), ("cut", 47), ("blur", 44),
               ("VLM: unreconstructable", 32), ("static camera", 14)]
    fig, axs = plt.subplots(1, 2, figsize=(21, 6.6), gridspec_kw={"width_ratios": [0.7, 1.4]})
    axs[0].bar([0, 1], [19.3, 44.0], color=[COL["da3"], COL["hybrid"]], width=0.6)
    axs[0].bar([0, 1], [12, 41], color="none", edgecolor=INK, width=0.6, ls="--", lw=1.5)
    for i, (v, v2) in enumerate([(19.3, 12), (44.0, 41)]):
        axs[0].text(i, v + 1, f"{v}%", ha="center", fontsize=17, fontweight="bold")
    axs[0].set_xticks([0, 1], ["passed", "rejected"], fontsize=16)
    axs[0].set_yticks([])
    axs[0].spines["left"].set_visible(False)
    axs[0].set_title("Downstream failure (430 clips; dashed = random 300)", fontsize=15)
    y = np.arange(len(reasons))
    axs[1].barh(y, [r[1] for r in reasons], color=COL["hybrid"], height=0.6, alpha=0.85)
    for i, r in enumerate(reasons):
        axs[1].text(r[1] + 1, i, f"{r[1]}%", va="center", fontsize=15)
    axs[1].set_yticks(y, [r[0] for r in reasons], fontsize=15)
    axs[1].invert_yaxis()
    axs[1].set_xlim(0, 110)
    axs[1].set_title("Annotation failure rate by rejection reason", fontsize=16)
    save(fig, "f11_filter_e2e.png")


# ---------------------------------------------------------------- 7. QC AUROC
def fig_qc():
    s = pd.read_csv(H0 / "outputs/qc/calib_signals.csv").set_index("signal")
    nm = {"xall_traj": "cross-model trajectory", "xba_traj": "trajectory vs BA methods", "samp5_bad": "Sampson epipolar (gap 5)",
          "iflow1_lt3": "induced vs RAFT flow <3px", "speed_spike": "speed spike", "rev_traj": "reverse-play rerun",
          "sub_traj": "half-frame rerun", "xff_depth": "depth vs feed-forward", "xall_depth": "depth vs all",
          "cons20": "self-consistency gap 20", "cons5": "self-consistency gap 5", "fsv5": "free-space violation",
          "xba_fov": "FOV vs BA methods", "iflow1_rel": "induced flow rel. error"}
    groups = [("Pose failure (base 12%)", "pose_auroc", ["xall_traj", "xba_traj", "samp5_bad", "iflow1_lt3", "speed_spike"]),
              ("Depth failure (base 35%)", "depth_auroc", ["xff_depth", "xall_depth", "cons20", "cons5", "fsv5"]),
              ("FOV failure (base 36%)", "fov_auroc", ["xba_fov", "iflow1_rel"])]
    fig, axs = plt.subplots(1, 3, figsize=(25, 6.6), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    for ax, (t, col, sig) in zip(axs, groups):
        sig = [x for x in sig if x in s.index and col in s.columns and pd.notna(s.loc[x, col])]
        v = [s.loc[x, col] for x in sig]
        cc = [COL["hybrid"] if x.startswith("x") else COL["vipe"] for x in sig]
        y = np.arange(len(sig))
        ax.barh(y, [x - 0.5 for x in v], left=0.5, color=cc, height=0.6)
        for i, x in enumerate(v):
            ax.text(x + 0.005, i, f"{x:.2f}", va="center", fontsize=15)
        ax.set_yticks(y, [nm.get(x, x) for x in sig], fontsize=15)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.02)
        ax.axvline(0.5, color=MUTED, lw=1)
        ax.set_title(t, fontsize=17)
        ax.set_xlabel("AUROC (0.5 = chance)")
    fig.text(0.5, -0.03, "red = cross-model signals, blue = single-model self-checks · A+B, 476 method×clip labels",
             ha="center", fontsize=15, color=MUTED)
    save(fig, "f12_qc_auroc.png")

    ret = [(20, 0), (30, 3.5), (40, 14), (50, 24), (100, 57)]
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.plot([r[0] for r in ret], [r[1] for r in ret], "-o", color=COL["hybrid"], lw=3, ms=9)
    for x, yv in ret:
        ax.annotate(f"{yv}%", (x, yv), xytext=(-8, 10), textcoords="offset points", fontsize=15)
    ax.scatter([36], [8.8], s=180, color=INK, zorder=5)
    ax.annotate("rule score > 1 rejects\n64% → kept fail 8.8%", (36, 8.8), xytext=(48, 2), fontsize=15,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("labels kept (%)")
    ax.set_ylabel("geometric failure among kept (%)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_title("Retention vs failure (A+B, leave-one-dataset-out)", fontsize=16)
    save(fig, "f13_qc_retention.png")


# ---------------------------------------------------------------- 8. funnel
def fig_funnel():
    st = [("searched videos", 921), ("CC-BY ≥1080p prefilter", 553), ("pool clips (436 videos)", 1835),
          ("pass rules + VLM", 668), ("QC kept (A/B rate 36%)", 240), ("QC kept (wild rate 20%)", 134)]
    fig, ax = plt.subplots(figsize=(14, 6.8))
    y = np.arange(len(st))[::-1]
    cc = [MUTED, MUTED, COL["vipe"], COL["da3"], COL["hybrid"], "#E58A9A"]
    mx = 1835
    for yi, (lab, v), c in zip(y, st, cc):
        w = v / mx
        ax.barh(yi, w, left=(1 - w) / 2, color=c, height=0.7)
        ax.text(0.5, yi, f"{lab}: {v:,}", ha="center", va="center", fontsize=16,
                color="white" if w > 0.35 else INK)
    ax.set_xlim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title("Yield funnel (last two rows are estimates: 668 × QC keep rate)", fontsize=16)
    save(fig, "f14_funnel.png")


ALL = [fig_rank, fig_hybrid, fig_filter, fig_qc, fig_funnel, fig_fov, fig_A, fig_scannetpp]
for f in [g for g in ALL if len(sys.argv) < 2 or g.__name__ in sys.argv[1:]]:
    try:
        f()
    except Exception:  # noqa: BLE001
        print("FAILED", f.__name__)
        traceback.print_exc()
print("ALL_DONE")
