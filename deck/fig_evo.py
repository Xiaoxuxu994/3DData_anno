"""Pipeline-evolution figures (line I/J numbers). Output: outputs/deck_figs/f16..f21"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path
EV = Path("/data/home/z00533370/outputs/evolution"); OUT = Path("/data/home/z00533370/outputs/deck_figs")
INK, MUTED, GRID = "#1B2430", "#5B6675", "#D9DEE5"
BLUE, RED, GREEN, ORANGE, GREY, PURPLE = "#2F6DB5", "#C8364A", "#3E9C6A", "#E07B39", "#9AA3AE", "#7B4FA0"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
  "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 110})
def save(fig, n): fig.savefig(OUT / n, bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", n)

# f16 K sources
src = [("ViPE", 10.9, BLUE), ("GeoCalib", 41.3, GREY), ("Pi3X self", 38.0, ORANGE), ("DA3 self", 42.4, GREEN),
       ("median of 4", 30.4, PURPLE), ("vote", 10.9, "#7FB0E0"), ("gate (adopted)", 12.0, RED)]
f = pd.read_csv(EV / "E1/fov_sources.csv"); w = f[f.bench == "wild"]
fig, axs = plt.subplots(1, 2, figsize=(21, 7), gridspec_kw={"width_ratios": [1, 1.05]})
ax = axs[0]; y = np.arange(len(src))
ax.barh(y, [s[1] for s in src], color=[s[2] for s in src], height=0.62)
for i, s in enumerate(src): ax.text(s[1] + 0.6, i, f"{s[1]}%", va="center", fontsize=15)
ax.set_yticks(y, [s[0] for s in src], fontsize=15); ax.invert_yaxis(); ax.set_xlim(0, 50)
ax.set_xlabel("HFOV error > 5° (92 clips with GT)"); ax.set_title("Which intrinsics source to trust", fontsize=17)
ax = axs[1]
ax.axhspan(25, 130, color=BLUE, alpha=0.06); ax.axhline(25, color=BLUE, lw=1, ls="--"); ax.axhline(130, color=BLUE, lw=1, ls="--")
vh = w.hfov_vipe.clip(-5, 175)
bad = (w.hfov_vipe < 25) | (w.hfov_vipe > 130)
ax.scatter(w.hfov_geo[~bad], vh[~bad], s=60, color=GREY, label="ViPE inside gate")
ax.scatter(w.hfov_geo[bad], vh[bad], s=110, color=RED, label=f"ViPE outside [25°,130°]: {bad.sum()} / {len(w)} wild clips (gate fires)")
ax.plot([20, 140], [20, 140], color=MUTED, lw=1)
ax.set_xlabel("GeoCalib HFOV (deg)"); ax.set_ylabel("ViPE HFOV (deg)"); ax.legend(frameon=False, fontsize=14, loc="upper left")
ax.set_title("Wild group C: ViPE focal collapses → gate to median(GeoCalib, Pi3X, DA3)", fontsize=16)
save(fig, "f16_k_sources.png")

# f17 compensation
r = pd.read_csv(EV / "E1/raw/A.csv")
p = r[r.method.isin(["pi3x", "pi3x_kvipe"])].pivot_table(index="clip", columns="method", values="sim3_scale")
d = p.join(f.set_index("clip")[["dataset", "hfov_gt", "hfov_pi3x"]]).dropna()
d["dfov"] = d.hfov_pi3x - d.hfov_gt
dsc = {"7scenes": BLUE, "tum": GREEN, "bonn": ORANGE, "kitti": RED}
fig, axs = plt.subplots(1, 2, figsize=(21, 7))
ax = axs[0]; ds = ["7scenes", "tum", "bonn", "kitti"]; x = np.arange(4)
a = [d[d.dataset == k].pi3x.median() for k in ds]; b = [d[d.dataset == k].pi3x_kvipe.median() for k in ds]
ax.bar(x - 0.2, a, 0.38, color=GREY, label="Pi3X image-only"); ax.bar(x + 0.2, b, 0.38, color=RED, label="Pi3X + ViPE K")
for xi, v in zip(x - 0.2, a): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=14)
for xi, v in zip(x + 0.2, b): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=14)
ax.axhline(1, color=INK, lw=1.2); ax.set_ylim(0.8, 1.42); ax.set_xticks(x, ["7-Scenes", "TUM", "Bonn", "KITTI"])
ax.set_ylabel("Sim3 scale to GT (1 = metric correct)"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Median scale per dataset (A group)", fontsize=17)
ax = axs[1]
for k, c in dsc.items():
    s = d[d.dataset == k]; ax.scatter(s.dfov, s.pi3x_kvipe / s.pi3x, s=90, color=c, label=k)
ax.axhline(1, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("image-only Pi3X HFOV − GT (deg; <0 = too narrow)"); ax.set_ylabel("scale with K ÷ scale image-only")
ax.legend(frameon=False); ax.grid(color=GRID, lw=0.8)
ax.set_title("Clips whose FOV was most wrong change scale most", fontsize=17)
save(fig, "f17_compensation.png")

# f18 v0 vs v1.3
M = [("A traj-aligned cloud F ↑", 0.618, 0.670, ""), ("A g20 EPE px ↓", 11.1, 7.03, ""), ("A SE3 ATE fail ↓", 10.7, 7.1, "%"),
     ("A scale fail ↓", 28.6, 25.0, "%"), ("C g20 self-consistency ↑", 0.654, 0.792, ""),
     ("A metric AbsRel ↓", 0.061, 0.069, ""), ("A FOV fail ↓", 19.6, 21.6, "%"), ("ScanNet++ g5 EPE px ↓", 0.81, 1.40, "")]
fig, axs = plt.subplots(2, 4, figsize=(22, 8.6))
for ax, (t, v0, v1, u) in zip(axs.flat, M):
    better = (v1 < v0) if "↓" in t else (v1 > v0)
    c = GREEN if better else RED
    ax.plot([0, 1], [v0, v1], color=c, lw=3); ax.scatter([0], [v0], s=160, color=GREY, zorder=3); ax.scatter([1], [v1], s=160, color=c, zorder=3)
    ax.text(-0.08, v0, f"{v0:g}{u}", ha="right", va="center", fontsize=16); ax.text(1.08, v1, f"{v1:g}{u}", ha="left", va="center", fontsize=16, color=c, fontweight="bold")
    ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1], ["v0", "v1.3"]); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    lo, hi = min(v0, v1), max(v0, v1); pad = (hi - lo) * 0.6 + 1e-9; ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(t, fontsize=15, color=INK)
fig.text(0.5, -0.03, "green = improved, red = regressed · A = 51 clips (no ETH3D), C = 72 wild clips, ScanNet++ = 28 clips", ha="center", color=MUTED)
fig.tight_layout(); save(fig, "f18_v0_v13.png")

# f19 shots
fig, axs = plt.subplots(1, 2, figsize=(21, 6.8), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axs[0]; st = [("cut-rejected clips", 574), ("shots found", 1764), ("shots ≥ 4 s kept", 633), ("shots passing filter", 220)]
y = np.arange(4); ax.barh(y, [s[1] for s in st], color=[GREY, GREY, BLUE, GREEN], height=0.6)
for i, s in enumerate(st): ax.text(s[1] + 20, i, f"{s[1]:,}", va="center", fontsize=16)
ax.set_yticks(y, [s[0] for s in st], fontsize=15); ax.invert_yaxis(); ax.set_xlim(0, 2100); ax.set_xticks([])
ax.spines["bottom"].set_visible(False); ax.set_title("+220 clips / +23,530 frames = +19.6% usable frames", fontsize=16)
ax = axs[1]; b = [("4–6 s", 27), ("6–8 s", 34), ("8–10 s", 24), ("10–12 s", 7)]
ax.bar(range(4), [v for _, v in b], color=[ORANGE, ORANGE, BLUE, BLUE], width=0.6)
for i, (_, v) in enumerate(b): ax.text(i, v + 0.8, f"{v}%", ha="center", fontsize=16)
ax.axhline(19.3, color=RED, ls="--", lw=2); ax.text(3.45, 20.2, "original passed clips 19.3%", ha="right", color=RED, fontsize=14)
ax.set_xticks(range(4), [k for k, _ in b]); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_title("Failure rate of passed shots by shot length", fontsize=16)
save(fig, "f19_shots.png")

# f20 cost
fig, axs = plt.subplots(1, 2, figsize=(21, 6.6), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axs[0]; c = [("per-clip CLI", 920), ("persistent worker", 711), ("worker, every 2nd frame", 458), ("2 workers / GPU (wall)", 526)]
y = np.arange(4); ax.barh(y, [v for _, v in c], color=[GREY, BLUE, GREEN, RED], height=0.6)
for i, (_, v) in enumerate(c): ax.text(v + 10, i, f"{v} s", va="center", fontsize=16)
ax.set_yticks(y, [k for k, _ in c], fontsize=15); ax.invert_yaxis(); ax.set_xlim(0, 1100); ax.set_xticks([])
ax.spines["bottom"].set_visible(False); ax.set_title("ViPE on the same 12 clips, one H200", fontsize=16)
ax = axs[1]; parts = [("ViPE (2 workers/GPU)", 43.8, BLUE), ("Pi3X K-conditioned", 19.3, ORANGE), ("DA3 on ViPE cams (QC)", 16, GREEN),
                      ("DA3 image-only (K gate)", 8.8, "#8FC9A8"), ("GeoCalib (K gate)", 1.1, GREY)]
left = 0
for n, v, col in parts:
    ax.barh([0], [v], left=left, color=col, height=0.5, edgecolor="white")
    if v > 5: ax.text(left + v / 2, 0, f"{v:g}", ha="center", va="center", color="white", fontsize=15, fontweight="bold")
    left += v
ax.text(left + 1.5, 0, f"≈ {left:.0f} GPU-s", va="center", fontsize=17, fontweight="bold")
ax.set_xlim(0, 112); ax.set_yticks([]); ax.set_xticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=p[2]) for p in parts], [p[0] for p in parts], frameon=False, loc="upper center",
          bbox_to_anchor=(0.45, -0.05), ncol=2, fontsize=14)
ax.set_title("v1.3 annotation cost per accepted 12 s clip", fontsize=16)
save(fig, "f20_cost.png")

# f21 scaling
fig, ax = plt.subplots(figsize=(13, 7))
D = np.logspace(np.log10(2e3), np.log10(1e7), 200)
for a, ls, lab, D1 in [(0.19, "-", "α ≈ 0.19 (2K→2M)", 2e6), (0.14, "--", "α ≈ 0.14 (if the top is 22M)", 2.2e7)]:
    ax.plot(D, 0.275 * (D / 2e3) ** (-a), ls, color=BLUE if a > 0.15 else PURPLE, lw=2.5, label=lab)
ax.scatter([2e3, 2e6], [0.275, 0.073], s=140, color=INK, zorder=5, label="VGGT-Ω reported endpoints")
for x, t in [(1e6, "our ~1M"), (3e6, "3M: −14–19%"), (1e7, "10M: −28–35%")]:
    ax.axvline(x, color=RED if x == 1e6 else GREY, lw=1.5, ls=":")
    ax.text(x * 1.05, 0.30, t, rotation=90, va="top", fontsize=14, color=RED if x == 1e6 else MUTED)
ax.axvspan(1e4, 1e6, color=GREEN, alpha=0.08); ax.text(1.1e4, 0.05, "nested subsets 10K…1M", color=GREEN, fontsize=14)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training sequences"); ax.set_ylabel("3D point error (VGGT-Ω, 6-dataset mean)")
ax.legend(frameon=False, fontsize=14, loc="lower left"); ax.grid(color=GRID, lw=0.8, which="both")
ax.set_title("Single power law through two published endpoints (our extrapolation)", fontsize=16)
save(fig, "f21_scaling.png")
