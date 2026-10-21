import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
INK, MUTED, GRID = "#1B2430", "#5B6675", "#D9DEE5"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
  "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 110})
M = [("Qwen3-VL-32B", "#C8364A", 6300, (0.67, 0.58), (0.72, 0.88), 36.4, 19.3, 44.0),
     ("Qwen3-VL-235B (TP4)", "#2F6DB5", 3300, (0.46, 0.63), (0.51, 0.83), 29.2, 15.5, 41.6),
     ("Qwen3.5-122B (TP2)", "#3E9C6A", 7750, (0.55, 0.50), (0.65, 0.83), 35.9, 19.9, 42.4)]
fig, axs = plt.subplots(1, 2, figsize=(21, 7.2), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axs[0]
for nm, c, tp, a, b, *_ in M:
    ax.scatter(a[1], a[0], s=tp / 14, color=c, alpha=0.35, edgecolor=c)
    ax.scatter(b[1], b[0], s=tp / 14, color=c, alpha=0.95)
    ax.annotate("", xy=(b[1], b[0]), xytext=(a[1], a[0]), arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax.annotate(f"{nm}\n{tp:,} clips/GPU·h", (b[1], b[0]), xytext=(14, -4), textcoords="offset points", fontsize=14, color=INK)
ax.scatter([0.79], [0.95], s=400, marker="s", color=INK)
ax.annotate("rules only", (0.79, 0.95), xytext=(14, -6), textcoords="offset points", fontsize=14)
ax.set_xlim(0.4, 1.12); ax.set_ylim(0.4, 1.0)
ax.set_xlabel("recall on group C"); ax.set_ylabel("precision"); ax.grid(color=GRID, lw=0.8)
ax.set_title("faded = VLM alone → solid = rules OR VLM (size = throughput)", fontsize=15)
ax = axs[1]
x = np.arange(3); w = 0.26
for j, (lab, idx, alpha) in enumerate([("final pass rate", 5, 0.45), ("ViPE fail · passed", 6, 0.95), ("ViPE fail · rejected", 7, 0.7)]):
    v = [m[idx] for m in M]
    bars = ax.bar(x + (j - 1) * w, v, w * 0.92, color=[m[1] for m in M], alpha=alpha, hatch="//" if j == 2 else None, edgecolor="white")
    for xi, vi in zip(x + (j - 1) * w, v):
        ax.text(xi, vi + 0.8, f"{vi}%", ha="center", fontsize=13)
ax.set_xticks(x, [m[0] for m in M], fontsize=14)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_title("per model: pass rate (light) · failure among passed (solid) · among rejected (hatched)", fontsize=14)
fig.savefig("/data/home/z00533370/outputs/deck_figs/f15_vlm_big.png", bbox_inches="tight", facecolor="white"); print("ok")
