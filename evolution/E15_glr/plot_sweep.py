"""plot_sweep.py SWEEP_RUNS_CSV OUT_DIR [--run v0a] - figures for the E15 sandbox sweep (matplotlib; reads the CSV written by summarize_sweep.py).

  fig_gate_signals.png         one panel per stage-1 gate signal: every clip of the easy set and of the hard controls on the signal's axis, the
                               gate threshold drawn as a solid line, the side that stops a clip shaded (thresholds come from glr_stage1_gate.DEFAULT_CFG)
  fig_variants_vs_reference.png  focal error against the silver reference (x) and ATE against it (y), one panel per camera-model variant;
                               the pre-declared 'silver accurate' box is shaded, marker = gate verdict
  fig_outcomes.png             per variant: how many easy clips are a correct pass / correct stop / false stop / silent error

Colour roles follow the dataviz skill (categorical slots 1-2 for the two clip groups, the fixed status palette for the gate verdicts and outcomes; status
colours always come with a marker shape or a label). Written for the light surface; nothing here is a result, it only draws the CSV.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import glr_stage1_gate as G  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"                      # categorical slots 1, 2: easy set, hard controls
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#86b6ef"                                       # sequential blue 250: 'correctly stopped' (not a status, not a failure)
VERDICT = {"ok": (GOOD, "o"), "suspect": (WARN, "^"), "stop": (CRIT, "s")}
ATE_MAX, F_MAX = 3.27, 10.0
VARIANTS = [("v0a", "v0a  PINHOLE, f init 1.2 x max(W,H)"), ("v1sp", "v1sp  SIMPLE_PINHOLE"), ("v2prior", "v2prior  PINHOLE, GeoCalib f as start"),
            ("v3fixed", "v3fixed  PINHOLE, GeoCalib f fixed")]


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if v in ("", None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
            if not r.get("role"):
                r["role"] = "control_negative" if str(r["clip"]).startswith("ctl_") else "easy"
            rows.append(r)
    return rows


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
                         "legend.frameon": False})


def _fails(v, thr, shade):
    return v < thr[0] if shade == "below" else v > thr[0] if shade == "above" else (v < thr[0] or v > thr[1])


def jitter(n, lane):
    """deterministic vertical stagger inside a lane (no random numbers: the figure is reproducible)."""
    return lane + 0.34 * (((np.arange(n) * 3) % 5) - 2) / 2.0


def fig_signals(rows, out, run, plt):
    cfg = G.DEFAULT_CFG
    fmin, fmax = G.focal_bounds(cfg)
    rs = [r for r in rows if r["run"] == run]
    # (title, rule text, value fn, threshold(s), shade 'below' / 'above' / 'outside', status colour, log x); only the first panel counts model-less clips as 0
    panels = [
        ("registered frames", f"stop if < {cfg['min_reg_frac']:.0%}", lambda r: r["reg_frac"] or 0.0, [cfg["min_reg_frac"]], "below", CRIT, False),
        ("points per registered image", f"stop if < {cfg['min_pts_per_img']}", lambda r: r["pts_img"], [cfg["min_pts_per_img"]], "below", CRIT, True),
        ("|fx / fy - 1|", f"stop if > {cfg['max_fx_fy_dev']}", lambda r: abs(r["fx_fy"] - 1) if r["fx_fy"] is not None else None, [cfg["max_fx_fy_dev"]], "above", CRIT, True),
        ("planar / panoramic pairs", f"stop if > {cfg['max_planar_frac']:.0%}", lambda r: r["planar"], [cfg["max_planar_frac"]], "above", CRIT, False),
        ("focal / max(W, H)", f"stop outside {fmin:.2f} .. {fmax:.1f}", lambda r: r["focal_w"], [fmin, fmax], "outside", CRIT, True),
        ("reprojection error (px)", f"suspect if > {cfg['max_reproj_px']}", lambda r: r["reproj"], [cfg["max_reproj_px"]], "above", WARN, False),
        ("median triangulation angle (deg)", f"suspect if < {cfg['min_median_tri_angle_deg']}", lambda r: r["tri50"], [cfg["min_median_tri_angle_deg"]], "below", WARN, False),
        ("pose step ratio (max / median)", f"suspect if > {cfg['max_step_ratio']}", lambda r: r["step_ratio"], [cfg["max_step_ratio"]], "above", WARN, True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.6))
    n_nomodel = {g: sum(1 for r in rs if r["role"] == g and not r.get("n_points")) for g in ("easy", "control_negative")}
    for ax, (title, rule, fn, thr, shade, col, logx) in zip(axes.ravel(), panels):
        vals = {}
        for g in ("easy", "control_negative"):
            vals[g] = [(fn(r), r["clip"]) for r in rs if r["role"] == g and fn(r) is not None and np.isfinite(fn(r))]
        allv = [v for g in vals.values() for v, _ in g]
        if not allv:
            ax.set_visible(False)
            continue
        if logx:
            pos = [v for v in allv if v > 0]
            lo, hi = (min(pos) / 1.6, max(max(pos), max(thr)) * 1.6) if pos else (0.01, 10)
            lo = min(lo, min(thr) / 1.6)
            ax.set_xscale("log")
        else:
            lo, hi = min(min(allv), min(thr)), max(max(allv), max(thr))
            pad = 0.06 * (hi - lo or 1.0)
            lo, hi = lo - pad, hi + pad
        ax.set_xlim(lo, hi)
        if shade == "below":
            ax.axvspan(lo, thr[0], color=col, alpha=0.10, lw=0)
        elif shade == "above":
            ax.axvspan(thr[0], hi, color=col, alpha=0.10, lw=0)
        else:
            ax.axvspan(lo, thr[0], color=col, alpha=0.10, lw=0)
            ax.axvspan(thr[1], hi, color=col, alpha=0.10, lw=0)
        for t in thr:
            ax.axvline(t, color=col, lw=1.6, solid_capstyle="round")
        for lane, g, c in ((1, "easy", BLUE), (0, "control_negative", ORANGE)):
            pts = sorted(vals[g])
            if not pts:
                continue
            x = np.array([v for v, _ in pts], float)
            if logx:
                x = np.maximum(x, lo * 1.01)
            ys = jitter(len(x), lane)
            ax.scatter(x, ys, s=34, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            if g == "easy":     # name the easy clips this rule would stop / flag (the controls are expected to fail; naming them adds clutter)
                names = sorted(name.replace("gov_grca_", "grca_") for v, name in pts if _fails(v, thr, shade))
                if names:
                    text = ", ".join(names)
                    if len(text) > 44:      # wrap into two lines of about the same length
                        cut = text.rfind(", ", 0, len(text) // 2 + 14) + 1
                        text = text[:cut] + "\n" + text[cut:].strip()
                    on_left = shade == "below"
                    ax.text(0.02 if on_left else 0.98, 0.5, ("easy, " + ("stopped: " if col == CRIT else "flagged: ")) + text, transform=ax.get_yaxis_transform(),
                            ha="left" if on_left else "right", va="center", fontsize=6.6, color=INK2, linespacing=1.15,
                            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.92}, zorder=4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["hard controls", "easy set"], color=INK2)
        ax.set_ylim(-0.7, 1.7)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", pad=15)
        ax.text(0.0, 1.015, rule, transform=ax.transAxes, va="bottom", ha="left", fontsize=8, color=INK2)
    ne, nc = n_nomodel["easy"], n_nomodel["control_negative"]
    n_e = sum(1 for r in rs if r["role"] == "easy")
    n_c = sum(1 for r in rs if r["role"] == "control_negative")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor=SURFACE, markersize=7, label=f"easy OOD set ({n_e} clips)"),
                        plt.Line2D([], [], marker="o", ls="", color=ORANGE, markeredgecolor=SURFACE, markersize=7, label=f"hard negative controls ({n_c} clips)"),
                        plt.Line2D([], [], color=CRIT, lw=1.6, label="stop threshold (shaded: stops the clip)"),
                        plt.Line2D([], [], color=WARN, lw=1.6, label="suspect threshold")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Stage-1 exit gate on the pycolmap replica ({run}): where each clip sits against each rule", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.01, 0.945, f"Clips without any model are drawn at 0 registered frames only (easy: {ne}, controls: {nc}); their other signals do not exist. "
             "Reduced-cost replica: 640 px, 60 frames, unguided matching.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.925), h_pad=2.2)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_variants(rows, out, plt):
    easy = [r for r in rows if r["role"] == "easy" and r.get("outcome")]
    have = [(k, t) for k, t in VARIANTS if any(r["run"] == k for r in easy)]
    if not have:
        return False
    fig, axes = plt.subplots(1, len(have), figsize=(3.9 * len(have) + 0.6, 4.6), squeeze=False, sharex=True, sharey=True)
    for ax, (key, title) in zip(axes[0], have):
        rs = [r for r in easy if r["run"] == key]
        todo = []
        ax.fill_between([0.1, F_MAX], 0.01, ATE_MAX, color=GOOD, alpha=0.10, lw=0)
        ax.axvline(F_MAX, color=GOOD, lw=1.2, alpha=0.7)
        ax.axhline(ATE_MAX, color=GOOD, lw=1.2, alpha=0.7)
        for r in rs:
            fe = abs(r["focal_err_pct"]) if r.get("focal_err_pct") is not None else None
            at = r.get("ate_vs_ref_pct")
            c, m = VERDICT[r["verdict"]]
            if fe is None or at is None or not np.isfinite(at):
                continue
            x, y = max(fe, 0.12), max(at, 0.012)
            ax.scatter([x], [y], s=46, color=c, marker=m, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            if r["outcome"] in ("silent_error", "false_stop"):
                todo.append((x, y, r["clip"].replace("gov_grca_", "grca_")))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 400)
        ax.set_ylim(0.01, 120)
        placed = []                     # label anchors in display pixels: shift a label down until it no longer sits on an earlier one
        for x, y, name in sorted(todo, key=lambda t: (-t[1], t[0])):
            px, py = ax.transData.transform((x, y))
            dy = 4
            while any(abs(px - qx) < 62 and abs((py + dy * 1.4) - qy) < 9 for qx, qy in placed):
                dy -= 7
            placed.append((px, py + dy * 1.4))
            ax.annotate(name, (x, y), xytext=(5, dy), textcoords="offset points", fontsize=6.8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("|focal error| vs reference (%)")
        ax.text(0.2, 0.012 * 1.15, "silver-accurate region (needs >= 90 % registered)", fontsize=7, color=INK2)
    axes[0][0].set_ylabel("Sim(3) ATE vs reference (% of path)")
    fig.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=c, markeredgecolor=SURFACE, markersize=8, label=f"gate: {k}") for k, (c, m) in VERDICT.items()],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Each easy clip against the silver reference, by stage-1 camera-model variant (same matching database)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def fig_outcomes(rows, out, plt):
    easy = [r for r in rows if r["role"] == "easy" and r.get("outcome")]
    order = [(k, t) for k, t in VARIANTS + [("v0b", "v0b  v0a, other mapper seed"), ("v0a_hq960", "v0a at 960 px (subset)")] if any(r["run"] == k for r in easy)]
    if not order:
        return False
    cats = [("true_pass", "correct pass (gate ok / suspect, silver-accurate)", GOOD), ("true_stop", "correct stop (gate stop, not silver-accurate)", NEUTRAL),
            ("false_stop", "false stop (gate stop, silver-accurate)", WARN), ("silent_error", "silent error (gate ok / suspect, not silver-accurate)", CRIT)]
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(order) + 2.4))
    for i, (key, title) in enumerate(order):
        rs = [r for r in easy if r["run"] == key]
        left = 0
        for cat, _, col in cats:
            n = sum(1 for r in rs if r["outcome"] == cat)
            if n:
                ax.barh(i, n - 0.06, left=left, height=0.5, color=col, edgecolor="none")
                if n >= 1:
                    ax.text(left + (n - 0.06) / 2, i, str(n), ha="center", va="center", fontsize=8.5, color="#0b0b0b" if cat in ("false_stop", "true_stop") else "#ffffff", fontweight="semibold")
            left += n
        ax.text(left + 0.25, i, f"n = {len(rs)}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([t for _, t in order], color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("easy-set clips")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(len([r for r in easy if r["run"] == k]) for k, _ in order) + 3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=t) for _, t, c in cats], loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2, fontsize=8)
    ax.set_title("What the stage-1 gate did, judged against the silver reference", loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--run", default="v0a", help="run drawn in the gate-signal figure")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    rows = load(a.sweep_csv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_signals(rows, out / "fig_gate_signals.png", a.run, plt)
    print("wrote", out / "fig_gate_signals.png")
    if fig_variants(rows, out / "fig_variants_vs_reference.png", plt):
        print("wrote", out / "fig_variants_vs_reference.png")
    else:
        print("no easy-clip run with a reference yet: fig_variants_vs_reference.png skipped")
    if fig_outcomes(rows, out / "fig_outcomes.png", plt):
        print("wrote", out / "fig_outcomes.png")
    else:
        print("no outcomes yet: fig_outcomes.png skipped")


if __name__ == "__main__":
    main()
