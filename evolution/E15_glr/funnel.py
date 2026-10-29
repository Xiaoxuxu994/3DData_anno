"""funnel.py DECISIONS_CSV SWEEP_CSV OUT_DIR [--manifest MANIFEST.json] [--route glr_route.csv] [--run v0a] - which layer stops each clip first (E15).

Joins up to four views of the same clips:
  DECISIONS_CSV  filter/decide.py output run with `--vlm TAG` (columns clip, rule_pass, rule_reasons, vlm_pass, reason ...): layer 1 = the traditional
                 rules (s1 cuts / s2 luminance, blur / s3 OCR / s4 flow / s5 GeoCalib), layer 2 = the VLM verdict (here: the labelled Claude surrogate)
  --route CSV    optional, pipeline/glr_prepare.py route output (glr_route.csv: clip, route, reasons): layer 3 = the router; a clip sent to the v1.3
                 annotator (route v13: dynamic, rotation-dominant, driving ... - GLRMap is for static scenes with parallax) is not a rejection,
                 it just does not become a GLR label
  SWEEP_CSV      summarize_sweep.py output: last layer = the stage-1 exit gate of GLRMap's first stage (rows of --run, the pycolmap replica)
  MANIFEST       optional: role (easy / control_negative), category / expected failure of each clip
and writes OUT_DIR/funnel.csv (one row per clip: the verdict of every layer and the first layer that stops it), funnel.md (counts per group and
per category) and fig_funnel.png (counts per category, coloured by the first stopping layer; grey = passed all three layers).
The layers are applied in the order the pipeline applies them, so "stopped by the gate" means "passed the rules and the VLM first"; the last column of
funnel.csv also tells how many layers WOULD have stopped the clip on their own (`n_layers_stopping`) - a clip that only one layer catches is the
clip a missing layer would let through.
"""

import argparse
import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path

LAYERS = ["rules", "vlm", "route", "gate"]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def truth(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def build(decisions, sweep, manifest, run="v0a", route=None):
    dec = {r["clip"]: r for r in decisions}
    rt = {r["clip"]: r for r in route} if route is not None else None
    gate = {r["clip"]: r for r in sweep if r["run"] == run}
    roles = {c["id"]: c for c in manifest.get("clips", [])}
    rows = []
    for clip in sorted(set(dec) | set(gate)):
        d, g, m = dec.get(clip), gate.get(clip), roles.get(clip, {})
        role = m.get("role") or (g or {}).get("role") or ("control_negative" if clip.startswith("ctl_") else "easy")
        row = OrderedDict(clip=clip, role=role, category=m.get("expected_failure") or m.get("category") or "")
        row["rules"] = ("" if d is None else "pass" if truth(d.get("rule_pass")) else "stop")
        row["rules_reason"] = (d or {}).get("rule_reasons", "") or ""
        row["vlm"] = ("" if d is None else "pass" if truth(d.get("vlm_pass")) else "stop")
        row["vlm_reason"] = ("" if d is None or truth(d.get("vlm_pass")) else (d.get("reason", "").split("vlm:")[-1] if "vlm:" in d.get("reason", "") else ""))
        r_ = (rt or {}).get(clip)
        row["route"] = ("" if rt is None or r_ is None else r_["route"])
        row["route_reason"] = (r_ or {}).get("reasons", "") or ""
        row["gate"] = ("" if g is None else "stop" if g["verdict"] == "stop" else "pass")
        row["gate_verdict"] = (g or {}).get("verdict", "")
        row["gate_flags"] = (g or {}).get("flags", "") or ""
        row_layer_route = "" if rt is None or not row["route"] else "pass" if row["route"] == "glr" else "stop"
        layer = {"rules": row["rules"], "vlm": row["vlm"], "route": row_layer_route, "gate": row["gate"]}
        used = [k for k in LAYERS if k != "route" or rt is not None]
        stops = [k for k in used if layer[k] == "stop"]
        known = [k for k in used if layer[k] != ""]
        row["first_stop"] = stops[0] if stops else ("passed_all" if len(known) == len(used) else "incomplete")
        row["n_layers_stopping"] = len(stops)
        rows.append(row)
    return rows


def summarize(rows):
    out = []
    for role in ("easy", "control_negative"):
        rs = [r for r in rows if r["role"] == role]
        if not rs:
            continue
        c = Counter(r["first_stop"] for r in rs)
        has_route = any(r["route"] for r in rows)
        alone = Counter(next(k for k in LAYERS if (r[k] == "stop" if k != "route" else r["route"] not in ("", "glr"))) for r in rs if r["n_layers_stopping"] == 1)
        out.append(f"**{role}** ({len(rs)} clips): first stopped by rules {c['rules']}, by the VLM {c['vlm']}, "
                   + (f"sent to the v1.3 annotator by the router {c['route']}, " if has_route else "") + f"by the stage-1 gate {c['gate']}; "
                   f"passed all {4 if has_route else 3} layers {c['passed_all']}" + (f"; incomplete (a layer has no result yet) {c['incomplete']}" if c["incomplete"] else "")
                   + (f". Stopped by exactly one layer: rules {alone['rules']}, VLM {alone['vlm']}, " + (f"router {alone['route']}, " if has_route else "")
                      + f"gate {alone['gate']}." if role == "control_negative" else "."))
    return out


def table(rows):
    cols = ["clip", "role", "category", "rules", "rules_reason", "vlm", "vlm_reason"] + (["route", "route_reason"] if any(r["route"] for r in rows) else []) + ["gate", "gate_flags", "first_stop"]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"
    colours = OrderedDict([("rules", "#2a78d6"), ("vlm", "#eb6834"), ("route", "#e87ba4"), ("gate", "#1baf7a"), ("passed_all", "#c3c2b7"), ("incomplete", "#e1e0d9")])
    labels = {"rules": "stopped by the rules (s1-s5)", "vlm": "stopped by the VLM verdict (Claude surrogate)", "route": "sent to the v1.3 annotator by the router",
              "gate": "stopped by the stage-1 gate (pycolmap replica)", "passed_all": "passed every layer", "incomplete": "a layer has no result yet"}
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink2,
                         "xtick.color": "#898781", "ytick.color": "#898781", "text.color": ink, "axes.grid": True, "grid.color": grid, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
    groups = [("easy", "Easy OOD set, by scene type"), ("control_negative", "Hard negative controls, by expected failure")]
    groups = [(k, t) for k, t in groups if any(r["role"] == k for r in rows)]
    heights = [max(len({r["category"] for r in rows if r["role"] == k}), 1) for k, _ in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(9.4, 1.25 + 0.46 * sum(heights) + 0.55 * len(groups)), gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (role, title) in zip(axes[:, 0], groups):
        cats = sorted({r["category"] for r in rows if r["role"] == role})
        for i, cat in enumerate(cats):
            rs = [r for r in rows if r["role"] == role and r["category"] == cat]
            c = Counter(r["first_stop"] for r in rs)
            left = 0
            for key, col in colours.items():
                n = c.get(key, 0)
                if n:
                    ax.barh(i, n - 0.06, left=left, height=0.56, color=col, edgecolor="none")
                    ax.text(left + (n - 0.06) / 2, i, str(n), ha="center", va="center", fontsize=8.5, fontweight="bold",
                            color="#ffffff" if key in ("rules", "vlm", "gate") else ink)
                left += n
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([c or "(none)" for c in cats], color=ink2)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    axes[-1, 0].set_xlabel("clips")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colours[k], label=labels[k]) for k in colours if any(r["first_stop"] == k for r in rows)],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("decisions")
    ap.add_argument("sweep_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--manifest")
    ap.add_argument("--route", help="glr_route.csv of pipeline/glr_prepare.py route: adds the router as a layer")
    ap.add_argument("--run", default="v0a")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else {}
    rows = build(read_csv(a.decisions), read_csv(a.sweep_csv), manifest, a.run, read_csv(a.route) if a.route else None)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "funnel.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = "\n\n".join(summarize(rows)) + "\n\n" + table(rows) + "\n"
    (out / "funnel.md").write_text(md)
    print(md)
    if not a.no_figure:
        figure(rows, out / "fig_funnel.png")


if __name__ == "__main__":
    main()
