"""Phase 0.2 of the GLRMap evaluation plan: replay the stage-1 gate signals offline on stage-1 outputs that already exist on disk.

Usage: python tools/glr_gate_replay.py ROOT [ROOT ...] [--labels labels.csv] [--out signals.csv] [--cfg configs/pipeline_glr_v0.yaml]
                                       [--min-frames 0] [--witness witness.csv]
  ROOT   directories searched (recursively) for stage-1 outputs: a directory that holds database.db and sparse/<k>/ (COLMAP layout,
         e.g. GLRMap's OUT/{images,database.db,sparse/0}). Nothing is modified; every database is opened read-only.
  --labels   CSV with columns `run,label` (run = path relative to its ROOT, or the directory name) and label in
             accurate | silent_wrong | degenerate | ... : enables the catch-rate report below.
  --witness  CSV `run,hfov_deg`: K witness per run (median HFOV of GeoCalib / Pi3X / DA3) -> signal k_witness_disagree.
Output: one row per stage-1 run with every signal of glr_stage1_gate, the gate verdict, and (with --labels) the label;
        a report: per rule and for the combined gate - how many non-accurate runs it flags (catch), how many accurate runs it stops
        (false stop). The pre-registered pass line (docs/GLR_INTEGRATION.md, Phase 0.2): >= 7 of 9 silent-wrong caught, >= 90 % of all
        non-accurate flagged (stop or suspect), accurate runs falsely STOPPED <= 2 of the accurate set. Thresholds come from the config
        and must not be tuned on the runs they are judged on - use leave-one-window-out folds (`--folds` groups runs by the first
        path component) when you adjust them.
Needs numpy + pycolmap (the GLRMap environment has both). Windows with several repeats of the same run directory are separate rows.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import glr_stage1_gate as G  # noqa: E402

RULES = ["registered_lt_half", "points_per_image", "hfov_out_of_range", "fx_fy_mismatch", "planar_or_panoramic_pairs",
         "reproj_gt_1px", "pose_step_outlier", "focal_at_colmap_default", "low_parallax", "k_witness_disagree"]


def find_runs(root):
    for db in sorted(Path(root).rglob("database.db")):
        d = db.parent
        models = sorted((d / "sparse").glob("*/cameras.*")) if (d / "sparse").exists() else []
        if models:
            yield d, db, models[0].parent


def load_csv(path, key, val):
    if not path:
        return {}
    with open(path) as fh:
        return {r[key]: r[val] for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--witness")
    ap.add_argument("--out", default="gate_signals.csv")
    ap.add_argument("--cfg")
    ap.add_argument("--min-frames", type=int, default=0)
    a = ap.parse_args()
    cfg = G._load_cfg(a.cfg)
    labels, wit = load_csv(a.labels, "run", "label"), load_csv(a.witness, "run", "hfov_deg")
    rows = []
    for root in a.roots:
        for d, db, model in find_runs(root):
            run = str(d.relative_to(root)) if d != Path(root) else d.name
            try:
                m, s = G.summarize_model(model), G.summarize_db(db)
                n = s["n_images"]
                if n < a.min_frames:
                    continue
                w = float(wit[run]) if run in wit else (float(wit[d.name]) if d.name in wit else None)
                v = G.gate(m, s, n, cfg, w)
                row = {"run": run, "n_frames": n, **{k: m.get(k) for k in ("n_reg", "n_points", "reproj", "focal_w", "fx_fy", "hfov_deg",
                                                                          "step_ratio", "tri_angle_p50")},
                       "pts_per_img": v["signals"].get("pts_per_img"), "planar_frac": s.get("planar_frac"), "kp_mean": s.get("kp_mean"),
                       "witness_dis": v["signals"].get("witness_hfov_dis"), "verdict": v["verdict"],
                       "stop": ";".join(v["stop"]), "suspect": ";".join(v["suspect"]), "label": labels.get(run, labels.get(d.name, ""))}
            except Exception as e:  # noqa: BLE001
                row = {"run": run, "verdict": "error", "stop": repr(e)[:120], "label": labels.get(run, "")}
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("run", "n_reg", "pts_per_img", "focal_w", "planar_frac", "verdict", "label")}, default=float))
    if not rows:
        sys.exit("no stage-1 outputs found")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out, len(rows), "runs")
    if not labels:
        return
    acc = [r for r in rows if r["label"] == "accurate"]
    bad = [r for r in rows if r["label"] and r["label"] != "accurate"]
    silent = [r for r in rows if r["label"] == "silent_wrong"]
    print(f"\nlabelled: {len(acc)} accurate, {len(bad)} non-accurate ({len(silent)} silent_wrong)")
    flag = lambda r, rule: rule in (r.get("stop", "") + ";" + r.get("suspect", "")).split(";")  # noqa: E731
    print(f"{'rule':30s} {'catch non-acc':>14s} {'catch silent':>13s} {'false flag acc':>15s}")
    for rule in RULES:
        c = sum(flag(r, rule) for r in bad)
        cs = sum(flag(r, rule) for r in silent)
        f = sum(flag(r, rule) for r in acc)
        print(f"{rule:30s} {c:>8d}/{len(bad):<5d} {cs:>7d}/{len(silent):<5d} {f:>9d}/{len(acc):<5d}")
    any_flag = lambda r: r.get("verdict") in ("stop", "suspect")  # noqa: E731
    stop = lambda r: r.get("verdict") == "stop"  # noqa: E731
    c_any, c_silent, fs = sum(any_flag(r) for r in bad), sum(any_flag(r) for r in silent), sum(stop(r) for r in acc)
    print(f"\ncombined gate: flags (stop or suspect) {c_any}/{len(bad)} non-accurate, {c_silent}/{len(silent)} silent_wrong; "
          f"false STOPS on accurate: {fs}/{len(acc)}")
    ok_sil = (len(silent) == 0) or (c_silent >= min(7, len(silent) * 7 / 9))
    ok_all = len(bad) == 0 or c_any / len(bad) >= 0.9
    ok_fs = fs <= 2
    print("pre-registered pass line (catch >= 7/9 silent, >= 90 % non-accurate, false stops <= 2):",
          "PASS" if (ok_sil and ok_all and ok_fs) else "FAIL", f"(silent {ok_sil}, all {ok_all}, false-stop {ok_fs})")
    by_win = defaultdict(list)
    for r in rows:
        by_win[r["run"].split("/")[0]].append(r)
    print(f"{len(by_win)} window groups (first path component) for leave-one-window-out threshold work")


if __name__ == "__main__":
    main()
