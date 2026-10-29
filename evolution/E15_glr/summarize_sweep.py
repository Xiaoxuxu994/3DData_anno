"""summarize_sweep.py S1_ROOT OUT_DIR [--decisions decisions.csv] [--manifest MANIFEST.json] [--clips a,b]  -> OUT_DIR/{sweep_runs.csv, tables.md}

Reads the outputs of tools/stage1_sfm.py laid out as S1_ROOT/<clip>/<run>/{stage1.json, sparse/0, database.db}, with the runs
  v0a        GLRMap stage 1 as described (PINHOLE, initial focal 1.2 x max(W, H), focal refined, principal point fixed), reduced cost config
  v0b        the same matching database, other mapper seed (stability)
  v1sp       SIMPLE_PINHOLE on the same database (fx = fy)
  v2prior    PINHOLE with the GeoCalib focal as initial value (still refined)
  v3fixed    PINHOLE with the GeoCalib focal fixed
  v0a_hq960  v0a at 960 px (resolution control for clips that stopped on resolution-dependent signals)
  ref        silver reference: every 6th frame, exhaustive matching, SIMPLE_RADIAL, incremental mapper
and writes one row per run (signals, gate verdict, agreement with the reference) plus the tables used in docs/experiments/LOG.md (E15).
Definitions (pre-declared in LOG.md before the reference results were read): a run is `silver_accurate` iff >= 90 % of its frames registered,
|f/W - f_ref/W| <= 10 % of f_ref/W for both fx and fy (the worse of the two) and the Sim(3) ATE to the reference over the common frames is <= 3.27 % of the reference path. Gate verdict
ok / suspect + not silver-accurate = silent error; gate stop + silver-accurate = false stop. The reference is a second SfM run, not ground truth.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import glr_common as C  # noqa: E402
import glr_stage1_gate as G  # noqa: E402

RUNS = ["v0a", "v0b", "v1sp", "v2prior", "v3fixed", "v0a_hq960", "ref"]
ATE_MAX, F_MAX, REG_MIN = 3.27, 10.0, 0.9


def frame_no(name):
    return int("".join(ch for ch in Path(name).stem if ch.isdigit()))


def load_run(clip_dir, run):
    d = clip_dir / run
    sj = d / "stage1.json"
    if not sj.exists():
        return None
    res = json.loads(sj.read_text())
    base = res["args"].get("reuse_db")
    stride = int((json.loads((Path(base) / "stage1.json").read_text()) if base else res)["args"].get("stride", 1))
    out = {"res": res, "stride": stride, "cams": None}
    if res.get("model") and (d / "sparse" / "0").exists():
        out["cams"] = G.load_cameras(d / "sparse" / "0")
        out["idx"] = np.array([(frame_no(n) - 1) * stride for n in out["cams"]["names"]])
    return out


def ate_pct(a, b):
    """Sim(3) ATE of the camera centres of run a vs run b over their common frames, % of b's path over those frames -> (pct, n_common)."""
    common = np.intersect1d(a["idx"], b["idx"])
    if len(common) < 5:
        return float("nan"), int(len(common))
    ia = {int(f): i for i, f in enumerate(a["idx"])}
    ib = {int(f): i for i, f in enumerate(b["idx"])}
    pa = np.array([a["cams"]["c2w"][ia[int(f)], :3, 3] for f in common])
    pb = np.array([b["cams"]["c2w"][ib[int(f)], :3, 3] for f in common])
    L = float(np.linalg.norm(np.diff(pb, axis=0), axis=1).sum())
    if L < 1e-9:
        return float("nan"), int(len(common))
    s, R, t = C.umeyama(pa, pb, True)
    return float(100 * np.sqrt(((s * pa @ R.T + t - pb) ** 2).sum(1).mean()) / L), int(len(common))


def load_witness(pool, clip):
    """GeoCalib's median focal / W of the clip (the only K witness the sandbox has) from POOL/<clip>/scores.json -> {focal_w, hfov (deg)} or None."""
    if not pool:
        return None
    try:
        fw = float(np.median(json.loads((Path(pool) / clip / "scores.json").read_text())["calib"]["focal_w"]))
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return {"focal_w": fw, "hfov": math.degrees(2 * math.atan(0.5 / fw))} if np.isfinite(fw) and fw > 0 else None


def row_for(clip, run, r, ref, witness=None):
    res = r["res"]
    m, db = res.get("model") or {}, res.get("db") or {}
    n_img = db.get("n_images") or 0
    row = {"clip": clip, "run": run, "n_img": n_img, "n_reg": m.get("n_reg", 0), "reg_frac": (m.get("n_reg", 0) / n_img) if n_img else None,
           "n_points": m.get("n_points"), "pts_img": (m["n_points"] / m["n_reg"]) if m.get("n_reg") else None, "reproj": m.get("reproj"),
           "focal_w": m.get("focal_w"), "fx_fy": m.get("fx_fy"), "hfov": m.get("hfov_deg"), "planar": db.get("planar_frac"),
           "tri50": m.get("tri_angle_p50"), "step_ratio": m.get("step_ratio"), "t_total_s": sum((res.get("timing_s") or {}).values())}
    if not m:
        row.update(verdict="stop", flags="no_model")
    else:
        g = G.gate(m, db, n_img)
        row.update(verdict=g["verdict"], flags=";".join(g["stop"] + g["suspect"]))
    if witness:
        row.update(witness_focal_w=witness["focal_w"], witness_hfov=witness["hfov"])
        if m:
            row["kdis_deg"] = abs(m["hfov_deg"] - witness["hfov"])            # what the v1c K-disagreement part would see against this witness
            gw = G.gate(m, db, n_img, witness_hfov=witness["hfov"])
            row.update(verdict_w=gw["verdict"], flags_w=";".join(gw["stop"] + gw["suspect"]))
        else:
            row.update(verdict_w="stop", flags_w="no_model")
    if ref is not None and run != "ref":
        fr = (ref["res"].get("model") or {}).get("focal_w")
        if m and fr:
            ex = m["focal_w"] / fr - 1                                          # fx against the reference's single focal ...
            ey = m["focal_w"] / m["fx_fy"] / fr - 1 if m.get("fx_fy") else ex   # ... and fy (= fx / (fx / fy)): a PINHOLE label is only as good as the worse one
            row["focal_err_pct"] = 100 * max(ex, ey, key=abs)
        if r["cams"] is not None and ref["cams"] is not None:
            row["ate_vs_ref_pct"], row["n_common"] = ate_pct(r, ref)
        acc = (row["reg_frac"] or 0) >= REG_MIN and abs(row.get("focal_err_pct", np.inf)) <= F_MAX and row.get("ate_vs_ref_pct", np.inf) <= ATE_MAX
        row["silver_accurate"] = bool(acc)
        row["outcome"] = ("silent_error" if row["verdict"] != "stop" and not acc else "false_stop" if row["verdict"] == "stop" and acc
                          else "true_pass" if acc else "true_stop")
    return row


def f(x, nd=2):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))


def md_table(rows, cols, head=None):
    head = head or cols
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("s1_root")
    ap.add_argument("out_dir")
    ap.add_argument("--decisions", help="filter/decide.py output (clip, decision, reason ...), joined into the per-clip table")
    ap.add_argument("--manifest", help="docs/experiments/E15_ood_manifest.json: roles (easy / control_negative) and categories")
    ap.add_argument("--pool", help="filter pool dir (POOL/<clip>/scores.json with the s5 GeoCalib stage): adds the K-witness columns and table E")
    ap.add_argument("--clips")
    a = ap.parse_args(argv)
    root, out = Path(a.s1_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roles = {c["id"]: c for c in json.loads(Path(a.manifest).read_text())["clips"]} if a.manifest else {}
    dec = {}
    if a.decisions:
        with open(a.decisions) as fh:
            dec = {r["clip"]: r for r in csv.DictReader(fh)}
    clips = a.clips.split(",") if a.clips else sorted(p.name for p in root.iterdir() if p.is_dir())
    rows = []
    for clip in clips:
        runs = {run: load_run(root / clip, run) for run in RUNS}
        ref = runs.get("ref")
        if ref is not None and ref["cams"] is None:
            ref = None
        for run, r in runs.items():
            if r is not None:
                row = row_for(clip, run, r, ref, load_witness(a.pool, clip))
                row["role"] = (roles.get(clip) or {}).get("role", "")
                rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "sweep_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(C.clean(rows))
    md = []
    byrun = lambda run, role=None: [r for r in rows if r["run"] == run and (role is None or r.get("role") == role)]  # noqa: E731
    # --- A: v0a per clip
    md.append("### A. Stage-1 replica (v0a) per clip: signals and gate\n")
    cols = ["clip", "reg", "pts/img", "reproj", "f/W", "fx/fy", "HFOV", "planar", "tri50", "gate", "flags", "decision"]
    tab = []
    for r in byrun("v0a"):
        d = dec.get(r["clip"], {})
        tab.append({"clip": r["clip"], "reg": f"{r['n_reg']}/{r['n_img']}", "pts/img": f(r["pts_img"], 0), "reproj": f(r["reproj"]), "f/W": f(r["focal_w"], 3),
                    "fx/fy": f(r["fx_fy"], 3), "HFOV": f(r["hfov"], 1), "planar": f(r["planar"]), "tri50": f(r["tri50"], 1), "gate": r["verdict"],
                    "flags": r["flags"], "decision": (d.get("decision", "") + (" " + d.get("reason", "") if d.get("reason") else "")).strip()})
    md.append(md_table(tab, cols))
    # --- B: variants vs reference
    md.append("\n### B. Camera-model / prior variants against the silver reference (focal error and ATE are vs `ref`)\n")
    cols = ["variant", "runs", "registered >= 90 %", "gate ok/suspect", "silver-accurate", "silent errors", "of which kdis <= 13.2 deg", "false stops", "median abs f err %", "median ATE %"]
    kmax = G.DEFAULT_CFG["max_witness_hfov_dis_deg"]
    tab = []
    for run in ("v0a", "v0b", "v1sp", "v2prior", "v3fixed", "v0a_hq960"):
        rs = [r for r in byrun(run, "easy" if roles else None) if "outcome" in r]
        if not rs:
            continue
        fe = [abs(r["focal_err_pct"]) for r in rs if r.get("focal_err_pct") is not None]
        at = [r["ate_vs_ref_pct"] for r in rs if r.get("ate_vs_ref_pct") is not None and np.isfinite(r["ate_vs_ref_pct"])]
        tab.append({"variant": run, "runs": len(rs), "registered >= 90 %": sum((r["reg_frac"] or 0) >= REG_MIN for r in rs),
                    "gate ok/suspect": sum(r["verdict"] != "stop" for r in rs), "silver-accurate": sum(r["silver_accurate"] for r in rs),
                    "silent errors": sum(r["outcome"] == "silent_error" for r in rs),
                    "of which kdis <= 13.2 deg": ("" if not any("kdis_deg" in r for r in rs) else
                                                  sum(r["outcome"] == "silent_error" and r.get("kdis_deg") is not None and r["kdis_deg"] <= kmax for r in rs)),
                    "false stops": sum(r["outcome"] == "false_stop" for r in rs),
                    "median abs f err %": f(float(np.median(fe)) if fe else None, 1), "median ATE %": f(float(np.median(at)) if at else None, 2)})
    md.append(md_table(tab, cols))
    # --- C: per clip variants
    md.append("\n### C. Per clip: focal error % / ATE % vs reference / gate (stop = S, suspect = s, ok = .)\n")
    cols = ["clip", "ref f/W"] + [v for v in ("v0a", "v0b", "v1sp", "v2prior", "v3fixed", "v0a_hq960")]
    tab = []
    for clip in clips:
        rr = {r["run"]: r for r in rows if r["clip"] == clip}
        if "ref" not in rr:
            continue
        t = {"clip": clip, "ref f/W": f(rr["ref"]["focal_w"], 3)}
        for v in cols[2:]:
            r = rr.get(v)
            t[v] = "" if r is None else f"{f(r.get('focal_err_pct'), 0)} / {f(r.get('ate_vs_ref_pct'), 1)} / {({'stop': 'S', 'suspect': 's', 'ok': '.'}[r['verdict']])}"
        tab.append(t)
    md.append(md_table(tab, cols))
    # --- D: controls
    ctl = [r for r in byrun("v0a") if r.get("role") == "control_negative"]
    if ctl:
        md.append("\n### D. Hard negative controls: filter decision and stage-1 gate\n")
        cols = ["clip", "category", "filter decision", "reg", "pts/img", "planar", "fx/fy", "gate", "flags"]
        tab = []
        for r in ctl:
            d = dec.get(r["clip"], {})
            tab.append({"clip": r["clip"], "category": (roles.get(r["clip"]) or {}).get("category", ""), "filter decision": (d.get("decision", "") + " " + d.get("reason", "")).strip(),
                        "reg": f"{r['n_reg']}/{r['n_img']}", "pts/img": f(r["pts_img"], 0), "planar": f(r["planar"]), "fx/fy": f(r["fx_fy"], 3), "gate": r["verdict"], "flags": r["flags"]})
        md.append(md_table(tab, cols))
    # --- E: three estimates of the focal length per clip (post-hoc, exploratory: not part of the pre-declared criteria)
    if any("witness_focal_w" in r for r in rows):
        md.append("\n### E. Focal length (f / W) three ways: replica fx, fy | silver reference | GeoCalib median (post-hoc, exploratory)\n")
        cols = ["clip", "v0a fx", "v0a fy", "ref", "GeoCalib", "v0a vs ref %", "v0a vs GeoCalib %", "ref vs GeoCalib %", "who agrees (10 %)"]
        tab = []
        for clip in clips:
            rr = {r["run"]: r for r in rows if r["clip"] == clip}
            a_, rf = rr.get("v0a"), rr.get("ref")
            if not a_ or not rf or a_.get("focal_w") is None or rf.get("focal_w") is None or a_.get("witness_focal_w") is None:
                continue
            fx, fy = a_["focal_w"], a_["focal_w"] / a_["fx_fy"]
            fr, fg = rf["focal_w"], a_["witness_focal_w"]
            worst = lambda x: max(abs(fx / x - 1), abs(fy / x - 1))  # noqa: E731
            d_ar, d_ag, d_rg = worst(fr), worst(fg), abs(fr / fg - 1)
            ok = lambda d: d <= F_MAX / 100  # noqa: E731
            who = ("all three agree" if ok(d_ar) and ok(d_ag) and ok(d_rg) else "replica = GeoCalib, reference differs" if ok(d_ag) and not ok(d_ar) and not ok(d_rg)
                   else "replica = reference, GeoCalib differs" if ok(d_ar) and not ok(d_ag) else "reference = GeoCalib, replica differs" if ok(d_rg) and not ok(d_ar)
                   else "no two agree")
            tab.append({"clip": clip, "v0a fx": f(fx, 3), "v0a fy": f(fy, 3), "ref": f(fr, 3), "GeoCalib": f(fg, 3), "v0a vs ref %": f(100 * d_ar, 0),
                        "v0a vs GeoCalib %": f(100 * d_ag, 0), "ref vs GeoCalib %": f(100 * d_rg, 0), "who agrees (10 %)": who})
        md.append(md_table(tab, cols))
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
