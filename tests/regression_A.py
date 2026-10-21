"""Compare a pipeline run on A-group GT clips with the recorded v1.3 results (E3b / E3 / E8 / E9).

Usage: cache/qc-venv/bin/python regression_A.py OUT [--strict]
For every clip in OUT/bench/clips.json:
  metrics : code/bench/eval3d_bench.eval_clip on the new label (OUT/bench/v13) vs the row "pi3x_kgate@vipe_td" of
            outputs/evolution/E3b/raw/A.csv (the adopted v1.3 setting; reference_metrics_A_v13 in pipeline_best.yaml)
  scale   : k vs k_vipe_td in outputs/evolution/E3/scales_A.csv
  QC      : xtraj_ba / xdepth_k / kdis / v1c / accept vs outputs/evolution/E1/qc_rule_cheap.csv (E8 inputs)
  flags   : jump60 / fov_out vs outputs/evolution/E9/flags.csv
ViPE is not bit-reproducible (its SLAM is stochastic), so values are compared with tolerances (TOL below). The QC
witness signals depend on the ViPE run (DA3 is conditioned on ViPE's cameras): when this run's ViPE trajectory differs
from the recorded ViPE run by more than VIPE_RERUN % (row vipe_rerun_traj_pct), a FAIL on xtraj_ba / xdepth_k / v1c
is reported as WARN, as long as the accept decision is unchanged (seen on KITTI 07: ViPE re-run differs by 2.9 %).
Exit code 0 = all PASS; WARN lines (a failure flag flipped while the metric is within tolerance of its threshold)
do not fail the test unless --strict. Writes OUT/regression.csv.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ANNO_ROOT", "/data/home/z00533370"))
for p in ("code/qc", "code/bench", "code/vipe_eval"):
    sys.path.insert(0, str(ROOT / p))
import eval3d_bench as E  # noqa: E402
import qc_cross as QX  # noqa: E402

OUT = Path(sys.argv[1])
STRICT = "--strict" in sys.argv
B = OUT / "bench"
EV = ROOT / "outputs/evolution"
REF = pd.read_csv(EV / "E3b/raw/A.csv")
REF = REF[REF["method"] == "pi3x_kgate@vipe_td"].set_index("clip")
SC = pd.read_csv(EV / "E3/scales_A.csv").set_index("clip")
QCR = pd.read_csv(EV / "E1/qc_rule_cheap.csv")
QCR = QCR[QCR["group"] == "A"].set_index("clip")
FL = pd.read_csv(EV / "E9/flags.csv")
FL = FL[FL["group"] == "A"].set_index("clip")

# (kind, tolerance): "abs" |new - ref| <= tol, "rel" |new - ref| <= tol * |ref| (or abs floor)
TOL = {"ate_sim3_cm": ("rel", 0.30, 0.5), "sim3_scale": ("abs", 0.03), "fov_err_deg": ("abs", 0.5),
       "depth_metric_absrel": ("abs", 0.015), "pc_fscore_5cm": ("abs", 0.03), "g20_epe_med_px": ("rel", 0.20, 0.5),
       "scale_k": ("rel", 0.03, 0.0), "xtraj_ba": ("rel", 0.30, 0.3), "xdepth_k": ("abs", 0.02), "kdis": ("abs", 1.0),
       "v1c": ("abs", 0.15)}
VIPE_RERUN = 1.0  # % of trajectory length
WITNESS = {"xtraj_ba", "xdepth_k", "v1c"}
FAILS = {"fail_ate_sim3": "ate_sim3_cm", "fail_scale": "sim3_scale", "fail_depth": "depth_metric_absrel", "fail_fov": "fov_err_deg"}


def ok(name, new, ref):
    if not (np.isfinite(new) and np.isfinite(ref)):
        return np.isfinite(new) == np.isfinite(ref)
    kind, tol, *floor = TOL[name]
    lim = tol if kind == "abs" else max(tol * abs(ref), floor[0] if floor else 0.0)
    return abs(new - ref) <= lim


rows, status = [], []
for c in json.loads((B / "clips.json").read_text()):
    clip = c["clip"]
    m = E.eval_clip((B, "v13", c))
    q = json.loads((OUT / "qc" / f"{clip}.json").read_text())
    ref = REF.loc[clip]
    new = {k: m.get(k, np.nan) for k in TOL if k in m}
    new.update(scale_k=q["scale"]["k"], xtraj_ba=q["xtraj_ba"], xdepth_k=q["xdepth_k"], kdis=q["kdis"], v1c=q["v1c"])
    refv = {k: ref.get(k, np.nan) for k in TOL if k in ref.index}
    refv["scale_k"] = SC.loc[clip, "k_vipe_td"]
    qr = QCR.loc[clip]
    refv.update(xtraj_ba=qr["xtraj_ba"], xdepth_k=qr["xdepth_k"], kdis=qr["kdis"])
    refv["v1c"] = float(np.nanmax([qr["xtraj_ba"] / 3.27, qr["xdepth_k"] / 0.194, qr["kdis"] / 13.2]))
    hw = np.load(B / "gt" / f"{clip}.npz", allow_pickle=True)["depth"].shape[1:]
    rerun = QX.traj_pct(E.load_method(B, "vipe", clip, c["n_frames"], *hw)[0][:, :3, 3],
                        E.load_method(ROOT / "outputs/bench/eval3d_A", "vipe", clip, c["n_frames"], *hw)[0][:, :3, 3])
    rows.append({"clip": clip, "item": "vipe_rerun_traj_pct", "new": rerun, "ref": 0.0, "result": "INFO"})
    ra = bool(refv["v1c"] <= 1 and not (FL.loc[clip, "jump60"] or FL.loc[clip, "fov_out"])) if clip in FL.index else None
    for k in TOL:
        a, b = float(new.get(k, np.nan) if new.get(k) is not None else np.nan), float(refv.get(k, np.nan))
        p = ok(k, a, b)
        res = "PASS" if p else ("WARN" if k in WITNESS and rerun > VIPE_RERUN and q["accept"] == ra else "FAIL")
        rows.append({"clip": clip, "item": k, "new": a, "ref": b, "result": res})
        status.append(res == "PASS" or (res == "WARN" and not STRICT))
    for f, src in FAILS.items():  # failure flags: a flip is a WARN when the metric itself is within tolerance
        a, b = m.get(f, np.nan), ref.get(f, np.nan)
        same = (a == b) or (np.isnan(a) and np.isnan(b))
        res = "PASS" if same else ("WARN" if ok(src, float(m.get(src, np.nan)), float(ref.get(src, np.nan))) else "FAIL")
        rows.append({"clip": clip, "item": f, "new": a, "ref": b, "result": res})
        status.append(res == "PASS" or (res == "WARN" and not STRICT))
    for f in ("jump60", "fov_out"):
        a, b = q["flags"][f], (int(FL.loc[clip, f]) if clip in FL.index else np.nan)
        rows.append({"clip": clip, "item": f, "new": a, "ref": b, "result": "PASS" if a == b else "FAIL"})
        status.append(a == b)
    near = abs(refv["v1c"] - 1) <= TOL["v1c"][1]
    res = "PASS" if q["accept"] == ra else ("WARN" if near else "FAIL")
    rows.append({"clip": clip, "item": "accept", "new": q["accept"], "ref": ra, "result": res})
    status.append(res == "PASS" or (res == "WARN" and not STRICT))

df = pd.DataFrame(rows)
df.to_csv(OUT / "regression.csv", index=False, float_format="%.5g")
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print("\nsummary:", df["result"].value_counts().to_dict())
print("REGRESSION", "PASS" if all(status) else "FAIL")
sys.exit(0 if all(status) else 1)
