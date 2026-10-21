"""Compare a pipeline run on C-group (wild, no GT) clips with the recorded experiment values.

Usage: cache/qc-venv/bin/python compare_C.py OUT
  filter : key stage scores + decision vs data/filter_pool/_calibC/<clip>/scores.json (+ decide.py on them)
  scale  : k vs k_vipe_td (outputs/evolution/E3/scales_wild.csv)
  QC     : xtraj_ba / xdepth_k / kdis / v1c vs outputs/evolution/E1/qc_rule_cheap.csv (group C)
  flags  : jump60 / fov_out vs outputs/evolution/E9/flags.csv
Prints one table (new vs recorded); informational, no pass/fail (C has no GT).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ANNO_ROOT", "/data/home/z00533370"))
sys.path.insert(0, str(ROOT / "code/filter"))
import decide as D  # noqa: E402

OUT = Path(sys.argv[1])
EV = ROOT / "outputs/evolution"
CAL = ROOT / "data/filter_pool/_calibC"
SC = pd.read_csv(EV / "E3/scales_wild.csv").set_index("clip")
QCR = pd.read_csv(EV / "E1/qc_rule_cheap.csv")
QCR = QCR[QCR["group"] == "C"].set_index("clip")
FL = pd.read_csv(EV / "E9/flags.csv")
FL = FL[FL["group"] == "C"].set_index("clip")
KEYS = ["cuts.tn_max", "rules.lum_mean", "rules.lap_var_median", "rules.aesthetic", "ocr.text_area_max",
        "flow.static_pair_frac", "flow.dyn_frac", "calib.focal_ratio_end", "calib.k1_median", "vlm_qwen3vl_32b.reconstructable"]

rows = []
dec = pd.read_csv(OUT / "filter/decisions.csv").set_index("clip") if (OUT / "filter/decisions.csv").exists() else None
for c in json.loads((OUT / "bench/clips_all.json" if (OUT / "bench/clips_all.json").exists() else OUT / "bench/clips.json").read_text()):
    clip = c["clip"]
    if dec is not None and (CAL / clip / "scores.json").exists():
        new = D.flatten(json.loads((OUT / "pool" / clip / "scores.json").read_text()))
        old = D.flatten(json.loads((CAL / clip / "scores.json").read_text()))
        for k in KEYS:
            rows.append({"clip": clip, "item": k, "new": new.get(k), "recorded": old.get(k)})
        od = D.apply_vlm(D.apply_rules(pd.DataFrame([dict(old, clip=clip)])), "qwen3vl_32b").iloc[0]
        rows.append({"clip": clip, "item": "filter decision", "new": dec.loc[clip, "decision"],
                     "recorded": "pass" if od["rule_pass"] and od["vlm_pass"] else "reject"})
    q = OUT / "qc" / f"{clip}.json"
    if not q.exists():
        continue
    q = json.loads(q.read_text())
    rows.append({"clip": clip, "item": "scale k", "new": q["scale"]["k"],
                 "recorded": SC.loc[clip, "k_vipe_td"] if clip in SC.index else None})  # shots (__s<k>) have no record
    if clip in QCR.index:
        r = QCR.loc[clip]
        for k in ("xtraj_ba", "xdepth_k", "kdis"):
            rows.append({"clip": clip, "item": k, "new": q[k], "recorded": r[k]})
        rows.append({"clip": clip, "item": "v1c", "new": q["v1c"],
                     "recorded": float(np.nanmax([r["xtraj_ba"] / 3.27, r["xdepth_k"] / 0.194, r["kdis"] / 13.2]))})
    if clip in FL.index:
        for k in ("jump60", "fov_out", "hfov", "max_step_deg"):
            rows.append({"clip": clip, "item": k, "new": q["flags"][k], "recorded": FL.loc[clip, k]})
    rows.append({"clip": clip, "item": "accept", "new": q["accept"], "recorded": None})
df = pd.DataFrame(rows)
df.to_csv(OUT / "compare_C.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
