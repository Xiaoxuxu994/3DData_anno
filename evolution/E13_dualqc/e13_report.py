"""E13: write outputs/evolution/E13/tables.md from raw/*.csv + summary.json (run after e13_analysis.py, e13_fsweep_eval.py)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

E13 = Path("/data/home/z00533370/outputs/evolution/E13")
R = E13 / "raw"
S = json.loads((E13 / "summary.json").read_text())


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


L = ["# E13 — dual-gate QC (gate 1 cross-model + gate 2 model-independent self-check)", ""]
m = S["main_rule"]
L += [f"Gate-2 rule chosen by the pre-declared criterion: reject if max_i(s_i / thr_i) > 1, signals {m['signals']} "
      f"(oriented: fsv1 = free-space violation share, cons5_occ -> 1 - occlusion-aware 5-frame depth consistency), "
      f"calibration `{m['calibration']}`, thresholds (fit on all 476 pool labels): "
      + ", ".join(f"{k} {v:.3f}" for k, v in m["thresholds"].items()), ""]
L += ["## 1. Single self-check signals on the 476 pool (AUROC; oriented higher = worse)", "", md(pd.read_csv(R / "pool_single_signals.csv")), ""]
v = pd.read_csv(R / "pool_rule_variants_lodo.csv")
L += ["## 2. Gate-2 rule variants, LODO on the 476 pool (gate 1 = LODO v0 rule: keeps 35.9 %, 8.8 % geo fail, 24.3 % false reject)", "",
      md(v[["rule", "thr", "g2_retention", "g2_geo_kept", "g2_false_reject", "g2_auroc", "dual_retention", "dual_geo_kept", "dual_false_reject", "ret_loss", "fr_add"]]), ""]
o = pd.read_csv(R / "operating_points.csv")
o = o[["set", "gate", "n", "retention", "geo_fail_base", "geo_fail_kept", "pose_fail_kept", "fov_fail_kept", "depth_fail_kept", "false_reject_good", "catch_bad", "auroc_geo"]]
L += ["## 3. Operating points (score <= 1): gate 1 / gate 2 / dual", "",
      "pool476 = 476 method x clip labels (gate 1 = v0 rule, LODO; gate 2 LODO). v0 / v13 composites: gate 1 = v0 rule / v1c, gate-2 thresholds "
      "from the full pool (out of sample for the composites). ABS = A + B + ScanNet++ (S). false_reject_good = share of geo-good labels rejected; "
      "catch_bad = share of geo-failed labels rejected.", "", md(o), ""]
cv = pd.read_csv(R / "composite_variants.csv")
L += ["## 4. All gate-2 variants on the composites (A+B+ScanNet++, n=92 each)", "", md(cv), ""]
d = S["decision"]
L += ["## 5. Decision against the pre-declared criterion", "",
      f"Criterion: {S['criterion']}", "", md(pd.DataFrame({k: v for k, v in d.items() if k != "adopt"}).T.reset_index().rename(columns={"index": "set"})),
      "", f"**adopt = {d['adopt']}**", ""]
c = pd.read_csv(R / "C_reject_by_cat.csv")
L += ["## 6. C group (72 wild clips, no GT): reject rate per category", "", md(c), ""]
L += ["Hard cases / line-G flags (v1.3 label): see raw/C_composites_decisions.csv.", "", "```", json.dumps(S["C"], indent=1, ensure_ascii=False), "```", ""]
b = pd.read_csv(R / "BTbkSEotmX0_gate2.csv")
L += ["## 7. BTbkSEotmX0 — gate 2 on the BA / feed-forward / composite labels", "", md(b), ""]
f = pd.read_csv(R / "fsweep_signals.csv")
sw = f[f.method.str.startswith("fs")]
rows = []
for clip, g in sw.groupby("clip"):
    r = {"clip": clip, "gt_hfov": g.gt_hfov.iloc[0], "label_hfov(s=1)": g[g.method == "fss1.00"].hfov.iloc[0]}
    for s_, best in (("photo1", "min"), ("photo5", "min"), ("iflow5_epe", "min"), ("iflow1_lt3", "max"), ("cons5_occ", "max"), ("cons20", "max"), ("fsv1", "min"), ("samp5_med", "min")):
        i = g[s_].idxmin() if best == "min" else g[s_].idxmax()
        edge = g.loc[i, "method"] in ("fss0.45", "fss1.60")
        r[f"argbest_{s_}"] = f"{g.loc[i, 'hfov']:.0f}{'*' if edge else ''}"
    rows.append(r)
L += ["## 8. Focal sweep (v1.3 re-solved with K x s, s = 0.45..1.6): HFOV (deg) at which each self-check is best", "",
      "`*` = at the sweep edge (no interior optimum). Controls KITTI / vKITTI2 have GT; Scene06 is almost static (unobservable).", "",
      md(pd.DataFrame(rows), "{:.1f}"), "", "Curves: fsweep_curves.png; raw: raw/fsweep_signals.csv", ""]
L += ["## 9. Cost", "", "```", json.dumps(S["cost"], indent=1), "```", "",
      "Cache (RAFT fw/bw gaps 1,5 + SuperPoint/LightGlue gaps 5,20, 512 px grid), ScanNet++ measured: logs/cache_S.log.", ""]
(E13 / "tables.md").write_text("\n".join(L))
print("\n".join(L[-40:]))
