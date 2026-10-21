"""E9: literature hard thresholds as extra QC flags (ideas.md #6), evaluated on the v1.3 label (offline).

python e9_hard_rules.py -> outputs/evolution/E9/flags.csv + printed summary
Flags computed from the label (pi3x_kgate, gated ViPE K; scale irrelevant for these):
  fov_out    HFOV outside [30, 120] deg                               (VGGT-Omega data engine)
  jump60     any adjacent-frame viewing-direction change > 60 deg      (Sekai)
  jump20     same with 20 deg (line G "suspect cut" flag, per-label)
  reversal   >= 2 heading reversals (> 150 deg between 1 s-apart motion directions) within 10 s   (Sekai)
  tiny_move  camera path < 2% of median scene depth (no parallax -> focal/scale unobservable)
Evaluated on C (hard_case vs normal clips; per hard type from the clip name) and on A+B (GT geometric failure).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/home/z00533370/code/bench")
from eval3d_bench import load_method  # noqa: E402

R = Path("/data/home/z00533370/outputs")
E = R / "evolution"
BEN = {"A": R / "bench/eval3d_A", "B": R / "bench/eval3d_B", "C": R / "bench/wild"}


def flags(T, K, D, W, fps):
    f = {}
    hf = float(np.degrees(2 * np.arctan(W / 2 / np.median(K[:, 0, 0]))))
    f["hfov"] = hf
    f["fov_out"] = int(not 30 <= hf <= 120)
    z = T[:, :3, 2]
    ang = np.degrees(np.arccos(np.clip((z[1:] * z[:-1]).sum(1), -1, 1)))
    f["max_step_deg"] = float(ang.max()) if len(ang) else 0.0
    f["jump60"], f["jump20"] = int(f["max_step_deg"] > 60), int(f["max_step_deg"] > 20)
    p = T[:, :3, 3]
    step = max(1, int(round(fps)))
    v = p[step:] - p[:-step]
    nv = np.linalg.norm(v, axis=1)
    ok = nv > 1e-3 * max(np.linalg.norm(np.diff(p, axis=0), axis=1).sum(), 1e-9)
    rev = 0
    if ok.sum() > step + 1:
        u = v[ok] / nv[ok][:, None]
        a = np.degrees(np.arccos(np.clip((u[step:] * u[:-step]).sum(1), -1, 1)))
        rev = int((a > 150).sum())
    f["reversals"] = rev
    f["reversal"] = int(rev >= 2)
    L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    md = float(np.nanmedian(D[D > 0])) if np.isfinite(D).any() else np.nan
    f["move_ratio"] = L / md if md and np.isfinite(md) else np.nan
    f["tiny_move"] = int(f["move_ratio"] < 0.02) if np.isfinite(f["move_ratio"]) else 0
    return f


rows = []
for G, B in BEN.items():
    for c in json.loads((B / "clips.json").read_text()):
        if c.get("dataset") == "eth3d_mvs":
            continue
        n = c["n_frames"]
        hw = c.get("hw", [480, 640])
        H, W = (hw[0] // 2 * 2, hw[1] // 2 * 2) if G != "C" else (540, 960)
        try:
            T, K, D, _ = load_method(B, "pi3x_kgate", c["clip"], n, H, W)
        except Exception as e:  # noqa: BLE001
            rows.append({"group": G, "clip": c["clip"], "error": str(e)[:100]})
            continue
        r = {"group": G, "clip": c["clip"], "hard_case": bool(c.get("hard_case", False)),
             "type": c["clip"].split("__")[1] if G == "C" and c["clip"].startswith("hard") else c.get("cat", c.get("dataset"))}
        r.update(flags(T, K, D, W, c.get("fps_out", 15) if G == "C" else 15))
        rows.append(r)
d = pd.DataFrame(rows)
gt = []
for g in "AB":
    x = pd.read_csv(E / "E3b" / "raw" / f"{g}.csv")
    x = x[x["method"] == "pi3x_kgate@vipe_td"].copy()
    x["geom_fail"] = ((x["fail_ate_sim3"] > 0) | (x["fail_fov"] > 0) | (x["fail_depth"].fillna(0) > 0)).astype(int)
    gt.append(x[["clip", "geom_fail"]].assign(group=g))
d = d.merge(pd.concat(gt), on=["group", "clip"], how="left")
(E / "E9").mkdir(exist_ok=True)
d.to_csv(E / "E9" / "flags.csv", index=False, float_format="%.4g")
F = ["fov_out", "jump60", "jump20", "reversal", "tiny_move"]
c = d[d["group"] == "C"]
print("C hard vs normal flag rates:")
print(c.groupby("hard_case")[F].mean().round(2).to_string())
print(c[c["hard_case"]].groupby("type")[F].mean().round(2).to_string())
ab = d[d["group"].isin(["A", "B"])]
print("A+B flag rate | failure rate when flagged / not flagged:")
for f in F:
    m = ab[f] == 1
    print(f, round(m.mean(), 3), round(ab.loc[m, "geom_fail"].mean(), 3) if m.any() else None, round(ab.loc[~m, "geom_fail"].mean(), 3))
