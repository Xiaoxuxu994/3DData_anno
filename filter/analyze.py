"""Calibration / comparison of the filter: rules and VLMs vs group C labels, VLM-VLM and VLM-rule agreement.

Usage: python analyze.py OUT_DIR TAG [TAG ...]
  Reads data/filter_pool/_calibC (group C, 72 clips) and data/filter_pool (pool) scores.
  Writes OUT_DIR/{calibC_pr.csv, vlm_agreement.csv, vlm_vs_rules.csv, summary.json}.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decide import RULES, apply_rules, load, vlm_field_reject  # noqa: E402

H = Path("/data/home/z00533370")
OUT, TAGS = Path(sys.argv[1]), sys.argv[2:]
OUT.mkdir(parents=True, exist_ok=True)

# Group C clips that are not flagged hard_case but, on visual inspection of the frames, should also be rejected
# (found while calibrating; the original label only marks the deliberately added hard cases).
REVISED_EXTRA = {
    "indoor_static__youtube_cc__gpKFsVg": "hard cut at frame 2-3",
    "outdoor_static__youtube_cc__IGYh7G": "hard cut at frame 166-167",
    "driving__youtube_cc__uRbDaKgz1_w": "cross-dissolve transition around frame 95-100",
    "embodied__droid__ep025_exterior": "static camera (robot arm moves)",
    "embodied__droid__ep028_exterior": "static camera (robot arm moves)",
    "embodied__droid__ep069_exterior": "static camera (robot arm moves)",
    "embodied__droid__ep096_exterior": "static camera (robot arm moves)",
    "embodied__youtube_cc__-jHQSVR8QT0": "static camera + subtitles / inset overlay",
    "indoor_dynamic__youtube_cc__wsNOEw": "static camera inside a train",
    "outdoor_static__sekai__FMopxWj7NS4": "near-static camera, far landscape",
    "outdoor_static__sekai__b06ThXzoqRI": "static camera + watermark",
    "embodied__droid__ep053_wrist": "almost black frames",
    "driving__youtube_cc__BTbkSEotmX0": "cross-dissolve transition around frame 112 (+ subtitles)",
    "outdoor_static__sekai__qzinq4hBvv4": "nadir aerial over water: planar scene, only boats move",
}


def label(df):
    orig = df.hard_case.astype(bool)
    extra = df["clip"].map(lambda c: any(c.startswith(k) for k in REVISED_EXTRA))
    return orig, orig | extra


def pr(pred, y):
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    return {"n_reject": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p and r and p + r else np.nan}


def vlm_reject(df, tag):
    c = f"vlm_{tag}.reconstructable"
    return df[c].map(lambda v: v != "yes") if c in df else pd.Series(False, index=df.index)


def vlm_fields(df, tag):
    return df.apply(lambda r: vlm_field_reject(r, tag), axis=1).astype(bool)


def cohen(a, b):
    m = a.notna() & b.notna()
    a, b = a[m].astype(str), b[m].astype(str)
    if not len(a):
        return np.nan, np.nan, 0
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == k).mean() * (b == k).mean() for k in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan, len(a)


FIELDS = ["scene_category", "indoor_outdoor", "dynamic_level", "camera_motion", "has_cuts", "is_cg_or_game", "fisheye",
          "low_light_or_blur", "reconstructable", "overlays.subtitles", "overlays.watermark", "overlays.picture_in_picture"]

summary = {}
C = apply_rules(load(H / "data/filter_pool/_calibC"))
y0, y1 = label(C)
rows = []
for name, f in RULES.items():
    fired = C.rule_reasons.str.split(";").map(lambda x, n=name: n in x)
    for lab, y in (("orig", y0), ("revised", y1)):
        rows.append({"filter": f"rule:{name}", "label": lab, **pr(fired, y)})
for lab, y in (("orig", y0), ("revised", y1)):
    rows.append({"filter": "rules_all", "label": lab, **pr(~C.rule_pass, y)})
    for t in TAGS:
        v = vlm_reject(C, t)
        rows.append({"filter": f"vlm:{t}", "label": lab, **pr(v, y)})
        rows.append({"filter": f"rules_or_vlm:{t}", "label": lab, **pr(v | ~C.rule_pass, y)})
        rows.append({"filter": f"rules_and_vlm:{t}", "label": lab, **pr(v & ~C.rule_pass, y)})
        vf = vlm_fields(C, t)
        rows.append({"filter": f"vlmF:{t}", "label": lab, **pr(vf, y)})
        rows.append({"filter": f"rules_or_vlmF:{t}", "label": lab, **pr(vf | ~C.rule_pass, y)})
        rows.append({"filter": f"rules_or_vlm_or_vlmF:{t}", "label": lab, **pr(v | vf | ~C.rule_pass, y)})
    if len(TAGS) >= 3:
        votes = sum(vlm_reject(C, t).astype(int) for t in TAGS)
        rows.append({"filter": "vlm_majority", "label": lab, **pr(votes * 2 > len(TAGS), y)})
        rows.append({"filter": "rules_or_vlm_majority", "label": lab, **pr((votes * 2 > len(TAGS)) | ~C.rule_pass, y)})
prdf = pd.DataFrame(rows)
prdf.to_csv(OUT / "calibC_pr.csv", index=False, float_format="%.3f")
summary["calibC_pr"] = prdf.round(3).to_dict("records")
summary["calibC_revised_extra"] = REVISED_EXTRA

# per-clip view on C for the README / visual check
keep = ["clip", "hard_case", "rule_reasons"] + [f"vlm_{t}.reconstructable" for t in TAGS if f"vlm_{t}.reconstructable" in C]
C[keep].assign(revised=y1).to_csv(OUT / "calibC_per_clip.csv", index=False)

# agreement on pool + C
P = load(H / "data/filter_pool")
P = apply_rules(P)
ag = []
for df_name, df in (("pool", P), ("C", C)):
    for a, b in itertools.combinations(TAGS, 2):
        for f in FIELDS:
            ca, cb = f"vlm_{a}.{f}", f"vlm_{b}.{f}"
            if ca in df and cb in df:
                po, k, n = cohen(df[ca], df[cb])
                ag.append({"set": df_name, "pair": f"{a}~{b}", "field": f, "agree": po, "kappa": k, "n": n})
agdf = pd.DataFrame(ag)
agdf.to_csv(OUT / "vlm_agreement.csv", index=False, float_format="%.3f")
if len(agdf):
    summary["vlm_agreement_mean_by_field"] = agdf[agdf.set == "pool"].groupby("field")[["agree", "kappa"]].mean().round(3).to_dict()

# VLM fields vs the matching expert rule
fired = lambda df, n: df.rule_reasons.str.split(";").map(lambda x: n in x)  # noqa: E731
vr = []
for df_name, df in (("pool", P), ("C", C)):
    for t in TAGS:
        p = f"vlm_{t}."
        if p + "reconstructable" not in df:
            continue
        pairs = {
            "has_cuts~rule:cut": (df[p + "has_cuts"] == True, fired(df, "cut")),  # noqa: E712
            "camera_static~rule:static_camera": (df[p + "camera_motion"] == "static", fired(df, "static_camera")),
            "camera_rotation~rule:no_parallax": (df[p + "camera_motion"] == "mostly_rotation", fired(df, "no_parallax")),
            "zoom~rule:zoom": (df[p + "camera_motion"] == "zoom", fired(df, "zoom")),
            "low_light_or_blur~rule:dark|blur": (df[p + "low_light_or_blur"] == True, fired(df, "dark") | fired(df, "blur")),  # noqa: E712
            "subtitles|watermark~rule:text_overlay": ((df[p + "overlays.subtitles"] == True) | (df[p + "overlays.watermark"] == True),  # noqa: E712
                                                      fired(df, "text_overlay")),
            "fisheye~rule:fisheye": (df[p + "fisheye"] == True, fired(df, "fisheye")),  # noqa: E712
            "dynamic_high~rule:too_dynamic": (df[p + "dynamic_level"] == "high", fired(df, "too_dynamic")),
            "reject~rules_any": (vlm_reject(df, t), ~df.rule_pass),
        }
        for k, (a, b) in pairs.items():
            a, b = a.fillna(False).astype(bool), b.fillna(False).astype(bool)
            vr.append({"set": df_name, "vlm": t, "pair": k, "agree": (a == b).mean(), "vlm_pos": int(a.sum()),
                       "rule_pos": int(b.sum()), "both": int((a & b).sum()), "n": len(a)})
vrdf = pd.DataFrame(vr)
vrdf.to_csv(OUT / "vlm_vs_rules.csv", index=False, float_format="%.3f")

# rule rejection rates on the pool
if len(P):
    summary["pool_n"] = len(P)
    summary["pool_rule_reject_rate"] = {n: float(fired(P, n).mean()) for n in RULES}
    summary["pool_rules_any_reject_rate"] = float((~P.rule_pass).mean())
    for t in TAGS:
        if f"vlm_{t}.reconstructable" in P:
            summary[f"pool_vlm_{t}_dist"] = P[f"vlm_{t}.reconstructable"].fillna("missing").value_counts().to_dict()
# stage runtimes from outputs/logs/filter/<stage>.w*.jsonl (one line per clip, both pool and group C runs)
rt = {}
for st in ("cuts", "rules", "ocr", "flow", "track", "calib"):
    secs = [json.loads(l)["sec"] for f in (H / "outputs/logs/filter").glob(f"{st}.w*.jsonl") for l in f.read_text().splitlines()]
    if secs:
        rt[st] = {"n": len(secs), "sec_per_clip_median": float(np.median(secs)), "sec_per_clip_mean": float(np.mean(secs)),
                  "clips_per_hour_1proc": 3600 / float(np.mean(secs))}
summary["stage_runtime"] = rt
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(prdf.round(3).to_string())
