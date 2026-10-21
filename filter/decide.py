"""Combine stage scores (+ VLM verdicts) into pass / reject decisions.

Usage: python decide.py POOL_DIR OUT_CSV [--vlm TAG]
  Flattens every scores.json into one row (stage.key columns), applies RULES (each rule = one reject reason) and,
  if --vlm is given, the VLM's "reconstructable" verdict. Final = pass iff no rule fires and the VLM says "yes".
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Reject rules on expert scores. Thresholds: SpatialVID (luminance 20-140, OCR <= 0.3, aesthetic >= 4),
# DynPose-100K (enough camera motion; cut / jump detection), the rest calibrated on group C (see README).
RULES = {
    "cut":            lambda r: r["cuts.tn_max"] > 0.5,
    "dark":           lambda r: r["rules.lum_mean"] < 20,
    "overexposed":    lambda r: r["rules.lum_mean"] > 200,
    "blur":           lambda r: r["rules.lap_var_median"] < 25,
    "text_overlay":   lambda r: r["ocr.text_area_max"] > 0.3 or r["ocr.static_text_area"] > 0.02,
    "static_camera":  lambda r: r["flow.static_pair_frac"] > 0.5,
    "zoom":           lambda r: (r["calib.focal_ratio_end"] > 1.4 or r["calib.focal_ratio_end"] < 0.6)
                                and abs(r["calib.focal_trend"]) > 0.8,
    "too_dynamic":    lambda r: r["flow.dyn_frac"] > 0.6,
    "fisheye":        lambda r: abs(r["calib.k1_median"]) > 0.3,
}
# Soft flags: recorded in decisions.csv but not used to reject (too many false positives on group C, see README).
SOFT = {
    "no_parallax":    lambda r: r["flow.static_pair_frac"] <= 0.5 and r["flow.rot_frac"] > 0.6,  # fires on planar highway driving
    "cut_track":      lambda r: r["cuts.ff_max"] > 0.5 and r["track.vis_drop_max"] > 0.8,      # fires on fast occluders
    "low_aesthetic":  lambda r: r["rules.aesthetic"] < 4.0,                                        # SpatialVID's threshold; DROID < 4
}
VLM_REJECT = {"no_unreconstructable", "no_inaccurate"}


def vlm_field_reject(r, tag):
    """Reject derived from the VLM's per-field answers instead of its own "reconstructable" verdict."""
    p = f"vlm_{tag}."
    if not isinstance(r.get(p + "camera_motion"), str):
        return False
    t = lambda k: r.get(p + k) in (True, "true", "True")  # noqa: E731
    return (r[p + "camera_motion"] in ("static", "zoom", "mostly_rotation") or t("has_cuts") or t("is_cg_or_game")
            or t("low_light_or_blur") or t("fisheye") or t("overlays.picture_in_picture"))


def flatten(s, prefix=""):
    out = {}
    for k, v in s.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, list):
            if prefix.startswith("vlm") or len(v) <= 8:
                out[f"{prefix}{k}"] = json.dumps(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def load(pool):
    pool = Path(pool)
    metas = (json.loads((pool / "pool.json").read_text()) if (pool / "pool.json").exists()
             else [json.loads(m.read_text()) for m in sorted(pool.glob("*/meta.json"))])
    rows = []
    for m in metas:
        d = pool / m["clip"]
        s = json.loads((d / "scores.json").read_text()) if (d / "scores.json").exists() else {}
        r = {"clip": m["clip"], "cat_hint": m.get("cat_hint"), "url": m.get("url") or m.get("url_or_ref"),
             "channel": m.get("channel"), "license": m.get("license"), "hard_case": m.get("hard_case")}
        r.update(flatten(s))
        rows.append(r)
    return pd.DataFrame(rows)


def apply_rules(df, rules=RULES):
    reasons = []
    for _, r in df.iterrows():
        fired = []
        for name, f in rules.items():
            try:
                if f(r):
                    fired.append(name)
            except (KeyError, TypeError):
                pass  # stage missing for this clip -> rule cannot fire
        reasons.append(fired)
    df = df.copy()
    df["rule_reasons"] = [";".join(x) for x in reasons]
    soft = []
    for _, r in df.iterrows():
        f = []
        for name, fn in SOFT.items():
            try:
                if fn(r):
                    f.append(name)
            except (KeyError, TypeError):
                pass
        soft.append(";".join(f))
    df["soft_flags"] = soft
    df["rule_pass"] = [len(x) == 0 for x in reasons]
    return df


def apply_vlm(df, tag):
    col = f"vlm_{tag}.reconstructable"
    df = df.copy()
    df["vlm_pass"] = df[col].map(lambda v: v == "yes") if col in df else False
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("out_csv")
    ap.add_argument("--vlm", default=None)
    a = ap.parse_args()
    df = apply_rules(load(a.pool))
    if a.vlm:
        df = apply_vlm(df, a.vlm)
        df["decision"] = ["pass" if rp and vp else "reject" for rp, vp in zip(df.rule_pass, df.vlm_pass)]
        vr = df.get(f"vlm_{a.vlm}.reason", pd.Series([""] * len(df)))
        df["reason"] = [";".join(x for x in [rr, "" if vp else f"vlm:{df.at[i, f'vlm_{a.vlm}.reconstructable']}"] if x)
                        for i, rr, vp in zip(df.index, df.rule_reasons, df.vlm_pass)]
    else:
        df["decision"] = ["pass" if p else "reject" for p in df.rule_pass]
        df["reason"] = df.rule_reasons
    front = ["clip", "decision", "reason", "soft_flags", "cat_hint", "url", "channel", "license"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(a.out_csv, index=False, float_format="%.5g")
    print(df.decision.value_counts().to_dict(), "->", a.out_csv)
