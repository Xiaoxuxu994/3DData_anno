"""E13 dual-gate QC analysis (CPU).  cache/qc-venv/bin/python e13_analysis.py
Inputs : outputs/qc/qc_A_B_oof.csv (476 pool: gate-2 signals, gate-1 = LODO v0 rule, GT failures), signals_C.csv, qc_C.csv,
         E13/raw/gate2_composites.csv (gate-2 signals of v0 / v13 on A, B, S=ScanNet++, C), E13/raw/gate1_S.csv,
         E1/qc_rule.csv (v0 rule on v0 label), E1/qc_rule_cheap.csv (v1c inputs; v1 = v13 up to scale, all signals scale-free),
         E3b/raw/{A,B,scannetpp}.csv (GT errors of the composites).
Gate 2 = max_i(signal_i / thr_i) > 1 over oriented (higher = worse) self-check signals; thresholds calibrated on the 476
pool with leave-one-dataset-out (LODO), final thresholds = fit on the whole pool, applied unchanged to the composites / C.
Outputs: E13/raw/*.csv, E13/tables.md, E13/retention_dual.png, E13/summary.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

H0 = Path("/data/home/z00533370")
QC, EV = H0 / "outputs/qc", H0 / "outputs/evolution"
E13 = EV / "E13"
RAW = E13 / "raw"
# oriented signal: (column, sign) ; value used = sign * x + (1 if sign < 0 else 0)  -> "1 - share" for share signals
SIG = {"samp5_med": 1, "samp20_med": 1, "iflow5_rel": 1, "iflow1_lt3": -1, "iflow5_lt3": -1, "cons20": -1, "cons5_occ": -1,
       "fsv5": 1, "fsv1": 1}
TYPES = ["pose", "fov", "depth", "geo"]
RULES = {"G2full": ["samp5_med", "iflow5_rel", "iflow1_lt3", "cons20", "fsv5"],
         "G2min": ["samp5_med", "iflow5_rel", "cons20"],          # pose (Sampson) + pose/FOV (induced flow) + depth
         "samp_flow": ["samp5_med", "iflow1_lt3"],                # pose only
         "flow_cons": ["iflow5_rel", "cons20"],
         "fsv_cons": ["fsv1", "cons5_occ"]}                      # occlusion-aware depth / free space
TARGET_FR = {"fr5": 0.05, "fr10": 0.10}  # union false-reject of good labels (one common quantile)
HOWS = ("youden", "q90", "q95", "q98", "fr5", "fr10")
# Pre-declared adoption criterion ("unreasonable loss" defined before looking at composites):
#   dual vs gate 1 must lower the kept geo-failure rate by >= 20 % relative AND >= 1.5 pp, while keeping >= 90 % of the
#   labels gate 1 keeps (retention loss <= 10 % relative) and adding <= 5 pp false rejects of good labels;
#   checked on the pool (LODO) and on the v1.3 composite (A+B+ScanNet++, thresholds from the pool).
MIN_REL_GAIN, MIN_ABS_GAIN, MAX_RET_LOSS, MAX_FR_ADD = 0.20, 0.015, 0.10, 0.05


def orient(df, c):
    x = df[c].to_numpy(float)
    return x if SIG[c] > 0 else 1 - x


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return np.nan
    return float(roc_auc_score(y[m], s[m]))


def youden(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    fpr, tpr, th = roc_curve(y[m], x[m])
    return float(th[np.argmax(tpr - fpr)])


def score(df, cols, thr):
    S = np.stack([orient(df, c) / thr[c] for c in cols], 1)
    with np.errstate(all="ignore"):
        s = np.nanmax(S, 1)
    bad = ~np.isfinite(s)
    if "nonfinite" in df:
        bad |= df["nonfinite"].fillna(0).to_numpy(float) > 0
    return np.where(bad, np.inf, s)  # no evidence / non-finite label -> reject


def fit(df, cols, how):
    y = df["geo_fail"].to_numpy(float)
    if how == "youden":  # per signal, against the failure type it targets
        tmap = {"samp5_med": "pose", "samp20_med": "pose", "iflow5_rel": "pose", "iflow1_lt3": "pose", "iflow5_lt3": "pose",
                "cons20": "depth", "cons5_occ": "depth", "fsv5": "depth", "fsv1": "depth"}
        return {c: youden(df[f"{tmap[c]}_fail"].to_numpy(float), orient(df, c)) for c in cols}
    good = df[y == 0]
    if how.startswith("q"):  # per-signal quantile of good labels
        return {c: float(np.nanquantile(orient(good, c), int(how[1:]) / 100)) for c in cols}
    lo, hi = 0.80, 1.0  # one common quantile q of the good-label distribution, union false-reject = target
    for _ in range(30):
        q = (lo + hi) / 2
        thr = {c: float(np.nanquantile(orient(good, c), q)) for c in cols}
        fr = np.mean(score(good, cols, thr) > 1)
        lo, hi = (q, hi) if fr > TARGET_FR[how] else (lo, q)
    return {c: float(np.nanquantile(orient(good, c), hi)) for c in cols}


def ops(df, s1, s2, tag):
    """Operating-point table for gate1 / gate2 / dual at threshold 1."""
    out = []
    keep = {"gate1": s1 <= 1, "gate2": s2 <= 1, "dual": (s1 <= 1) & (s2 <= 1)}
    for g, k in keep.items():
        r = {"set": tag, "gate": g, "n": len(df), "retention": k.mean()}
        for t in TYPES:
            y = df[f"{t}_fail"].to_numpy(float)
            r[f"{t}_fail_base"] = np.nanmean(y)
            r[f"{t}_fail_kept"] = np.nanmean(y[k]) if k.any() else np.nan
        good = df["geo_fail"].to_numpy(float) == 0
        r["false_reject_good"] = float(np.mean(~k[good])) if good.any() else np.nan
        bad = df["geo_fail"].to_numpy(float) == 1
        r["catch_bad"] = float(np.mean(~k[bad])) if bad.any() else np.nan
        r["n_kept"], r["n_bad_kept"] = int(k.sum()), int((k & bad).sum())
        out.append(r)
    s = np.maximum(s1, s2)
    for g, sc in (("gate1", s1), ("gate2", s2), ("dual", s)):
        next(r for r in out if r["gate"] == g)["auroc_geo"] = auc(df["geo_fail"], np.where(np.isfinite(sc), sc, 1e9))
    return out


def curve(y, s, fr=np.linspace(0.1, 1, 19)):
    ok = np.isfinite(y)
    y, s = y[ok], np.where(np.isfinite(s[ok]), s[ok], 1e9)
    o = np.argsort(s, kind="stable")
    return [(f, float(np.mean(y[o[:max(1, int(round(f * len(o))))]]))) for f in fr]


def gtfail(kind, method):
    x = pd.read_csv(EV / "E3b/raw" / f"{kind}.csv")
    x = x[x["method"] == method].copy()
    x["pose_fail"] = (x["ate_sim3_cm"] / 100 > 0.05 * x["traj_len_m"].clip(lower=1e-6)).astype(float)
    x["fov_fail"] = (x["fov_err_deg"] > 5).astype(float)
    x["depth_fail"] = np.where(x["depth_seq_absrel"].isna(), np.nan, x["depth_seq_absrel"] > 0.15)
    x["geo_fail"] = x[["pose_fail", "fov_fail", "depth_fail"]].max(axis=1)
    return x[["clip", "pose_fail", "fov_fail", "depth_fail", "geo_fail", "ate_sim3_cm", "fov_err_deg", "depth_seq_absrel"]]


def main():
    res = {}
    P = pd.read_csv(QC / "qc_A_B_oof.csv")
    ds = P["dataset"].to_numpy()
    # ---------- 1. single signals on the pool
    rows = []
    for c in SIG:
        rows.append({"signal": c, **{f"{t}_auroc": auc(P[f"{t}_fail"], orient(P, c)) for t in TYPES}})
    single = pd.DataFrame(rows)
    single.to_csv(RAW / "pool_single_signals.csv", index=False, float_format="%.3f")
    # ---------- 2. LODO calibration of rule variants on the pool
    var_rows, oof = [], {}
    for rn, cols in RULES.items():
        for how in HOWS:
            p = np.full(len(P), np.nan)
            for d in np.unique(ds):
                tr = ds != d
                p[ds == d] = score(P[ds == d], cols, fit(P[tr], cols, how))
            oof[(rn, how)] = p
            o = ops(P, P["oof_geo_rule"].to_numpy(float), p, "pool")
            r2, rd, r1 = o[1], o[2], o[0]
            var_rows.append({"rule": rn, "thr": how, "g2_retention": r2["retention"], "g2_geo_kept": r2["geo_fail_kept"],
                             "g2_false_reject": r2["false_reject_good"], "g2_auroc": r2["auroc_geo"],
                             "dual_retention": rd["retention"], "dual_geo_kept": rd["geo_fail_kept"],
                             "dual_false_reject": rd["false_reject_good"], "g1_retention": r1["retention"], "g1_geo_kept": r1["geo_fail_kept"],
                             "g1_false_reject": r1["false_reject_good"]})
    var = pd.DataFrame(var_rows)
    thr_all = {f"{rn}_{how}": fit(P, RULES[rn], how) for rn in RULES for how in HOWS}
    res["thresholds_full_pool"] = thr_all
    # main variant: largest pool (LODO) failure reduction among variants within the retention / false-reject budget;
    # if none qualifies, the one with the smallest dual failure rate among those with retention loss <= 10 %
    var["ret_loss"] = 1 - var.dual_retention / var.g1_retention
    var["fr_add"] = var.dual_false_reject - var.g1_false_reject
    ok = var[(var.ret_loss <= MAX_RET_LOSS) & (var.fr_add <= MAX_FR_ADD)]
    best = (ok if len(ok) else var).sort_values("dual_geo_kept").iloc[0]
    MAIN = (best["rule"], best["thr"])
    rn, how = MAIN
    var.to_csv(RAW / "pool_rule_variants_lodo.csv", index=False, float_format="%.4f")
    THR = thr_all[f"{rn}_{how}"]
    COLS = RULES[rn]
    res["main_rule"] = {"signals": COLS, "thresholds": THR, "calibration": how}
    P["gate1"], P["gate2"] = P["oof_geo_rule"], oof[MAIN]
    P[["group", "method", "clip", "dataset", "gate1", "gate2", "pose_fail", "fov_fail", "depth_fail", "geo_fail"] + COLS].to_csv(
        RAW / "pool_scores.csv", index=False, float_format="%.4f")
    OPS = ops(P, P["gate1"].to_numpy(float), P["gate2"].to_numpy(float), "pool476")
    # per method on the pool
    pm = []
    for m, g in P.groupby("method"):
        for r in ops(g, g["gate1"].to_numpy(float), g["gate2"].to_numpy(float), m):
            pm.append(r)
    pd.DataFrame(pm).to_csv(RAW / "pool_by_method.csv", index=False, float_format="%.4f")
    # ---------- 3. composites
    G2 = pd.read_csv(RAW / "gate2_composites.csv")
    G2["gate2"] = score(G2, COLS, THR)
    q0 = pd.read_csv(EV / "E1/qc_rule.csv")
    q0 = q0[q0.label == "v0"].assign(label="v0")
    q0["gate1"] = np.nanmax(np.stack([q0.xall_traj / 3.5, q0.xba_fov / 4.1, q0.xff_depth / 0.157], 1), 1)
    q1 = pd.read_csv(EV / "E1/qc_rule_cheap.csv").assign(label="v13")
    q1["gate1"] = np.nanmax(np.stack([q1.xtraj_ba / 3.27, q1.xdepth_k / 0.194, q1.kdis / 13.2], 1), 1)
    gS = pd.read_csv(RAW / "gate1_S.csv")
    g1 = pd.concat([q0[["group", "clip", "label", "gate1"]], q1[["group", "clip", "label", "gate1"]], gS[["group", "clip", "label", "gate1"]]])
    CP = G2.rename(columns={"method": "label"}).merge(g1, on=["group", "clip", "label"], how="left")
    gts = []
    for G, kind in (("A", "A"), ("B", "B"), ("S", "scannetpp")):
        for lab, meth in (("v0", "H:vipe+pi3x+med"), ("v13", "pi3x_kgate@vipe_td")):
            gts.append(gtfail(kind, meth).assign(group=G, label=lab))
    CP = CP.merge(pd.concat(gts), on=["group", "clip", "label"], how="left")
    CP.to_csv(RAW / "composites_scores.csv", index=False, float_format="%.4f")
    for lab in ("v0", "v13"):
        for sets, name in ((("A", "B"), "AB"), (("S",), "S"), (("A", "B", "S"), "ABS")):
            d = CP[(CP.label == lab) & CP.group.isin(sets)]
            OPS += ops(d, d["gate1"].to_numpy(float), d["gate2"].to_numpy(float), f"{lab}_{name}")
    # every variant on the composites (thresholds from the full pool), for transparency
    cv_rows = []
    for key, thr in thr_all.items():
        r_, h_ = key.rsplit("_", 1)
        for lab in ("v0", "v13"):
            d = CP[(CP.label == lab) & CP.group.isin(("A", "B", "S"))]
            o = ops(d, d["gate1"].to_numpy(float), score(d, RULES[r_], thr), lab)
            cv_rows.append({"rule": r_, "thr": h_, "label": lab, "g1_ret": o[0]["retention"], "g1_fail": o[0]["geo_fail_kept"],
                            "dual_ret": o[2]["retention"], "dual_fail": o[2]["geo_fail_kept"], "g2_ret": o[1]["retention"],
                            "g2_fail": o[1]["geo_fail_kept"], "dual_fr": o[2]["false_reject_good"], "g1_fr": o[0]["false_reject_good"]})
    pd.DataFrame(cv_rows).to_csv(RAW / "composite_variants.csv", index=False, float_format="%.4f")
    ot = pd.DataFrame(OPS)

    def verdict(tag):
        g1, du = (ot[(ot.set == tag) & (ot.gate == g)].iloc[0] for g in ("gate1", "dual"))
        gain = g1.geo_fail_kept - du.geo_fail_kept
        rel = gain / g1.geo_fail_kept if g1.geo_fail_kept > 0 else 0.0
        loss = 1 - du.retention / g1.retention
        fr = du.false_reject_good - g1.false_reject_good
        return {"fail_gate1": g1.geo_fail_kept, "fail_dual": du.geo_fail_kept, "abs_gain": gain, "rel_gain": rel,
                "retention_gate1": g1.retention, "retention_dual": du.retention, "retention_loss_rel": loss, "fr_added": fr,
                "passes": bool(rel >= MIN_REL_GAIN and gain >= MIN_ABS_GAIN and loss <= MAX_RET_LOSS and fr <= MAX_FR_ADD)}
    res["decision"] = {t: verdict(t) for t in ("pool476", "v13_ABS", "v0_ABS")}
    res["decision"]["adopt"] = bool(res["decision"]["pool476"]["passes"] and res["decision"]["v13_ABS"]["passes"])
    res["criterion"] = {"min_rel_gain": MIN_REL_GAIN, "min_abs_gain_pp": MIN_ABS_GAIN * 100, "max_retention_loss_rel": MAX_RET_LOSS,
                        "max_false_reject_added_pp": MAX_FR_ADD * 100}
    ot.to_csv(RAW / "operating_points.csv", index=False, float_format="%.4f")
    # ---------- 4. retention curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    col = {"gate1": "#2a78d6", "gate2": "#eb6834", "dual": "#1baf7a"}
    for ax, (tag, d) in zip(axes, (("476 single-method labels (A+B)", P), ("v0 composite (A+B+ScanNet++)", CP[(CP.label == "v0") & (CP.group != "C")]),
                                   ("v1.3 composite (A+B+ScanNet++), gate 1 = v1c", CP[(CP.label == "v13") & (CP.group != "C")]))):
        y = d["geo_fail"].to_numpy(float)
        s1, s2 = d["gate1"].to_numpy(float), d["gate2"].to_numpy(float)
        for g, s in (("gate1", s1), ("gate2", s2), ("dual", np.maximum(s1, s2))):
            cv = curve(y, s)
            ax.plot([a * 100 for a, _ in cv], [b * 100 for _, b in cv], color=col[g], lw=2, label=f"{g} (sweep)")
            k = s <= 1
            ax.plot(k.mean() * 100, np.nanmean(y[k]) * 100, "o", ms=9, color=col[g], mec="white", mew=2)
            ax.annotate(f"{g} @1", (k.mean() * 100, np.nanmean(y[k]) * 100), textcoords="offset points", xytext={"gate1": (8, 10), "gate2": (6, -12), "dual": (8, -16)}[g], fontsize=8, color="#333")
        ax.axhline(np.nanmean(y) * 100, color="#999", lw=0.8, ls=":")
        ax.set_title(f"{tag}, n={len(d)}, base {np.nanmean(y) * 100:.0f}%", fontsize=10)
        ax.set_xlabel("labels kept (%), lowest score first")
        ax.set_ylabel("geo failure among kept (%)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"E13 dual-gate QC: gate 1 = cross-model rule, gate 2 = self-check max({', '.join(COLS)}) / thr ({how}); dots = operating point (score <= 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(E13 / "retention_dual.png", dpi=130)
    # ---------- 5. C group
    C = pd.read_csv(QC / "qc_C.csv")
    C["gate1"], C["gate2"] = C["rule_geo"], score(C, COLS, THR)
    CC = CP[CP.group == "C"].copy()
    import json as _j
    meta = {c["clip"]: c for c in _j.loads((H0 / "outputs/bench/wild/clips.json").read_text())}
    cflags = C.drop_duplicates("clip").set_index("clip")["clip_flags"].fillna("")
    for d in (C, CC):
        d["cat"] = d["clip"].map(lambda c: meta[c]["cat"])
        d["hard"] = d["clip"].map(lambda c: bool(meta[c].get("hard_case")))
        d["notes"] = d["clip"].map(lambda c: meta[c].get("notes", ""))
        d["clip_flags"] = d["clip"].map(cflags)
        d["rej1"], d["rej2"] = d["gate1"] > 1, d["gate2"] > 1
        d["rej_dual"] = d["rej1"] | d["rej2"]
    crow = []
    for tag, d in (("pool7_C", C), ("v0_C", CC[CC.label == "v0"]), ("v13_C", CC[CC.label == "v13"])):
        for cat, g in list(d.groupby("cat")) + [("ALL", d)]:
            crow.append({"set": tag, "cat": cat, "n": len(g), "rej_gate1": g.rej1.mean(), "rej_gate2": g.rej2.mean(), "rej_dual": g.rej_dual.mean(),
                         "gate2_only_rejects": (g.rej2 & ~g.rej1).mean()})
    ct = pd.DataFrame(crow)
    ct.to_csv(RAW / "C_reject_by_cat.csv", index=False, float_format="%.3f")
    keepc = ["label", "clip", "cat", "hard", "gate1", "gate2", "rej1", "rej2", "clip_flags", "notes"] + COLS
    CC[keepc].sort_values(["label", "hard", "cat"]).to_csv(RAW / "C_composites_decisions.csv", index=False, float_format="%.3f")
    C[["method", "clip", "cat", "hard", "gate1", "gate2", "rej1", "rej2", "clip_flags"] + COLS + ["hfov"]].to_csv(RAW / "C_pool_decisions.csv", index=False, float_format="%.3f")
    res["C"] = {}
    for tag, d in (("pool7", C), ("v0", CC[CC.label == "v0"]), ("v13", CC[CC.label == "v13"])):
        res["C"][tag] = {"hard_vs_normal_rej_gate1": d.groupby("hard").rej1.mean().round(3).to_dict(),
                         "hard_vs_normal_rej_gate2": d.groupby("hard").rej2.mean().round(3).to_dict(),
                         "hard_vs_normal_rej_dual": d.groupby("hard").rej_dual.mean().round(3).to_dict(),
                         "hard_auroc_gate1": auc(d.hard, d.gate1.replace(np.inf, 1e9)), "hard_auroc_gate2": auc(d.hard, d.gate2.replace(np.inf, 1e9)),
                         "hard_caught_by_gate2_missed_by_gate1": sorted(d[d.hard & d.rej2 & ~d.rej1]["clip"].unique().tolist()),
                         "hard_missed_by_both": sorted(d[d.hard & ~d.rej_dual]["clip"].unique().tolist()),
                         "flagged_normal_caught_by_gate2_only": sorted(d[~d.hard & (d.clip_flags != "") & d.rej2 & ~d.rej1]["clip"].unique().tolist())}
    # BTbkSEotmX0
    bt = "driving__youtube_cc__BTbkSEotmX0"
    b1 = C[C["clip"] == bt][["method", "hfov", "gate1", "gate2"] + COLS + ["iflow1_lt3", "fsv5", "samp20_med"]]
    b2 = CC[CC["clip"] == bt][["label", "gate1", "gate2"] + COLS + ["iflow1_lt3", "fsv5", "samp20_med", "hfov"]].rename(columns={"label": "method"})
    btab = pd.concat([b1, b2])
    btab.to_csv(RAW / "BTbkSEotmX0_gate2.csv", index=False, float_format="%.4f")
    # ---------- 6. cost
    cost = {"gate2_cpu_s_per_label_med": float(G2["cpu_s"].median()), "gate2_wall_s_per_label_med": float(G2["wall_s"].median()),
            "gate2_cpu_s_by_group": G2.groupby("group")["cpu_s"].median().round(2).to_dict()}
    res["cost"] = cost
    res["pool_variants"] = var.round(4).to_dict(orient="records")
    res["operating_points"] = ot.round(4).to_dict(orient="records")
    res["single_signals"] = single.round(3).to_dict(orient="records")
    (E13 / "summary.json").write_text(json.dumps(res, indent=1, default=lambda o: None if o != o else str(o)))
    pd.set_option("display.width", 250)
    print(single.round(3).to_string())
    print(var.round(3).to_string())
    print(ot[["set", "gate", "n", "retention", "geo_fail_base", "geo_fail_kept", "pose_fail_kept", "fov_fail_kept", "depth_fail_kept", "false_reject_good", "catch_bad", "auroc_geo"]].round(3).to_string())
    print(ct.round(2).to_string())
    print(json.dumps(res["C"], indent=1, default=str))
    print(btab.round(3).to_string())
    print(cost)
    print(MAIN, json.dumps(THR))
    print(json.dumps(res["decision"], indent=1, default=str))


if __name__ == "__main__":
    main()
