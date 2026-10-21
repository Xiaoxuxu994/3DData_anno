"""Calibrate the no-GT QC signals against GT label errors on A (rank) + B (diagnostic), apply to C (wild).

  python qc_calib.py  -> outputs/qc/{calib_signals.csv, calib_by_method.csv, calib_models.json, retention_AB.png,
                                     qc_A_B_oof.csv, qc_C.csv, qc_C_by_clip.csv, qc_models.pkl}
Failure labels per annotator x clip (GT errors from eval3d_bench.py per_clip.csv):
  pose_fail   = ATE_Sim3 > 5 % of trajectory length (scale-free trajectory shape)
  fov_fail    = HFOV error > 5 deg
  depth_fail  = sequence-aligned (one scale per clip) depth AbsRel > 0.15
  metric_fail = metric method on metric GT: ATE_SE3 > 5 % length or |Sim3 scale - 1| > 10 % or metric AbsRel > 0.2
  any_fail    = union of the above (= the agreed definition)
Continuous targets for Spearman: pose ATE_Sim3 / length (%), fov error, seq AbsRel, |log scale|.
Signals are oriented so that higher = worse. Combined models (logistic regression, shallow GBDT) are trained with
leave-one-dataset-out (LODO); C is scored with the average of the LODO fold models (same distribution as the OOF scores).
"""
import json
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from qc_common import BENCH, METHODS, METRIC, QC, clips  # noqa: E402

warnings.filterwarnings("ignore")
GOOD = ("cons", "_lt1", "_lt3", "valid_frac", "match_cnt", "fb_ok_frac")  # higher = better -> negated
DROP = {"group", "method", "clip", "dataset", "n_frames", "error", "nonfinite", "rev_error", "sub_error", "cat",
        "hard_case", "depth_med", "notes"}
LABELS = {"pose": "pose_err", "fov": "fov_err_deg", "depth": "depth_err", "geo": "depth_err", "metric": "scale_err", "any": "pose_err"}
RULE = {"pose": "xall_traj", "fov": "xba_fov", "depth": "xff_depth"}
COMPACT = ["xall_traj", "xff_traj", "xba_traj", "xba_fov", "xall_fov", "fov_spread", "xff_depth", "xall_depth", "cons5", "fsv5",
           "iflow5_rel", "samp20_excess", "xmetric_scale", "static_cam", "focal_cv"]


def load(G):
    s = pd.read_csv(QC / f"signals_{G}.csv")
    df = s.merge(pd.read_csv(QC / f"cross_{G}.csv"), on=["group", "method", "clip"], how="left")
    st = pd.read_csv(QC / f"stab_{G}.csv")
    st = st.drop(columns=[c for c in st.columns if c.endswith("error")])
    return df.merge(st, on=["group", "method", "clip"], how="left")


def gt(G):
    g = pd.read_csv(BENCH[G] / "eval" / "per_clip.csv")
    g = g[g["method"].isin(METHODS)].copy()
    meta = {c["clip"]: c for c in clips(G)}
    L = g["traj_len_m"].clip(lower=1e-6)
    mm = g["method"].isin(METRIC) & g["clip"].map(lambda c: bool(meta[c].get("metric")))
    g["pose_err"] = g["ate_sim3_cm"] / L
    g["depth_err"] = g["depth_seq_absrel"]
    g["scale_err"] = np.where(mm, np.abs(np.log(g["sim3_scale"])), np.nan)
    g["pose_fail"] = (g["ate_sim3_cm"] / 100 > 0.05 * L).astype(float)
    g["fov_fail"] = (g["fov_err_deg"] > 5).astype(float)
    g["depth_fail"] = np.where(g["depth_seq_absrel"].isna(), np.nan, g["depth_seq_absrel"] > 0.15)
    mf = (g["ate_se3_cm"] / 100 > 0.05 * L) | ((g["sim3_scale"] - 1).abs() > 0.10) | (g["depth_metric_absrel"] > 0.2)
    g["metric_fail"] = np.where(mm, mf, np.nan)
    g["geo_fail"] = g[["pose_fail", "fov_fail", "depth_fail"]].max(axis=1)
    g["any_fail"] = g[["pose_fail", "fov_fail", "depth_fail", "metric_fail"]].max(axis=1)
    return g[["method", "clip", "pose_err", "fov_err_deg", "depth_err", "scale_err"] + [f"{t}_fail" for t in LABELS]]


def orient(df, cols):
    out = df[cols].copy()
    for c in cols:
        if any(k in c for k in GOOD):
            out[c] = -out[c]
    return out


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2 or min(np.bincount(y[m].astype(int))) < 3:
        return np.nan
    return float(roc_auc_score(y[m], s[m]))


def spear(e, s):
    e, s = np.asarray(e, float), np.asarray(s, float)
    m = np.isfinite(e) & np.isfinite(s)
    return float(spearmanr(e[m], s[m]).statistic) if m.sum() >= 10 else np.nan


class _Cols:
    """Column selector usable inside a sklearn Pipeline."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[c for c in self.cols if c in X.columns]]

    def get_params(self, deep=True):
        return {"cols": self.cols}


def make(name):
    if name == "compact":
        return make_pipeline(_Cols(COMPACT), SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced"))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=12,
                                          l2_regularization=1.0, class_weight="balanced")


def design(df, feats):
    X = orient(df, feats)
    for m in METHODS:
        X[f"is_{m}"] = (df["method"] == m).astype(float)
    return X


def retention(y, err, score, fr=np.round(np.linspace(0.1, 1, 19), 2)):
    ok = np.isfinite(y)
    y, err, score = y[ok], err[ok], np.where(np.isfinite(score[ok]), score[ok], np.inf)
    o = np.argsort(score, kind="stable")
    return pd.DataFrame([{"keep": f, "fail_rate": float(np.mean(y[o[:max(1, int(round(f * len(o))))]])),
                          "err_med": float(np.nanmedian(err[o[:max(1, int(round(f * len(o))))]]))} for f in fr])


def main():
    AB = pd.concat([load("A"), load("B")], ignore_index=True)
    G = pd.concat([gt("A"), gt("B")], ignore_index=True)
    AB = AB.merge(G, on=["method", "clip"], how="inner")
    feats = [c for c in AB.columns if c not in DROP and c not in G.columns and AB[c].dtype != object]
    stab = [c for c in feats if c.startswith(("rev_", "sub_"))]
    base = [c for c in feats if c not in stab]
    O = orient(AB, feats)
    # ---- single-signal calibration
    rows, rows_m = [], []
    for c in feats:
        r = {"signal": c, "n": int(O[c].notna().sum())}
        for t, e in LABELS.items():
            r[f"{t}_auroc"] = auc(AB[f"{t}_fail"], O[c])
            if t != "any":
                r[f"{t}_spearman"] = spear(AB[e], O[c])
        rows.append(r)
        for m, g in AB.groupby("method"):
            rows_m.append({"signal": c, "method": m, **{f"{t}_auroc": auc(g[f"{t}_fail"], O.loc[g.index, c]) for t in LABELS}})
    cs = pd.DataFrame(rows)
    cs.to_csv(QC / "calib_signals.csv", index=False, float_format="%.3f")
    pd.DataFrame(rows_m).to_csv(QC / "calib_by_method.csv", index=False, float_format="%.3f")
    # ---- combined models, leave-one-dataset-out
    res = {"n_rows": len(AB), "datasets": sorted(AB["dataset"].unique()), "features": base,
           "fail_rate": {t: float(AB[f"{t}_fail"].mean()) for t in LABELS},
           "fail_rate_by_method": AB.groupby("method")[[f"{t}_fail" for t in LABELS]].mean().round(3).to_dict(orient="index")}
    X = design(AB, base)
    ds = AB["dataset"].to_numpy()
    folds = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    for ax, (t, e) in zip(axes, LABELS.items()):
        y = AB[f"{t}_fail"].to_numpy(float)
        ok = np.isfinite(y)
        res[t] = {"n": int(ok.sum()), "fail_rate": float(np.nanmean(y))}
        oof = {}
        for name in ("compact", "logreg", "gbdt"):
            p = np.full(len(y), np.nan)
            fm = []
            for d in np.unique(ds[ok]):
                tr, te = ok & (ds != d), ok & (ds == d)
                if len(np.unique(y[tr])) < 2:
                    continue
                mdl = make(name).fit(X[tr], y[tr])
                p[te] = mdl.predict_proba(X[te])[:, 1]
                fm.append(mdl)
            oof[name] = p
            folds[(t, name)] = fm
            AB[f"oof_{t}_{name}"] = p
            res[t][name] = {"auroc_oof": auc(y, p), "by_method": {m: auc(y[AB.method == m], p[AB.method == m]) for m in METHODS}}
        best = cs.sort_values(f"{t}_auroc", ascending=False)
        best = best[best["n"] > 300].iloc[0]["signal"]  # exclude the 2-method stability signals
        res[t]["best_single"] = {"signal": best, "auroc": auc(y, O[best])}
        res[t]["chosen"] = max(("compact", "logreg", "gbdt"), key=lambda n: res[t][n]["auroc_oof"] or 0)
        err = AB[e].to_numpy(float)
        for lab, s, kw in (("best single: " + best, O[best].to_numpy(float), {}), ("logreg (LODO)", oof["logreg"], {"lw": 1.8}),
                           ("gbdt (LODO)", oof["gbdt"], {"lw": 1.8}), ("compact (LODO)", oof["compact"], {"lw": 1.8}), ("oracle", y + 1e-3 * np.nan_to_num(err), {"ls": "--"})):
            rc = retention(y, err, s)
            ax.plot(rc["keep"] * 100, rc["fail_rate"] * 100, label=lab, **kw)
            if "LODO" in lab:
                res[t][lab.split()[0]]["retention"] = rc.round(4).to_dict(orient="records")
        ax.axhline(np.nanmean(y) * 100, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("labels kept (%), lowest predicted risk first")
        ax.set_ylabel("failure rate among kept (%)")
        ax.set_title(f"{t}_fail (A+B, n={ok.sum()}, base {np.nanmean(y)*100:.0f}%)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(QC / "retention_AB.png", dpi=120)
    plt.close(fig)
    # ---- transparent rule score: max over (best signal / its Youden threshold) for pose, fov, depth
    def youden(y, x):
        m = np.isfinite(y) & np.isfinite(x)
        fpr, tpr, th = roc_curve(y[m], x[m])
        return th[np.argmax(tpr - fpr)]

    def rule_score(df, thr):
        O_ = orient(df, list(RULE.values()))
        return np.nanmax(np.stack([O_[sig].to_numpy(float) / thr[t] for t, sig in RULE.items()]), 0)
    p = np.full(len(AB), np.nan)
    for d in np.unique(ds):
        tr = ds != d
        thr_f = {t: youden(AB.loc[tr, f"{t}_fail"].to_numpy(float), O.loc[tr, sig].to_numpy(float)) for t, sig in RULE.items()}
        p[ds == d] = rule_score(AB[ds == d], thr_f)
    AB["oof_geo_rule"] = p
    rule_thr = {t: float(youden(AB[f"{t}_fail"].to_numpy(float), O[sig].to_numpy(float))) for t, sig in RULE.items()}
    res["rule"] = {"signals": RULE, "thresholds_raw": rule_thr, "geo_auroc_oof": auc(AB["geo_fail"], p),
                   "any_auroc_oof": auc(AB["any_fail"], p),
                   "geo_retention": retention(AB["geo_fail"].to_numpy(float), AB["depth_err"].to_numpy(float), p).round(4).to_dict(orient="records"),
                   "reject_rate_at_1": float(np.nanmean(p > 1)),
                   "geo_fail_kept_at_1": float(AB.loc[p <= 1, "geo_fail"].mean()), "geo_fail_rejected_at_1": float(AB.loc[p > 1, "geo_fail"].mean())}
    # thresholds on the any_fail risk
    ch = res["geo"]["chosen"]
    AB["risk"] = AB[f"oof_geo_{ch}"]  # QC score = geometric failure risk; metric scale is scored separately (p_metric)
    thr = {f"keep{k}": float(np.nanquantile(AB["risk"], k / 100)) for k in (50, 60, 80)}
    res["thresholds"] = thr
    for k, v in thr.items():
        kept = AB["risk"] <= v
        res[f"at_{k}"] = {**{f"{t}_fail_kept": float(AB.loc[kept, f"{t}_fail"].mean()) for t in LABELS},
                          **{f"{t}_fail_rejected": float(AB.loc[~kept, f"{t}_fail"].mean()) for t in LABELS},
                          "kept_by_method": AB[kept].groupby("method").size().reindex(METHODS).fillna(0).astype(int).to_dict()}
    AB.to_csv(QC / "qc_A_B_oof.csv", index=False, float_format="%.4f")
    res["stability"] = {m: {c: {t: auc(AB.loc[AB.method == m, f"{t}_fail"], AB.loc[AB.method == m, c]) for t in LABELS}
                            for c in stab} for m in ("vipe", "vggt_omega")}
    # ---- apply to C: average of LODO fold models
    C = load("C")
    cm = {c["clip"]: c for c in clips("C")}
    C["cat"] = C["clip"].map(lambda c: cm[c]["cat"])
    C["hard_case"] = C["clip"].map(lambda c: bool(cm[c].get("hard_case")))
    C["notes"] = C["clip"].map(lambda c: cm[c].get("notes", ""))
    XC = design(C, base)
    for t in LABELS:
        C[f"p_{t}"] = np.mean([m.predict_proba(XC)[:, 1] for m in folds[(t, res[t]["chosen"])]], 0)
    C["risk"] = C["p_geo"]
    C["rule_geo"] = rule_score(C, rule_thr)
    C["rule_reject"] = C["rule_geo"] > 1
    for k, v in thr.items():
        C[k] = C["risk"] <= v
    # explanations: largest positive standardized contributions in the logistic model trained on all of A+B
    lr = make("logreg").fit(X[AB["geo_fail"].notna()], AB.loc[AB["geo_fail"].notna(), "geo_fail"])
    Z = lr[:-1].transform(XC) * lr[-1].coef_[0]
    names = list(lr[0].get_feature_names_out(XC.columns))
    C["reasons"] = [";".join(names[j] for j in np.argsort(-row)[:3] if row[j] > 0.3) for row in Z]
    coef = pd.Series(lr[-1].coef_[0], index=names).sort_values()
    res["logreg_any_coef_top"] = {"risk_up": coef.tail(10).round(3).to_dict(), "risk_down": coef.head(10).round(3).to_dict()}

    def flags(r):
        f = []
        if r["baseline_ratio"] < 0.05:
            f.append("focal_unobservable(static camera)")
        if r["fov_out"] > 0:
            f.append("fov_implausible")
        # thresholds: ~95-99th percentile of A/B labels without cuts (ETH3D excluded); > 20 deg in one frame = cut / jump
        if r["angvel_max"] > 20 or (r["speed_spike"] > 30 and r["baseline_ratio"] > 0.2):
            f.append("trajectory_jump")
        if r["focal_cv"] > 0.07:
            f.append("focal_varies")
        if r["samp20_bad"] > 0.5:
            f.append("epipolar_violation")
        return f
    fl = C.apply(flags, axis=1)
    # clip-level evidence shared by all annotators: a majority (>= 4 of 7) sees a jump (cut?) / varying focal (zoom?) /
    # a static camera
    agg = pd.DataFrame({"clip": C["clip"], "jump": fl.map(lambda f: "trajectory_jump" in f),
                        "zoom": fl.map(lambda f: "focal_varies" in f), "static": fl.map(lambda f: any("static" in x for x in f))})
    agg = agg.groupby("clip").agg(jump=("jump", "sum"), zoom=("zoom", "sum"), static=("static", "mean"))
    clipf = {c: [n for n, ok in (("cut_suspected", r.jump >= 4), ("zoom_suspected", r.zoom >= 4), ("static_camera", r.static > 0.5)) if ok]
             for c, r in agg.iterrows()}
    C["flags"] = fl.map(";".join)
    C["clip_flags"] = C["clip"].map(lambda c: ";".join(clipf[c]))
    C["flag_any"] = (C["flags"] != "") | (C["clip_flags"] != "")
    C["reject80"] = (~C["keep80"]) | (C["clip_flags"] != "")
    C.to_csv(QC / "qc_C.csv", index=False, float_format="%.4f")
    by = C.pivot_table(index="clip", columns="method", values="risk").round(3)
    by["min_risk"], by["best_method"] = by.min(1), by.idxmin(1)
    by["cat"] = by.index.map(lambda c: cm[c]["cat"])
    by["hard"] = by.index.map(lambda c: bool(cm[c].get("hard_case")))
    by["notes"] = by.index.map(lambda c: cm[c].get("notes", ""))
    by.sort_values("min_risk", ascending=False).to_csv(QC / "qc_C_by_clip.csv")
    res["C"] = {
        "keep80_rate_by_method": C.groupby("method")["keep80"].mean().round(3).to_dict(),
        "keep50_rate_by_method": C.groupby("method")["keep50"].mean().round(3).to_dict(),
        "risk_median_by_cat": C.pivot_table(index="cat", columns="method", values="risk", aggfunc="median").round(3).to_dict(),
        "hard_vs_normal_risk_median": C.groupby("hard_case")["risk"].median().round(3).to_dict(),
        "hard_auroc_by_method": {m: auc(g["hard_case"], g["risk"]) for m, g in C.groupby("method")},
        "hard_auroc_all": auc(C["hard_case"], C["risk"]),
        "rule_hard_auroc": auc(C["hard_case"], C["rule_geo"]),
        "rule_reject_rate_by_method": C.groupby("method")["rule_reject"].mean().round(3).to_dict(),
        "rule_reject_rate_by_cat": C.groupby("cat")["rule_reject"].mean().round(3).to_dict(),
        "rule_reject_rate_hard_vs_normal": C.groupby("hard_case")["rule_reject"].mean().round(3).to_dict(),
        "hard_auroc_by_label": {t: auc(C["hard_case"], C[f"p_{t}"]) for t in LABELS},
        "hard_auroc_flag_any": auc(C["hard_case"], C["flag_any"]),
        "flag_rate_hard_vs_normal": C.groupby("hard_case")["flag_any"].mean().round(3).to_dict(),
        "clip_flag_rate_hard_vs_normal": C.drop_duplicates("clip").assign(f=lambda d: d.clip_flags != "").groupby("hard_case")["f"].mean().round(3).to_dict(),
        "reject80_rate_hard_vs_normal": C.groupby("hard_case")["reject80"].mean().round(3).to_dict(),
        "reject80_rate_by_cat": C.groupby("cat")["reject80"].mean().round(3).to_dict(),
        "hard_auroc_min_risk_per_clip": auc(by["hard"], by["min_risk"]),
    }
    with open(QC / "qc_models.pkl", "wb") as f:
        pickle.dump({"features": base, "methods": METHODS, "fold_models": folds, "chosen": {t: res[t]["chosen"] for t in LABELS},
                     "thresholds": thr, "explainer_logreg": lr}, f)
    (QC / "calib_models.json").write_text(json.dumps(res, indent=1, default=lambda o: None if o != o else str(o)))
    for t in LABELS:
        print(t, {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("retention", "by_method")}) for k, v in res[t].items()})
    print(json.dumps({k: res[k] for k in ("thresholds", "at_keep50", "at_keep80")}, default=str))
    print(json.dumps({k: v for k, v in res["rule"].items() if k != "geo_retention"}, default=str))
    print(json.dumps(res["C"], default=str))


if __name__ == "__main__":
    main()
