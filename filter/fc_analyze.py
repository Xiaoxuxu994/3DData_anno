"""End-to-end check: does the filter predict annotation failure?

Usage: FC_BENCH=outputs/bench/filter_check python fc_analyze.py decisions.csv OUT_DIR
No-GT QC per clip (fc_eval.py = yt_eval.py): ViPE multi-view self-consistency / photometric warp error (gap 5, 20),
HFOV, plus pose agreement ViPE ~ VGGT-Omega (Sim3 ATE as % of trajectory length).
Annotation failure := ViPE crashed, or g5_selfcons < 0.5, or ATE(ViPE~VGGT-Omega) > 10 % of trajectory length,
or HFOV outside [20, 130] deg. Reports failure rate for pass vs reject, per reject reason, and the AUROC of each score.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fc_eval as E  # noqa: E402

dec = pd.read_csv(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
clips = json.loads((E.B / "clips.json").read_text())
st = {json.loads(l)["clip"]: json.loads(l) for l in (E.B / "vipe/logs/status.jsonl").read_text().splitlines()}
clips = [c for c in clips if c["clip"] in st]


def one(c):
    r = E.per_clip(("vipe", c))
    r["vipe_rc"] = st[c["clip"]]["rc"]
    try:
        a = E.load("vipe", c["clip"], c["n_frames"])[0][:, :3, 3]
        b = E.load("vggt_omega", c["clip"], c["n_frames"])[0][:, :3, 3]
        s, R, t = E.umeyama(a, b, True)
        L = np.linalg.norm(np.diff(b, axis=0), axis=1).sum()
        r["ate_pct_vipe_vggto"] = 100 * np.sqrt(((s * a @ R.T + t - b) ** 2).sum(1).mean()) / max(L, 1e-9)
        r["traj_len_vggto"] = L
    except Exception as e:  # noqa: BLE001
        r["ate_err"] = str(e)[:120]
    return r


QC = E.B / "eval" / "fc_qc_per_clip.csv"  # cached: QC depends only on the annotations, not on the decisions
old = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["clip"])
new = [c for c in clips if c["clip"] not in set(old["clip"])]
if new:
    with ProcessPoolExecutor(24) as ex:
        old = pd.concat([old, pd.DataFrame(list(ex.map(one, new)))], ignore_index=True)
    QC.parent.mkdir(exist_ok=True)
    old.to_csv(QC, index=False, float_format="%.4f")
qc = old[old["clip"].isin([c["clip"] for c in clips])].copy()
qc.to_csv(OUT / "fc_qc_per_clip.csv", index=False, float_format="%.4f")


def failed(r):
    return bool(r.get("vipe_rc", 0) != 0 or not np.isfinite(r.get("g5_selfcons", np.nan)) or r["g5_selfcons"] < 0.5
                or (np.isfinite(r.get("ate_pct_vipe_vggto", np.nan)) and r["ate_pct_vipe_vggto"] > 10)
                or not (20 <= r.get("hfov_deg", 60) <= 130))


qc["fail"] = qc.apply(failed, axis=1)
df = qc.merge(dec, on="clip", how="left", suffixes=("", "_dec"))
df["reject"] = df.decision == "reject"
df["rule_reject"] = ~df.rule_pass.astype(bool)


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if y.all() or (~y).all():
        return np.nan
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())


summ = {"n": len(df), "fail_rate_all": df.fail.mean(),
        "fail_rate_pass": df[~df.reject].fail.mean(), "fail_rate_reject": df[df.reject].fail.mean(),
        "n_pass": int((~df.reject).sum()), "n_reject": int(df.reject.sum()),
        "median_qc_pass": df[~df.reject][["g5_selfcons", "g20_selfcons", "g5_photo", "ate_pct_vipe_vggto"]].median().to_dict(),
        "median_qc_reject": df[df.reject][["g5_selfcons", "g20_selfcons", "g5_photo", "ate_pct_vipe_vggto"]].median().to_dict()}
y = df.fail
# precision / recall of the filter as a failure predictor
allr = pd.Series(True, index=df.index)
for name, pred, valid in (("final", df.reject, allr), ("rules", df.rule_reject, allr)) + tuple(
        (c, df[c] != "yes", df[c].notna()) for c in df.columns if c.startswith("vlm_") and c.endswith(".reconstructable")):
    p_, y_ = pred[valid], y[valid]  # clips the VLM has not judged (or failed to parse) are left out
    tp, fp, fn = int((p_ & y_).sum()), int((p_ & ~y_).sum()), int((~p_ & y_).sum())
    summ[f"predict_fail:{name}"] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                                    "n_reject": int(p_.sum()), "n": int(valid.sum()),
                                    "fail_rate_if_pass": float(y_[~p_].mean()), "fail_rate_if_reject": float(y_[p_].mean())}
# per reject reason
rs = df[df.reject].reason.fillna("").str.split(";").explode()
summ["fail_rate_by_reason"] = {k: {"n": int((rs == k).sum()), "fail_rate": float(df.loc[rs[rs == k].index.unique(), "fail"].mean())}
                               for k in rs.unique() if k}
# AUROC of individual scores for predicting failure (> 0.5: higher score -> more failures)
cand = [c for c in df.columns if c.split(".")[0] in ("cuts", "rules", "ocr", "flow", "track", "calib")
        and pd.api.types.is_numeric_dtype(df[c])]
summ["auroc_scores"] = {c: auroc(df[c], y) for c in cand}
summ["auroc_scores"] = dict(sorted(summ["auroc_scores"].items(), key=lambda kv: -abs((kv[1] or 0.5) - 0.5)))
df.to_csv(OUT / "fc_joined.csv", index=False, float_format="%.4g")
(OUT / "fc_summary.json").write_text(json.dumps(summ, indent=1, default=float))
print(json.dumps({k: v for k, v in summ.items() if k != "auroc_scores"}, indent=1, default=float))
print("top AUROC:", list(summ["auroc_scores"].items())[:12])
