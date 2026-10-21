"""E12 scale tables from scales_<kind>.csv (CPU, analytic; no re-evaluation needed for the scale metric).

  python e12_scale_report.py -> outputs/evolution/E12/raw/scale_{A,B,scannetpp,wild}.csv, E12/scale_tables.md
Scaling the v1.3 base (pi3x_kgate cameras + depth) by k turns its Sim3 scale onto GT from s_pi3x into s_pi3x / k, so
  traj scale error = |s_pi3x / k - 1|   (the eval3d fail_scale definition: > 10% -> failure; metric GT clips only)
  depth scale error = |k / gtd - 1|     (gtd = median GT / median Pi3X depth: how well k fixes the depth alone)
Rows: own (k = 1, raw Pi3X), ViPE / DA3 raw trajectories (s_vipe, s_da3), single sources, votes, and the
'agree' gate (use sqrt(k_v13 * k_moge) when MoGe agrees with v1.3 within T, else flag the clip).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import report_hybrid as R  # noqa: E402

E12 = Path("/data/home/z00533370/outputs/evolution/E12")
UNSEEN = ["1c7a683c92", "6c14d5fd01", "a4c043ac48"]
MOGE = [f"moge_{m}_{v}" for m in ("m2l", "m3l", "m3g") for v in ("free", "kgate")]
TS = (0.05, 0.10, 0.15, 0.20)


def sources(d):
    k = {"own (raw Pi3X)": pd.Series(1.0, index=d.index)}
    for c, name in (("vd", "ViPE depth ratio"), ("vt", "ViPE traj"), ("v13", "v1.3 sqrt(ViPE d*t)"), ("da3", "DA3 depth ratio")):
        k[name] = d[f"k_{c}"]
    for m in MOGE:
        if f"k_{m}" in d:
            k[m] = d[f"k_{m}"]
    for m in MOGE:
        if f"k_{m}" not in d:
            continue
        k[f"med3(v13,{m},da3)"] = pd.Series(np.nanmedian(d[["k_v13", f"k_{m}", "k_da3"]].values, 1), index=d.index)
        k[f"gm(v13,{m})"] = np.sqrt(d["k_v13"] * d[f"k_{m}"])
        k[f"med3(vd,vt,{m})"] = pd.Series(np.nanmedian(d[["k_vd", "k_vt", f"k_{m}"]].values, 1), index=d.index)
    return k


def scale_table(d):
    d = d[d["metric"] == 1].copy() if "metric" in d else d
    rows = []
    raw = {"ViPE own trajectory": d.get("s_vipe"), "DA3 own trajectory": d.get("s_da3")}
    for name, s in raw.items():
        if s is None:
            continue
        e = (s - 1).abs()
        rows.append({"source": name, "n": int(e.notna().sum()), "traj_err_med": e.median(), "traj_err_mean": e.mean(),
                     "fail_scale": (e > 0.1).mean()})
    for name, k in sources(d).items():
        k = k.where(np.isfinite(k) & (k > 0), d["k_v13"])
        e = (d["s_pi3x"] / k - 1).abs()
        ed = (k / d["gtd"] - 1).abs()
        rows.append({"source": name, "n": int(e.notna().sum()), "traj_err_med": e.median(), "traj_err_mean": e.mean(),
                     "fail_scale": (e > 0.1).mean(), "depth_err_med": ed.median(), "depth_fail": (ed > 0.1).mean()})
    for m in MOGE:  # MoGe scale applied directly to ViPE's own trajectory (Pi3X-free path)
        if f"vm_{m}" in d and "s_vipe" in d:
            e = (d["s_vipe"] / d[f"vm_{m}"] - 1).abs()
            rows.append({"source": f"ViPE traj x {m}", "n": int(e.notna().sum()), "traj_err_med": e.median(),
                         "traj_err_mean": e.mean(), "fail_scale": (e[e.notna()] > 0.1).mean()})
    for m in MOGE:
        if f"k_{m}" not in d:
            continue
        dis = np.abs(np.log(d[f"k_{m}"] / d["k_v13"]))
        lab = np.sqrt(d["k_v13"] * d[f"k_{m}"])
        for T in TS:
            keep = dis < np.log(1 + T)
            e = (d["s_pi3x"] / lab - 1).abs()
            ev = (d["s_pi3x"] / d["k_v13"] - 1).abs()
            rows.append({"source": f"agree({m},{int(T * 100)}%)", "n": int(keep.sum()), "flag_rate": 1 - keep.mean(),
                         "traj_err_med": e[keep].median(), "fail_scale": (e[keep] > 0.1).mean(),
                         "fail_scale_v13_on_kept": (ev[keep] > 0.1).mean(),
                         "fail_caught": int(((ev > 0.1) & ~keep).sum()), "fail_total_v13": int((ev > 0.1).sum())})
    return pd.DataFrame(rows).set_index("source")


def wild_table(d):
    rows = []
    for m in MOGE:
        if f"k_{m}" not in d:
            continue
        dis = np.abs(np.log(d[f"k_{m}"] / d["k_v13"]))
        r = {"moge": m, "n": int(dis.notna().sum()), "disagree_med": float(np.exp(dis.median()) - 1),
             "cv_frame_med": float(d[f"cv_{m}"].median())}
        for T in TS:
            r[f"flag@{int(T * 100)}%"] = float((dis >= np.log(1 + T)).mean())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("moge")
    by = {}
    for m in ("moge_m3l_kgate", "moge_m3g_kgate"):
        if f"k_{m}" in d:
            dis = np.abs(np.log(d[f"k_{m}"] / d["k_v13"]))
            by[m] = d.assign(dis=dis).groupby("dataset")["dis"].apply(lambda x: float((x >= np.log(1.1)).mean()))
    return t, pd.DataFrame(by)


def md(t):
    return R.to_md(t.round(3))


def main():
    out, summ = ["# E12 — metric-scale tables (analytic, see e12_scale_report.py)\n"], {}
    for kind in ("A", "B", "scannetpp", "wild"):
        f = E12 / f"scales_{kind}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()] if "error" in d else d
        if kind == "wild":
            t, by = wild_table(d)
            t.to_csv(E12 / "raw" / "scale_wild.csv")
            out.append("\n## C (wild) — MoGe vs v1.3 scale disagreement |k_moge/k_v13 - 1| and flag rates\n\n" + md(t))
            out.append("\n### C: flag rate at 10% by category\n\n" + md(by))
            summ["wild"] = json.loads(t.to_json(orient="index"))
            continue
        t = scale_table(d)
        t.to_csv(E12 / "raw" / f"scale_{kind}.csv")
        out.append(f"\n## {kind} — metric clips (n={int((d.get('metric', 1) == 1).sum())})\n\n" + md(t))
        summ[kind] = json.loads(t.to_json(orient="index"))
        if kind in ("A", "B"):  # per dataset: failure rate and signed depth-scale bias median(k / gtd)
            m1 = d[d["metric"] == 1]
            cols = ["v13", "da3"] + [m for m in MOGE if f"k_{m}" in d]
            rows = {}
            for ds, g in m1.groupby("dataset"):
                for c in cols:
                    e = (g["s_pi3x"] / g[f"k_{c}"] - 1).abs()
                    rows[(ds, c)] = {"n": len(g), "traj_fail": (e > 0.1).mean(), "traj_err_med": e.median(),
                                     "depth_bias_med(k/gtd)": (g[f"k_{c}"] / g["gtd"]).median(),
                                     "frame_scale_cv": g[f"cv_{c}"].median() if f"cv_{c}" in g else np.nan}
            out.append(f"\n### {kind} per dataset: scale source -> fail rate / signed depth bias\n\n"
                       + md(pd.DataFrame(rows).T))
        if kind == "scannetpp":
            u = d[d["scene"].isin(UNSEEN)]
            tu = scale_table(u)
            out.append(f"\n### scannetpp unseen-3 (n={len(u)})\n\n" + md(tu.iloc[:12]))
            summ["scannetpp_unseen3"] = json.loads(tu.to_json(orient="index"))
    (E12 / "scale_tables.md").write_text("\n".join(out))
    (E12 / "scale_summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(out)[:12000])


if __name__ == "__main__":
    main()
