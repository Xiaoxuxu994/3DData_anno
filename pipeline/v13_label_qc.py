"""v1.3 metric scale, final label and QC (CPU).

Usage: cache/qc-venv/bin/python v13_label_qc.py OUT
Inputs (OUT/bench): vipe/, pi3x/, da3/, pi3x_kgate/, da3_pkvipe/, gate/ (see run_batch.sh).

1. Metric scale (E3b): k = sqrt(k_traj * k_depth) with k_depth = median(ViPE depth) / median(Pi3X depth) and
   k_traj = Sim(3) scale of the Pi3X camera centres onto ViPE's (fallbacks as in E3). Computed by
   code/evolution/e3_scale_vote.py (functions `one` and `votes`, column k_vipe_td) on the same evaluation grid it used
   (GT depth resolution for GT clips, long side <= 960 otherwise).
2. Label: Pi3X (gated-K conditioned) poses and depth with translation and depth multiplied by k
   -> OUT/labels/<clip>.npz and OUT/bench/v13/<clip>/pred.npz (bench layout, used by the regression test).
3. QC rule v1c (E8, code/evolution/e8_qc_cheap.py / E13 e13_gate1_S.py), all signals scale-free:
     xtraj_ba = Sim(3) trajectory ATE % vs ViPE                              threshold 3.27
     xdepth_k = depth AbsRel after median scaling vs DA3 on ViPE cameras     threshold 0.194
                (8 frames, QC grid long side 512, sky / dynamic pixels removed with ViPE's masks)
     kdis     = |HFOV ViPE - median(GeoCalib, Pi3X, DA3)| (deg)            threshold 13.2
     v1c = nanmax(xtraj_ba / 3.27, xdepth_k / 0.194, kdis / 13.2); > 1 -> reject
   Flags (E9, code/evolution/e9_hard_rules.py `flags`, copied): reject on jump60 (adjacent-frame view change > 60 deg)
   and fov_out (label HFOV outside [30, 120] deg). Info only: k_gate_fired, suspect_cut (view change > 20 deg),
   focal_unobservable (camera path < 2% of median depth), suspect_zoom (ViPE per-frame focal CV > 7%).
   accept = v1c <= 1 and no reject flag (v1c NaN = not checkable -> reject).
   -> OUT/qc/<clip>.json, OUT/results.csv
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ANNO_ROOT", "/data/home/z00533370"))
for p in ("code/qc", "code/bench", "code/vipe_eval", "code/hybrid", "code/evolution"):
    sys.path.insert(0, str(ROOT / p))
OUT = Path(sys.argv[1])
B = OUT / "bench"
_argv, sys.argv = sys.argv, ["e3_scale_vote.py", "A"]
import e3_scale_vote as ESV  # noqa: E402
sys.argv = _argv
import qc_cross as QX  # noqa: E402
from eval3d_bench import load_method  # noqa: E402
from qc_common import LONG, vipe_static_mask  # noqa: E402
from wild_eval import grid as wild_grid  # noqa: E402

THR = {"xtraj_ba": 3.27, "xdepth_k": 0.194, "kdis": 13.2}


def eval_grid(c):
    """Grid the experiments evaluated on: GT depth resolution if the clip has GT, else long side <= 960."""
    g = B / "gt" / f"{c['clip']}.npz"
    if g.exists():
        return np.load(g, allow_pickle=True)["depth"].shape[1:]
    return wild_grid(c)


def qc_grid(h, w):  # code/qc/qc_common.qc_grid on a frame size
    s = LONG / max(h, w)
    return max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)


def e9_flags(T, K, D, W, fps):  # copy of code/evolution/e9_hard_rules.py flags()
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


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def one(c):
    clip, n = c["clip"], c["n_frames"]
    H, W = eval_grid(c)
    ESV.loader = lambda c_: (lambda m: load_method(B, m, clip, n, H, W))  # noqa: E731
    # ---- 1. metric scale (E3b)
    sc = ESV.votes(pd.DataFrame([ESV.one(c)])).iloc[0]
    k = float(sc["k_vipe_td"])
    k = k if np.isfinite(k) and k > 0 else 1.0
    # ---- 2. label at Pi3X resolution
    z = np.load(B / "pi3x_kgate" / clip / "pred.npz")
    c2w = z["c2w"].copy()
    c2w[:, :3, 3] *= k
    depth = z["depth"] * np.float32(k)
    gate = json.loads((B / "gate" / f"{clip}.json").read_text())
    lab = dict(c2w=c2w, K=z["K"], depth=depth.astype(np.float32), conf=z["conf"], scale=k,
               k_depth=float(sc.get("k_vipe_d", np.nan)), k_traj=float(sc.get("k_vipe_t", np.nan)),
               hfov=float(gate["hfov_label"]), k_source=gate["k_source"], frame_hw=np.array(c["hw"]))
    (OUT / "labels").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "labels" / f"{clip}.npz", **lab)
    (B / "v13" / clip).mkdir(parents=True, exist_ok=True)
    np.savez(B / "v13" / clip / "pred.npz", c2w=c2w, K=z["K"], depth=depth, conf=z["conf"],
             sec=float(z["sec"]), scale=k)
    # ---- 3. QC v1c on the QC grid
    hw = qc_grid(*c["hw"])
    ks = np.linspace(0, n - 1, 8).astype(int)
    m = vipe_static_mask(B, clip, n, hw)
    static = None if m is None else m[ks] == 0
    T, K, D, _ = load_method(B, "pi3x_kgate", clip, n, *hw)
    T = T.copy()
    T[:, :3, 3] *= k
    D = D * k
    q = {"clip": clip}
    try:
        Tv = load_method(B, "vipe", clip, n, *hw)[0]
        q["xtraj_ba"] = QX.traj_pct(T[:, :3, 3], Tv[:, :3, 3])
    except Exception as e:  # noqa: BLE001
        q["xtraj_ba"], q["xtraj_error"] = np.nan, str(e)[:150]
    try:
        q["xdepth_k"] = QX.depth_cmp(D[ks], load_method(B, "da3_pkvipe", clip, n, *hw)[2][ks], static)[0]
    except Exception as e:  # noqa: BLE001
        q["xdepth_k"], q["xdepth_error"] = np.nan, str(e)[:150]
    q["kdis"] = gate["kdis"] if gate["kdis"] is not None else np.nan
    ratios = [q[s] / t for s, t in THR.items()]
    q["v1c"] = float(np.nanmax(ratios)) if np.isfinite(ratios).any() else np.nan
    q["v1c_parts"] = {s: q[s] / t for s, t in THR.items()}
    # ---- flags (E9) on the label at the experiments' evaluation grid
    Hg, Wg = (c["hw"][0] // 2 * 2, c["hw"][1] // 2 * 2) if (B / "gt" / f"{clip}.npz").exists() else wild_grid(c)
    Tg, Kg, Dg, _ = load_method(B, "pi3x_kgate", clip, n, Hg, Wg)
    f = e9_flags(Tg, Kg, Dg, Wg, c.get("fps_out", 15))
    fv = np.load(B / "vipe" / "intrinsics" / f"{clip}.npz")["data"][:, 0]
    f["vipe_focal_cv"] = float(np.std(fv) / np.mean(fv))
    reject_flags = [x for x in ("jump60", "fov_out") if f[x]]
    info_flags = [name for name, on in (("k_gate_fired", gate["fired"]), ("suspect_cut", f["jump20"] and not f["jump60"]),
                                        ("focal_unobservable", f["tiny_move"]), ("suspect_zoom", f["vipe_focal_cv"] > 0.07)) if on]
    q.update(flags=f, reject_flags=reject_flags, info_flags=info_flags,
             accept=bool(np.isfinite(q["v1c"]) and q["v1c"] <= 1 and not reject_flags),
             scale={"k": k, "k_depth": lab["k_depth"], "k_traj": lab["k_traj"], "s_vipe": sc.get("s_vipe"),
                    "r_vipe": sc.get("r_vipe")},
             gate=gate)
    (OUT / "qc").mkdir(exist_ok=True)
    (OUT / "qc" / f"{clip}.json").write_text(json.dumps(clean(q), indent=1))
    return {"clip": clip, "accept": q["accept"], "v1c": q["v1c"], "xtraj_ba": q["xtraj_ba"], "xdepth_k": q["xdepth_k"],
            "kdis": q["kdis"], "reject_flags": ";".join(reject_flags), "info_flags": ";".join(info_flags),
            "scale": k, "hfov": lab["hfov"], "k_source": lab["k_source"]}


if __name__ == "__main__":
    rows = []
    for c in json.loads((B / "clips.json").read_text()):
        if not (B / "pi3x_kgate" / c["clip"] / "pred.npz").exists():
            rows.append({"clip": c["clip"], "accept": False, "error": "no pi3x_kgate prediction"})
            continue
        try:
            rows.append(one(c))
        except Exception as e:  # noqa: BLE001
            rows.append({"clip": c["clip"], "accept": False, "error": repr(e)[:300]})
        print(json.dumps(clean(rows[-1])), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False, float_format="%.5g")
