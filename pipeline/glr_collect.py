"""GLR track, phase C (H200 side): GLRMap result -> label in the v1.3 label contract + cross-witness QC + tier.

Usage: python glr_collect.py OUT [--cfg FILE] [--clips a,b] [--witness-dir DIR]          (cache/qc-venv; CPU)
  --witness-dir  read the witness bundle of each clip from DIR/<clip>.npz (see load_witness_npz) instead of the experiments' bench layout
Inputs (per clip; OUT = a run directory of pipeline/run_batch.sh followed by glr_prepare.py + glr_worker.sh):
  OUT/glr_jobs/<clip>/{job.json, RESULT_DIR, <result dir>/{result.json, gate.json, sparse model, fused depth ...}}
  OUT/bench/<vipe, da3_pkvipe, geocalib, pi3x, da3, gate, pi3x_kgate, frames>   (v1.3 witnesses; same layout as the experiments)
Outputs:
  OUT/labels_glr/<clip>.npz   c2w (N,4,4) OpenCV camera-to-world, metres (first registered camera = world origin), K (N,3,3) at the
                              depth grid, depth (N,h,w) metres, conf, scale, k_depth, k_traj, hfov, k_source, frame_hw (as v1.3)
                              + frame_idx (index of each label frame in the 15 fps clip), n_orig, annotator, tier, depth_source
  OUT/qc_glr/<clip>.json      v1c, v1c_parts, xtraj_ba, xdepth_k, kdis, flags, reject_flags, info_flags, accept (as v1.3)
                              + tier (T1 | T2 | T3), stage1 (gate verdict + signals), glr (status, depth source ...), agreement_v13
  OUT/results_glr.csv         one row per GLR clip
The QC is the v1c rule of pipeline/v13_label_qc.py with the GLRMap label in place of the Pi3X label and the same independent witnesses:
  xtraj_ba = Sim(3) ATE % of the GLRMap cameras vs ViPE (3.27)   xdepth_k = depth AbsRel (one median scale) vs DA3 conditioned on ViPE
  cameras (0.194)   kdis = |HFOV(GLRMap) - median(GeoCalib, Pi3X, DA3)| (13.2 deg); v1c = max of the three ratios, > 1 rejects.
Tiers: T3 reject (v1c > 1 or not checkable, reject flag jump60 / fov_out, stage-1 gate stop, GLRMap failed); T1 accepted with every part <= 0.5 x its
threshold, no stage-1 suspect flag, depth present and >= 90 % of the frames registered; T2 accepted otherwise. A tier says "no known failure
signature fired", not "accurate" - the witnesses are noisier than GLRMap's good case (docs/GLR_INTEGRATION.md).
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glr_common as C  # noqa: E402
import glr_stage1_gate as G  # noqa: E402

BAD_STATUS = {"failed", "stopped_at_gate", "native_gate_stop", "complete_gate_stop"}


# ----------------------------------------------------------------------------- GLRMap result -> arrays
def _frame_no(name):
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return int(digits) if digits else None


def find_model_dir(res):
    res = Path(res)
    try:
        rj = json.loads((res / "result.json").read_text())
        for rel in rj.get("found", {}).get("sparse_models", []):
            return (res / rel).parent
    except Exception:  # noqa: BLE001
        pass
    hits = sorted(res.glob("**/sparse/*/cameras.*"), key=lambda p: len(str(p)))
    return hits[0].parent if hits else None


def read_depth_maps(res, nos):
    """Per-frame PDA depth: <res>/**/fused/<...number...>.npz with `fine` (else `coarse`). -> (dict frame_no -> 2D array, source)."""
    out, src = {}, "none"
    files = {}
    for p in Path(res).glob("**/fused/*.npz"):
        no = _frame_no(p.name)
        if no is not None:
            files[no] = p
    for no in nos:
        p = files.get(no)
        if p is None:
            continue
        z = np.load(p)
        key = "fine" if "fine" in z.files else ("coarse" if "coarse" in z.files else None)
        if key is None:
            arrs = [k for k in z.files if z[k].ndim == 2 and z[k].dtype.kind == "f"]
            key = arrs[0] if arrs else None
        if key is None:
            continue
        out[no] = z[key].astype(np.float32)
        src = f"pda_{key}" if key in ("fine", "coarse") else f"pda_{key}"
    return out, src


def read_glr(res, job, label_long=512):
    """-> dict(ok, why, names, k_idx, frame_idx, c2w (first camera = origin), K (frame grid), W, H, depth (n,h,w) on the label grid | None,
              K_label, depth_source, conf, n_reg)"""
    res = Path(res)
    mdir = find_model_dir(res)
    if mdir is None:
        return {"ok": False, "why": "no_sparse_model"}
    cams = G.load_cameras(mdir)
    if not len(cams["names"]):
        return {"ok": False, "why": "no_registered_images"}
    names_job = job["image_names"]
    k_idx = []
    for nm in cams["names"]:
        no = _frame_no(nm)
        k_idx.append(no - 1 if no is not None and 1 <= no <= len(names_job) else -1)
    keep = [i for i, k in enumerate(k_idx) if k >= 0]
    c2w = cams["c2w"][keep]
    K = cams["K"][keep]
    k_idx = np.array([k_idx[i] for i in keep])
    order = np.argsort(k_idx)
    c2w, K, k_idx = c2w[order], K[order], k_idx[order]
    c2w = np.linalg.inv(c2w[0])[None] @ c2w  # first registered camera = world origin (as the Pi3X label)
    frame_idx = np.array(job["frame_idx"])[k_idx]
    W, H = cams["W"], cams["H"]
    out = {"ok": True, "why": "", "names": [names_job[k] for k in k_idx], "k_idx": k_idx, "frame_idx": frame_idx, "c2w": c2w, "K": K,
           "W": W, "H": H, "model": cams["model"], "n_reg": int(len(k_idx)), "depth": None, "K_label": K, "depth_source": "none",
           "conf": None}
    maps, src = read_depth_maps(res, [k + 1 for k in k_idx])
    if maps:
        h0, w0 = next(iter(maps.values())).shape
        Hl, Wl = C.qc_grid(H, W, label_long)
        D = np.stack([maps.get(int(k) + 1, np.full((h0, w0), np.nan, np.float32)) for k in k_idx])
        Kn = K.copy()
        Kn[:, 0, :] *= w0 / W
        Kn[:, 1, :] *= h0 / H
        out["depth"], out["K_label"] = C.to_grid(D, Kn, Hl, Wl)
        out["depth_source"] = src
        out["frame_with_depth"] = int(sum(int(k) + 1 in maps for k in k_idx))
    return out


# ----------------------------------------------------------------------------- QC
def e9_flags(T, K, D, W, fps):
    """Copy of pipeline/v13_label_qc.e9_flags (= evolution/e9_hard_rules.flags). D may be None (then move_ratio is NaN)."""
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
    nv = np.linalg.norm(v, axis=1) if len(v) else np.array([])
    ok = nv > 1e-3 * max(np.linalg.norm(np.diff(p, axis=0), axis=1).sum(), 1e-9) if len(v) else np.array([], bool)
    rev = 0
    if ok.sum() > step + 1:
        u = v[ok] / nv[ok][:, None]
        a = np.degrees(np.arccos(np.clip((u[step:] * u[:-step]).sum(1), -1, 1)))
        rev = int((a > 150).sum())
    f["reversals"] = rev
    f["reversal"] = int(rev >= 2)
    L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    md = float(np.nanmedian(D[D > 0])) if D is not None and np.isfinite(D).any() and (D > 0).any() else np.nan
    f["move_ratio"] = float(L / md) if md and np.isfinite(md) else np.nan
    f["tiny_move"] = int(f["move_ratio"] < 0.02) if np.isfinite(f["move_ratio"]) else 0
    return f


def qc_one(job, glr, stage1, wit, cfg):
    """Pure QC/label computation (no file access).
    glr    read_glr() result (ok): c2w / K at the GLRMap frame grid, depth + K_label on the label grid, frame_idx into the 15 fps clip.
    stage1 stage-1 gate dict {verdict, stop, suspect, signals} (or None).
    wit    witness bundle on the QC grid, all frames of the clip: c2w, K, D (ViPE, metric), da3_D (DA3 conditioned on ViPE cameras | None),
           static (n,h,w) bool | None, gate (K-gate json with others_med | None), vipe_focal_cv, v13 (c2w, K, D, W of the v1.3 label | None).
    -> (label dict, qc dict)"""
    q = cfg["qc"]
    thr = q["thr"]
    idx = np.asarray(glr["frame_idx"])
    n = len(idx)
    sub = lambda a: None if a is None else a[idx]  # noqa: E731
    cs = glr["c2w"].copy()
    D = glr["depth"]
    has_d = D is not None and bool(np.isfinite(D).any())
    ref_c2w, ref_D = (sub(wit["c2w"]) if wit.get("c2w") is not None else None), (sub(wit["D"]) if wit.get("D") is not None else None)
    # ---- metric scale: the E3b rule against ViPE on the frames GLRMap used
    if ref_c2w is not None:
        D_x = None
        if has_d and ref_D is not None:
            D_x = D if D.shape[1:] == ref_D.shape[1:] else C.to_grid(D, glr["K_label"], *ref_D.shape[1:])[0]
        sc = C.metric_scale(cs, D_x, ref_c2w, ref_D, q["n_scale_frames"])
    else:
        sc = {"k": 1.0, "k_traj": 1.0, "k_depth": 1.0, "s_vipe": np.nan, "r_vipe": np.nan}
    k = sc["k"] if np.isfinite(sc["k"]) and sc["k"] > 0 else 1.0
    cs[:, :3, 3] *= k
    Dm = None if D is None else (D * np.float32(k))
    # ---- v1c parts (all scale-free)
    parts = {"xtraj_ba": np.nan, "xdepth_k": np.nan, "kdis": np.nan}
    if ref_c2w is not None:
        parts["xtraj_ba"] = C.traj_pct(cs[:, :3, 3], ref_c2w[:, :3, 3])
    if has_d and wit.get("da3_D") is not None:
        ks = np.linspace(0, n - 1, q["n_depth_frames"]).astype(int)
        gd = wit["da3_D"].shape[1:]
        Dq = Dm if Dm.shape[1:] == tuple(gd) else C.to_grid(Dm, glr["K_label"], *gd)[0]
        st = None if wit.get("static") is None else sub(wit["static"])[ks]
        parts["xdepth_k"] = C.depth_cmp(Dq[ks], sub(wit["da3_D"])[ks], st)[0]
    hf = C.hfov_deg(float(np.median(glr["K"][:, 0, 0])), glr["W"])
    others = (wit.get("gate") or {}).get("others_med")
    if others is not None and np.isfinite(others):
        parts["kdis"] = abs(hf - float(others))
    ratios = {s: parts[s] / thr[s] for s in thr}
    v1c = float(np.nanmax(list(ratios.values()))) if np.isfinite(list(ratios.values())).any() else float("nan")
    # ---- flags (E9) on the label; effective frame rate = clip fps / stride
    fps_eff = job.get("fps_orig", 15) / max(job.get("stride", 1), 1)
    D_flag = Dm if has_d else ref_D  # ViPE depth is metric already; used only for the camera-path / scene-depth ratio
    f = e9_flags(cs, glr["K"], D_flag, glr["W"], fps_eff)
    f["vipe_focal_cv"] = float(wit.get("vipe_focal_cv", np.nan))
    reject_flags = [x for x in q["reject_flags"] if f.get(x)]
    s1 = stage1 or {"verdict": "ok", "stop": [], "suspect": [], "signals": {}}
    reg_frac = glr["n_reg"] / max(len(job["frame_idx"]), 1)
    info = [name for name, on in (("suspect_cut", f["jump20"] and not f["jump60"]), ("focal_unobservable", f["tiny_move"]),
                                  ("suspect_zoom", f["vipe_focal_cv"] > 0.07), ("partial_registration", reg_frac < 0.9),
                                  ("depth_missing", not has_d)) if on]
    info += [f"s1_{x}" for x in s1.get("suspect", [])]
    # ---- decision
    why = []
    if not np.isfinite(v1c):
        why.append("v1c_not_checkable")
    elif v1c > 1:
        why.append("v1c_gt_1")
    why += reject_flags
    if s1.get("verdict") == "stop":
        why += [f"s1_{x}" for x in s1.get("stop", [])] or ["s1_stop"]
    accept = not why
    tier = "T3"
    if accept:
        small = bool(np.all(np.nan_to_num([ratios[s] for s in thr], nan=0.0) <= q["tier1_part"]))
        tier = "T1" if (small and not s1.get("suspect") and has_d and reg_frac >= 0.9) else "T2"
    qc = {"clip": job["clip"], "annotator": "glr", **parts, "v1c": v1c, "v1c_parts": ratios, "flags": f, "reject_flags": reject_flags,
          "info_flags": info, "accept": bool(accept), "tier": tier, "reject_reasons": why,
          "scale": {k_: sc.get(k_) for k_ in ("k", "k_depth", "k_traj", "s_vipe", "r_vipe")}, "stage1": s1,
          "glr": {"n_reg": glr["n_reg"], "n_sampled": len(job["frame_idx"]), "stride": job.get("stride"), "model": glr.get("model"),
                  "depth_source": glr["depth_source"], "hfov": hf}}
    nan11 = np.full((n, 1, 1), np.nan, np.float32)
    label = {"c2w": cs, "K": glr["K_label"], "depth": Dm if Dm is not None else nan11,
             "conf": glr["conf"] if glr.get("conf") is not None else (np.isfinite(Dm).astype(np.float32) if Dm is not None else nan11),
             "scale": k, "k_depth": sc.get("k_depth", np.nan), "k_traj": sc.get("k_traj", np.nan), "hfov": hf,
             "k_source": "glr_sfm", "frame_hw": np.array([glr["H"], glr["W"]]), "frame_idx": idx, "n_orig": job["n_orig"],
             "annotator": "glr", "tier": tier, "depth_source": glr["depth_source"]}
    # ---- agreement with the v1.3 label of the same clip (information only; the Pi3X label is conditioned on ViPE's K, so it is a
    #      partially independent second annotator, not a witness)
    v13 = wit.get("v13")
    if v13 is not None:
        ag = {"traj_pct": C.traj_pct(cs[:, :3, 3], v13["c2w"][idx][:, :3, 3]),
              "hfov_diff": abs(hf - C.hfov_deg(float(np.median(v13["K"][:, 0, 0])), v13["W"]))}
        if has_d:
            ks = np.linspace(0, n - 1, q["n_depth_frames"]).astype(int)
            gd = v13["D"].shape[1:]
            Dq = Dm if Dm.shape[1:] == tuple(gd) else C.to_grid(Dm, glr["K_label"], *gd)[0]
            ag["depth_absrel"] = C.depth_cmp(Dq[ks], v13["D"][idx][ks])[0]
        qc["agreement_v13"] = ag
    return label, qc


def failed_qc(job, status, s1):
    why = [f"glr_{status}"] + [f"s1_{x}" for x in (s1 or {}).get("stop", [])]
    return {"clip": job["clip"], "annotator": "glr", "accept": False, "tier": "T3", "v1c": float("nan"), "reject_reasons": why,
            "stage1": s1, "glr": {"status": status}}


# ----------------------------------------------------------------------------- real witnesses (v1.3 bench layout)
def load_witnesses(out, job, cfg):
    """Needs the experiment code on ANNO_ROOT/code (same imports as pipeline/v13_label_qc.py) and cache/qc-venv (OpenEXR)."""
    for p in ("code/qc", "code/bench", "code/vipe_eval", "code/hybrid", "code/evolution"):
        sys.path.insert(0, str(C.ROOT / p))
    from eval3d_bench import load_method  # noqa: E402
    from qc_common import vipe_static_mask  # noqa: E402
    B, clip, n = out / "bench", job["clip"], job["n_orig"]
    hw = C.qc_grid(*job["hw"], cfg["qc"]["qc_long"])
    W = {"hw": hw}
    Tv, Kv, Dv, _ = load_method(B, "vipe", clip, n, *hw)
    W.update(c2w=Tv, K=Kv, D=Dv)
    try:
        W["da3_D"] = load_method(B, "da3_pkvipe", clip, n, *hw)[2]
    except Exception:  # noqa: BLE001
        W["da3_D"] = None
    m = vipe_static_mask(B, clip, n, hw)
    W["static"] = None if m is None else (m == 0)
    gp = B / "gate" / f"{clip}.json"
    W["gate"] = json.loads(gp.read_text()) if gp.exists() else None
    fv = np.load(B / "vipe" / "intrinsics" / f"{clip}.npz")["data"][:, 0]
    W["vipe_focal_cv"] = float(np.std(fv) / np.mean(fv))
    if (B / "pi3x_kgate" / clip / "pred.npz").exists():
        T, K, D, _ = load_method(B, "pi3x_kgate", clip, n, *hw)
        W["v13"] = {"c2w": T, "K": K, "D": D, "W": hw[1]}
    return W


def load_witness_npz(path):
    """Witness bundle from a .npz written by another tool (offline tests, or witnesses computed elsewhere): c2w (n,4,4), K (n,3,3), D (n,h,w) of the
    metric reference (ViPE); optional da3_D (n,h,w), static (n,h,w) bool, v13_{c2w,K,D}; scalars others_med (deg, K-source median), vipe_focal_cv."""
    z = np.load(path)
    W = {"c2w": z["c2w"], "K": z["K"], "D": z["D"], "da3_D": z["da3_D"] if "da3_D" in z.files else None,
         "static": z["static"].astype(bool) if "static" in z.files else None,
         "gate": {"others_med": float(z["others_med"])} if "others_med" in z.files else None,
         "vipe_focal_cv": float(z["vipe_focal_cv"]) if "vipe_focal_cv" in z.files else 0.0}
    if "v13_c2w" in z.files:
        W["v13"] = {"c2w": z["v13_c2w"], "K": z["v13_K"], "D": z["v13_D"], "W": z["D"].shape[2]}
    return W


def main():
    a = sys.argv[1:]
    out = Path(a[0])
    opt = dict(x[2:].split("=", 1) if "=" in x else (x[2:], "1") for x in a[1:] if x.startswith("--"))
    cfg = C.load_cfg(opt.get("cfg"))
    want = set(opt["clips"].split(",")) if "clips" in opt else None
    (out / "labels_glr").mkdir(exist_ok=True)
    (out / "qc_glr").mkdir(exist_ok=True)
    rows = []
    for jd in sorted((out / "glr_jobs").iterdir()):
        if not (jd / "job.json").exists() or (want and jd.name not in want):
            continue
        job = json.loads((jd / "job.json").read_text())
        clip = job["clip"]
        try:
            rd = jd / (jd / "RESULT_DIR").read_text().strip()
            rj = json.loads((rd / "result.json").read_text())
            s1 = json.loads((rd / "gate.json").read_text()) if (rd / "gate.json").exists() else None
            status = rj["status"]
            if status in BAD_STATUS and status != "complete_gate_stop":
                qc = failed_qc(job, status, s1)
            else:
                glr = read_glr(rd, job, cfg["qc"]["label_long"])
                if not glr["ok"]:
                    qc = failed_qc(job, glr["why"], s1)
                else:
                    wit = load_witness_npz(Path(opt["witness-dir"]) / f"{clip}.npz") if "witness-dir" in opt else load_witnesses(out, job, cfg)
                    label, qc = qc_one(job, glr, s1, wit, cfg)
                    np.savez_compressed(out / "labels_glr" / f"{clip}.npz", **label)
        except Exception as e:  # noqa: BLE001
            qc = {"clip": clip, "annotator": "glr", "accept": False, "tier": "T3", "v1c": float("nan"),
                  "reject_reasons": ["collect_error"], "error": repr(e)[:300]}
        C.dump_json(out / "qc_glr" / f"{clip}.json", qc)
        rows.append({"clip": clip, "annotator": "glr", "tier": qc["tier"], "accept": qc["accept"], "v1c": qc.get("v1c"),
                     "xtraj_ba": qc.get("xtraj_ba"), "xdepth_k": qc.get("xdepth_k"), "kdis": qc.get("kdis"),
                     "reasons": ";".join(qc.get("reject_reasons", [])), "info_flags": ";".join(qc.get("info_flags", [])),
                     "scale": (qc.get("scale") or {}).get("k"), "hfov": (qc.get("glr") or {}).get("hfov"),
                     "s1_gate": (qc.get("stage1") or {}).get("verdict"), "depth_source": (qc.get("glr") or {}).get("depth_source")})
        print(json.dumps(C.clean(rows[-1])), flush=True)
    if rows:
        with open(out / "results_glr.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(C.clean(rows))


if __name__ == "__main__":
    main()
