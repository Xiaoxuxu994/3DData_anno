"""GLR track, phase C (H200 side): GLRMap result -> label in the v1.3 label contract + cross-witness QC + tier.

Usage: python glr_collect.py OUT [--cfg FILE] [--clips a,b] [--witness-dir DIR]          (cache/qc-venv; CPU)
  --clips        only (re)collect these clips; the other rows of results_glr.csv are kept
  --witness-dir  read the witness bundle of each clip from DIR/<clip>.npz (see load_witness_npz) instead of the experiments' bench layout
Inputs (per clip; OUT = a run directory of pipeline/run_batch.sh followed by glr_prepare.py + glr_worker.sh):
  OUT/glr_jobs/<clip>/{job.json, RESULT_DIR, <result dir>/{result.json, gate.json, sparse model, fused depth ...}}
  A job counts as finished when its result directory has result.json (RESULT_DIR names it; if the pointer is missing, e.g. after a
  copy-back that skipped it, the highest-numbered result*/ with a result.json is used and the QC json says so). Jobs without a
  finished result are reported as glr_incomplete (T3) - the merge then falls back to the v1.3 label for them. No glr_jobs/ at all
  (nothing was routed to GLRMap) gives an empty results_glr.csv.
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
Tiers: T3 reject (v1c > 1, any v1c part or the metric scale not checkable, reject flag jump60 / fov_out, stage-1 gate stop / error / missing,
no GLRMap depth, GLRMap failed or unfinished); T1 accepted with all three v1c parts finite and <= 0.5 x their threshold, stage-1 gate ok,
depth on >= 90 % of the label frames and >= 90 % of the sampled frames registered; T2 accepted otherwise. A tier says "no known failure
signature fired", not "accurate" - the witnesses are noisier than GLRMap's good case (docs/GLR_INTEGRATION.md).
The GLR track is deliberately stricter than v1.3 about what it cannot check (configs/pipeline_glr_v0.yaml, qc.require_*): GLRMap's failures
are silent, and a check that did not run must not look like a check that passed.
Jump flags (jump20 / jump60) are evaluated between consecutive LABEL frames, i.e. `stride` source frames apart: at stride 4 a smooth pan of
more than 225 deg/s trips jump60 (v1.3: 900 deg/s between adjacent 15 fps frames). That is deliberate - a pose glitch between two sampled
frames should not be diluted - and the router picks small strides for fast footage; it is recorded here so nobody is surprised by it.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glr_common as C  # noqa: E402
import glr_stage1_gate as G  # noqa: E402

# worker statuses that never give a usable label: the QC is a stub with the reason glr_<status> (`complete_gate_stop` / `complete_gate_error`
# still carry a full label that is QC-ed and rejected with the gate's reasons, so it can be inspected)
BAD_STATUS = {"failed", "stopped_at_gate", "native_gate_stop", "gate_error", "interrupted", "stage1_only"}
ROW_FIELDS = ["clip", "annotator", "tier", "accept", "v1c", "xtraj_ba", "xdepth_k", "kdis", "reasons", "info_flags", "scale", "hfov", "s1_gate",
              "depth_source"]


# ----------------------------------------------------------------------------- GLRMap result -> arrays
def _frame_no(name):
    """The frame number in a file name: the longest run of digits, the last one on a tie ('000012.npz', 'depth_000012.npz',
    'clip2_000012.jpg' -> 12; 'a_0003_0007.npz' -> 7), else None."""
    runs = re.findall(r"\d+", Path(name).stem)
    return int(sorted(runs, key=len)[-1]) if runs else None


class MissingDependency(RuntimeError):
    """The environment cannot read what the job needs (no cameras.npz and no pycolmap): not a property of one clip, so the run stops."""


def load_cameras(res):
    """Registered cameras of a result directory: cameras.npz (written by the stage-1 gate on the worker host from the very model it
    checked - no pycolmap needed here) or, failing that, the COLMAP model itself (needs pycolmap). -> (cams dict like G.load_cameras, source)."""
    npz = Path(res) / "cameras.npz"
    if npz.exists():
        z = np.load(npz)
        return {"names": [str(x) for x in z["names"]], "c2w": z["c2w"], "K": z["K"], "W": int(z["W"]), "H": int(z["H"]),
                "model": str(z["model"])}, "cameras_npz"
    mdir = find_model_dir(res)
    if mdir is None:
        return None, "no_sparse_model"
    try:
        import pycolmap  # noqa: F401
    except ImportError as e:
        raise MissingDependency(f"{res}: no cameras.npz and pycolmap cannot be imported ({e}); run phase c with the GLRMap python (PY_GLR), "
                                f"or use a worker whose gate writes cameras.npz") from e
    return G.load_cameras(mdir), "colmap_model"


def find_model_dir(res):
    """The sparse model the stage-1 gate looked at (result.json: gate_model), else the shallowest sparse/<k>/cameras.* (ties by name) -
    the same rule as glr_worker.sh, so the label and the gate verdict always describe the same model."""
    res = Path(res)
    try:
        rel = json.loads((res / "result.json").read_text()).get("gate_model")
        if rel and (res / rel).is_dir():
            return res / rel
    except Exception:  # noqa: BLE001
        pass
    hits = sorted(res.glob("**/sparse/*/cameras.*"), key=lambda p: (len(str(p)), str(p)))
    return hits[0].parent if hits else None


def _result_no(d):
    m = re.fullmatch(r"result(?:_(\d+))?", d.name)
    return None if m is None else int(m.group(1) or 0)


def find_result_dir(jd):
    """-> (result dir | None, note). RESULT_DIR is written by the worker after result.json; without a usable pointer the highest-numbered
    result*/ that has a result.json is taken (note says so)."""
    jd = Path(jd)
    note = "no_result"
    ptr = jd / "RESULT_DIR"
    if ptr.exists():
        rd = jd / Path(ptr.read_text().strip()).name
        if (rd / "result.json").exists():
            return rd, ""
        note = "result_dir_without_result_json"
    cands = sorted((d for d in jd.glob("result*") if d.is_dir() and _result_no(d) is not None and (d / "result.json").exists()), key=_result_no)
    if cands:
        return cands[-1], "result_dir_guessed"
    return None, note


def read_depth_maps(res, nos):
    """Per-frame PDA depth: <res>/**/fused/<...number...>.npz with `fine` (else `coarse`). -> (dict frame_no -> 2D array, source, n_resized).
    The same number in several fused/ directories resolves to the shallowest path. Maps whose shape differs from the most common one are
    resized to it (nearest) and counted; the source is 'pda_fine' / 'pda_coarse' / 'pda_mixed' (what was actually read), 'none' without maps."""
    files = {}
    for p in sorted(Path(res).glob("**/fused/*.npz"), key=lambda q: (len(q.parts), str(q))):
        no = _frame_no(p.name)
        if no is not None:
            files.setdefault(no, p)
    maps, keys = {}, set()
    for no in nos:
        p = files.get(no)
        if p is None:
            continue
        z = np.load(p)
        key = "fine" if "fine" in z.files else ("coarse" if "coarse" in z.files else None)
        if key is None:
            arrs = [k for k in z.files if z[k].ndim == 2 and z[k].dtype.kind == "f"]
            key = arrs[0] if arrs else None
        if key is None or z[key].ndim != 2:
            continue
        maps[no] = z[key].astype(np.float32)
        keys.add(key)
    n_resized = 0
    if maps:
        shape = Counter(m.shape for m in maps.values()).most_common(1)[0][0]
        for no, m in maps.items():
            if m.shape != shape:
                import cv2
                maps[no] = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                n_resized += 1
    src = "none" if not maps else ("pda_" + next(iter(keys)) if len(keys) == 1 else "pda_mixed")
    return maps, src, n_resized


def read_glr(res, job, label_long=512):
    """-> dict(ok, why, names, k_idx, frame_idx, c2w (first camera = origin), K (frame grid), W, H, depth (n,h,w) on the label grid | None,
              K_label, depth_source, conf, n_reg)"""
    res = Path(res)
    cams, src = load_cameras(res)
    if cams is None:
        return {"ok": False, "why": "no_sparse_model"}
    if not len(cams["names"]):
        return {"ok": False, "why": "no_registered_images"}
    names_job = job["image_names"]
    pos = {nm: i for i, nm in enumerate(names_job)}
    k_idx, mapped_by = [pos.get(nm, -1) for nm in cams["names"]], "name"   # model image -> position in the job, by exact name
    if not any(k >= 0 for k in k_idx):  # GLRMap renamed the images: accept the frame number in the name, but only if that is one-to-one
        nos = [_frame_no(nm) for nm in cams["names"]]
        if all(n is not None and 1 <= n <= len(names_job) for n in nos) and len(set(nos)) == len(nos):
            k_idx, mapped_by = [n - 1 for n in nos], "number"
    n_unmapped = int(sum(k < 0 for k in k_idx))
    keep = [i for i, k in enumerate(k_idx) if k >= 0]
    if not keep:
        return {"ok": False, "why": "no_image_maps_to_the_job"}
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
           "conf": None, "mapped_by": mapped_by, "n_unmapped": n_unmapped, "n_depth_resized": 0, "cameras_from": src}
    maps, src, out["n_depth_resized"] = read_depth_maps(res, [k + 1 for k in k_idx])
    if maps:
        h0, w0 = next(iter(maps.values())).shape
        Hl, Wl = C.qc_grid(H, W, label_long)
        D = np.stack([maps.get(int(k) + 1, np.full((h0, w0), np.nan, np.float32)) for k in k_idx])
        Kn = K.copy()
        Kn[:, 0, :] *= w0 / W
        Kn[:, 1, :] *= h0 / H
        out["depth"], out["K_label"] = C.to_grid(D, Kn, Hl, Wl)
        out["depth_source"] = src if np.isfinite(out["depth"]).any() else "none"   # all-NaN maps are no depth
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
    stage1 stage-1 gate dict {verdict, stop, suspect, signals} as written by glr_worker.sh, or None when there is none (then the label is
           rejected as s1_gate_missing unless qc.require_gate is false).
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
    depth_cov = float(np.isfinite(D).reshape(n, -1).any(1).mean()) if has_d else 0.0   # label frames that have any depth
    ref_c2w, ref_D = (sub(wit["c2w"]) if wit.get("c2w") is not None else None), (sub(wit["D"]) if wit.get("D") is not None else None)
    # ---- metric scale: the E3b rule against ViPE on the frames GLRMap used. GLRMap's SfM scale is arbitrary, so "no evidence" is NaN
    #      (not E3b's neutral 1.0, which would export SfM units as metres): the label is rejected as scale_unverified.
    nan = float("nan")
    if ref_c2w is not None:
        D_x = None
        if has_d and ref_D is not None:
            D_x = D if D.shape[1:] == ref_D.shape[1:] else C.to_grid(D, glr["K_label"], *ref_D.shape[1:])[0]
        sc = C.metric_scale(cs, D_x, ref_c2w, ref_D, q["n_scale_frames"], neutral=nan)
    else:
        sc = {"k": nan, "k_traj": nan, "k_depth": nan, "s_vipe": nan, "r_vipe": nan}
    scale_ok = bool(np.isfinite(sc["k"]) and sc["k"] > 0)
    k = sc["k"] if scale_ok else 1.0
    cs[:, :3, 3] *= k
    Dm = None if D is None else (D * np.float32(k))
    # ---- v1c parts (all scale-free)
    parts = {"xtraj_ba": nan, "xdepth_k": nan, "kdis": nan}
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
    v1c = float(np.nanmax(list(ratios.values()))) if np.isfinite(list(ratios.values())).any() else nan
    # ---- flags (E9) on the label; effective frame rate = clip fps / stride
    fps_eff = job.get("fps_orig", 15) / max(job.get("stride", 1), 1)
    D_flag = Dm if has_d else ref_D  # ViPE depth is metric already; used only for the camera-path / scene-depth ratio
    f = e9_flags(cs, glr["K"], D_flag, glr["W"], fps_eff)
    f["vipe_focal_cv"] = float(wit.get("vipe_focal_cv", np.nan))
    reject_flags = [x for x in q["reject_flags"] if f.get(x)]
    s1 = stage1 if stage1 is not None else {"verdict": "missing" if q.get("require_gate", True) else "skipped"}
    s1 = {"stop": [], "suspect": [], "signals": {}, **s1}
    verdict = s1.get("verdict")
    reg_frac = glr["n_reg"] / max(len(job["frame_idx"]), 1)
    min_cov = q.get("min_depth_cov", 0.9)
    info = [name for name, on in (("suspect_cut", f["jump20"] and not f["jump60"]), ("focal_unobservable", f["tiny_move"]),
                                  ("suspect_zoom", f["vipe_focal_cv"] > 0.07), ("partial_registration", reg_frac < 0.9),
                                  ("depth_missing", not has_d), ("depth_partial", has_d and depth_cov < min_cov),
                                  ("unmapped_images", glr.get("n_unmapped", 0) > 0), ("depth_resized", glr.get("n_depth_resized", 0) > 0)) if on]
    info += [f"s1_{x}" for x in s1.get("suspect", [])]
    # ---- decision: a check that did not run is a reason to reject, not a pass
    why = []
    if not np.isfinite(v1c):
        why.append("v1c_not_checkable")
    elif v1c > 1:
        why.append("v1c_gt_1")
    if q.get("require_all_parts", True):
        why += [f"{s}_not_checkable" for s in thr if not np.isfinite(parts[s]) and not (s == "xdepth_k" and not has_d)]
    why += reject_flags
    if verdict == "stop":
        why += [f"s1_{x}" for x in s1.get("stop", [])] or ["s1_stop"]
    elif verdict not in ("ok", "suspect", "skipped"):
        why.append("s1_gate_missing" if verdict == "missing" else "s1_gate_error")
    if not has_d and not q.get("allow_no_depth", False):
        why.append("depth_missing")
    if not scale_ok:
        why.append("scale_unverified")
    accept = not why
    tier = "T3"
    if accept:
        vals = np.array([ratios[s] for s in thr], float)
        small = bool(np.isfinite(vals).all() and (vals <= q["tier1_part"]).all())
        tier = "T1" if (small and verdict == "ok" and not s1.get("suspect") and has_d and depth_cov >= min_cov and reg_frac >= 0.9) else "T2"
    qc = {"clip": job["clip"], "annotator": "glr", **parts, "v1c": v1c, "v1c_parts": ratios, "flags": f, "reject_flags": reject_flags,
          "info_flags": info, "accept": bool(accept), "tier": tier, "reject_reasons": why,
          "scale": {k_: sc.get(k_) for k_ in ("k", "k_depth", "k_traj", "s_vipe", "r_vipe")}, "stage1": s1,
          "glr": {"n_reg": glr["n_reg"], "n_sampled": len(job["frame_idx"]), "stride": job.get("stride"), "model": glr.get("model"),
                  "depth_source": glr["depth_source"], "depth_cov": depth_cov, "hfov": hf, "mapped_by": glr.get("mapped_by", "name"),
                  "cameras_from": glr.get("cameras_from", ""),
                  "n_unmapped": glr.get("n_unmapped", 0), "n_depth_resized": glr.get("n_depth_resized", 0)}}
    nan11 = np.full((n, 1, 1), np.nan, np.float32)
    label = {"c2w": cs, "K": glr["K_label"], "depth": Dm if Dm is not None else nan11,
             "conf": glr["conf"] if glr.get("conf") is not None else (np.isfinite(Dm).astype(np.float32) if Dm is not None else nan11),
             "scale": k if scale_ok else nan, "k_depth": sc.get("k_depth", nan), "k_traj": sc.get("k_traj", nan), "hfov": hf,
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


def failed_qc(job, status, s1, note=""):
    why = [f"glr_{status}"] + [f"s1_{x}" for x in (s1 or {}).get("stop", [])]
    return {"clip": job["clip"], "annotator": "glr", "accept": False, "tier": "T3", "v1c": float("nan"), "reject_reasons": why,
            "stage1": s1, "glr": {"status": status, "note": note}}


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


def read_gate(rd, rj):
    """gate.json of a result directory, cross-checked with the verdict the worker recorded in result.json. None = no gate.json."""
    gp = Path(rd) / "gate.json"
    s1 = None
    if gp.exists():
        try:
            s1 = json.loads(gp.read_text())
            if not isinstance(s1, dict) or "verdict" not in s1:
                s1 = {"verdict": "error", "error": "gate.json has no verdict"}
        except Exception as e:  # noqa: BLE001
            s1 = {"verdict": "error", "error": f"gate.json unreadable: {e!r}"[:200]}
    rg = rj.get("gate")
    if s1 is None and rg in ("error", "stop"):
        s1 = {"verdict": rg, "stop": [], "suspect": [], "signals": {}, "error": "gate.json missing"}
    elif s1 is not None and rg and s1.get("verdict") != rg:
        s1 = {**s1, "verdict": "error", "error": f"gate.json says {s1.get('verdict')!r} but result.json says {rg!r}"}
    return s1


def collect_clip(out, jd, cfg, witness_dir=None):
    """-> qc dict of one job directory (writes labels_glr/<clip>.npz when a label could be computed)."""
    job = json.loads((jd / "job.json").read_text())
    rd, note = find_result_dir(jd)
    if rd is None:
        return failed_qc(job, "incomplete", None, note)
    rj = json.loads((rd / "result.json").read_text())
    s1 = read_gate(rd, rj)
    status = rj["status"]
    if status in BAD_STATUS:
        return failed_qc(job, status, s1, note)
    glr = read_glr(rd, job, cfg["qc"]["label_long"])
    if not glr["ok"]:
        return failed_qc(job, glr["why"], s1, note)
    wit = load_witness_npz(Path(witness_dir) / f"{job['clip']}.npz") if witness_dir else load_witnesses(out, job, cfg)
    label, qc = qc_one(job, glr, s1, wit, cfg)
    if note:
        qc["glr"]["note"] = note
    np.savez_compressed(out / "labels_glr" / f"{job['clip']}.npz", **label)
    return qc


def main(argv=None):
    ap = argparse.ArgumentParser(description="GLR track phase C: GLRMap results -> labels_glr/, qc_glr/, results_glr.csv")
    ap.add_argument("out", help="run directory (OUT of run_batch_glr.sh)")
    ap.add_argument("--cfg", help="config yaml (default configs/pipeline_glr_v0.yaml); must exist")
    ap.add_argument("--clips", help="comma separated: only (re)collect these clips, keep the other rows of results_glr.csv")
    ap.add_argument("--witness-dir", help="witness bundles DIR/<clip>.npz (see load_witness_npz) instead of the experiments' bench layout")
    a = ap.parse_args(argv)
    out = Path(a.out)
    cfg = C.load_cfg(a.cfg)
    want = set(a.clips.split(",")) if a.clips else None
    (out / "labels_glr").mkdir(exist_ok=True)
    (out / "qc_glr").mkdir(exist_ok=True)
    root = out / "glr_jobs"
    jds = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []   # nothing routed to GLRMap: no glr_jobs/ at all
    rows = {}
    for jd in jds:
        if not (jd / "job.json").exists() or (want and jd.name not in want):
            continue
        clip = json.loads((jd / "job.json").read_text())["clip"]
        try:
            qc = collect_clip(out, jd, cfg, a.witness_dir)
        except MissingDependency as e:
            sys.exit(f"glr_collect: {e}")
        except Exception as e:  # noqa: BLE001
            qc = {"clip": clip, "annotator": "glr", "accept": False, "tier": "T3", "v1c": float("nan"),
                  "reject_reasons": ["collect_error"], "error": repr(e)[:300]}
        C.dump_json(out / "qc_glr" / f"{clip}.json", qc)
        rows[clip] = C.clean({"clip": clip, "annotator": "glr", "tier": qc["tier"], "accept": qc["accept"], "v1c": qc.get("v1c"),
                              "xtraj_ba": qc.get("xtraj_ba"), "xdepth_k": qc.get("xdepth_k"), "kdis": qc.get("kdis"),
                              "reasons": ";".join(qc.get("reject_reasons", [])), "info_flags": ";".join(qc.get("info_flags", [])),
                              "scale": (qc.get("scale") or {}).get("k"), "hfov": (qc.get("glr") or {}).get("hfov"),
                              "s1_gate": (qc.get("stage1") or {}).get("verdict"), "depth_source": (qc.get("glr") or {}).get("depth_source")})
        print(json.dumps(rows[clip]), flush=True)
    path = out / "results_glr.csv"
    keep = {}
    if want and path.exists():   # a partial re-collect must not drop the other clips from the table the merge reads
        with open(path) as fh:
            keep = {r["clip"]: r for r in csv.DictReader(fh) if r["clip"] not in rows}
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, restval="")
        w.writeheader()
        w.writerows(sorted({**keep, **rows}.values(), key=lambda r: r["clip"]))


if __name__ == "__main__":
    main()
