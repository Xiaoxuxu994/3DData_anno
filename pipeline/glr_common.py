"""Shared helpers of the GLR track (GLRMap annotator + 3DData_anno front/back-end filters).

Pure numpy (cv2 only inside `to_grid`). `umeyama`, `traj_pct`, `depth_cmp`, `to_grid` and the metric-scale rule are copies of
vipe_eval/eval_vipe.py, qc/qc_cross.py, bench/eval3d_bench.py and evolution/e3_scale_vote.py (E3b: k = sqrt(k_traj * k_depth)),
so the GLR code runs without those scripts' hard-coded paths; tests/glr/test_fidelity.py extracts the original functions from their
source files (AST, no imports of the hard-coded paths) and compares the copies with them on random inputs. `metric_scale` is
identical to E3b where E3b has evidence and differs only where GLRMap's label is not a Pi3X label (see its docstring).

Config: DEFAULTS below are the single source of truth; configs/pipeline_glr_v0.yaml overrides them (loaded with PyYAML when it is
installed, else with a small parser for the subset that file uses).
"""

import copy
import json
import math
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ANNO_ROOT", "/data/home/z00533370"))
CODE = ROOT / "code"

DEFAULTS = {
    "version": "glr-v0",
    # frames handed to GLRMap: from the 15 fps clip, every `stride`-th frame; n in [min_frames, max_frames]
    "sampling": {"target_frames": 100, "min_frames": 60, "max_frames": 120, "max_stride": 4,
                 "target_step_frac": 0.02, "default_stride": 2},
    # which clips go to GLRMap (static scene + real parallax) and which stay with the feed-forward v1.3 annotator
    "route": {"max_dyn_frac": 0.30, "max_rot_frac": 0.60, "min_parallax_px": 0.5, "max_static_pair_frac": 0.20,
              "min_mag_median_frac": 0.002, "force": None},
    # stage-1 exit gate: defaults live in glr_stage1_gate.DEFAULT_CFG (the worker host only has that file)
    "stage1_gate": {},
    "qc": {"thr": {"xtraj_ba": 3.27, "xdepth_k": 0.194, "kdis": 13.2},  # v1c thresholds (E8), reused unchanged
           "tier1_part": 0.5,        # T1 needs every v1c part finite and <= tier1_part * threshold
           "reject_flags": ["jump60", "fov_out"],
           # GLRMap's failures are silent and mostly about K, so the GLR track is stricter than v1.3 about what it cannot check:
           "require_gate": True,      # no readable stage-1 gate.json (or a gate that errored) -> reject (s1_gate_missing / s1_gate_error)
           "require_all_parts": True,  # a v1c part that cannot be computed (no K witness, no DA3 depth ...) -> reject (<part>_not_checkable)
           "allow_no_depth": False,   # a label without GLRMap depth violates the label contract -> reject (depth_missing)
           "min_depth_cov": 0.9,      # fraction of label frames with depth: below -> info flag depth_partial, and no T1
           "label_long": 512, "qc_long": 512, "n_depth_frames": 8, "n_scale_frames": 10},
}


def _deep_merge(a, b):
    """a with b merged in; the result shares no mutable object with either input."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _scalar(s):
    s = s.strip()
    if s in ("", "~", "null", "Null", "None"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s[0] == s[-1]) and s[0] in "\"'" and len(s) >= 2:
        return s[1:-1]
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", s):
        return float(s)
    return s


def _split_top(s, sep=","):
    out, depth, cur, q = [], 0, "", None
    for ch in s:
        if q:
            cur += ch
            q = None if ch == q else q
        elif ch in "\"'":
            q = ch
            cur += ch
        elif ch in "[{":
            depth += 1
            cur += ch
        elif ch in "]}":
            depth -= 1
            cur += ch
        elif ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _value(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [_value(x) for x in _split_top(s[1:-1])]
    if s.startswith("{") and s.endswith("}"):
        d = {}
        for item in _split_top(s[1:-1]):
            k, v = item.split(":", 1)
            d[k.strip()] = _value(v)
        return d
    return _scalar(s)


def mini_yaml(text):
    """Parse the YAML subset used by configs/pipeline_glr_v0.yaml: nested maps by 2-space indentation, inline [lists] and
    {maps}, scalars, '#' comments. No anchors, multi-line scalars or block lists."""
    root, stack = {}, [(-1, None)]
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = re.sub(r"\s+#.*$", "", raw) if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip(" "))
        key, _, val = line.strip().partition(":")
        while stack and stack[-1][0] >= ind:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            child = {}
            parent[key.strip()] = child
            stack.append((ind, child))
        else:
            parent[key.strip()] = _value(val)
    return root


def load_cfg(path=None):
    """DEFAULTS merged with the YAML at `path`. An explicit `path` must exist (a typo must not silently fall back to the defaults);
    without one, CODE/configs/pipeline_glr_v0.yaml is used if it exists."""
    cfg = _deep_merge(DEFAULTS, {})
    try:
        import glr_stage1_gate as G
        cfg["stage1_gate"] = _deep_merge(G.DEFAULT_CFG, cfg.get("stage1_gate") or {})
    except Exception:  # noqa: BLE001
        pass
    p = Path(path) if path else CODE / "configs" / "pipeline_glr_v0.yaml"
    if path and not p.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    if p.exists():
        text = p.read_text()
        try:
            import yaml
            d = yaml.safe_load(text)
        except ImportError:
            d = mini_yaml(text)
        cfg = _deep_merge(cfg, d if isinstance(d, dict) else {})
    return cfg


# ----------------------------------------------------------------------------- geometry (copies, see module docstring)
def umeyama(src, dst, with_scale=True):
    """s, R, t minimising ||dst - (s R src + t)||; src, dst (N, 3). Copy of vipe_eval/eval_vipe.umeyama."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def traj_pct(pa, pb):
    """Symmetric Sim(3) trajectory ATE as % of trajectory length. Copy of qc/qc_cross.traj_pct."""
    La, Lb = np.linalg.norm(np.diff(pa, axis=0), axis=1).sum(), np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()
    if min(La, Lb) < 1e-9 or not (np.isfinite(pa).all() and np.isfinite(pb).all()):
        return float("nan")
    out = []
    for x, y, L in ((pa, pb, Lb), (pb, pa, La)):
        s, R, t = umeyama(x, y, True)
        out.append(100 * np.sqrt(((s * x @ R.T + t - y) ** 2).sum(1).mean()) / L)
    return float(np.mean(out))


def depth_cmp(Da, Db, static=None):
    """(AbsRel of Da scaled by one median ratio vs Db, |log ratio|). Copy of qc/qc_cross.depth_cmp."""
    m = np.isfinite(Da) & np.isfinite(Db) & (Da > 0) & (Db > 0)
    if static is not None:
        m &= static
    if m.sum() < 1000:
        return float("nan"), float("nan")
    ratio = np.median(Db[m]) / np.median(Da[m])
    return float(np.mean(np.abs(Da[m] * ratio - Db[m]) / Db[m])), float(abs(np.log(ratio)))


def hfov_deg(fx, W):
    return float(np.degrees(2 * np.arctan(W / 2 / fx)))


def qc_grid(h, w, long=512):
    """QC grid: long side `long`, both sides multiples of 8 (copy of v13_label_qc.qc_grid)."""
    s = long / max(h, w)
    return max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)


def to_grid(depth, K, H, W):
    """Nearest-neighbour resize of (N,h,w) depth to (H,W) and the matching K (copy of eval3d_bench.to_grid)."""
    import cv2
    n, h, w = depth.shape
    sx, sy = W / w, H / h
    D = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth])
    K = K.copy()
    K[:, 0, :] *= sx
    K[:, 1, :] *= sy
    K[:, 0, 2] += 0.5 * sx - 0.5
    K[:, 1, 2] += 0.5 * sy - 0.5
    return D, K


def metric_scale(c2w_x, D_x, c2w_ref, D_ref, n_depth_frames=10, neutral=1.0):
    """Scale k that brings annotator X (cameras + depth, arbitrary scale) to the metric reference (ViPE), E3b rule:
         s = Sim(3) scale of X's camera centres onto the reference's (NaN when the reference path is < 2 % of the scene depth
             or the aligned residual is > 20 % of its path),  r = median(D_ref) / median(D_x) over co-valid pixels of
         `n_depth_frames` frames,  k_traj = s (else r, else `neutral`),  k_depth = r (else k_traj),  k = sqrt(k_traj * k_depth).
    All inputs refer to the same frames / grid. Where E3b has evidence this is E3b exactly (tests/glr/test_fidelity.py). Differences,
    all for a label whose depth can be missing (GLRMap) rather than a Pi3X label that always has depth:
      * the `n_depth_frames` frames are spread over the frames where D_x has any finite pixel (all frames when D_x is complete);
      * a missing depth ratio makes k_depth = k_traj (E3b: 1.0, which would halve the log of the trajectory scale; in E3b a missing
        ratio always comes with a missing trajectory vote, so the two rules only differ in the case of the next bullet);
      * the scene depth for the 2 %-path rule comes from the reference depth alone when D_x is None or has no co-valid pixels
        (E3b: no trajectory vote then);
      * `neutral` is what E3b calls 1.0 (no evidence at all). The GLR track passes NaN, because GLRMap's SfM scale is arbitrary:
        1.0 would export SfM units as metres.  D_x may be None (no depth): then only the trajectory vote is used."""
    r = s = float("nan")
    dm = float("inf")
    if D_ref is not None:
        ks = np.linspace(0, len(D_ref) - 1, n_depth_frames).astype(int)
        a = D_ref[ks]
        mr = np.isfinite(a) & (a > 0)
        if mr.sum() > 1000:  # scene depth from the reference alone (replaced by the co-valid median below when X has depth)
            dm = float(np.median(a[mr]))
        if D_x is not None:
            have = np.flatnonzero(np.isfinite(D_x).reshape(len(D_x), -1).any(1))
            if len(have):
                kx = have[np.linspace(0, len(have) - 1, n_depth_frames).astype(int)]
                a, b = D_ref[kx], D_x[kx]
                mk = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)  # co-valid pixels of both depth maps (the E3b rule)
                if mk.sum() > 1000:
                    r = float(np.median(a[mk]) / np.median(b[mk]))
                    dm = float(np.median(a[mk]))
    pb, pm = c2w_x[:, :3, 3], c2w_ref[:, :3, 3]
    Lm = float(np.linalg.norm(np.diff(pm, axis=0), axis=1).sum())
    if np.isfinite(pb).all() and np.isfinite(pm).all() and len(pb) >= 2:
        s_raw, R, t = umeyama(pb, pm, True)
        res = float(np.sqrt(((s_raw * pb @ R.T + t - pm) ** 2).sum(1).mean()) / max(Lm, 1e-9))
    else:  # a NaN camera centre (frame the witness does not cover) makes the SVD fail: no trajectory vote, like traj_pct's guard
        s_raw, res = float("nan"), float("nan")
    move = Lm / dm if np.isfinite(dm) else float("nan")
    s = float(s_raw) if (Lm > 0.02 * dm and res < 0.2 and s_raw > 0) else float("nan")
    k_t = s if np.isfinite(s) else (r if np.isfinite(r) else neutral)
    k_d = r if np.isfinite(r) else k_t
    return {"k": float(math.sqrt(k_t * k_d)), "k_traj": float(k_t), "k_depth": float(k_d), "s_vipe": s, "r_vipe": r,
            "res_vipe": res, "move_vipe": move}


def clean(o):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> python, floats rounded to 6 digits."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def dump_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False))
