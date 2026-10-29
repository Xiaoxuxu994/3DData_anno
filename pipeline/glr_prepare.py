"""GLR track, phase A (H200 side): route the clips that passed the filter and write one job package per GLRMap clip.

Usage:  python glr_prepare.py OUT route [--cfg FILE] [--vlm TAG] [--glr-all]
        python glr_prepare.py OUT jobs  [--cfg FILE] [--copy] [--clips a,b]
A job package is built in a hidden temporary directory and renamed to glr_jobs/<clip> only when complete (a crash leaves no half-written
job); a clip that fails (no frames, unreadable files ...) is reported and skipped, the others are still prepared, and the exit code is 1.
Inputs  (written by pipeline/run_batch.sh; OUT = a run directory):
  OUT/bench/clips.json        clips that passed the filter (the annotate list) - OUT/bench/clips_all.json has every clip
  OUT/bench/frames/<clip>/    15 fps frames 000000.jpg ...
  OUT/pool/<clip>/scores.json filter stage scores (flow / rules / calib / vlm_<tag>)
  OUT/filter/decisions.csv    filter verdicts (clips missing here, e.g. --skip-filter runs, count as passed)
  OUT/bench/{geocalib,pi3x,da3}  image-only K sources (give the K prior / K witness of the job)
Outputs:
  route -> OUT/filter/glr_route.csv  clip, route (reject | glr | v13), reasons, the routing features
  jobs  -> OUT/glr_jobs/<clip>/{images/000001.jpg ..., job.json}   images = every `stride`-th frame (symlinks; use
           `rsync -aL` to move a job to the GLRMap host, or --copy)

Routing (configs/pipeline_glr_v0.yaml `route`, provisional until calibrated on the stage-1 corpus, docs/GLR_INTEGRATION.md):
GLRMap only pays off when stage 1 (SIFT + global SfM, camera fixed afterwards) can succeed: a static scene seen with real parallax.
Clips that pass the filter but are rotation-dominated, dynamic, slow-moving or driving/first-person footage stay with the
feed-forward v1.3 annotator (route v13); the filter's own rejects stay rejects.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glr_common as C  # noqa: E402


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def vlm_block(scores, tag=None):
    keys = [k for k in scores if k.startswith("vlm_")]
    if tag:
        keys = [k for k in keys if k == f"vlm_{tag}"] or keys
    return scores[keys[0]] if keys and isinstance(scores[keys[0]], dict) else {}


def route_clip(scores, cfg, filter_pass=True, vlm_tag=None, force=None):
    """-> (route, reasons, features). Missing stage scores never fire a rule (same convention as filter/decide.py)."""
    r = cfg["route"]
    f = {"dyn_frac": _get(scores, "flow", "dyn_frac"), "rot_frac": _get(scores, "flow", "rot_frac"),
         "parallax_px": _get(scores, "flow", "parallax_px"), "static_pair_frac": _get(scores, "flow", "static_pair_frac"),
         "mag_median_frac": _get(scores, "flow", "mag_median_frac"), "lum_mean": _get(scores, "rules", "lum_mean")}
    v = vlm_block(scores, vlm_tag)
    f["vlm_scene"], f["vlm_dynamic"], f["vlm_motion"] = v.get("scene_category"), v.get("dynamic_level"), v.get("camera_motion")
    if not filter_pass:
        return "reject", ["filter"], f
    why = []
    if f["dyn_frac"] is not None and f["dyn_frac"] > r["max_dyn_frac"]:
        why.append("dynamic")
    if f["rot_frac"] is not None and f["rot_frac"] > r["max_rot_frac"]:
        why.append("rotation_dominant")
    if f["parallax_px"] is not None and f["parallax_px"] < r["min_parallax_px"]:
        why.append("low_parallax")
    if f["static_pair_frac"] is not None and f["static_pair_frac"] > r["max_static_pair_frac"]:
        why.append("slow_or_static_segments")
    if f["mag_median_frac"] is not None and f["mag_median_frac"] < r["min_mag_median_frac"]:
        why.append("too_slow")
    if f["vlm_dynamic"] in ("medium", "high"):
        why.append("vlm_dynamic")
    if f["vlm_scene"] in ("driving", "embodied", "indoor_dynamic", "outdoor_dynamic"):
        why.append(f"vlm_{f['vlm_scene']}")
    forced = force or r.get("force")
    if forced in ("glr", "v13"):
        return forced, (["forced"] + why), f
    return ("v13" if why else "glr"), why, f


def choose_stride(n_frames, mag_median_frac, cfg):
    """Every `stride`-th frame of the 15 fps clip goes to GLRMap: n in [min_frames, max_frames] and the motion between
    consecutive GLRMap frames near target_step_frac of the image width (mag_median_frac = median mean-|flow| per 3 frames,
    fraction of width, from filter stage s4). Without flow stats: default_stride when that keeps n in the window, otherwise the
    stride whose n is closest to target_frames. A clip too long for max_stride to reach max_frames gets a larger stride (SfM cost
    grows quickly with the image count; v1.3 clips are 12 s = 180 frames, so this only matters for custom --max-frames)."""
    s = cfg["sampling"]
    lo, hi, tgt, smax = int(s["min_frames"]), int(s["max_frames"]), s["target_frames"], int(s["max_stride"])
    n_of = lambda k: math.ceil(n_frames / k)  # noqa: E731

    def closest():
        k = min(range(1, smax + 1), key=lambda k: abs(n_of(k) - tgt))
        while n_of(k) > hi:  # even max_stride leaves too many images
            k += 1
        return k

    if mag_median_frac is None:
        d = int(s["default_stride"])
        return d if lo <= n_of(d) <= hi else closest()
    per_frame = mag_median_frac / 3.0
    best = None
    for stride in range(1, smax + 1):
        n = n_of(stride)
        step = per_frame * stride
        if n < lo or (n > hi and not (stride == 1 and n <= 2 * hi and per_frame * 2 > 2.5 * s["target_step_frac"])):
            continue  # stride 1 may exceed max_frames (up to twice) only when stride 2 would move more than 5 % of the width per step
        score = abs(math.log(max(step, 1e-6) / s["target_step_frac"]))
        if best is None or score < best[0]:
            best = (score, stride)
    return best[1] if best is not None else closest()


def k_prior(bench, clip):
    """Median HFOV (deg) of the image-only K sources GeoCalib / Pi3X / DA3 (the v1.3 K-gate sources) -> dict or None."""
    h = {}
    g = bench / "geocalib" / f"{clip}.json"
    if g.exists():
        h["geocalib"] = json.loads(g.read_text()).get("hfov")
    for m in ("pi3x", "da3"):
        p = bench / m / clip / "pred.npz"
        if p.exists():
            z = np.load(p)
            h[m] = C.hfov_deg(float(np.median(z["K"][:, 0, 0])), z["depth"].shape[2])
    vals = [x for x in h.values() if x is not None and np.isfinite(x)]
    return {"hfov_deg": float(np.median(vals)), "sources": h} if vals else None


def build_job(out, bench, c, route_row, cfg, copy=False):
    """Write glr_jobs/<clip>/{images/, job.json} atomically -> (stride, n_images). Raises on any problem; leaves no partial job behind."""
    name = c["clip"]
    frames = sorted((bench / "frames" / name).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {bench / 'frames' / name}")
    mag = route_row.get("mag_median_frac")
    mag = float(mag) if mag not in (None, "", "None") else None
    stride = choose_stride(len(frames), mag, cfg)
    idx = list(range(0, len(frames), stride))
    try:
        kp = k_prior(bench, name)
    except Exception as e:  # noqa: BLE001  - the K prior is only the gate's witness; a broken witness file must not block the job
        print(f"warning: no K prior for {name} ({e!r})")
        kp = None
    job = {"clip": name, "annotator": "glr", "version": cfg["version"], "n_orig": len(frames),
           "fps_orig": c.get("fps_out", 15), "stride": stride, "frame_idx": idx,
           "image_names": [f"{k + 1:06d}.jpg" for k in range(len(idx))], "hw": c.get("hw"),
           "route_reasons": route_row.get("reasons", ""), "k_prior": kp, "stage1_gate_cfg": cfg["stage1_gate"]}
    root = out / "glr_jobs"
    root.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    try:
        (tmp / "images").mkdir()
        for k, i in enumerate(idx):
            dst = tmp / "images" / f"{k + 1:06d}.jpg"
            if copy:
                dst.write_bytes(frames[i].read_bytes())
            else:
                dst.symlink_to(frames[i].resolve())
        C.dump_json(tmp / "job.json", job)
        final = root / name
        if final.exists():  # a half-written job of an older version (images/ without job.json): move it aside, never delete
            stale, n = root / f".{name}.stale", 0
            while stale.exists():
                n += 1
                stale = root / f".{name}.stale{n}"
            final.rename(stale)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)   # our own temporary directory: symlinks / copies made a moment ago
        raise
    return stride, len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GLR track phase A: route the clips and write the GLRMap job packages")
    ap.add_argument("out", help="run directory (OUT of run_batch.sh)")
    ap.add_argument("mode", choices=("route", "jobs"))
    ap.add_argument("--cfg", help="config yaml (default configs/pipeline_glr_v0.yaml); must exist")
    ap.add_argument("--vlm", help="route: VLM tag whose verdict is used (default: the first vlm_* block)")
    ap.add_argument("--glr-all", action="store_true", help="route: send every clip that passed the filter to GLRMap")
    ap.add_argument("--copy", action="store_true", help="jobs: copy the frames instead of symlinking them")
    ap.add_argument("--clips", help="jobs: comma separated clips")
    a = ap.parse_args(argv)
    out, mode = Path(a.out), a.mode
    cfg = C.load_cfg(a.cfg)
    B, P = out / "bench", out / "pool"
    allc = json.loads((B / "clips_all.json").read_text()) if (B / "clips_all.json").exists() else json.loads((B / "clips.json").read_text())
    passed_names = {c["clip"] for c in json.loads((B / "clips.json").read_text())}
    dec = {}
    if (out / "filter" / "decisions.csv").exists():
        with open(out / "filter" / "decisions.csv") as fh:
            dec = {r["clip"]: r for r in csv.DictReader(fh)}

    if mode == "route":
        rows = []
        for c in allc:
            name = c["clip"]
            sp = P / name / "scores.json"
            scores = json.loads(sp.read_text()) if sp.exists() else {}
            ok = (dec[name]["decision"] == "pass") if name in dec else (name in passed_names)
            route, why, f = route_clip(scores, cfg, ok, a.vlm, "glr" if a.glr_all else None)
            rows.append({"clip": name, "route": route, "reasons": ";".join(why), **f,
                         "filter_reasons": dec.get(name, {}).get("reason", "")})
        (out / "filter").mkdir(exist_ok=True)
        with open(out / "filter" / "glr_route.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["clip", "route"])
            w.writeheader()
            w.writerows(rows)
        cnt = {k: sum(r["route"] == k for r in rows) for k in ("reject", "glr", "v13")}
        print(json.dumps(cnt))
        return 0
    with open(out / "filter" / "glr_route.csv") as fh:
        routes = {r["clip"]: r for r in csv.DictReader(fh)}
    want = set(a.clips.split(",")) if a.clips else None
    n_jobs, failed = 0, []
    for c in allc:
        name = c["clip"]
        if routes.get(name, {}).get("route") != "glr" or (want and name not in want):
            continue
        if (out / "glr_jobs" / name / "job.json").exists():
            print("exists, skipped:", name)
            continue
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            print(f"warning: clip name {name!r} has characters outside [A-Za-z0-9._-]; the worker handles them, GLRMap's own scripts may not")
        try:
            stride, n = build_job(out, B, c, routes[name], cfg, a.copy)
        except Exception as e:  # noqa: BLE001
            print(f"job FAILED: {name}: {e!r}")
            failed.append(name)
            continue
        n_jobs += 1
        print(name, "stride", stride, "frames", n)
    print("jobs written:", n_jobs)
    if failed:
        print("jobs failed:", ",".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
