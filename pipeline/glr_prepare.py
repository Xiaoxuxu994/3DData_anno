"""GLR track, phase A (H200 side): route the clips that passed the filter and write one job package per GLRMap clip.

Usage:  python glr_prepare.py OUT route [--cfg FILE] [--vlm TAG] [--glr-all]
        python glr_prepare.py OUT jobs  [--cfg FILE] [--copy] [--clips a,b]
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

import csv
import json
import math
import os
import sys
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
    fraction of width, from filter stage s4). Without flow stats: default_stride."""
    s = cfg["sampling"]
    if mag_median_frac is None:
        return int(s["default_stride"])
    per_frame = mag_median_frac / 3.0
    best = None
    for stride in range(1, int(s["max_stride"]) + 1):
        n = math.ceil(n_frames / stride)
        step = per_frame * stride
        if n < s["min_frames"] or (n > s["max_frames"] and not (stride == 1 and per_frame * 2 > 2.5 * s["target_step_frac"])):
            continue  # stride 1 may exceed max_frames only when stride 2 would move more than 5 % of the width per step
        score = abs(math.log(max(step, 1e-6) / s["target_step_frac"]))
        if best is None or score < best[0]:
            best = (score, stride)
    if best is not None:
        return best[1]
    return min(range(1, int(s["max_stride"]) + 1), key=lambda k: abs(math.ceil(n_frames / k) - s["target_frames"]))


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


def main():
    a = sys.argv[1:]
    out, mode = Path(a[0]), a[1]
    opt = dict(x[2:].split("=", 1) if "=" in x else (x[2:], "1") for x in a[2:] if x.startswith("--"))
    cfg = C.load_cfg(opt.get("cfg"))
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
            route, why, f = route_clip(scores, cfg, ok, opt.get("vlm"), "glr" if "glr-all" in opt else None)
            rows.append({"clip": name, "route": route, "reasons": ";".join(why), **f,
                         "filter_reasons": dec.get(name, {}).get("reason", "")})
        (out / "filter").mkdir(exist_ok=True)
        with open(out / "filter" / "glr_route.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["clip", "route"])
            w.writeheader()
            w.writerows(rows)
        cnt = {k: sum(r["route"] == k for r in rows) for k in ("reject", "glr", "v13")}
        print(json.dumps(cnt))
    elif mode == "jobs":
        with open(out / "filter" / "glr_route.csv") as fh:
            routes = {r["clip"]: r for r in csv.DictReader(fh)}
        want = set(opt["clips"].split(",")) if "clips" in opt else None
        n_jobs = 0
        for c in allc:
            name = c["clip"]
            if routes.get(name, {}).get("route") != "glr" or (want and name not in want):
                continue
            jd = out / "glr_jobs" / name
            if (jd / "job.json").exists():
                print("exists, skipped:", name)
                continue
            frames = sorted((B / "frames" / name).glob("*.jpg"))
            mag = routes[name].get("mag_median_frac")
            mag = float(mag) if mag not in (None, "", "None") else None
            stride = choose_stride(len(frames), mag, cfg)
            idx = list(range(0, len(frames), stride))
            (jd / "images").mkdir(parents=True)
            for k, i in enumerate(idx):
                dst = jd / "images" / f"{k + 1:06d}.jpg"
                if "copy" in opt:
                    dst.write_bytes(frames[i].read_bytes())
                else:
                    dst.symlink_to(frames[i].resolve())
            job = {"clip": name, "annotator": "glr", "version": cfg["version"], "n_orig": len(frames),
                   "fps_orig": c.get("fps_out", 15), "stride": stride, "frame_idx": idx,
                   "image_names": [f"{k + 1:06d}.jpg" for k in range(len(idx))], "hw": c.get("hw"),
                   "route_reasons": routes[name].get("reasons", ""), "k_prior": k_prior(B, name),
                   "stage1_gate_cfg": cfg["stage1_gate"]}
            C.dump_json(jd / "job.json", job)
            n_jobs += 1
            print(name, "stride", stride, "frames", len(idx))
        print("jobs written:", n_jobs)
    else:
        sys.exit("mode must be route or jobs")


if __name__ == "__main__":
    main()
