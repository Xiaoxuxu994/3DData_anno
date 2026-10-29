"""tools/stage1_sfm.py - pycolmap replica of GLRMap stage 1 (frames -> SIFT -> sequential matching -> global SfM).

GLRMap's reconstruction_flow.glomap (as described in the evaluation sessions): ffmpeg frames at 10 fps (-q:v 2) ->
colmap feature_extractor (SIFT, --ImageReader.single_camera 1, camera_model PINHOLE) -> sequential_matcher (overlap 20,
quadratic_overlap 0, loop_detection 0, guided_matching 1) -> glomap mapper (optimize_intrinsics 1, optimize_principal_point 0,
ba_iteration_num 4, retriangulation 2). This script does the same with pycolmap's global_mapping, so it needs no GLRMap
code, and adds the knobs the ablations need (camera model, initial focal, fixed intrinsics, seed, image size).

  python stage1_sfm.py --out OUT (--video V --start S --duration D | --images DIR [--stride N]) [options]
    --fps 10 --long 1920          frame extraction (long side cap; 0 = keep); --images: every N-th image of DIR, resized to --long
    --camera-model PINHOLE|SIMPLE_PINHOLE|SIMPLE_RADIAL
    --overlap 20 --guided 1       sequential matching
    --focal-w F                   initial focal as a multiple of the image WIDTH (default: COLMAP's 1.2 x max(W,H))
    --fixed-focal 1               do not refine focal / principal point / extra params (needs --focal-w)
    --seed N                      random seed of the global mapper (-1 = nondeterministic)
    --threads 4  --max-features 8192
    --reuse-db DIR                reuse images/ + database.db (features + matches) of an earlier run, map only; with it
                                  --db-camera PINHOLE|SIMPLE_PINHOLE rewrites the shared camera (fx=fy for SIMPLE_PINHOLE) and --focal-w sets
                                  its initial focal (x image width), so camera-model / K-prior ablations share one expensive matching run
    --mapper global|incremental   (incremental = the 'reference' mode used for the silver reference)
    --matcher sequential|exhaustive
Outputs: OUT/images/, OUT/database.db, OUT/sparse/<k>/, OUT/stage1.json (summary, timings, camera, gate-relevant stats)
The summary uses glr_stage1_gate.summarize_model so that the gate and this script read the model the same way.
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import glr_stage1_gate as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--video")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--duration", type=float, default=10.0)
ap.add_argument("--images")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--long", type=int, default=1920)
ap.add_argument("--camera-model", default="PINHOLE")
ap.add_argument("--overlap", type=int, default=20)
ap.add_argument("--guided", type=int, default=1)
ap.add_argument("--focal-w", type=float, default=0.0)
ap.add_argument("--fixed-focal", type=int, default=0)
ap.add_argument("--seed", type=int, default=-1)
ap.add_argument("--threads", type=int, default=int(os.environ.get("ANNO_THREADS", 4)))
ap.add_argument("--max-features", type=int, default=8192)
ap.add_argument("--reuse-db")
ap.add_argument("--db-camera", default="")
ap.add_argument("--mapper", default="global")
ap.add_argument("--matcher", default="sequential")
ap.add_argument("--tag", default="")
a = ap.parse_args()

OUT = Path(a.out)
if OUT.exists() and any(OUT.iterdir()):
    sys.exit(f"{OUT} exists and is not empty (never overwrite; pick a new directory)")
OUT.mkdir(parents=True, exist_ok=True)
IM, DB, SP = OUT / "images", OUT / "database.db", OUT / "sparse"
SP.mkdir()
T = {}
t0 = time.time()


def lap(name):
    global t0
    T[name] = round(time.time() - t0, 2)
    t0 = time.time()


MODEL_ID = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}
# parameter layout of the COLMAP camera models a database can hold (model id -> names); the first parameter is always a focal length
LAYOUT = {0: ("f", "cx", "cy"), 1: ("fx", "fy", "cx", "cy"), 2: ("f", "cx", "cy", "k1"), 3: ("f", "cx", "cy", "k1", "k2"),
          4: ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")}


def rewrite_camera(db, model, focal_w):
    """Rewrite the (single, shared) camera of a copied database. `model` PINHOLE / SIMPLE_PINHOLE converts the camera (distortion is
    dropped, the principal point is kept); an empty `model` keeps the camera model and only sets the initial focal (focal_w x width)."""
    con = sqlite3.connect(str(db))
    cid, mid, w, h, blob = con.execute("select camera_id, model, width, height, params from cameras").fetchone()
    if mid not in LAYOUT:
        con.close()
        sys.exit(f"{db}: camera model id {mid} is not supported by --reuse-db rewriting (known: {sorted(LAYOUT)})")
    names = LAYOUT[mid]
    p = np.frombuffer(blob, np.float64).copy()
    cx, cy = p[names.index("cx")], p[names.index("cy")]
    fx = focal_w * w if focal_w > 0 else p[0]
    if model:
        if model not in MODEL_ID:
            con.close()
            sys.exit(f"--db-camera must be one of {sorted(MODEL_ID)}")
        mid = MODEL_ID[model]
        p = np.array([fx, cx, cy] if model == "SIMPLE_PINHOLE" else [fx, fx, cx, cy], np.float64)
    else:
        p[0] = fx
        if names[1] == "fy":
            p[1] = fx
    con.execute("update cameras set model=?, params=?, prior_focal_length=0 where camera_id=?", (int(mid), p.tobytes(), cid))
    con.commit()
    con.close()


# ---------------------------------------------------------------- frames / database
if a.reuse_db:
    R = Path(a.reuse_db)
    shutil.copytree(R / "images", IM)
    shutil.copy(R / "database.db", DB)
    if a.db_camera or a.focal_w > 0:
        rewrite_camera(DB, a.db_camera, a.focal_w)
    lap("reuse")
else:
    IM.mkdir()
    if a.images:
        import cv2
        srcs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[::max(a.stride, 1)]
        for i, p in enumerate(srcs):
            im = cv2.imread(str(p))
            if im is None:
                sys.exit(f"cannot read image {p}")
            if a.long and max(im.shape[:2]) > a.long:
                sc = a.long / max(im.shape[:2])
                im = cv2.resize(im, (int(round(im.shape[1] * sc / 2)) * 2, int(round(im.shape[0] * sc / 2)) * 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(IM / f"{i + 1:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        vf = f"fps={a.fps:g}"
        if a.long:
            vf += f",scale='if(gt(iw,ih),min({a.long},iw),-2)':'if(gt(iw,ih),-2,min({a.long},ih))':flags=lanczos"
        subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{a.start}", "-i",
                        a.video, "-t", f"{a.duration}", "-vf", vf, "-q:v", "2", "-start_number", "1", str(IM / "%06d.jpg")],
                       check=True)
    lap("frames")
    n_img = len(list(IM.glob("*.jpg")))
    ropt = pycolmap.ImageReaderOptions()
    ropt.camera_model = a.camera_model
    if a.focal_w > 0:
        from PIL import Image
        w0, h0 = Image.open(sorted(IM.glob("*.jpg"))[0]).size
        ropt.default_focal_length_factor = a.focal_w * w0 / max(w0, h0)
    eopt = pycolmap.FeatureExtractionOptions()
    eopt.num_threads = a.threads
    eopt.sift.max_num_features = a.max_features
    pycolmap.extract_features(str(DB), str(IM), camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ropt,
                              extraction_options=eopt, device=pycolmap.Device.cpu)
    lap("extract")
    mopt = pycolmap.FeatureMatchingOptions()
    mopt.num_threads = a.threads
    mopt.guided_matching = bool(a.guided)
    if a.matcher == "sequential":
        popt = pycolmap.SequentialPairingOptions()
        popt.overlap = a.overlap
        popt.quadratic_overlap = False
        popt.loop_detection = False
        popt.num_threads = a.threads
        pycolmap.match_sequential(str(DB), matching_options=mopt, pairing_options=popt, device=pycolmap.Device.cpu)
    else:
        pycolmap.match_exhaustive(str(DB), matching_options=mopt, device=pycolmap.Device.cpu)
    lap("match")

# ---------------------------------------------------------------- mapping
if a.mapper == "global":
    o = pycolmap.GlobalPipelineOptions()
    o.num_threads = a.threads
    o.mapper.num_threads = a.threads
    o.random_seed = a.seed
    o.mapper.random_seed = a.seed
    ba = o.mapper.bundle_adjustment
    ba.refine_focal_length = not a.fixed_focal
    ba.refine_principal_point = False
    ba.refine_extra_params = not a.fixed_focal
    recs = pycolmap.global_mapping(str(DB), str(IM), str(SP), o)
else:
    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads = a.threads
    o.random_seed = a.seed
    o.ba_refine_focal_length = not a.fixed_focal
    o.ba_refine_principal_point = False
    o.ba_refine_extra_params = not a.fixed_focal
    recs = pycolmap.incremental_mapping(str(DB), str(IM), str(SP), o)
lap("map")

# ---------------------------------------------------------------- summary
res = {"args": vars(a), "timing_s": T, "n_models": len(recs)}
if recs:
    best = max(recs, key=lambda k: recs[k].num_reg_images())
    mdir = SP / str(best)
    mdir.mkdir(exist_ok=True)
    recs[best].write(str(mdir))
    res["model_dir"] = str(mdir)
    res["model"] = G.summarize_model(str(mdir))
res["db"] = G.summarize_db(str(DB))
res["n_images"] = len(list(IM.glob("*.jpg")))
(OUT / "stage1.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res.get(k) for k in ("n_images", "n_models", "timing_s")}),
      json.dumps({k: (res.get("model") or {}).get(k) for k in ("n_reg", "n_points", "reproj", "focal_w", "fx_fy")}), flush=True)
