"""Profile one ViPE run stage-by-stage and dump intermediate results for documentation.

Usage (inside container, from the vipe repo):
    .venv/bin/python profile_vipe.py VIDEO OUT_DIR
"""

import dataclasses
import functools
import inspect
import json
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

VIDEO, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
DUMP = OUT / "_profile"
DUMP.mkdir(exist_ok=True)
SAMPLE_FRAMES = {0, 60, 121}

T = defaultdict(float)  # label -> seconds (inclusive)
N = defaultdict(int)  # label -> call count
REC: dict = defaultdict(list)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(label):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            sync()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            sync()
            T[label] += time.perf_counter() - t0
            N[label] += 1
            return out

        return wrapper

    return deco


def timed_iter(label):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            it = fn(*a, **kw)
            while True:
                sync()
                t0 = time.perf_counter()
                try:
                    x = next(it)
                except StopIteration:
                    sync()
                    T[label] += time.perf_counter() - t0
                    return
                sync()
                T[label] += time.perf_counter() - t0
                N[label] += 1
                yield x

        return wrapper

    return deco


def patch(obj, name, deco):
    setattr(obj, name, deco(getattr(obj, name)))


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return x


# ---------------------------------------------------------------- imports
from vipe.utils import model_cache as mc  # noqa: E402

_orig_get = mc.ModelCache.get


def _get(self, key, builder):
    def b():
        sync()
        t0 = time.perf_counter()
        out = builder()
        sync()
        T[f"load:{key.split('/')[0]}/{key.split('/')[1] if '/' in key else ''}"] += time.perf_counter() - t0
        return out

    return _orig_get(self, key, b)


mc.ModelCache.get = _get

import vipe.pipeline.processors as P  # noqa: E402
import vipe.priors.track_anything as TA  # noqa: E402
from vipe.priors.track_anything import detector as TAdet  # noqa: E402
from vipe.priors.track_anything import segmentor as TAseg  # noqa: E402
from vipe.priors.track_anything import seg_tracker as TAst  # noqa: E402
from vipe.slam import system as S  # noqa: E402
from vipe.slam.components import backend as SB  # noqa: E402
from vipe.slam.components import frontend as SF  # noqa: E402
from vipe.slam.components import inner_filler as SI  # noqa: E402
from vipe.slam.components import motion_filter as SM  # noqa: E402
from vipe.priors.depth.priorda import PriorDAModel  # noqa: E402
from vipe.priors.depth.unidepth import UniDepth2Model  # noqa: E402
from vipe.utils import io as vio  # noqa: E402

# ---------------------------------------------------------------- GeoCalib
_geo_init = P.GeoCalibIntrinsicsProcessor.__init__


def geo_init(self, *a, **kw):
    sync()
    t0 = time.perf_counter()
    _geo_init(self, *a, **kw)
    sync()
    T["1.geocalib(incl load)"] += time.perf_counter() - t0
    REC["geocalib"] = {"fov_y_deg": float(np.degrees(self.fov_y)), "sample_frame_inds": list(self.sample_frame_inds)}


P.GeoCalibIntrinsicsProcessor.__init__ = geo_init

# ---------------------------------------------------------------- TrackAnything
_ta_iter = P.TrackAnythingProcessor.update_iterator


def ta_iter(self, prev, pass_idx):
    it = _ta_iter(self, prev, pass_idx)
    idx = 0
    while True:
        sync()
        t0 = time.perf_counter()
        try:
            fr = next(it)
        except StopIteration:
            return
        sync()
        T["2.track_anything(total)"] += time.perf_counter() - t0
        if idx in SAMPLE_FRAMES:
            np.savez_compressed(
                DUMP / f"ta_{idx:04d}.npz",
                rgb=(to_np(fr.rgb) * 255).astype(np.uint8),
                instance=to_np(fr.instance).astype(np.uint8),
                mask=to_np(fr.mask).astype(bool),
                intrinsics=to_np(fr.intrinsics),
            )
            REC["phrases"].append({"frame": idx, "phrases": {int(k): v for k, v in fr.instance_phrases.items()}})
        if idx == 0:
            REC["frame0_intrinsics"] = to_np(fr.intrinsics).tolist()
        idx += 1
        yield fr


P.TrackAnythingProcessor.update_iterator = ta_iter

_det = TAdet.Detector.run_grounding_tensor


def det(self, *a, **kw):
    sync()
    t0 = time.perf_counter()
    out = _det(self, *a, **kw)
    sync()
    T["2a.groundingdino"] += time.perf_counter() - t0
    N["2a.groundingdino"] += 1
    try:
        _, boxes, phrases = out
        REC["gdino"].append({"call": N["2a.groundingdino"], "boxes_xyxy": to_np(boxes).round(1).tolist(), "phrases": list(phrases)})
    except Exception as e:  # noqa: BLE001
        REC["gdino"].append({"err": str(e)})
    return out


TAdet.Detector.run_grounding_tensor = det
patch(TAseg.Segmentor, "segment_with_box_tensor", timed("2b.sam"))
patch(TAst.SegTracker, "track", timed("2c.deaot_track"))
patch(TAst.SegTracker, "add_reference", timed("2c.deaot_add_reference"))
if hasattr(TAst.SegTracker, "encode_aot_frames"):
    patch(TAst.SegTracker, "encode_aot_frames", timed("2c.deaot_encode"))

# ---------------------------------------------------------------- SLAM
patch(S.SLAMSystem, "run", timed("3.slam_run(incl init stream)"))
patch(S.SLAMSystem, "_precompute_features", timed("3a.precompute_features(fnet/cnet)"))
patch(S.SLAMSystem, "_add_keyframe", timed("3c.add_keyframe(incl depth prior)"))
patch(SF.SLAMFrontend, "run", timed("3d.frontend"))
patch(SB.SLAMBackend, "run", timed("3e.backend_global_ba"))
patch(SB.SLAMBackend, "run_if_necessary", timed("3e.backend_mid"))
patch(SI.InnerFiller, "compute", timed("3f.inner_filler"))
patch(UniDepth2Model, "estimate", timed("depth:unidepth.estimate"))

# MotionFilter.check: re-exec source with a hook that records the motion score.
src = "".join(l[4:] if l.startswith("    ") else l.lstrip(" ") for l in inspect.getsourcelines(SM.MotionFilter.check)[0])
import re  # noqa: E402

src = re.sub(
    r"^( *)(dense_motion_score = dense_motion_score\.min\(\)\.item\(\))$",
    r"\1\2\n\1_MOTION_REC.append(dense_motion_score)",
    src,
    flags=re.M,
)
assert "_MOTION_REC" in src
SM._MOTION_REC = REC["motion_scores"]
ns: dict = {}
exec(compile(src, SM.__file__, "exec"), SM.__dict__, ns)
SM.MotionFilter.check = timed("3b.motion_filter")(ns["check"])

_slam_run = S.SLAMSystem.run


def slam_run(self, *a, **kw):
    out = _slam_run(self, *a, **kw)
    m = out.slam_map
    xyz, rgb = to_np(m.dense_disp_xyz), to_np(m.dense_disp_rgb)
    sel = np.random.default_rng(0).choice(len(xyz), size=min(len(xyz), 40000), replace=False)
    np.savez_compressed(
        DUMP / "slam.npz",
        c2w=to_np(out.trajectory.matrix()),
        intrinsics=to_np(out.intrinsics),
        kf=np.array(m.dense_disp_frame_inds),
        xyz=xyz[sel],
        rgb=rgb[sel],
        n_points=len(xyz),
        motion=np.array(REC["motion_scores"]),
        slam_hw=np.array([self.config.height, self.config.width]),
    )
    REC["slam"] = {
        "n_keyframes": len(m.dense_disp_frame_inds),
        "keyframes": list(map(int, m.dense_disp_frame_inds)),
        "final_intrinsics": to_np(out.intrinsics).tolist(),
        "slam_hw": [int(self.config.height), int(self.config.width)],
        "n_map_points": int(len(xyz)),
    }
    return out


S.SLAMSystem.run = slam_run

# ---------------------------------------------------------------- depth post
patch(P.AdaptiveDepthProcessor, "__init__", timed("4.depth_init(incl load)"))
patch(P.AdaptiveDepthProcessor, "update_iterator", timed_iter("4.depth_post(total)"))

_pda = PriorDAModel.estimate
_pda_i = [0]


def pda(self, src):
    sync()
    t0 = time.perf_counter()
    out = _pda(self, src)
    sync()
    T["4b.priorda"] += time.perf_counter() - t0
    N["4b.priorda"] += 1
    i = _pda_i[0]
    if i in SAMPLE_FRAMES:
        d = {f.name: to_np(getattr(src, f.name)) for f in dataclasses.fields(src) if isinstance(getattr(src, f.name), torch.Tensor)}
        d.update({"out_" + f.name: to_np(getattr(out, f.name)) for f in dataclasses.fields(out) if isinstance(getattr(out, f.name), torch.Tensor)})
        np.savez_compressed(DUMP / f"priorda_{i:04d}.npz", **d)
    _pda_i[0] += 1
    return out


PriorDAModel.estimate = pda

_align = P.align_inv_depth_to_depth


def align(*a, **kw):
    sync()
    t0 = time.perf_counter()
    out = _align(*a, **kw)
    sync()
    T["4c.align_lstsq"] += time.perf_counter() - t0
    REC["align_sb"].append([float(to_np(out[1]).reshape(-1)[0]), float(to_np(out[2]).reshape(-1)[0])])
    return out


P.align_inv_depth_to_depth = align

_cvda = P.AdaptiveDepthProcessor._compute_video_da


def cvda(self, it):
    res, frames = _cvda(self, it)
    for i in SAMPLE_FRAMES:
        if i < len(res):
            np.save(DUMP / f"vda_{i:04d}.npy", to_np(res[i]))
    return res, frames


P.AdaptiveDepthProcessor._compute_video_da = timed("4a.video_depth_anything")(cvda)

patch(vio, "save_artifacts", timed("5.save_artifacts(incl depth post)"))

# ---------------------------------------------------------------- run
from vipe.cli.main import main  # noqa: E402

sync()
t_all = time.perf_counter()
try:
    main(["infer", VIDEO, "-o", str(OUT)], standalone_mode=False)
finally:
    total = time.perf_counter() - t_all
    REC["timing"] = {k: round(v, 3) for k, v in sorted(T.items())}
    REC["counts"] = dict(N)
    REC["total_wall_s"] = round(total, 3)
    REC["gpu"] = torch.cuda.get_device_name(0)
    REC["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (DUMP / "profile.json").write_text(json.dumps(REC, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: REC[k] for k in ["timing", "counts", "total_wall_s", "peak_mem_gb"]}, indent=1))
