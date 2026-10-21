"""E6b: persistent ViPE worker — build the pipeline once, then run many frame directories through it.

Usage: CUDA_VISIBLE_DEVICES=g code/vipe/.venv/bin/python vipe_worker.py TAG OUT_DIR LIST_FILE [WORKER N_WORKERS]
LIST_FILE: one frame directory per line (artifact name = directory name). Same overrides as `vipe infer --image-dir`
(pipeline=default, save_artifacts=true, save_viz=false). Appends {clip, sec, n_frames} to OUT_DIR/logs/status.jsonl;
`sec` is wall time of pipeline.run including frame reading — model loading is paid once (logged as load_sec).
Env: VIPE_PIPELINE (default "default"), VIPE_OVERRIDES (extra hydra overrides, ";"-separated),
     VIPE_DAV3_WEIGHTS (local DA3METRIC-LARGE safetensors for slam.keyframe_depth=dav3; HF is too slow from this host).
"""

import os

import json
import sys
import time
from pathlib import Path

from vipe import make_pipeline
from vipe.config import parse_typed_config
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.logging import configure_logging

TAG, OUT, LST = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
WID, NW = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
configure_logging()
dirs = [Path(x) for x in LST.read_text().split()][WID::NW]
(OUT / "logs").mkdir(parents=True, exist_ok=True)
t0 = time.time()
if os.environ.get("VIPE_DAV3_WEIGHTS"):
    import vipe.priors.depth.dav3 as _d3
    _init = _d3.DepthAnything3Model.__init__
    _d3.DepthAnything3Model.__init__ = lambda self, weights_path=None: _init(self, os.environ["VIPE_DAV3_WEIGHTS"])
extra = [o for o in os.environ.get("VIPE_OVERRIDES", "").split(";") if o]
args = parse_typed_config("default", hydra_args=[f"pipeline={os.environ.get('VIPE_PIPELINE', 'default')}",
                                                f"pipeline.output.path={OUT}", "pipeline.output.save_artifacts=true",
                                                "pipeline.output.save_viz=false"] + extra)
pipe = make_pipeline(args.pipeline)
load_sec = time.time() - t0
first = True
for d in dirs:
    if (OUT / "pose" / f"{d.name}.npz").exists():
        continue
    t1 = time.time()
    rec = {"clip": d.name, "tag": TAG, "worker": WID, "n_frames": len(list(d.glob("*.jpg")))}
    try:
        stream = ProcessedVideoStream(FrameDirStream(d), []).cache(desc="Reading image frames")
        pipe.run(stream)
        rec.update(ok=True)
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=repr(e)[:300])
    rec.update(sec=round(time.time() - t1, 2), load_sec=round(load_sec, 1) if first else 0.0)
    first = False
    with open(OUT / "logs" / "status.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
print("WORKER_DONE", TAG, WID, flush=True)
