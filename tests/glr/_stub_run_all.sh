#!/bin/bash
# stand-in for GLRMap's data_generate_demo/run_all.sh (reads IMAGES, OUTPUT from the environment)
HERE=$(cd "$(dirname "$0")" && pwd)
${PY_GLR:-python3} - <<PY
import json, os, sys
sys.path.insert(0, "$HERE")
import numpy as np, _fix
from pathlib import Path
out, images = Path(os.environ["OUTPUT"]), Path(os.environ["IMAGES"])
job = images.parent / "job.json"
names = json.loads(job.read_text())["image_names"] if job.exists() else sorted(p.name for p in images.glob("*.jpg"))
n = max(len(names), 8)
names = (names + [f"{i+1:06d}.jpg" for i in range(len(names), n)])[:n]
K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
s1 = out / "base" / "scene"
_fix.write_text_model(s1 / "sparse" / "0", _fix.look_at_rig(n, 0.25), K, 640, 360, _fix.scene_points(3000), names)
_fix.write_db(s1 / "database.db", n)
(out / "pda" / "fused").mkdir(parents=True)
np.savez(out / "pda" / "fused" / "000001.npz", fine=np.ones((36, 64), np.float32))
PY
exit ${STUB_RC:-0}
