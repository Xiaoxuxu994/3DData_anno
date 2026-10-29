"""Stand-in for GLRMap stage 1 in the worker tests: python _stub_stage1.py IMAGES S1_DIR [good|planar|fail|nodb|garbage|few|two_models]
The model registers every image of the job (IMAGES/../job.json), like a stage 1 that worked."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fix  # noqa: E402

images, s1, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "good"
if kind == "fail":
    sys.exit(7)
job = images.parent / "job.json"
names = json.loads(job.read_text())["image_names"] if job.exists() else sorted(p.name for p in images.glob("*.jpg"))
n = max(len(names), 8)
names = (names + [f"{i + 1:06d}.jpg" for i in range(len(names), n)])[:n]
if kind == "few":   # stage 1 that registered only a third of the frames
    names = names[:max(len(names) // 3, 8)]
    n = len(names)
K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
_fix.write_text_model(s1 / "sparse" / "0", _fix.look_at_rig(n, 0.25), K, 640, 360, _fix.scene_points(3000), names)
if kind == "two_models":   # a second, deeper model that must not be the one the gate looks at
    _fix.write_text_model(s1 / "deeper" / "x" / "sparse" / "0", _fix.look_at_rig(n, 0.25), K, 640, 360, _fix.scene_points(300), names)
if kind != "nodb":
    _fix.write_db(s1 / "database.db", n, planar_frac=1.0 if kind == "planar" else 0.0)
if kind == "garbage":   # a database file that is not a database: the gate must fail loudly, never pass
    (s1 / "database.db").write_bytes(b"this is not sqlite")
