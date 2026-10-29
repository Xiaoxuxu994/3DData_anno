"""Stand-in for GLRMap stage 1 in the worker tests: python _stub_stage1.py IMAGES S1_DIR [good|planar|fail]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fix  # noqa: E402

images, s1, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "good"
if kind == "fail":
    sys.exit(7)
names = sorted(p.name for p in images.glob("*.jpg"))
n = max(len(names), 8)
names = (names + [f"{i + 1:06d}.jpg" for i in range(len(names), n)])[:n]
K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
_fix.write_text_model(s1 / "sparse" / "0", _fix.look_at_rig(n, 0.25), K, 640, 360, _fix.scene_points(3000), names)
_fix.write_db(s1 / "database.db", n, planar_frac=1.0 if kind == "planar" else 0.0)
