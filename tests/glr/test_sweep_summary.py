import importlib.util
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix
import glr_stage1_gate as G

REPO = Path(__file__).resolve().parents[2]
HAS_PYCOLMAP = importlib.util.find_spec("pycolmap") is not None


def make_run(root, clip, run, c2w, K, names, stride, planar=0.0, cam="PINHOLE", scale=1.0, reuse=None, best="0", n_img=None):
    """best != "0": the largest model sits at sparse/<best> (as the incremental mapper writes it) and sparse/0 is a tiny decoy; n_img: images in the database"""
    d = Path(root) / clip / run
    _fix.write_text_model(d / "sparse" / best, c2w, K, 640, 360, _fix.scene_points(12000), names, cam=cam, scale=scale)
    if best != "0":
        _fix.write_text_model(d / "sparse" / "0", c2w[:2], K, 640, 360, _fix.scene_points(12000), names[:2], cam=cam, scale=scale)
    _fix.write_db(d / "database.db", n_img or len(c2w), planar_frac=planar)
    args = {"stride": stride}
    if reuse:
        args["reuse_db"] = str(reuse)
    (d / "stage1.json").write_text(json.dumps({"args": args, "model": G.summarize_model(d / "sparse" / best), "db": G.summarize_db(d / "database.db"),
                                               "timing_s": {"map": 1.0}, "n_models": 1 if best == "0" else 2, "model_dir": str(d / "sparse" / best)}, default=float))
    return d


@unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
class SweepSummary(unittest.TestCase):
    def test_numbers_and_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "s1"
            K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
            traj = _fix.look_at_rig(60, 0.1, curve=0.01)                       # 60 frames of the clip
            nm = lambda idx: [f"{i + 1:06d}.jpg" for i in range(len(idx))]  # noqa: E731
            f3, f6 = np.arange(0, 60, 3), np.arange(0, 60, 6)
            # reference: stride 6, SIMPLE_RADIAL-like (written as SIMPLE_PINHOLE here), focal 500
            make_run(root, "good", "ref", traj[f6], K, nm(f6), 6, cam="SIMPLE_PINHOLE")
            a = make_run(root, "good", "v0a", traj[f3], K, nm(f3), 3, scale=0.37)           # same cameras, arbitrary SfM scale: accurate
            make_run(root, "good", "v1sp", traj[f3], K * np.diag([1.05, 1.05, 1]), nm(f3), 3, cam="SIMPLE_PINHOLE", reuse=a)  # focal +5 %: still accurate
            K2 = K.copy()
            K2[0, 0] = K2[1, 1] = 650.0                                                   # focal +30 %
            make_run(root, "good", "v3fixed", traj[f3], K2, nm(f3), 3, reuse=a)            # gate passes, silver says no -> silent error
            wrong = traj[f3].copy()
            wrong[:, 1, 3] += 0.5 * np.sin(np.linspace(0, 3 * np.pi, len(f3)))             # a wobble no Sim(3) can absorb
            make_run(root, "good", "v2prior", wrong, K, nm(f3), 3, reuse=a)
            make_run(root, "good", "v0b", traj[f3], K, nm(f3), 3, planar=1.0, scale=2.0, reuse=a)      # accurate but 100 % planar pairs: false stop
            K3 = K.copy()
            K3[1, 1] = 555.0                                                              # fx right, fy +11 %: fx / fy = 0.901 stays under the gate's 10 % rule
            make_run(root, "good", "v0a_hq960", traj[f3], K3, nm(f3), 3, reuse=a)
            # a clip whose reference has the largest model at sparse/1 (sparse/0 is a 2-image decoy): ATE must still be computed over the common frames
            make_run(root, "second", "ref", traj[f6], K, nm(f6), 6, cam="SIMPLE_PINHOLE", best="1")
            make_run(root, "second", "v0a", traj[f3], K, nm(f3), 3)
            # a clip whose reference failed (registered 4 of its 10 frames): its runs must not be scored against it
            make_run(root, "refbad", "ref", traj[f6][:4], K, nm(f6)[:4], 6, cam="SIMPLE_PINHOLE", n_img=len(f6))
            make_run(root, "refbad", "v0a", traj[f3], K, nm(f3), 3)
            out = Path(td) / "out"
            pool = Path(td) / "pool" / "good"
            pool.mkdir(parents=True)
            (pool / "scores.json").write_text(json.dumps({"calib": {"focal_w": [0.78125] * 5}}))             # GeoCalib: f / W = 500 / 640, the true K
            r = subprocess.run([sys.executable, str(REPO / "evolution" / "E15_glr" / "summarize_sweep.py"), str(root), str(out), "--pool", str(Path(td) / "pool")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out / "sweep_runs.csv") as fh:
                allrows = list(csv.DictReader(fh))
            rows = {x["run"]: x for x in allrows if x["clip"] == "good"}
            sec = {x["run"]: x for x in allrows if x["clip"] == "second"}
            bad = {x["run"]: x for x in allrows if x["clip"] == "refbad"}
            self.assertEqual(sec["v0a"]["n_common"], "10")                                     # the 10-image model at sparse/1, not the 2-image decoy at sparse/0
            self.assertLess(float(sec["v0a"]["ate_vs_ref_pct"]), 0.01)
            self.assertEqual(sec["v0a"]["outcome"], "true_pass")
            self.assertEqual(bad["v0a"]["ref_failed"], "True")                                 # reference registered 4 / 10: no outcome for the run
            self.assertEqual(bad["v0a"].get("outcome", ""), "")
            self.assertEqual(bad["v0a"].get("silver_accurate", ""), "")
            self.assertAlmostEqual(float(rows["v0a"]["focal_err_pct"]), 0.0, places=3)
            self.assertLess(float(rows["v0a"]["ate_vs_ref_pct"]), 0.01)                   # scale-free: 0.37 x SfM scale does not matter
            self.assertEqual(rows["v0a"]["silver_accurate"], "True")
            self.assertEqual(rows["v0a"]["outcome"], "true_pass")
            self.assertEqual(rows["v1sp"]["outcome"], "true_pass")
            self.assertAlmostEqual(float(rows["v1sp"]["focal_err_pct"]), 5.0, places=1)
            self.assertEqual(rows["v3fixed"]["outcome"], "silent_error")                  # +30 % focal passes the gate but disagrees with the reference
            self.assertAlmostEqual(float(rows["v3fixed"]["focal_err_pct"]), 30.0, places=1)
            self.assertEqual(rows["v2prior"]["outcome"], "silent_error")                  # wobble: ATE above 3.27 %
            self.assertGreater(float(rows["v2prior"]["ate_vs_ref_pct"]), 3.27)
            self.assertEqual(rows["v0b"]["verdict"], "stop")
            self.assertEqual(rows["v0b"]["outcome"], "false_stop")
            self.assertEqual(rows["v0a_hq960"]["verdict"], "ok")                          # a PINHOLE label is only as good as the worse of fx and fy:
            self.assertAlmostEqual(float(rows["v0a_hq960"]["focal_err_pct"]), 11.0, places=1)
            self.assertEqual(rows["v0a_hq960"]["outcome"], "silent_error")                # fy is 11 % off although fx is exact
            self.assertEqual(rows["ref"]["outcome"], "")                                   # the reference is not scored against itself
            self.assertAlmostEqual(float(rows["v0a"]["kdis_deg"]), 0.0, places=2)           # K witness columns (GeoCalib, post-hoc table E)
            self.assertAlmostEqual(float(rows["v3fixed"]["kdis_deg"]), 12.5, places=0)      # focal +30 %: 12.5 deg HFOV apart, under the 13.2 deg rule ...
            self.assertEqual(rows["v3fixed"]["verdict_w"], "ok")                            # ... so the witness does not catch this silent error either
            self.assertEqual(rows["v3fixed"]["outcome"], "silent_error")
            md = (out / "tables.md").read_text()
            self.assertIn("### A.", md)
            self.assertIn("### B.", md)
            self.assertIn("### C.", md)
            self.assertIn("### E.", md)
            self.assertIn("| good | 0.781 | 0.781 | 0.781 | 0.781 | 0 | 0 | 0 | all three agree |", md)
            self.assertIn("of which kdis <= 13.2 deg", md)


if __name__ == "__main__":
    unittest.main()
