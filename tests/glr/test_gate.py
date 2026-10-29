import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix
import glr_stage1_gate as G

# Numbers from the evaluation sessions (docs/GLR_INTEGRATION.md, section 2) re-used as regression fixtures: (model summary, db summary)
KITTI_0001 = ({"n_reg": 100, "n_points": 41926, "reproj": 0.43, "hfov_deg": 82.0, "fx_fy": 698 / 709, "focal_w": 0.56, "step_ratio": 1.2,
               "tri_angle_p50": 6.0}, {"planar_frac": 0.0})
KITTI_0005 = ({"n_reg": 100, "n_points": 26461, "reproj": 0.81, "hfov_deg": 82.0, "fx_fy": 697 / 712, "focal_w": 0.57, "step_ratio": 1.2,
               "tri_angle_p50": 5.0}, {"planar_frac": 0.17})
DRONE = ({"n_reg": 100, "n_points": 0, "reproj": None, "hfov_deg": 45.0, "fx_fy": 1.0, "focal_w": 1.2}, {"planar_frac": 1.0})
INDOOR = ({"n_reg": 100, "n_points": 21, "reproj": 1.18, "hfov_deg": 70.0, "fx_fy": 754 / 147, "focal_w": 0.59, "step_ratio": 8.3},
          {"planar_frac": 1.0})
TOKYO = ({"n_reg": 100, "n_points": 1306, "reproj": 1.18, "hfov_deg": 45.0, "fx_fy": 2246 / 2304, "focal_w": 1.18, "step_ratio": 3.2,
          "tri_angle_p50": 3.4}, {"planar_frac": 0.032})
AHLBECK = ({"n_reg": 100, "n_points": 26609, "reproj": 0.75, "hfov_deg": 45.0, "fx_fy": 1497 / 1546, "focal_w": 1.19, "step_ratio": 15.1,
            "tri_angle_p50": 6.0}, {"planar_frac": 0.0})
W06 = ({"n_reg": 90, "n_points": 22000, "reproj": 1.05, "hfov_deg": 45.0, "fx_fy": 689 / 721, "focal_w": 0.36, "step_ratio": 1.3,
        "tri_angle_p50": 4.0}, {"planar_frac": 0.0})  # KITTI w03: plausible-but-wrong, the gate cannot see it (documented limit)


class GateRules(unittest.TestCase):
    def v(self, case, n=100, **kw):
        return G.gate(case[0], case[1], n, **kw)

    def test_good_models_pass(self):
        for c in (KITTI_0001, KITTI_0005):
            self.assertEqual(self.v(c)["verdict"], "ok", c)

    def test_hard_failures_stop(self):
        self.assertIn("planar_or_panoramic_pairs", self.v(DRONE)["stop"])
        self.assertIn("points_per_image", self.v(DRONE)["stop"])
        r = self.v(INDOOR)
        self.assertEqual(r["verdict"], "stop")
        self.assertIn("fx_fy_mismatch", r["stop"])
        self.assertIn("planar_or_panoramic_pairs", r["stop"])

    def test_suspicious_models(self):
        r = self.v(TOKYO)
        self.assertIn("points_per_image", r["stop"])  # 13 points per frame
        self.assertIn("focal_at_colmap_default", r["suspect"])
        self.assertIn("reproj_gt_1px", r["suspect"])
        r = self.v(AHLBECK)
        self.assertEqual(r["verdict"], "suspect")
        self.assertIn("pose_step_outlier", r["suspect"])
        self.assertIn("focal_at_colmap_default", r["suspect"])

    def test_plausible_but_wrong_is_invisible_without_a_witness(self):
        self.assertIn(self.v(W06)["verdict"], ("ok", "suspect"))  # documented limitation, see KNOWN_ISSUES / GLR_INTEGRATION
        r = self.v(W06, witness_hfov=75.0)  # the K witness (GeoCalib / Pi3X / DA3 median) is what can catch it
        self.assertIn("k_witness_disagree", r["suspect"])

    def test_registration_fraction(self):
        m = dict(KITTI_0001[0], n_reg=40)
        self.assertIn("registered_lt_half", G.gate(m, KITTI_0001[1], 100)["stop"])

    def test_missing_signals_never_fire(self):
        r = G.gate({"n_reg": 50, "n_points": 50000}, None, 100)
        self.assertEqual(r["verdict"], "ok")
        self.assertEqual(G.gate(None, None, 100)["stop"], ["no_model"])

    def test_cfg_override(self):
        r = G.gate(*KITTI_0005, 100, cfg={"max_planar_frac": 0.1})
        self.assertIn("planar_or_panoramic_pairs", r["stop"])


@unittest.skipUnless(__import__("importlib").util.find_spec("pycolmap"), "pycolmap not installed")
class RealModelIO(unittest.TestCase):
    """Synthetic scene -> COLMAP text model + database -> summarize_model / summarize_db / load_cameras / CLI."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name)
        W, H = 640, 360
        cls.K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
        cls.c2w = _fix.look_at_rig(20, 0.25)
        X = _fix.scene_points(3000)
        _fix.write_text_model(cls.root / "m", cls.c2w, cls.K, W, H, X)
        _fix.write_db(cls.root / "database.db", 20, planar_frac=0.0)
        cls.W, cls.H = W, H

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_summaries(self):
        m = G.summarize_model(self.root / "m")
        self.assertEqual(m["n_reg"], 20)
        self.assertGreater(m["n_points"], 1000)
        self.assertAlmostEqual(m["fx"], 500.0)
        self.assertAlmostEqual(m["hfov_deg"], float(np.degrees(2 * np.arctan(320 / 500))), places=4)
        self.assertAlmostEqual(m["step_ratio"], 1.0, places=3)  # constant-speed rig
        self.assertGreater(m["tri_angle_p50"], 1.0)
        d = G.summarize_db(self.root / "database.db")
        self.assertEqual(d["planar_frac"], 0.0)
        self.assertGreater(d["n_pairs"], 10)

    def test_load_cameras_matches_input(self):
        cams = G.load_cameras(self.root / "m")
        self.assertEqual(cams["names"][0], "000001.jpg")
        self.assertTrue(np.allclose(cams["c2w"], self.c2w, atol=1e-5))
        self.assertTrue(np.allclose(cams["K"][0], self.K))
        self.assertEqual((cams["W"], cams["H"]), (self.W, self.H))

    def test_cli_exit_codes(self):
        import subprocess
        import sys
        job = {"stage1_gate_cfg": {}, "k_prior": {"hfov_deg": 70.0}}
        (self.root / "job.json").write_text(json.dumps(job))
        base = [sys.executable, str(_fix.PIPE / "glr_stage1_gate.py"), str(self.root / "m"), str(self.root / "database.db"), "20"]
        r = subprocess.run(base, capture_output=True, text=True)
        self.assertIn(r.returncode, (0, 1), r.stderr)  # n_points/frame high enough; may be 'suspect' (default-focal coincidence etc.)
        out = json.loads(r.stdout)
        self.assertIn("gate", out)
        r = subprocess.run(base + ["--job", str(self.root / "job.json"), "--witness-hfov", "20"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)  # K witness disagrees by > 13.2 deg -> suspect
        self.assertIn("k_witness_disagree", json.loads(r.stdout)["gate"]["suspect"])


if __name__ == "__main__":
    unittest.main()
