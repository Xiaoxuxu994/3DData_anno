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

    @staticmethod
    def cam(W, H, f, **kw):
        return dict({"n_reg": 100, "n_points": 30000, "reproj": 0.5, "fx": f, "fy": f, "fx_fy": 1.0, "W": W, "H": H,
                     "focal_w": f / W, "focal_rel": f / max(W, H), "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / f))),
                     "tri_angle_p50": 5.0, "step_ratio": 1.2}, **kw)

    def test_focal_rules_are_relative_to_the_long_side(self):
        # COLMAP's default focal is 1.2 x max(W, H): untouched focals are flagged in portrait clips too, legitimate ones are not
        land = G.gate(self.cam(640, 360, 768.0), {"planar_frac": 0.0}, 100)
        port = G.gate(self.cam(360, 640, 768.0), {"planar_frac": 0.0}, 100)
        self.assertIn("focal_at_colmap_default", land["suspect"])
        self.assertIn("focal_at_colmap_default", port["suspect"])
        legit = G.gate(self.cam(360, 640, 432.0), {"planar_frac": 0.0}, 100)   # fx / W = 1.2, fx / max(W, H) = 0.675
        self.assertEqual(legit["verdict"], "ok", legit)
        # sane range 0.3 .. 4 x max(W, H) (the HFOV range [14.3, 118.1] of a landscape frame), same focal bounds for every shape
        wide = G.gate(self.cam(1080, 1920, 480.0), {"planar_frac": 0.0}, 100)      # 0.25 x max(W, H): HFOV 96.7 deg in portrait
        self.assertIn("hfov_out_of_range", wide["stop"])
        self.assertNotIn("hfov_out_of_range", G.gate(self.cam(1080, 1920, 620.0), {"planar_frac": 0.0}, 100)["stop"])   # 0.32
        self.assertIn("hfov_out_of_range", G.gate(self.cam(1920, 1080, 560.0), {"planar_frac": 0.0}, 100)["stop"])      # 0.29: HFOV 119 deg
        self.assertNotIn("hfov_out_of_range", G.gate(self.cam(1920, 1080, 700.0), {"planar_frac": 0.0}, 100)["stop"])   # HFOV 108 deg
        self.assertIn("hfov_out_of_range", G.gate(self.cam(1920, 1080, 8000.0), {"planar_frac": 0.0}, 100)["stop"])      # 4.17 x

    def test_legacy_model_dicts_without_wh_still_work(self):
        self.assertIn("hfov_out_of_range", G.gate({"n_reg": 90, "n_points": 9000, "hfov_deg": 125.0}, None, 100)["stop"])
        self.assertNotIn("hfov_out_of_range", G.gate({"n_reg": 90, "n_points": 9000, "hfov_deg": 60.0}, None, 100)["stop"])


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

    def cli(self, *args):
        import subprocess
        import sys
        return subprocess.run([sys.executable, str(_fix.PIPE / "glr_stage1_gate.py"), *map(str, args)], capture_output=True, text=True)

    def test_cli_exit_codes(self):
        job = {"stage1_gate_cfg": {}, "k_prior": {"hfov_deg": 70.0}}
        (self.root / "job.json").write_text(json.dumps(job))
        base = [self.root / "m", self.root / "database.db", "20"]
        r = self.cli(*base)
        self.assertIn(r.returncode, (0, 1), r.stderr)  # n_points/frame high enough; may be 'suspect' (default-focal coincidence etc.)
        out = json.loads(r.stdout)
        self.assertIn("gate", out)
        r = self.cli(*base, "--job", self.root / "job.json", "--witness-hfov", "20")
        self.assertEqual(r.returncode, 1)  # K witness disagrees by > 13.2 deg -> suspect
        self.assertIn("k_witness_disagree", json.loads(r.stdout)["gate"]["suspect"])
        w = self.root / "gate_compact.json"
        r = self.cli(*base, "--job", self.root / "job.json", "--witness-hfov", "20", "--require-db", "--write", w)
        compact = json.loads(w.read_text())
        self.assertEqual((compact["verdict"], r.returncode), ("suspect", 1))
        self.assertIn("n_reg", compact["model"])
        self.assertIn("planar_frac", compact["db"])

    def test_gate_failures_are_an_error_verdict_with_exit_4(self):
        w = self.root / "err.json"
        bad = self.root / "bad.db"
        bad.write_bytes(b"this is not a database")
        r = self.cli(self.root / "m", bad, "20", "--write", w)
        self.assertEqual(r.returncode, 4)
        self.assertEqual(json.loads(r.stdout)["gate"]["verdict"], "error")
        self.assertEqual(json.loads(w.read_text())["verdict"], "error")
        self.assertTrue(json.loads(w.read_text())["error"])
        r = self.cli(self.root / "m", self.root / "missing.db", "20", "--require-db")      # strict: a missing database is a failure
        self.assertEqual((r.returncode, json.loads(r.stdout)["gate"]["verdict"]), (4, "error"))
        r = self.cli(self.root / "m", self.root / "missing.db", "20")                      # the replay tool's lenient mode
        self.assertIn(r.returncode, (0, 1))
        r = self.cli(self.root / "no_such_model", self.root / "database.db", "20")         # unreadable model
        self.assertEqual(r.returncode, 4)
        self.assertFalse((self.root / "missing.db").exists())                              # and nothing was created next to the data

    def test_database_path_with_uri_characters(self):
        import shutil
        for name in ("x #shorts", "what?now", "100%", "a b"):
            d = self.root / name
            d.mkdir()
            shutil.copy(self.root / "database.db", d / "database.db")
            before = sorted(p.name for p in self.root.iterdir())
            s = G.summarize_db(d / "database.db")
            self.assertEqual(s["n_images"], 20, name)
            self.assertEqual(sorted(p.name for p in self.root.iterdir()), before, f"stray file created for {name!r}")
            self.assertEqual([p.name for p in d.iterdir()], ["database.db"])

    def test_portrait_default_focal_is_flagged_end_to_end(self):
        c2w = _fix.look_at_rig(12, 0.25)
        for (W, H), tag in (((640, 360), "land"), ((360, 640), "port")):
            K = np.array([[768.0, 0, W / 2], [0, 768.0, H / 2], [0, 0, 1]])           # untouched COLMAP default 1.2 x max(W, H)
            _fix.write_text_model(self.root / f"def_{tag}", c2w, K, W, H, _fix.scene_points(3000, depth=(8.0, 20.0)))
            m = G.summarize_model(self.root / f"def_{tag}")
            self.assertAlmostEqual(m["focal_rel"], 1.2, places=3)
            self.assertIn("focal_at_colmap_default", G.gate(m, None, 12)["suspect"], tag)


if __name__ == "__main__":
    unittest.main()
