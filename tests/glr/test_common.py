import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix  # noqa: F401  (puts pipeline/ on sys.path)
import glr_common as C


class MiniYaml(unittest.TestCase):
    def test_subset(self):
        t = """# comment
version: glr-v0
sampling:
  target_frames: 100   # trailing comment
  target_step_frac: 0.02
route:
  force: null
  max_rot_frac: 0.6
qc:
  thr: {xtraj_ba: 3.27, xdepth_k: 0.194, kdis: 13.2}
  reject_flags: [jump60, fov_out]
  neg: -1.5e-3
  flag: true
"""
        d = C.mini_yaml(t)
        self.assertEqual(d["version"], "glr-v0")
        self.assertEqual(d["sampling"]["target_frames"], 100)
        self.assertIsNone(d["route"]["force"])
        self.assertEqual(d["qc"]["thr"]["kdis"], 13.2)
        self.assertEqual(d["qc"]["reject_flags"], ["jump60", "fov_out"])
        self.assertAlmostEqual(d["qc"]["neg"], -1.5e-3)
        self.assertIs(d["qc"]["flag"], True)

    def test_load_cfg_override_and_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.yaml"
            p.write_text("route:\n  max_dyn_frac: 0.5\nqc:\n  thr: {kdis: 20.0}\n")
            cfg = C.load_cfg(p)
            self.assertEqual(cfg["route"]["max_dyn_frac"], 0.5)
            self.assertEqual(cfg["route"]["max_rot_frac"], C.DEFAULTS["route"]["max_rot_frac"])  # untouched keys keep the default
            self.assertEqual(cfg["qc"]["thr"]["kdis"], 20.0)
            self.assertEqual(cfg["qc"]["thr"]["xtraj_ba"], 3.27)
            self.assertIn("min_pts_per_img", cfg["stage1_gate"])  # filled from glr_stage1_gate.DEFAULT_CFG

    def test_repo_config_parses_like_pyyaml(self):
        p = Path(__file__).resolve().parents[2] / "configs" / "pipeline_glr_v0.yaml"
        if not p.exists():
            self.skipTest("config not written yet")
        mine = C.mini_yaml(p.read_text())
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        theirs = yaml.safe_load(p.read_text())
        theirs["date"] = str(theirs["date"])  # PyYAML turns 2026-10-29 into a date object
        self.assertEqual(mine, theirs)


class Geometry(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.P = np.cumsum(self.rng.normal(size=(60, 3)), 0)

    def test_umeyama_recovers_sim3(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.random(random_state=3).as_matrix()
        s, t = 0.37, np.array([1.0, -2.0, 3.0])
        Q = s * self.P @ R.T + t
        s2, R2, t2 = C.umeyama(self.P, Q, True)
        self.assertAlmostEqual(s2, s, places=9)
        self.assertTrue(np.allclose(R2, R, atol=1e-9))
        self.assertTrue(np.allclose(t2, t, atol=1e-9))

    def test_traj_pct_scale_free_and_sensitive(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.random(random_state=5).as_matrix()
        Q = 3.1 * self.P @ R.T + 4.0
        self.assertLess(C.traj_pct(self.P, Q), 1e-8)
        noisy = Q + self.rng.normal(scale=0.05 * np.linalg.norm(np.diff(Q, axis=0), axis=1).mean(), size=Q.shape)
        self.assertGreater(C.traj_pct(self.P, noisy), 0.01)
        self.assertTrue(np.isnan(C.traj_pct(self.P, self.P[:1].repeat(60, 0))))

    def test_depth_cmp_global_scale_invariant(self):
        D = self.rng.uniform(1, 10, size=(3, 60, 60)).astype(np.float32)
        self.assertLess(C.depth_cmp(D, 7.3 * D)[0], 1e-6)
        self.assertTrue(np.isnan(C.depth_cmp(D[:, :5, :5], D[:, :5, :5])[0]))  # < 1000 valid pixels

    def test_metric_scale_recovers_k(self):
        T = _fix.look_at_rig(30, 0.3, curve=0.05)
        Dv = np.full((30, 40, 60), 6.0, np.float32)
        s_glr = 0.37  # GLRMap's arbitrary SfM scale: its cameras / depth are 0.37 x the metric ones
        Tg = T.copy()
        Tg[:, :3, 3] *= s_glr
        r = C.metric_scale(Tg, Dv * s_glr, T, Dv)
        self.assertAlmostEqual(r["k"], 1 / s_glr, places=5)
        self.assertAlmostEqual(r["k_traj"], 1 / s_glr, places=5)
        self.assertAlmostEqual(r["k_depth"], 1 / s_glr, places=5)
        r2 = C.metric_scale(Tg, None, T, Dv)  # no GLRMap depth: the trajectory vote carries the scale
        self.assertAlmostEqual(r2["k"], 1 / s_glr, places=5)

    def test_metric_scale_falls_back_when_path_is_tiny(self):
        T = _fix.look_at_rig(30, 1e-4)
        Dv = np.full((30, 40, 60), 6.0, np.float32)
        r = C.metric_scale(T, Dv * 0.5, T, Dv)
        self.assertTrue(np.isnan(r["s_vipe"]))  # path < 2 % of the scene depth: no trajectory evidence
        self.assertAlmostEqual(r["k"], 2.0, places=5)  # depth ratio alone

    def test_to_grid_scales_k(self):
        D = np.ones((2, 20, 40), np.float32)
        K = np.tile(np.array([[40.0, 0, 20], [0, 40, 10], [0, 0, 1]]), (2, 1, 1))
        D2, K2 = C.to_grid(D, K, 40, 80)
        self.assertEqual(D2.shape, (2, 40, 80))
        self.assertAlmostEqual(K2[0, 0, 0], 80.0)
        self.assertAlmostEqual(K2[0, 0, 2], 20 * 2 + 0.5 * 2 - 0.5)

    def test_qc_grid_multiple_of_8(self):
        h, w = C.qc_grid(1080, 1920, 512)
        self.assertEqual((h % 8, w % 8), (0, 0))
        self.assertEqual(w, 512)

    def test_clean_json(self):
        o = C.clean({"a": np.float32(1.23456789), "b": float("nan"), "c": np.array([1.0, np.inf]), "d": np.int64(3), "e": np.bool_(True)})
        self.assertEqual(json.loads(json.dumps(o)), {"a": 1.234568, "b": None, "c": [1.0, None], "d": 3, "e": True})


class ConfigFile(unittest.TestCase):
    def test_repo_yaml_equals_defaults(self):
        """configs/pipeline_glr_v0.yaml must not drift from glr_common.DEFAULTS / glr_stage1_gate.DEFAULT_CFG."""
        import glr_stage1_gate as G
        p = Path(__file__).resolve().parents[2] / "configs" / "pipeline_glr_v0.yaml"
        d = C.mini_yaml(p.read_text())
        for sec in ("sampling", "route", "qc"):
            self.assertEqual(d[sec], C.DEFAULTS[sec], sec)
        self.assertEqual(d["stage1_gate"], G.DEFAULT_CFG)
        self.assertEqual(d["version"], C.DEFAULTS["version"])

    def test_load_cfg_is_isolated_and_strict(self):
        c = C.load_cfg(None)
        c["qc"]["require_all_parts"] = False
        c["qc"]["thr"]["kdis"] = 1.0
        c["sampling"]["min_frames"] = 1
        fresh = C.load_cfg(None)                              # mutating one config must never leak into DEFAULTS or the next config
        self.assertTrue(fresh["qc"]["require_all_parts"])
        self.assertEqual(fresh["qc"]["thr"]["kdis"], 13.2)
        self.assertEqual(C.DEFAULTS["sampling"]["min_frames"], 60)
        with self.assertRaises(FileNotFoundError):
            C.load_cfg("/nonexistent/typo.yaml")              # an explicit path that does not exist is an error, not the defaults


class ScaleGuards(unittest.TestCase):
    def test_nan_camera_centre_gives_no_trajectory_vote_instead_of_an_exception(self):
        T = _fix.look_at_rig(30, 0.3)
        Dv = np.full((30, 40, 60), 6.0, np.float32)
        ref = T.copy()
        ref[5, :3, 3] = np.nan                                 # the witness does not cover this frame
        r = C.metric_scale(T * 1.0, Dv * 0.5, ref, Dv)
        self.assertTrue(np.isnan(r["s_vipe"]))
        self.assertAlmostEqual(r["k"], 2.0, places=5)          # the depth ratio still gives the scale
        r = C.metric_scale(T * 1.0, None, ref, Dv, neutral=float("nan"))
        self.assertTrue(np.isnan(r["k"]))                      # no evidence at all


if __name__ == "__main__":
    unittest.main()
