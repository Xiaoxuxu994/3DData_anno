"""Tests that pin what a consumer of the label sees and what the QC compares: label contract, frame alignment of the witnesses, the registered subset
path, tier boundaries, flag bounds, geometry conventions (rotated cameras, fx != fy, an arbitrary world frame) and the no-pycolmap collect path.
Written against the findings of the independent test-adequacy review (mutants of these lines used to survive the suite)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import _fix
import glr_collect as GC
import glr_common as C
import glr_stage1_gate as G
from test_collect import CFG, FX, HW, H_IMG, IDX, N, S_GLR, W_IMG, case

HAS_PYCOLMAP = importlib.util.find_spec("pycolmap") is not None


class LabelContract(unittest.TestCase):
    def test_label_is_the_glr_result_times_k(self):
        job, glr, s1, wit = case()
        glr["depth"] = (glr["depth"] / S_GLR * 0.30).astype(np.float32)      # the depth scale differs from the trajectory scale: k_traj != k_depth
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        kt, kd = 1 / S_GLR, 1 / 0.30
        self.assertAlmostEqual(label["k_traj"], kt, places=4)
        self.assertAlmostEqual(label["k_depth"], kd, places=4)
        self.assertAlmostEqual(label["scale"], float(np.sqrt(kt * kd)), places=4)   # geometric mean, not the arithmetic mean (3.02)
        np.testing.assert_allclose(label["depth"], glr["depth"] * np.float32(label["scale"]), rtol=1e-6)          # the depth is in metres
        np.testing.assert_allclose(label["c2w"][:, :3, 3], glr["c2w"][:, :3, 3] * label["scale"], rtol=1e-6)
        np.testing.assert_allclose(label["c2w"][:, :3, :3], glr["c2w"][:, :3, :3])                                # rotations are untouched
        self.assertEqual(label["frame_hw"].tolist(), [H_IMG, W_IMG])                                              # [H, W] like v1.3
        self.assertAlmostEqual(label["hfov"], 60.0, places=3)                                                      # from fx and the image width
        self.assertEqual(label["K"].shape, (len(IDX), 3, 3))
        self.assertAlmostEqual(float(label["K"][0, 0, 0]), FX * HW[1] / W_IMG, places=3)                          # K lives on the depth grid
        np.testing.assert_array_equal(label["conf"], np.isfinite(label["depth"]).astype(np.float32))
        self.assertEqual((label["n_orig"], str(label["annotator"]), str(label["k_source"]), str(label["depth_source"])), (N, "glr", "glr_sfm", "pda_fine"))
        self.assertEqual(str(label["tier"]), qc["tier"])
        self.assertEqual(label["frame_idx"].tolist(), IDX.tolist())

    def test_qc_json_carries_the_numbers_a_reviewer_needs(self):
        job, glr, s1, wit = case(v13=True)
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        for k in ("v1c", "v1c_parts", "xtraj_ba", "xdepth_k", "kdis", "flags", "reject_flags", "info_flags", "accept", "tier", "reject_reasons", "scale",
                  "stage1", "glr", "agreement_v13"):
            self.assertIn(k, qc)
        self.assertEqual(set(qc["v1c_parts"]), {"xtraj_ba", "xdepth_k", "kdis"})
        self.assertEqual(qc["glr"]["n_reg"], len(IDX))
        self.assertEqual(qc["glr"]["stride"], 2)


class WitnessAlignment(unittest.TestCase):
    """The witnesses cover all frames of the clip; GLRMap used every stride-th frame: a wrong index or grid turns xdepth_k into noise."""

    def varying_case(self):
        job, glr, s1, wit = case()
        t = np.arange(N)
        yy, xx = np.mgrid[0:HW[0], 0:HW[1]]
        Dv = (5.0 + 0.4 * t[:, None, None] + 0.5 * np.sin(xx / 9.0)[None] + 0.3 * np.cos(yy / 7.0)[None]).astype(np.float32)
        wit["D"] = Dv
        glr["depth"] = (Dv[IDX] * S_GLR).astype(np.float32)
        band = np.zeros((N, *HW), bool)
        for f in range(N):
            band[f, :, (f % 4) * 20:(f % 4) * 20 + 20] = True        # a different 20-column band is "not static" in every frame
        da3 = Dv * 1.1
        da3[band] *= 4.0                                              # DA3 is garbage inside the band
        wit["da3_D"], wit["static"] = da3.astype(np.float32), ~band
        return job, glr, s1, wit

    def test_static_mask_and_da3_are_indexed_by_the_sampled_frames(self):
        job, glr, s1, wit = self.varying_case()
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertLess(qc["xdepth_k"], 0.1)
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=3)
        wit["static"] = np.ones_like(wit["static"])                                       # without the mask the garbage counts: the mask is used
        _, qc2 = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertGreater(qc2["xdepth_k"], 0.2)

    def test_label_grid_differs_from_the_witness_grid(self):
        job, glr, s1, wit = self.varying_case()
        glr["depth"] = glr["depth"][:, ::2, ::2].copy()                                   # 32 x 48 label grid, 64 x 96 witnesses
        glr["K_label"] = glr["K_label"].copy()
        glr["K_label"][:, :2, :] *= 0.5
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertLess(qc["xdepth_k"], 0.1)
        self.assertEqual(label["depth"].shape[1:], (HW[0] // 2, HW[1] // 2))
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=3)


class RegisteredSubset(unittest.TestCase):
    def test_partial_registration_uses_the_right_witness_frames(self):
        job, glr, s1, wit = case()
        keep = np.r_[0:20, 30:90]                                                         # label frames 20..29 were not registered
        for key in ("c2w", "K", "K_label", "depth"):
            glr[key] = glr[key][keep]
        glr["frame_idx"], glr["n_reg"] = IDX[keep], len(keep)
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertIn("partial_registration", qc["info_flags"])
        self.assertEqual(qc["tier"], "T2")                                                # 80 / 90 registered < 90 %
        self.assertEqual(label["frame_idx"].tolist(), IDX[keep].tolist())
        self.assertEqual((len(label["c2w"]), len(label["depth"])), (80, 80))
        self.assertLess(qc["xtraj_ba"], 1e-6)                                             # the ViPE frames were taken at the registered indices
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=4)
        self.assertEqual(qc["glr"]["n_sampled"], len(IDX))


class TierBoundaries(unittest.TestCase):
    def test_half_threshold_is_the_t1_line(self):
        inside = GC.qc_one(*case(gate_med=60.0 + 5.0), CFG)[1]                            # kdis 5 deg = 0.38 x 13.2
        outside = GC.qc_one(*case(gate_med=60.0 + 8.0), CFG)[1]                           # 8 deg = 0.61 x 13.2: accepted, not T1
        self.assertEqual((inside["accept"], inside["tier"]), (True, "T1"))
        self.assertEqual((outside["accept"], outside["tier"]), (True, "T2"))
        self.assertGreater(outside["v1c_parts"]["kdis"], 0.5)
        over = GC.qc_one(*case(gate_med=60.0 + 14.0), CFG)[1]                             # 14 deg > 13.2: rejected
        self.assertEqual((over["accept"], over["tier"]), (False, "T3"))

    def test_the_tier_threshold_is_the_configured_one(self):
        cfg = C.load_cfg(None)
        cfg["qc"]["tier1_part"] = 0.3
        self.assertEqual(GC.qc_one(*case(gate_med=60.0 + 5.0), cfg)[1]["tier"], "T2")     # 0.38 > 0.3

    def test_flag_bounds(self):
        T = _fix.look_at_rig(40, 0.3)
        D = np.full((40, 8, 8), 5.0, np.float32)

        def flags(hfov=60.0, yaw=0.0, step=0.3):
            Tt = _fix.look_at_rig(40, step)
            Tt[20:, :3, :3] = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
            f = 320 / np.tan(np.radians(hfov / 2))
            return GC.e9_flags(Tt, np.tile(np.array([[f, 0, 320], [0, f, 180], [0, 0, 1]]), (40, 1, 1)), D, 640, 15)
        self.assertEqual([flags(hfov=h)["fov_out"] for h in (25, 35, 115, 125)], [1, 0, 0, 1])      # [30, 120] deg
        self.assertEqual([(flags(yaw=y)["jump20"], flags(yaw=y)["jump60"]) for y in (15, 25, 55, 65)], [(0, 0), (1, 0), (1, 0), (1, 1)])
        self.assertEqual(flags(step=0.3)["tiny_move"], 0)
        self.assertEqual(flags(step=0.0003)["tiny_move"], 1)                                          # camera path < 2 % of the scene depth
        self.assertEqual(T.shape, (40, 4, 4))

    def test_info_flags_reach_the_qc_json(self):
        job, glr, s1, wit = case()
        wit["vipe_focal_cv"] = 0.10
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertIn("suspect_zoom", qc["info_flags"])
        job, glr, s1, wit = case()
        c = glr["c2w"].copy()
        c[40:, :3, :3] = c[40:, :3, :3] @ Rotation.from_euler("y", 30, degrees=True).as_matrix()   # 30 deg between two label frames
        glr["c2w"] = c
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertIn("suspect_cut", qc["info_flags"])
        self.assertNotIn("jump60", qc["reject_reasons"])
        self.assertTrue(qc["accept"], qc["reject_reasons"])


@unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
class GeometryConventions(unittest.TestCase):
    """Rotated cameras in an arbitrary world frame, fx != fy: every other fixture is the identity world with square pixels."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def model(self, name="m", n=12, cam="PINHOLE", K=None):
        K = K if K is not None else np.array([[500.0, 0, 320], [0, 540.0, 180], [0, 0, 1]])
        c2w, W0 = _fix.rotated_rig(n)
        X = _fix.scene_points(3000)
        Xw = X @ W0[:3, :3].T + W0[:3, 3]
        _fix.write_text_model(self.root / name, c2w, K, 640, 360, Xw, cam=cam)
        return self.root / name, c2w, K

    def test_load_cameras_returns_the_input_poses_and_intrinsics(self):
        mdir, c2w, K = self.model()
        cams = G.load_cameras(mdir)
        np.testing.assert_allclose(cams["c2w"], c2w, atol=1e-5)
        np.testing.assert_allclose(cams["K"][0], K, atol=1e-6)
        self.assertEqual((cams["K"][0, 0, 0], cams["K"][0, 1, 1]), (500.0, 540.0))
        m = G.summarize_model(mdir)
        self.assertAlmostEqual(m["fx_fy"], 500 / 540, places=6)
        self.assertAlmostEqual(m["focal_w"], 500 / 640, places=6)
        self.assertAlmostEqual(m["focal_rel"], 500 / 640, places=6)
        self.assertLess(m["step_ratio"], 1.1)                                             # rotated cameras: centres, not rotations, are compared
        self.assertGreater(m["tri_angle_p50"], 1.0)

    def test_simple_pinhole_models_are_read_with_cx_cy_in_the_right_order(self):
        mdir, c2w, K = self.model("sp", cam="SIMPLE_PINHOLE", K=np.array([[500.0, 0, 300], [0, 500.0, 170], [0, 0, 1]]))
        cams = G.load_cameras(mdir)
        np.testing.assert_allclose(cams["K"][0], [[500, 0, 300], [0, 500, 170], [0, 0, 1]], atol=1e-6)
        self.assertEqual(cams["model"], "SIMPLE_PINHOLE")

    def test_read_glr_normalises_the_world_frame_to_the_first_camera(self):
        mdir, c2w, K = self.model(n=8)
        res = self.root / "result"
        (res / "stage1" / "sparse").mkdir(parents=True)
        mdir.rename(res / "stage1" / "sparse" / "0")
        names = [f"{k + 1:06d}.jpg" for k in range(8)]
        job = {"clip": "c", "image_names": names, "frame_idx": list(range(0, 16, 2))}
        g = GC.read_glr(res, job, 512)
        self.assertTrue(g["ok"])
        np.testing.assert_allclose(g["c2w"], np.linalg.inv(c2w[0])[None] @ c2w, atol=1e-5)    # c2w_0^-1 @ c2w_i, in this order
        self.assertAlmostEqual(float(np.abs(g["c2w"][0] - np.eye(4)).max()), 0.0, places=6)
        np.testing.assert_allclose(g["K"][0], K, atol=1e-6)

    def test_cameras_npz_makes_phase_c_independent_of_pycolmap(self):
        mdir, c2w, K = self.model(n=8)
        res = self.root / "result"
        (res / "stage1" / "sparse").mkdir(parents=True)
        mdir.rename(res / "stage1" / "sparse" / "0")
        cams = G.load_cameras(res / "stage1" / "sparse" / "0")
        np.savez(res / "cameras.npz", names=np.array(cams["names"]), c2w=cams["c2w"], K=cams["K"], W=cams["W"], H=cams["H"], model=cams["model"])
        job = {"clip": "c", "image_names": [f"{k + 1:06d}.jpg" for k in range(8)], "frame_idx": list(range(0, 16, 2))}
        with_model = GC.read_glr(res, job, 512)
        hidden = sys.modules.get("pycolmap", False)
        sys.modules["pycolmap"] = None                                                      # `import pycolmap` now raises ImportError
        try:
            from_npz = GC.read_glr(res, job, 512)
            self.assertEqual(from_npz["cameras_from"], "cameras_npz")
            np.testing.assert_allclose(from_npz["c2w"], with_model["c2w"], atol=1e-12)
            np.testing.assert_allclose(from_npz["K"], with_model["K"], atol=1e-12)
            (res / "cameras.npz").unlink()                                                  # no npz and no pycolmap: say so, do not guess
            with self.assertRaises(GC.MissingDependency):
                GC.read_glr(res, job, 512)
            jd = self.root / "out" / "glr_jobs" / "c"
            jd.mkdir(parents=True)
            (jd / "job.json").write_text(json.dumps(dict(job, n_orig=16, hw=[360, 640])))
            res.rename(jd / "result")
            (jd / "result" / "result.json").write_text(json.dumps({"status": "complete", "gate": "ok"}))
            (jd / "RESULT_DIR").write_text("result\n")
            with self.assertRaises(SystemExit) as cm:                                       # the CLI stops the whole run instead of one collect_error per clip
                GC.main([str(self.root / "out"), "--witness-dir", str(self.root / "w")])
            self.assertIn("pycolmap", str(cm.exception))
        finally:
            if hidden is False:
                sys.modules.pop("pycolmap", None)
            else:
                sys.modules["pycolmap"] = hidden


if __name__ == "__main__":
    unittest.main()
