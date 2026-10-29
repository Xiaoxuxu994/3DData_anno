"""Table-driven tests of the stage-1 gate: for every rule one perturbation just beyond its threshold must fire exactly that rule (with the right severity)
and one just inside must not. Deleting a rule, flipping a comparison, dropping an abs() or changing a default threshold fails here
(the independent test-adequacy review found these mutants alive in the first suite)."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix
import glr_common as C
import glr_stage1_gate as G

HAS_PYCOLMAP = importlib.util.find_spec("pycolmap") is not None
BASE_MODEL = {"n_reg": 100, "n_points": 40000, "reproj": 0.4, "hfov_deg": 70.0, "fx_fy": 1.0, "focal_w": 0.6, "step_ratio": 1.2, "tri_angle_p50": 6.0}
BASE_DB = {"planar_frac": 0.1}

# (rule, severity, model/db perturbation that must fire it, one just inside that must stay quiet)
RULES = [
    ("registered_lt_half", "stop", {"n_reg": 49}, {"n_reg": 51}),
    ("points_per_image", "stop", {"n_points": 4900}, {"n_points": 5100}),
    ("hfov_out_of_range", "stop", {"hfov_deg": 10.0}, {"hfov_deg": 20.0}),          # lower bound 14.3 deg
    ("hfov_out_of_range", "stop", {"hfov_deg": 125.0}, {"hfov_deg": 110.0}),        # upper bound 118.1 deg
    ("fx_fy_mismatch", "stop", {"fx_fy": 1.2}, {"fx_fy": 1.05}),
    ("fx_fy_mismatch", "stop", {"fx_fy": 0.8}, {"fx_fy": 0.95}),                    # fx < fy: needs the abs()
    ("planar_or_panoramic_pairs", "stop", {"_planar": 0.6}, {"_planar": 0.4}),
    ("reproj_gt_1px", "suspect", {"reproj": 1.2}, {"reproj": 0.9}),
    ("pose_step_outlier", "suspect", {"step_ratio": 6.0}, {"step_ratio": 4.0}),
    ("focal_at_colmap_default", "suspect", {"focal_w": 1.17}, {"focal_w": 1.13}),   # |f - 1.2| < 0.06
    ("focal_at_colmap_default", "suspect", {"focal_w": 1.23}, {"focal_w": 1.27}),
    ("low_parallax", "suspect", {"tri_angle_p50": 1.5}, {"tri_angle_p50": 2.5}),
]


def run_gate(perturb, **kw):
    model, db = dict(BASE_MODEL), dict(BASE_DB)
    for k, v in perturb.items():
        if k == "_planar":
            db["planar_frac"] = v
        else:
            model[k] = v
    return G.gate(model, db, 100, **kw)


class RuleTable(unittest.TestCase):
    def test_baseline_is_quiet(self):
        g = run_gate({})
        self.assertEqual((g["verdict"], g["stop"], g["suspect"]), ("ok", [], []))

    def test_every_rule_fires_beyond_and_not_inside_its_threshold(self):
        for rule, sev, bad, good in RULES:
            with self.subTest(rule=rule, bad=bad):
                g = run_gate(bad)
                self.assertEqual(g[sev], [rule], g)
                self.assertEqual(g["verdict"], "stop" if sev == "stop" else "suspect")
                other = "suspect" if sev == "stop" else "stop"
                self.assertEqual(g[other], [])
            with self.subTest(rule=rule, good=good):
                self.assertEqual(run_gate(good)["verdict"], "ok", run_gate(good))

    def test_witness_disagreement_threshold(self):
        self.assertEqual(run_gate({}, witness_hfov=70.0 + 13.0)["suspect"], [])
        self.assertEqual(run_gate({}, witness_hfov=70.0 + 13.5)["suspect"], ["k_witness_disagree"])
        self.assertEqual(run_gate({}, witness_hfov=70.0 - 13.5)["suspect"], ["k_witness_disagree"])
        self.assertEqual(run_gate({}, witness_hfov=70.0 + 13.5)["verdict"], "suspect")

    def test_thresholds_come_from_the_cfg_and_the_defaults_are_the_documented_ones(self):
        self.assertEqual(G.DEFAULT_CFG["default_focal_tol"], 0.06)
        self.assertEqual(G.DEFAULT_CFG["max_fx_fy_dev"], 0.1)
        self.assertEqual(G.DEFAULT_CFG["hfov_range_deg"], [14.3, 118.1])
        self.assertEqual(G.gate(BASE_MODEL, BASE_DB, 100, cfg={"max_reproj_px": 0.3})["suspect"], ["reproj_gt_1px"])
        self.assertEqual(G.gate(dict(BASE_MODEL, focal_w=1.17), BASE_DB, 100, cfg={"default_focal_tol": 0.02})["suspect"], [])
        self.assertEqual(C.load_cfg(None)["stage1_gate"], G.DEFAULT_CFG)

    def test_registration_fraction_uses_the_frames_given(self):
        self.assertEqual(G.gate(dict(BASE_MODEL, n_reg=60), BASE_DB, 100)["verdict"], "ok")
        self.assertEqual(G.gate(dict(BASE_MODEL, n_reg=60), BASE_DB, 130)["stop"], ["registered_lt_half"])      # 60 of 130


@unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
class RuleInputs(unittest.TestCase):
    def test_step_ratio_is_normalised_by_the_frame_gaps(self):
        with tempfile.TemporaryDirectory() as td:
            names = [f"{k:06d}.jpg" for k in (1, 2, 3, 6, 7, 8, 12, 13, 14, 15)]           # unregistered frames in between
            idx = np.array([1, 2, 3, 6, 7, 8, 12, 13, 14, 15]) - 1
            c2w = _fix.look_at_rig(16, 0.25)[idx]                                              # constant speed per frame, positions follow the frame number
            _fix.write_text_model(Path(td) / "m", c2w, np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]]), 640, 360, _fix.scene_points(3000), names)
            self.assertAlmostEqual(G.summarize_model(Path(td) / "m")["step_ratio"], 1.0, places=3)

    def test_database_summary_counts_the_valid_pair_configs(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "database.db"
            _fix.write_db(db, 10, planar_frac=0.0, n_pairs=0)
            con = sqlite3.connect(db)
            rows = [(1, 100, 2), (2, 100, 2), (3, 100, 6), (4, 100, 6), (5, 100, 6), (6, 100, 1), (7, 100, 0), (8, 100, 7)]   # 1 = DEGENERATE, 0 = UNDEFINED, 7 = WATERMARK-like
            con.executemany("insert into two_view_geometries values (?, ?, 2, NULL, ?)", rows)
            con.commit()
            con.close()
            s = G.summarize_db(db)
            self.assertEqual(s["pair_configs"], {2: 2, 6: 3, 1: 1, 0: 1, 7: 1})
            self.assertEqual(s["n_pairs"], 8)
            self.assertEqual(s["n_pairs_valid"], 5)                                           # only CALIBRATED / UNCALIBRATED / PLANAR / PANORAMIC / PLANAR_OR_PANORAMIC / MULTIPLE
            self.assertAlmostEqual(s["planar_frac"], 3 / 5)                                   # over the verified pairs ...
            self.assertAlmostEqual(s["planar_frac_all"], 3 / 8)                               # ... and over all pairs

    def test_thresholds_of_the_job_reach_the_cli(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
            _fix.write_text_model(td / "m", _fix.look_at_rig(20, 0.25), K, 640, 360, _fix.scene_points(6000))
            _fix.write_db(td / "database.db", 20, planar_frac=0.3)
            run = lambda job: subprocess.run([sys.executable, str(_fix.PIPE / "glr_stage1_gate.py"), str(td / "m"), str(td / "database.db"), "20",  # noqa: E731
                                              "--job", str(job)], capture_output=True, text=True)
            (td / "job_default.json").write_text(json.dumps({"stage1_gate_cfg": {}}))
            (td / "job_strict.json").write_text(json.dumps({"stage1_gate_cfg": {"max_planar_frac": 0.1}}))
            self.assertNotIn("planar_or_panoramic_pairs", json.loads(run(td / "job_default.json").stdout)["gate"]["stop"])
            r = run(td / "job_strict.json")
            self.assertEqual(r.returncode, 3)
            self.assertIn("planar_or_panoramic_pairs", json.loads(r.stdout)["gate"]["stop"])


if __name__ == "__main__":
    unittest.main()
