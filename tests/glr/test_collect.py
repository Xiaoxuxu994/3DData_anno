import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix
import glr_collect as GC
import glr_common as C
import glr_stage1_gate as G

CFG = C.load_cfg(None)
N, STRIDE, S_GLR = 180, 2, 0.37
IDX = np.arange(0, N, STRIDE)
HW = (64, 96)          # QC / label grid of the synthetic case
W_IMG, H_IMG = 960, 540
FX = W_IMG / 2 / np.tan(np.radians(30))  # HFOV 60 deg


def vipe_depth(n=N):
    yy, xx = np.mgrid[0:HW[0], 0:HW[1]]
    base = 5.0 + 2.0 * np.sin(xx / 9.0) + 1.5 * np.cos(yy / 7.0)
    return np.stack([base + 0.01 * i for i in range(n)]).astype(np.float32)


OK_GATE = {"verdict": "ok", "stop": [], "suspect": [], "signals": {}}


def case(stage1=OK_GATE, glr_scale=S_GLR, drift=0.0, gate_med=60.3, with_depth=True, da3_noise=0.02, v13=False, hfov_img=None, depth_frames=None):
    rng = np.random.default_rng(0)
    Tv = _fix.look_at_rig(N, 0.05, curve=0.02)       # metric ViPE cameras, all 180 frames
    Dv = vipe_depth()
    Tg = Tv[IDX].copy()
    Tg[:, :3, 3] *= glr_scale                          # GLRMap: arbitrary SfM scale
    if drift:
        Tg[:, 0, 3] += drift * (np.arange(len(IDX)) / len(IDX)) ** 2
    Kf = np.array([[FX, 0, W_IMG / 2], [0, FX, H_IMG / 2], [0, 0, 1.0]])
    if hfov_img:
        f = W_IMG / 2 / np.tan(np.radians(hfov_img / 2))
        Kf = np.array([[f, 0, W_IMG / 2], [0, f, H_IMG / 2], [0, 0, 1.0]])
    K = np.tile(Kf, (len(IDX), 1, 1))
    Kl = K.copy()
    Kl[:, 0, :] *= HW[1] / W_IMG
    Kl[:, 1, :] *= HW[0] / H_IMG
    depth = (Dv[IDX] * glr_scale) if with_depth else None
    if with_depth and depth_frames is not None:   # PDA depth only on these label frames, NaN on the others
        keep = np.zeros(len(IDX), bool)
        keep[list(depth_frames)] = True
        depth = np.where(keep[:, None, None], depth, np.nan).astype(np.float32)
    glr = {"ok": True, "frame_idx": IDX, "c2w": np.linalg.inv(Tg[0])[None] @ Tg, "K": K, "K_label": Kl, "W": W_IMG, "H": H_IMG,
           "n_reg": len(IDX), "depth": depth, "depth_source": "pda_fine" if with_depth else "none",
           "conf": None, "model": "PINHOLE"}
    da3 = (Dv * 1.1 * (1 + da3_noise * rng.normal(size=Dv.shape))).astype(np.float32)
    wit = {"c2w": Tv, "K": np.tile(Kl[0], (N, 1, 1)), "D": Dv, "da3_D": da3, "static": np.ones_like(Dv, bool),
           "gate": {"others_med": gate_med}, "vipe_focal_cv": 0.01}
    if v13:
        wit["v13"] = {"c2w": Tv, "K": np.tile(Kl[0], (N, 1, 1)), "D": Dv, "W": HW[1]}
    job = {"clip": "c", "n_orig": N, "fps_orig": 15, "stride": STRIDE, "frame_idx": IDX.tolist(),
           "image_names": [f"{k + 1:06d}.jpg" for k in range(len(IDX))], "hw": [H_IMG, W_IMG]}
    return job, glr, stage1, wit


class QC(unittest.TestCase):
    def test_clean_case_is_t1_and_metric(self):
        job, glr, s1, wit = case()
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertEqual(qc["tier"], "T1")
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=4)
        Tv = _fix.look_at_rig(N, 0.05, curve=0.02)[IDX]
        L_label = np.linalg.norm(np.diff(label["c2w"][:, :3, 3], axis=0), axis=1).sum()
        L_vipe = np.linalg.norm(np.diff(Tv[:, :3, 3], axis=0), axis=1).sum()
        self.assertAlmostEqual(L_label / L_vipe, 1.0, places=4)       # the label is in metres
        self.assertLess(qc["xtraj_ba"], 1e-6)
        self.assertLess(qc["xdepth_k"], 0.1)
        self.assertLess(qc["kdis"], 1.0)
        self.assertEqual(label["frame_idx"].tolist(), IDX.tolist())
        self.assertEqual(label["depth"].shape, (len(IDX), *HW))
        self.assertEqual(set(["c2w", "K", "depth", "conf", "scale", "k_depth", "k_traj", "hfov", "k_source", "frame_hw"]) - set(label), set())

    def test_json_roundtrip(self):
        job, glr, s1, wit = case(v13=True)
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        json.dumps(C.clean(qc))
        self.assertIn("agreement_v13", qc)
        self.assertLess(qc["agreement_v13"]["traj_pct"], 1e-6)
        self.assertLess(qc["agreement_v13"]["depth_absrel"], 1e-3)

    def test_wrong_k_is_caught_by_the_witness(self):
        job, glr, s1, wit = case(gate_med=90.0)   # the image-only K sources say 90 deg, GLRMap's camera says 60
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertFalse(qc["accept"])
        self.assertEqual(qc["tier"], "T3")
        self.assertIn("v1c_gt_1", qc["reject_reasons"])
        self.assertGreater(qc["v1c_parts"]["kdis"], 1)

    def test_trajectory_drift_is_caught(self):
        job, glr, s1, wit = case(drift=3.0)
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertFalse(qc["accept"])
        self.assertGreater(qc["v1c_parts"]["xtraj_ba"], 1)

    def test_depth_disagreement_is_caught(self):
        job, glr, s1, wit = case(da3_noise=0.6)
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertGreater(qc["v1c_parts"]["xdepth_k"], 1)
        self.assertFalse(qc["accept"])

    def test_stage1_stop_rejects_even_if_parts_are_fine(self):
        s1 = {"verdict": "stop", "stop": ["planar_or_panoramic_pairs"], "suspect": [], "signals": {}}
        _, qc = GC.qc_one(*case(stage1=s1), CFG)
        self.assertFalse(qc["accept"])
        self.assertIn("s1_planar_or_panoramic_pairs", qc["reject_reasons"])

    def test_stage1_suspect_demotes_to_t2(self):
        s1 = {"verdict": "suspect", "stop": [], "suspect": ["focal_at_colmap_default"], "signals": {}}
        _, qc = GC.qc_one(*case(stage1=s1), CFG)
        self.assertTrue(qc["accept"])
        self.assertEqual(qc["tier"], "T2")
        self.assertIn("s1_focal_at_colmap_default", qc["info_flags"])

    def test_missing_depth_is_rejected_unless_allowed(self):
        label, qc = GC.qc_one(*case(with_depth=False), CFG)
        self.assertFalse(qc["accept"])                      # a label without depth breaks the label contract
        self.assertIn("depth_missing", qc["reject_reasons"])
        self.assertNotIn("xdepth_k_not_checkable", qc["reject_reasons"])   # no depth is reported once, not twice
        self.assertIn("depth_missing", qc["info_flags"])
        self.assertTrue(np.isnan(qc["xdepth_k"]))
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=4)   # trajectory vote alone
        self.assertEqual(label["depth_source"], "none")
        cfg = C.load_cfg(None)
        cfg["qc"]["allow_no_depth"] = True                  # explicit switch (camera-only experiments): accepted, but never T1
        _, qc = GC.qc_one(*case(with_depth=False), cfg)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertEqual(qc["tier"], "T2")

    def test_missing_or_failed_gate_is_a_rejection_never_an_ok(self):
        for s1, reason in ((None, "s1_gate_missing"), ({"verdict": "missing"}, "s1_gate_missing"),
                           ({"verdict": "error", "error": "boom"}, "s1_gate_error"), ({"verdict": "weird"}, "s1_gate_error")):
            _, qc = GC.qc_one(*case(stage1=s1), CFG)
            self.assertFalse(qc["accept"], s1)
            self.assertIn(reason, qc["reject_reasons"])
            self.assertEqual(qc["tier"], "T3")
        cfg = C.load_cfg(None)
        cfg["qc"]["require_gate"] = False                   # explicit switch for GLR results made without the worker
        _, qc = GC.qc_one(*case(stage1=None), cfg)
        self.assertTrue(qc["accept"])
        self.assertEqual(qc["tier"], "T2")                  # an unchecked stage 1 is never T1
        self.assertEqual(qc["stage1"]["verdict"], "skipped")

    def test_t1_needs_every_part_finite(self):
        job, glr, s1, wit = case()
        wit["gate"] = None                                  # no K witness: kdis cannot be computed
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertFalse(qc["accept"])
        self.assertIn("kdis_not_checkable", qc["reject_reasons"])
        cfg = C.load_cfg(None)
        cfg["qc"]["require_all_parts"] = False              # the lenient v1.3 reading: v1c = max of the parts that exist
        _, qc = GC.qc_one(job, glr, s1, wit, cfg)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertEqual(qc["tier"], "T2")                  # ... but a NaN part never counts as "small enough" for T1
        job, glr, s1, wit = case()
        wit["da3_D"] = None
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertIn("xdepth_k_not_checkable", qc["reject_reasons"])

    def test_partial_depth_coverage(self):
        job, glr, s1, wit = case(depth_frames=range(0, 27))        # 30 % of the label frames
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertTrue(qc["accept"], qc["reject_reasons"])
        self.assertEqual(qc["tier"], "T2")
        self.assertIn("depth_partial", qc["info_flags"])
        self.assertAlmostEqual(qc["glr"]["depth_cov"], 0.3, places=3)
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=3)
        _, qc = GC.qc_one(*case(depth_frames=range(0, 85)), CFG)   # 94 %
        self.assertEqual(qc["tier"], "T1")
        self.assertNotIn("depth_partial", qc["info_flags"])

    def test_scale_is_found_when_depth_exists_only_off_the_scale_frames(self):
        # GLR depth on 12 % of the frames and none on the frames an evenly spread pick over ALL frames would have used
        job, glr, s1, wit = case(depth_frames=range(3, 14))
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertAlmostEqual(label["scale"], 1 / S_GLR, places=3)
        self.assertNotIn("scale_unverified", qc["reject_reasons"])

    def test_no_scale_evidence_is_rejected_not_exported_as_metres(self):
        job, glr, s1, wit = case()
        wit["D"] = np.full_like(wit["D"], np.nan)          # the reference has no depth: the 2 %-of-scene-depth rule and the ratio both fail
        label, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertFalse(qc["accept"])
        self.assertIn("scale_unverified", qc["reject_reasons"])
        self.assertTrue(np.isnan(label["scale"]))

    def test_nothing_checkable_is_rejected(self):
        job, glr, s1, wit = case()
        wit = {"gate": None, "vipe_focal_cv": 0.0}
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertFalse(qc["accept"])
        self.assertIn("v1c_not_checkable", qc["reject_reasons"])

    def test_jump60_rejects(self):
        job, glr, s1, wit = case()
        c = glr["c2w"].copy()
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler("y", 80, degrees=True).as_matrix()
        c[40:, :3, :3] = c[40:, :3, :3] @ R   # an 80 deg view jump between two GLRMap frames
        glr["c2w"] = c
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertIn("jump60", qc["reject_reasons"])
        self.assertFalse(qc["accept"])

    def test_fov_out_rejects(self):
        job, glr, s1, wit = case(hfov_img=130.0, gate_med=130.0)
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertIn("fov_out", qc["reject_reasons"])

    def test_partial_registration_demotes(self):
        job, glr, s1, wit = case()
        job["frame_idx"] = job["frame_idx"] + [179, 181][:0]  # unchanged length; emulate 20 % unregistered below
        glr["n_reg"] = int(0.8 * len(IDX))
        _, qc = GC.qc_one(job, glr, s1, wit, CFG)
        self.assertIn("partial_registration", qc["info_flags"])
        self.assertEqual(qc["tier"], "T2")


@unittest.skipUnless(__import__("importlib").util.find_spec("pycolmap"), "pycolmap not installed")
class ReadGlr(unittest.TestCase):
    def test_read_glr_from_text_model_and_fused_depth(self):
        with tempfile.TemporaryDirectory() as td:
            res = Path(td) / "result"
            n = 12
            idx = list(range(0, 24, 2))
            names = [f"{k + 1:06d}.jpg" for k in range(n)]
            c2w = _fix.look_at_rig(n, 0.25)
            K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
            _fix.write_text_model(res / "stage1" / "sparse" / "0", c2w, K, 640, 360, _fix.scene_points(2000), names)
            (res / "dense" / "pda" / "fused").mkdir(parents=True)
            for k in range(n):
                if k == 5:
                    continue  # PDA skipped this frame (fewer than 5 anchors)
                np.savez(res / "dense" / "pda" / "fused" / f"{k + 1:06d}.npz", coarse=np.full((360, 640), 3.0, np.float32),
                         fine=np.full((360, 640), 4.0, np.float32))
            (res / "result.json").write_text(json.dumps({"found": {"sparse_models": ["stage1/sparse/0/cameras.txt"]}}))
            job = {"clip": "c", "image_names": names, "frame_idx": idx}
            g = GC.read_glr(res, job, 512)
            self.assertTrue(g["ok"])
            self.assertEqual(g["n_reg"], n)
            self.assertTrue(np.allclose(g["c2w"][0], np.eye(4), atol=1e-6))                 # first camera = world origin
            self.assertEqual(g["frame_idx"].tolist(), idx)
            self.assertEqual(g["depth_source"], "pda_fine")
            Hl, Wl = C.qc_grid(360, 640, 512)
            self.assertEqual(g["depth"].shape, (n, Hl, Wl))
            self.assertTrue(np.isnan(g["depth"][5]).all())                                  # skipped frame -> NaN map
            self.assertAlmostEqual(float(g["depth"][0, 0, 0]), 4.0)
            self.assertAlmostEqual(g["K_label"][0, 0, 0], 500.0 * Wl / 640, places=3)

    def test_no_model(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(GC.read_glr(Path(td), {"image_names": [], "frame_idx": []})["ok"])


class CollectFiles(unittest.TestCase):
    """glr_collect.py on job directories (no pycolmap needed: only paths whose result is a stub QC are exercised)."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def job(self, clip):
        return _fix.make_job(self.out, clip, n_orig=40)[0]

    @staticmethod
    def result(jd, name="result", status="stopped_at_gate", gate="stop", gate_json=None, pointer=True):
        rd = jd / name
        rd.mkdir()
        (rd / "result.json").write_text(json.dumps({"clip": jd.name, "status": status, "gate": gate}))
        if gate_json is not None:
            (rd / "gate.json").write_text(json.dumps(gate_json))
        if pointer:
            (jd / "RESULT_DIR").write_text(name + "\n")
        return rd

    def collect(self, *args):
        return subprocess.run([sys.executable, str(_fix.PIPE / "glr_collect.py"), str(self.out), *args], capture_output=True, text=True)

    def rows(self):
        with open(self.out / "results_glr.csv") as fh:
            return {r["clip"]: r for r in csv.DictReader(fh)}

    def qc(self, clip):
        return json.loads((self.out / "qc_glr" / f"{clip}.json").read_text())

    def test_no_glr_jobs_directory_gives_an_empty_table(self):
        r = self.collect()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out / "results_glr.csv") as fh:
            self.assertEqual(next(csv.reader(fh)), GC.ROW_FIELDS)
            self.assertEqual(list(csv.reader(fh)), [])

    def test_unfinished_jobs_are_incomplete_not_collect_errors(self):
        self.job("never_ran")
        jd = self.job("killed")
        (jd / "result").mkdir()                                   # started, no result.json (killed hard), no pointer
        jd = self.job("pointer_only")
        (jd / "result").mkdir()
        (jd / "RESULT_DIR").write_text("result\n")                # pointer to a directory without result.json (copy-back in progress)
        r = self.collect()
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = self.rows()
        for clip in ("never_ran", "killed", "pointer_only"):
            self.assertEqual((rows[clip]["accept"], rows[clip]["tier"], rows[clip]["reasons"]), ("False", "T3", "glr_incomplete"), clip)
        self.assertEqual(self.qc("pointer_only")["glr"]["note"], "result_dir_without_result_json")

    def test_failed_statuses_become_stub_rejections(self):
        for status, gate in (("interrupted", ""), ("gate_error", "error"), ("stage1_only", "ok"), ("failed", ""), ("native_gate_stop", "stop")):
            self.result(self.job(status), status=status, gate=gate, gate_json=None if not gate else {"verdict": gate, "stop": [], "suspect": []})
        self.assertEqual(self.collect().returncode, 0)
        rows = self.rows()
        for status in ("interrupted", "gate_error", "stage1_only", "failed", "native_gate_stop"):
            self.assertEqual(rows[status]["reasons"].split(";")[0], f"glr_{status}")
            self.assertEqual(rows[status]["accept"], "False")

    def test_pointer_is_optional_and_the_newest_result_wins(self):
        jd = self.job("c")
        self.result(jd, "result", status="failed", gate="", pointer=False)
        self.result(jd, "result_1", status="interrupted", gate="", pointer=False)
        self.result(jd, "result_2", status="gate_error", gate="error", gate_json={"verdict": "error"}, pointer=False)
        (jd / "result_10").mkdir()                                  # no result.json: ignored
        self.assertEqual(self.collect().returncode, 0)
        self.assertEqual(self.rows()["c"]["reasons"], "glr_gate_error")
        self.assertEqual(self.qc("c")["glr"]["note"], "result_dir_guessed")
        self.result(jd, "result_3", status="failed", gate="", pointer=True)
        (jd / "RESULT_DIR").write_text("result_1\n")               # the pointer wins over the guess
        self.collect()
        self.assertEqual(self.rows()["c"]["reasons"], "glr_interrupted")

    def test_gate_json_is_cross_checked_with_result_json(self):
        rd = self.out / "rd"
        rd.mkdir()
        (rd / "gate.json").write_text(json.dumps({"verdict": "ok", "stop": [], "suspect": []}))
        self.assertEqual(GC.read_gate(rd, {"gate": "ok"})["verdict"], "ok")
        self.assertEqual(GC.read_gate(rd, {"gate": ""})["verdict"], "ok")
        bad = GC.read_gate(rd, {"gate": "stop"})                      # the worker saw a stop, the file says ok: do not trust either
        self.assertEqual(bad["verdict"], "error")
        self.assertIn("result.json says 'stop'", bad["error"])
        (rd / "gate.json").write_text("{not json")
        self.assertEqual(GC.read_gate(rd, {"gate": "ok"})["verdict"], "error")
        (rd / "gate.json").write_text("[1, 2]")
        self.assertEqual(GC.read_gate(rd, {"gate": "ok"})["verdict"], "error")
        (rd / "gate.json").unlink()
        self.assertIsNone(GC.read_gate(rd, {"gate": "ok"}))           # absent: qc_one rejects it as s1_gate_missing
        self.assertEqual(GC.read_gate(rd, {"gate": "error"})["verdict"], "error")

    def test_partial_recollect_keeps_the_other_rows(self):
        self.result(self.job("a"), status="failed", gate="")
        self.result(self.job("b"), status="interrupted", gate="")
        self.collect()
        self.assertEqual(self.rows()["b"]["reasons"], "glr_interrupted")
        self.result(self.out / "glr_jobs" / "a", "result_1", status="gate_error", gate="error", gate_json={"verdict": "error"})
        self.assertEqual(self.collect("--clips", "a").returncode, 0)
        rows = self.rows()
        self.assertEqual((rows["a"]["reasons"], rows["b"]["reasons"]), ("glr_gate_error", "glr_interrupted"))
        self.assertEqual(list(rows), ["a", "b"])

    def test_cfg_option(self):
        self.result(self.job("a"), status="failed", gate="")
        r = self.collect("--cfg", str(self.out / "typo.yaml"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("typo.yaml", r.stderr)
        good = self.out / "c.yaml"
        good.write_text("qc:\n  tier1_part: 0.4\n")
        self.assertEqual(self.collect("--cfg", str(good)).returncode, 0)
        self.assertEqual(self.collect(f"--cfg={good}").returncode, 0)

    def test_model_selection_matches_the_workers_rule(self):
        rd = self.out / "rd"
        for rel in ("run/deep/er/sparse/0", "stage1/sparse/1", "stage1/sparse/0"):
            (rd / rel).mkdir(parents=True)
            (rd / rel / "cameras.bin").write_bytes(b"")
        self.assertEqual(GC.find_model_dir(rd), rd / "stage1" / "sparse" / "0")          # shallowest, ties by name
        (rd / "result.json").write_text(json.dumps({"gate_model": "run/deep/er/sparse/0"}))
        self.assertEqual(GC.find_model_dir(rd), rd / "run" / "deep" / "er" / "sparse" / "0")   # what the gate actually looked at
        (rd / "result.json").write_text(json.dumps({"gate_model": "gone/sparse/0"}))
        self.assertEqual(GC.find_model_dir(rd), rd / "stage1" / "sparse" / "0")


class WitnessNpz(unittest.TestCase):
    def test_roundtrip_and_cli(self):
        import subprocess
        import sys
        job, glr, s1, wit = case(v13=True)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            np.savez(td / "c.npz", c2w=wit["c2w"], K=wit["K"], D=wit["D"], da3_D=wit["da3_D"], static=wit["static"], others_med=60.3,
                     vipe_focal_cv=0.01, v13_c2w=wit["v13"]["c2w"], v13_K=wit["v13"]["K"], v13_D=wit["v13"]["D"])
            w2 = GC.load_witness_npz(td / "c.npz")
            _, qc = GC.qc_one(job, glr, s1, w2, CFG)
            self.assertTrue(qc["accept"])
            self.assertIn("agreement_v13", qc)


if __name__ == "__main__":
    unittest.main()
