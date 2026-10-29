import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _fix

WORKER = _fix.PIPE / "glr_worker.sh"
HERE = Path(__file__).resolve().parent
HAS_PYCOLMAP = __import__("importlib").util.find_spec("pycolmap") is not None


def run(job, *args, env=None, glr_home=None):
    e = dict(os.environ, PY_GLR=sys.executable, GLRMAP_HOME=str(glr_home or job.parent), ANNO_THREADS="2")
    for k in ("GLR_STAGE1_CMD", "GLR_DENSE_CMD", "GLR_RUN_CMD", "GLR_TASKSET", "GLR_FORCE", "GLR_TIMEOUT"):
        e.pop(k, None)
    e.update(env or {})
    return subprocess.run(["bash", str(WORKER), str(job), *args], capture_output=True, text=True, env=e)


def stub(kind):
    return f"{{py}} {HERE}/_stub_stage1.py {{images}} {{s1}} {kind}"


@unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
class Worker(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = Path(self.td.name)
        self.jd, self.job = _fix.make_job(self.out, "clip", k_prior={"hfov_deg": 70.0})

    def tearDown(self):
        self.td.cleanup()

    def result(self):
        name = (self.jd / "RESULT_DIR").read_text().strip()
        return self.jd / name, json.loads((self.jd / name / "result.json").read_text())

    def test_split_stage1_only_ok(self):
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good")})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d, j = self.result()
        self.assertEqual((j["status"], j["gate"], j["dense_ran"]), ("stage1_only", "ok", False))
        self.assertTrue((d / "gate.json").exists())
        self.assertIn("stage1", j["commands"][0])
        self.assertEqual(len(j["found"]["sparse_models"]), 1)

    def test_split_gate_stop_then_force(self):
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("planar"), "GLR_DENSE_CMD": "touch {out}/dense_marker"})
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        d, j = self.result()
        self.assertEqual((j["status"], j["gate"]), ("stopped_at_gate", "stop"))
        self.assertFalse((d / "dense_marker").exists())            # the expensive stage did not run
        g = json.loads((d / "gate.json").read_text())
        self.assertIn("planar_or_panoramic_pairs", g["stop"])
        r = run(self.jd, "--mode=split", "--force", env={"GLR_STAGE1_CMD": stub("planar"), "GLR_DENSE_CMD": "touch {out}/dense_marker"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d2, j2 = self.result()
        self.assertNotEqual(d2.name, d.name)                        # never overwritten: result, result_1
        self.assertEqual((j2["status"], j2["gate"], j2["dense_ran"]), ("complete", "stop", True))
        self.assertTrue((d2 / "dense_marker").exists())

    def test_stage1_failure(self):
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("fail")})
        self.assertEqual(r.returncode, 5)
        self.assertEqual(self.result()[1]["status"], "failed")

    def test_gate_uses_job_k_witness(self):
        job = json.loads((self.jd / "job.json").read_text())
        job["k_prior"] = {"hfov_deg": 20.0}   # the stub camera has HFOV 65.2 deg -> disagreement
        (self.jd / "job.json").write_text(json.dumps(job))
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good")})
        self.assertEqual(r.returncode, 0)
        g = json.loads((self.result()[0] / "gate.json").read_text())
        self.assertEqual(g["verdict"], "suspect")
        self.assertIn("k_witness_disagree", g["suspect"])

    def test_mono_mode(self):
        r = run(self.jd, "--mode=mono", env={"GLR_RUN_CMD": f"bash {HERE}/_stub_run_all.sh"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d, j = self.result()
        self.assertEqual((j["status"], j["gate"], j["dense_ran"]), ("complete", "ok", True))
        self.assertEqual(len(j["found"]["pda_fused"]), 1)

    def test_mono_native_gate_stop(self):
        r = run(self.jd, "--mode=mono", env={"GLR_RUN_CMD": f"STUB_RC=3 bash {HERE}/_stub_run_all.sh"})
        self.assertEqual(r.returncode, 0)   # worker itself succeeded; the status carries the outcome
        self.assertEqual(self.result()[1]["status"], "native_gate_stop")

    def test_dry_run_creates_nothing(self):
        r = run(self.jd, "--mode=split", "--dry-run", env={"GLR_STAGE1_CMD": stub("good"), "GLR_DENSE_CMD": "echo dense {s1}"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("stage1", r.stdout)
        self.assertIn("dense", r.stdout)
        self.assertFalse((self.jd / "RESULT_DIR").exists())
        self.assertFalse((self.jd / "result").exists())

    def test_bad_arguments(self):
        self.assertEqual(run(self.jd, "--mode=bogus").returncode, 4)
        self.assertEqual(run(self.out / "nope").returncode, 4)
        e = {"GLRMAP_HOME": ""}
        self.assertEqual(run(self.jd, env=e).returncode, 4)


if __name__ == "__main__":
    unittest.main()
