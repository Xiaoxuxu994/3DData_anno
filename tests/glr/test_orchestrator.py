"""pipeline/run_batch_glr.sh: usage guards, phase handling, GPU / CPU-cap forwarding, job skipping and failure reporting.
Runs the real scripts against a throw-away ANNO_ROOT (code -> this checkout, cache/qc-venv/bin/python -> the interpreter running the tests)."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _fix

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "pipeline" / "run_batch_glr.sh"
HERE = Path(__file__).resolve().parent
HAS_PYCOLMAP = __import__("importlib").util.find_spec("pycolmap") is not None


class Orchestrator(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name) / "root"
        (self.root / "code").parent.mkdir(parents=True)
        (self.root / "code").symlink_to(REPO)
        py = self.root / "cache" / "qc-venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')   # a wrapper, not a symlink: a venv python needs its own pyvenv.cfg
        py.chmod(0o755)
        self.out = self.root / "outputs" / "run"
        self.home = Path(self.td.name) / "glrmap"
        self.home.mkdir()

    def tearDown(self):
        self.td.cleanup()

    def env(self, **kw):
        e = dict(os.environ, ANNO_ROOT=str(self.root), ANNO_CONTAINER="", ANNO_THREADS="2", PY_GLR=sys.executable, GLRMAP_HOME=str(self.home))
        for k in ("GLR_TASKSET", "GLR_ALLOW_UNCAPPED", "GLR_RUN_CMD", "GLR_STAGE1_CMD", "GLR_DENSE_CMD", "GLR_STAGE1_ONLY", "CUDA_VISIBLE_DEVICES",
                  "GLR_TIMEOUT", "GLR_FORCE"):
            e.pop(k, None)
        e.update({k: v for k, v in kw.items() if v is not None})
        return e

    def run_batch(self, *args, **env):
        return subprocess.run(["bash", str(SCRIPT), *map(str, args)], capture_output=True, text=True, env=self.env(**env), cwd=self.td.name)

    def make_run(self, glr_clips=("c1",), v13_clips=()):
        """OUT of a finished phase a: route table, v1.3 results and labels, and job packages for the GLR clips."""
        import numpy as np
        for d in ("filter", "labels"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        rows = [{"clip": c, "route": "glr"} for c in glr_clips] + [{"clip": c, "route": "v13"} for c in v13_clips]
        with open(self.out / "filter" / "glr_route.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["clip", "route"])
            w.writeheader()
            w.writerows(rows)
        with open(self.out / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["clip", "accept", "v1c"])
            w.writeheader()
            w.writerows([{"clip": r["clip"], "accept": "True", "v1c": "0.3"} for r in rows])
        for r in rows:
            np.savez(self.out / "labels" / f"{r['clip']}.npz", tag=r["clip"])
        for c in glr_clips:
            _fix.make_job(self.out, c, n_orig=40)

    def job(self, clip="c1"):
        return self.out / "glr_jobs" / clip

    # ---- usage -------------------------------------------------------------------------------------------------------------------
    def test_usage_errors(self):
        r = self.run_batch()
        self.assertEqual(r.returncode, 2)
        self.assertIn("Usage:", r.stdout)
        r = self.run_batch("")                                                      # empty OUT must not create /logs
        self.assertEqual(r.returncode, 2)
        r = self.run_batch(self.out)                                                # --phase=all needs inputs
        self.assertEqual(r.returncode, 2)
        self.assertFalse(self.out.exists())
        for bad in ("--phase=x", "--worker-mode=splt", "--bogus"):
            self.assertEqual(self.run_batch(self.out, "a.mp4", bad).returncode, 2, bad)
        self.assertFalse(self.out.exists())

    def test_phase_c_without_inputs_and_without_any_glr_job(self):
        self.make_run(glr_clips=(), v13_clips=("d1", "d2"))
        r = self.run_batch(self.out, "--phase=c")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(self.out / "final" / "results.csv") as fh:
            rows = {x["clip"]: x for x in csv.DictReader(fh)}
        self.assertEqual({k: v["accept"] for k, v in rows.items()}, {"d1": "True", "d2": "True"})
        self.assertTrue((self.out / "final" / "labels" / "d1.npz").exists())
        with open(self.out / "results_glr.csv") as fh:
            self.assertEqual(list(csv.reader(fh))[1:], [])

    def test_dry_run_creates_nothing_and_prints_the_plan(self):
        self.make_run()
        before = sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*"))
        r = self.run_batch(self.out, "--phase=b", "--dry-run", "--worker-mode=split", GLR_DENSE_CMD="echo dense {s1}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("[dry-run] GLRMap: c1", r.stdout)
        self.assertIn("dense", r.stdout)
        self.assertEqual(sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*")), before)     # no logs/, no result dirs
        r = self.run_batch(self.out, "--phase=c", "--dry-run")
        self.assertIn("[dry-run]", r.stdout)
        self.assertNotIn("--witness-dir", r.stdout)
        r = self.run_batch(self.out, "--phase=c", "--dry-run", "--witness-dir=/some/dir")
        self.assertIn("glr_collect.py", r.stdout)
        self.assertIn("--witness-dir /some/dir", r.stdout)
        self.assertEqual(sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*")), before)

    # ---- phase b preconditions ---------------------------------------------------------------------------------------------------
    def test_phase_b_refuses_to_start_without_the_cpu_cap_gpu_and_dense_command(self):
        self.make_run()
        r = self.run_batch(self.out, "--phase=b")
        self.assertEqual(r.returncode, 2)
        self.assertIn("GLR_TASKSET", r.stdout)
        r = self.run_batch(self.out, "--phase=b", GLR_ALLOW_UNCAPPED="1", GLRMAP_HOME="")
        self.assertEqual(r.returncode, 2)
        self.assertIn("GLRMAP_HOME", r.stdout)
        r = self.run_batch(self.out, "--phase=b", "--worker-mode=split", GLR_ALLOW_UNCAPPED="1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("GLR_DENSE_CMD", r.stdout)
        self.assertFalse((self.job() / "result").exists())

    @unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
    def test_phase_b_runs_skips_and_reruns_and_forwards_gpus(self):
        self.make_run()
        cmd = f'mkdir -p "$OUTPUT"; echo "$CUDA_VISIBLE_DEVICES" > "$OUTPUT/cvd"; bash {HERE}/_stub_run_all.sh'
        kw = dict(GLR_ALLOW_UNCAPPED="1", GLR_RUN_CMD=cmd)
        r = self.run_batch(self.out, "--phase=b", "--gpus=3,4", **kw)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.job() / "result"
        self.assertEqual((res / "run" / "cvd").read_text().strip(), "3,4")                 # --gpus reaches the GLRMap command
        self.assertEqual(json.loads((res / "result.json").read_text())["status"], "complete")
        r = self.run_batch(self.out, "--phase=b", "--gpus=3", **kw)
        self.assertIn("has a result, skipped: c1", r.stdout)
        self.assertFalse((self.job() / "result_1").exists())
        # an interrupted result is not a result: the job runs again, into a new directory
        (res / "result.json").write_text(json.dumps({"clip": "c1", "status": "interrupted", "gate": ""}))
        r = self.run_batch(self.out, "--phase=b", "--gpus=3", **kw)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.job() / "result_1" / "result.json").exists())
        self.assertEqual((self.job() / "RESULT_DIR").read_text().strip(), "result_1")

    @unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
    def test_failed_jobs_are_reported_and_retried_only_on_request(self):
        self.make_run(glr_clips=("c1", "c2"))
        r = self.run_batch(self.out, "--phase=b", GLR_ALLOW_UNCAPPED="1", GLR_RUN_CMD="exit 7")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)                              # worker rc 5 for both jobs -> orchestrator fails
        self.assertIn("GLRMap jobs that failed", r.stdout)
        self.assertIn("c1(rc=5)", r.stdout)
        self.assertIn("c2(rc=5)", r.stdout)
        r = self.run_batch(self.out, "--phase=b", GLR_ALLOW_UNCAPPED="1", GLR_RUN_CMD="exit 7")
        self.assertEqual(r.returncode, 0, r.stdout)                                          # failed jobs are not retried automatically
        self.assertIn("has a result, skipped: c1", r.stdout)
        r = self.run_batch(self.out, "--phase=b", "--glr-retry", GLR_ALLOW_UNCAPPED="1", GLR_RUN_CMD=f"bash {HERE}/_stub_run_all.sh")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads((self.job("c2") / "result_1" / "result.json").read_text())["status"], "complete")

    @unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
    def test_stopped_at_the_gate_is_a_normal_outcome(self):
        self.make_run()
        cmd = f"STUB_RC=3 bash {HERE}/_stub_run_all.sh"          # GLRMap's own camera_sanity stop
        r = self.run_batch(self.out, "--phase=b", GLR_ALLOW_UNCAPPED="1", GLR_RUN_CMD=cmd)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("rc=3", r.stdout)
        self.assertNotIn("failed", r.stdout)

    @unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
    def test_whole_phase_c_after_a_worker_run(self):
        """worker result -> collect (real witnesses are not available: the clip is reported as such, never as accepted) -> merge falls back to v1.3."""
        self.make_run(glr_clips=("c1",), v13_clips=("d1",))
        self.assertEqual(self.run_batch(self.out, "--phase=b", GLR_ALLOW_UNCAPPED="1", GLR_RUN_CMD=f"bash {HERE}/_stub_run_all.sh").returncode, 0)
        r = self.run_batch(self.out, "--phase=c")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(self.out / "final" / "results.csv") as fh:
            rows = {x["clip"]: x for x in csv.DictReader(fh)}
        self.assertEqual(rows["d1"]["annotator"], "v13")
        self.assertEqual(rows["c1"]["accept"], "True")
        self.assertEqual((rows["c1"]["annotator"], rows["c1"]["fallback"]), ("v13", "True"))   # GLR label rejected (no witnesses), v1.3 label used
        with open(self.out / "results_glr.csv") as fh:
            glr = {x["clip"]: x for x in csv.DictReader(fh)}
        self.assertEqual(glr["c1"]["accept"], "False")
        self.assertEqual(rows["c1"]["reason"].split(";")[0][:4], "glr:")


if __name__ == "__main__":
    unittest.main()
