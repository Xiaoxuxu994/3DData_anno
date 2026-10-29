import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import _fix

WORKER = _fix.PIPE / "glr_worker.sh"
HERE = Path(__file__).resolve().parent
HAS_PYCOLMAP = __import__("importlib").util.find_spec("pycolmap") is not None
CPU0 = str(sorted(os.sched_getaffinity(0))[0]) if hasattr(os, "sched_getaffinity") else "0"


def environ(env=None, glr_home=None, job=None):
    e = dict(os.environ, PY_GLR=sys.executable, GLRMAP_HOME=str(glr_home or job.parent), ANNO_THREADS="2", GLR_ALLOW_UNCAPPED="1")
    for k in ("GLR_STAGE1_CMD", "GLR_DENSE_CMD", "GLR_RUN_CMD", "GLR_TASKSET", "GLR_FORCE", "GLR_TIMEOUT", "GLR_CAMERA_MODEL"):
        e.pop(k, None)
    e.update(env or {})
    for k in [k for k, v in e.items() if v is None]:
        del e[k]
    return e


def run(job, *args, env=None, glr_home=None, worker=WORKER):
    return subprocess.run(["bash", str(worker), str(job), *args], capture_output=True, text=True,
                          env=environ(env, glr_home, Path(job)))


def stub(kind):
    return f"{{py}} {HERE}/_stub_stage1.py {{images}} {{s1}} {kind}"


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


@unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
class Worker(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = Path(self.td.name)
        self.jd, self.job = _fix.make_job(self.out, "clip", n_orig=60, k_prior={"hfov_deg": 70.0})

    def tearDown(self):
        self.td.cleanup()

    def result(self, jd=None):
        jd = jd or self.jd
        name = (jd / "RESULT_DIR").read_text().strip()
        return jd / name, json.loads((jd / name / "result.json").read_text())

    def test_split_stage1_only_ok(self):
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good")})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d, j = self.result()
        self.assertEqual((j["status"], j["gate"], j["dense_ran"]), ("stage1_only", "ok", False))
        self.assertTrue((d / "gate.json").exists())
        self.assertIn("stage1", j["commands"][0])
        self.assertEqual(len(j["found"]["sparse_models"]), 1)
        self.assertEqual(j["n_frames"], 30)   # the job's frame count (not the number of image files that happen to exist)
        self.assertFalse((d / "RUNNING").exists())

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
        self.assertEqual(self.result()[1]["status"], "failed")     # a failed job still gets result.json + RESULT_DIR

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
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)      # stopped at the gate = exit 3, as in split mode
        self.assertEqual(self.result()[1]["status"], "native_gate_stop")

    def test_dry_run_creates_nothing(self):
        r = run(self.jd, "--mode=split", "--dry-run", env={"GLR_STAGE1_CMD": stub("good"), "GLR_DENSE_CMD": "echo dense {s1}",
                                                           "GLR_ALLOW_UNCAPPED": None})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("stage1", r.stdout)
        self.assertIn("dense", r.stdout)
        self.assertFalse((self.jd / "RESULT_DIR").exists())
        self.assertFalse((self.jd / "result").exists())

    def test_bad_arguments(self):
        self.assertEqual(run(self.jd, "--mode=bogus").returncode, 4)
        self.assertEqual(run(self.out / "nope").returncode, 4)
        self.assertEqual(run(self.jd, env={"GLRMAP_HOME": ""}).returncode, 4)
        self.assertEqual(run(self.jd, env={"GLR_CAMERA_MODEL": "PINHOLE; touch x"}).returncode, 4)
        self.assertFalse((self.jd / "result").exists())

    # ---- review fixes -----------------------------------------------------------------------------------------------------------
    def test_uncapped_host_is_refused_unless_acknowledged(self):
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good"), "GLR_ALLOW_UNCAPPED": None})
        self.assertEqual(r.returncode, 4)
        self.assertIn("GLR_TASKSET", r.stdout)
        self.assertFalse((self.jd / "result").exists())
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good"), "GLR_ALLOW_UNCAPPED": None, "GLR_TASKSET": CPU0})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_taskset_and_timeout_cover_the_whole_template(self):
        if not shutil.which("taskset") or not shutil.which("timeout"):
            self.skipTest("taskset / timeout missing")
        cmd = ("FOO=1 touch {out}/foo_marker; taskset -pc $$ > {out}/aff1; taskset -pc $$ > {out}/aff2 && "
               f"{{py}} -c 'import os; print(sorted(os.sched_getaffinity(0)))' > {{out}}/aff3")
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good") + "; " + cmd, "GLR_TASKSET": CPU0})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d, j = self.result()
        self.assertTrue((d / "foo_marker").exists())               # `VAR=x cmd` first word works under the wrapper
        for f in ("aff1", "aff2"):
            self.assertTrue((d / f).read_text().strip().endswith(f": {CPU0}"), (d / f).read_text())
        self.assertEqual((d / "aff3").read_text().strip(), f"[{CPU0}]")  # a later command of the template is capped too
        self.assertEqual(j["taskset"], CPU0)

    def test_timeout_kills_the_whole_template(self):
        if not shutil.which("timeout"):
            self.skipTest("timeout missing")
        t0 = time.time()
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": "sleep 30; touch {out}/after_sleep", "GLR_TIMEOUT": "2"})
        self.assertLess(time.time() - t0, 25)
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        d, j = self.result()
        self.assertEqual(j["status"], "failed")
        self.assertFalse((d / "after_sleep").exists())

    def test_hostile_clip_names_are_data_not_code(self):
        names = ["walk 4K", "walk (4K)", "a&b", "it's", "x;touch INJECTED;y", "drone#1", "q?x=1", 'dq"x', "back\\slash", "p%20q"]
        for name in names:
            with self.subTest(name=name):
                jd, _ = _fix.make_job(self.out, name, n_orig=40)
                home = self.out / "home"
                home.mkdir(exist_ok=True)
                r = run(jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good")}, glr_home=home)
                self.assertEqual(r.returncode, 0, name + "\n" + r.stdout + r.stderr)
                d, j = self.result(jd)
                self.assertEqual((j["status"], j["gate"], j["clip"]), ("stage1_only", "ok", name))
                self.assertEqual(sorted(p.name for p in home.iterdir()), [])  # nothing was injected into GLRMAP_HOME
                self.assertFalse(any(p.name == "INJECTED" for p in self.out.rglob("*")))   # `touch INJECTED` did not run
                self.assertTrue(all(p.is_dir() for p in (self.out / "glr_jobs").iterdir()))   # no stray file from a truncated sqlite URI

    def test_placeholders_are_quote_aware(self):
        jd, _ = _fix.make_job(self.out, "a b", n_orig=40)
        cmd = ("printf '%s' \"{images}|{threads}\" > \"{out}/p1\"; printf '%s' {s1}/sparse > {out}/p2; printf '%s' '{images}' > {out}/p3; "
               "printf '%s' {py} > {out}/p4; printf '%s|%s' {focal_w} {hfov} > {out}/p5")
        jj = json.loads((jd / "job.json").read_text())
        jj["k_prior"] = {"hfov_deg": 70.0}
        (jd / "job.json").write_text(json.dumps(jj))
        r = run(jd, "--mode=split", env={"GLR_STAGE1_CMD": cmd})
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)    # no model -> the gate stops
        d, j = self.result(jd)
        self.assertEqual((d / "p1").read_text(), f"{os.path.realpath(jd)}/images|2")
        self.assertEqual((d / "p2").read_text(), f"{os.path.realpath(d)}/stage1/sparse")
        self.assertEqual((d / "p3").read_text(), "{images}")       # single quotes: left alone
        self.assertEqual((d / "p4").read_text(), sys.executable)
        fw, hf = (d / "p5").read_text().split("|")
        self.assertAlmostEqual(float(fw), 0.5 / __import__("math").tan(__import__("math").radians(35.0)), places=4)   # the job's K prior
        self.assertEqual(float(hf), 70.0)
        self.assertEqual(j["status"], "stopped_at_gate")
        self.assertEqual(json.loads((d / "gate.json").read_text())["stop"], ["no_model"])

    def test_gate_error_is_never_ok_and_blocks_the_dense_stage(self):
        for kind in ("garbage", "nodb"):
            with self.subTest(kind=kind):
                jd, _ = _fix.make_job(self.out, f"g_{kind}", n_orig=40)
                env = {"GLR_STAGE1_CMD": stub(kind), "GLR_DENSE_CMD": "touch {out}/dense_marker"}
                r = run(jd, "--mode=split", env=env, glr_home=self.out)
                self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
                d, j = self.result(jd)
                self.assertEqual((j["status"], j["gate"], j["dense_ran"]), ("gate_error", "error", False))
                self.assertFalse((d / "dense_marker").exists())
                g = json.loads((d / "gate.json").read_text())
                self.assertEqual(g["verdict"], "error")
                self.assertTrue(g.get("error"))
                r = run(jd, "--mode=split", "--force", env=env, glr_home=self.out)   # --force does not override a failed gate
                self.assertEqual(r.returncode, 5)
                self.assertFalse((self.result(jd)[0] / "dense_marker").exists())

    def test_gate_that_crashes_without_output_is_an_error(self):
        wdir = self.out / "bin"
        wdir.mkdir()
        shutil.copy(WORKER, wdir / "glr_worker.sh")
        (wdir / "glr_stage1_gate.py").write_text("import sys\nsys.exit(1)\n")     # traceback-like: exit 1, no JSON
        env = {"GLR_STAGE1_CMD": stub("good"), "GLR_DENSE_CMD": "touch {out}/dense_marker"}
        r = run(self.jd, "--mode=split", env=env, worker=wdir / "glr_worker.sh")
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        d, j = self.result()
        self.assertEqual((j["status"], j["gate"]), ("gate_error", "error"))
        self.assertFalse((d / "dense_marker").exists())
        self.assertEqual(json.loads((d / "gate.json").read_text())["verdict"], "error")
        (wdir / "glr_stage1_gate.py").write_text("import json, sys\nprint(json.dumps({'gate': {'verdict': 'ok'}}))\n"
                                                 "open(sys.argv[sys.argv.index('--write') + 1], 'w').write('{\"verdict\": \"ok\"}')\nsys.exit(3)\n")
        r = run(self.jd, "--mode=split", env=env, worker=wdir / "glr_worker.sh")   # verdict and exit code disagree
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assertEqual(self.result()[1]["gate"], "error")

    def test_mono_gate_error_is_recorded(self):
        wdir = self.out / "bin"
        wdir.mkdir()
        shutil.copy(WORKER, wdir / "glr_worker.sh")
        (wdir / "glr_stage1_gate.py").write_text("import sys\nsys.exit(1)\n")
        r = run(self.jd, "--mode=mono", env={"GLR_RUN_CMD": f"bash {HERE}/_stub_run_all.sh"}, worker=wdir / "glr_worker.sh")
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assertEqual(self.result()[1]["status"], "complete_gate_error")

    def test_result_dir_is_written_last_and_sigterm_stops_the_whole_tree(self):
        cmd = 'mkdir -p "$OUTPUT"; echo $$ > "$OUTPUT/pid"; sleep 300 & wait'
        p = subprocess.Popen(["bash", str(WORKER), str(self.jd), "--mode=mono"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=environ({"GLR_RUN_CMD": cmd}, job=self.jd))
        try:
            pidf = self.jd / "result" / "run" / "pid"
            for _ in range(100):
                if pidf.exists() and pidf.read_text().strip():
                    break
                time.sleep(0.1)
            self.assertTrue(pidf.exists(), "the GLRMap command did not start")
            self.assertTrue((self.jd / "result" / "RUNNING").exists())
            self.assertFalse((self.jd / "RESULT_DIR").exists())      # still running: not a result yet
            self.assertFalse((self.jd / "result" / "result.json").exists())
            shell_pid = int(pidf.read_text())
            kids = subprocess.run(["pgrep", "-P", str(shell_pid)], capture_output=True, text=True).stdout.split()
            self.assertTrue(kids, "no child process found")
            p.send_signal(signal.SIGTERM)
            p.communicate(timeout=40)
        finally:
            if p.poll() is None:
                p.kill()
        self.assertEqual(p.returncode, 143)
        d, j = self.result()
        self.assertEqual(j["status"], "interrupted")
        self.assertFalse((d / "RUNNING").exists())
        time.sleep(0.3)
        self.assertFalse(alive(shell_pid))
        for k in kids:
            self.assertFalse(alive(int(k)), "an orphaned GLRMap child is still running")

    def test_broken_symlink_images_and_missing_python_fail_early(self):
        (self.jd / "images" / "000001.jpg").unlink()
        (self.jd / "images" / "000001.jpg").symlink_to("/nonexistent/host/frame.jpg")
        r = run(self.jd, "--mode=split", env={"GLR_STAGE1_CMD": stub("good")})
        self.assertEqual(r.returncode, 4)
        self.assertIn("rsync -aL", r.stdout)
        self.assertFalse((self.jd / "result").exists())
        r = run(self.jd, "--mode=split", env={"PY_GLR": "/bin/false", "GLR_STAGE1_CMD": stub("good")})
        self.assertEqual(r.returncode, 4)
        self.assertFalse((self.jd / "result").exists())


if __name__ == "__main__":
    unittest.main()
