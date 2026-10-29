import importlib.util
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix

REPO = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("pycolmap"), "pycolmap not installed")
class GateReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name) / "runs"
        K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
        X = _fix.scene_points(3000)
        runs = (("w1/good", 0.0), ("w1/planar", 1.0), ("w2/good2", 0.0), ("w3/silent1", 1.0), ("w3/silent2", 1.0), ("w3/silent3", 1.0))
        for run, planar in runs:
            d = cls.root / run
            _fix.write_text_model(d / "sparse" / "0", _fix.look_at_rig(20, 0.25), K, 640, 360, X)
            _fix.write_db(d / "database.db", 20, planar_frac=planar)
        with open(cls.root / "labels.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "label"])
            w.writerows([["w1/good", "accurate"], ["w1/planar", "degenerate"], ["w2/good2", "accurate"],
                         ["w3/silent1", "silent_wrong"], ["w3/silent2", "silent_wrong"], ["w3/silent3", "silent_wrong"]])
        for name, rows in (("labels_no_silent.csv", [["w1/good", "accurate"], ["w1/planar", "degenerate"]]),
                           ("labels_typo.csv", [["w1/gooood", "accurate"], ["nowhere/planar", "silent_wrong"]])):
            with open(cls.root / name, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["run", "label"])
                w.writerows(rows)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def tool(self, *args):
        return subprocess.run([sys.executable, str(REPO / "tools" / "glr_gate_replay.py"), *map(str, args)], capture_output=True, text=True)

    def test_replay_report_and_output_safety(self):
        out = Path(self.td.name) / "new_dir" / "deeper" / "signals.csv"       # the directory does not exist yet
        r = self.tool(self.root, "--labels", self.root / "labels.csv", "--out", out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(out) as fh:
            rows = {x["run"]: x for x in csv.DictReader(fh)}
        self.assertEqual(sorted(rows), ["w1/good", "w1/planar", "w2/good2", "w3/silent1", "w3/silent2", "w3/silent3"])
        self.assertEqual(rows["w1/planar"]["verdict"], "stop")
        self.assertIn("planar_or_panoramic_pairs", rows["w1/planar"]["stop"])
        self.assertIn(rows["w1/good"]["verdict"], ("ok", "suspect"))
        self.assertAlmostEqual(float(rows["w1/good"]["focal_rel"]), 500 / 640, places=3)
        self.assertIn("catch non-acc", r.stdout)
        self.assertIn("pass line (catch >= 7/9 silent, >= 90 % non-accurate, false stops <= 2): PASS", r.stdout)
        self.assertIn("3 window groups", r.stdout)
        before = out.read_text()
        r = self.tool(self.root, "--out", out)                                 # never overwrites a result
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exists", r.stderr)
        self.assertEqual(out.read_text(), before)
        self.assertNotEqual(self.tool(self.root).returncode, 0)                # --out is required: no default file in the cwd
        self.assertEqual([p.name for p in self.root.rglob("*") if p.name.startswith("database") and p.suffix not in (".db",)], [])   # read-only

    def test_the_pass_line_is_never_vacuous(self):
        out = Path(self.td.name) / "inc1.csv"
        r = self.tool(self.root, "--labels", self.root / "labels_no_silent.csv", "--out", out)       # no silent_wrong run labelled
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("INCONCLUSIVE - no labelled silent_wrong runs", r.stdout)
        self.assertNotIn(": PASS", r.stdout)
        out = Path(self.td.name) / "inc2.csv"
        r = self.tool(self.root, "--labels", self.root / "labels_typo.csv", "--out", out)            # keys that match nothing
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("WARNING: label keys that matched no run", r.stdout)
        self.assertIn("nowhere/planar", r.stdout)
        self.assertIn("0/2 label keys matched", r.stdout)
        # a gate that misses the silent runs FAILS: relabel the good runs as silent_wrong
        with open(self.root / "labels_fail.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "label"])
            w.writerows([["w1/good", "silent_wrong"], ["w2/good2", "silent_wrong"], ["w3/silent1", "accurate"], ["w3/silent2", "accurate"],
                         ["w3/silent3", "accurate"], ["w1/planar", "degenerate"]])
        r = self.tool(self.root, "--labels", self.root / "labels_fail.csv", "--out", Path(self.td.name) / "fail.csv")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn(": FAIL", r.stdout)

    def test_unreadable_run_is_reported_not_silently_dropped(self):
        root = Path(self.td.name) / "runs_bad"
        K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
        _fix.write_text_model(root / "r" / "sparse" / "0", _fix.look_at_rig(20, 0.25), K, 640, 360, _fix.scene_points(500))
        (root / "r" / "database.db").write_bytes(b"not a database")
        out = Path(self.td.name) / "bad.csv"
        r = self.tool(root, "--out", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out) as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["verdict"], "error")
        self.assertIn("WARNING: 1 runs could not be read", r.stdout)


class RewriteCamera(unittest.TestCase):
    """tools/stage1_sfm.py rewrite_camera (the ablation switch behind --reuse-db), extracted from the script: it parses its arguments at import."""

    @classmethod
    def setUpClass(cls):
        import ast
        import sqlite3
        src = ast.parse((REPO / "tools" / "stage1_sfm.py").read_text())
        keep = [n for n in src.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("MODEL_ID", "LAYOUT") for t in n.targets))
                or (isinstance(n, ast.FunctionDef) and n.name in ("rewrite_camera", "put_model_first"))]
        cls.ns = {"np": np, "sqlite3": sqlite3, "sys": sys}
        exec(compile(ast.Module(keep, []), "stage1_sfm.py", "exec"), cls.ns)

    def db(self, td, model_id, params, w=640, h=360):
        import sqlite3
        path = Path(td) / "database.db"
        con = sqlite3.connect(path)
        con.execute("create table cameras (camera_id integer primary key, model integer, width integer, height integer, params blob, prior_focal_length integer)")
        con.execute("insert into cameras values (1, ?, ?, ?, ?, 1)", (model_id, w, h, np.array(params, np.float64).tobytes()))
        con.commit()
        con.close()
        return path

    def read(self, path):
        import sqlite3
        mid, blob, prior = sqlite3.connect(path).execute("select model, params, prior_focal_length from cameras").fetchone()
        return mid, np.frombuffer(blob, np.float64), prior

    def test_largest_model_is_put_at_sparse_0(self):
        """the incremental mapper numbers models in build order; sparse/0 must be the one stage1.json describes (the gate / worker read the shortest path)"""
        put = self.ns["put_model_first"]
        with tempfile.TemporaryDirectory() as td:
            sp = Path(td) / "sparse"
            for k in (0, 1, 2):
                (sp / str(k)).mkdir(parents=True)
                (sp / str(k) / "who").write_text(f"model{k}")
            z = put(sp, 2)
            self.assertEqual(z, sp / "0")
            self.assertEqual((sp / "0" / "who").read_text(), "model2")            # the best is now at 0 ...
            self.assertEqual((sp / "2" / "who").read_text(), "model0")            # ... the old 0 took its number, nothing was lost
            self.assertEqual((sp / "1" / "who").read_text(), "model1")
            self.assertFalse((sp / "_swap").exists())
            self.assertEqual(put(sp, 0), sp / "0")                                # already first: untouched
            self.assertEqual((sp / "0" / "who").read_text(), "model2")
            only = Path(td) / "only"
            (only / "3").mkdir(parents=True)                                      # a single model that is not numbered 0
            self.assertEqual(put(only, 3), only / "0")
            self.assertTrue((only / "0").is_dir() and not (only / "3").exists())

    def test_conversions_keep_the_principal_point(self):
        rw = self.ns["rewrite_camera"]
        with tempfile.TemporaryDirectory() as td:
            d = self.db(td, 2, [500.0, 321.0, 181.0, -0.05])                  # SIMPLE_RADIAL: [f, cx, cy, k1] - the old code read cy from k1
            rw(d, "PINHOLE", 0.9)
            mid, p, prior = self.read(d)
            self.assertEqual((mid, prior), (1, 0))
            self.assertTrue(np.allclose(p, [576.0, 576.0, 321.0, 181.0]))
        with tempfile.TemporaryDirectory() as td:
            d = self.db(td, 1, [700.0, 710.0, 320.0, 180.0])                  # PINHOLE -> SIMPLE_PINHOLE keeps fx
            rw(d, "SIMPLE_PINHOLE", 0.0)
            mid, p, _ = self.read(d)
            self.assertEqual(mid, 0)
            self.assertTrue(np.allclose(p, [700.0, 320.0, 180.0]))
        with tempfile.TemporaryDirectory() as td:
            d = self.db(td, 3, [500.0, 320.0, 180.0, -0.05, 0.01])            # no --db-camera: keep the model (and its distortion), set the focal
            rw(d, "", 1.2)
            mid, p, _ = self.read(d)
            self.assertEqual(mid, 3)
            self.assertTrue(np.allclose(p, [768.0, 320.0, 180.0, -0.05, 0.01]))
        with tempfile.TemporaryDirectory() as td:
            d = self.db(td, 1, [700.0, 710.0, 320.0, 180.0])
            rw(d, "", 1.0)
            self.assertTrue(np.allclose(self.read(d)[1], [640.0, 640.0, 320.0, 180.0]))   # PINHOLE: fx = fy = the prior

    def test_unsupported_models_are_refused(self):
        rw = self.ns["rewrite_camera"]
        with tempfile.TemporaryDirectory() as td:
            d = self.db(td, 99, [1.0, 2.0, 3.0])
            with self.assertRaises(SystemExit):
                rw(d, "PINHOLE", 1.0)
        with tempfile.TemporaryDirectory() as td:
            d = self.db(td, 1, [700.0, 710.0, 320.0, 180.0])
            with self.assertRaises(SystemExit):
                rw(d, "RADIAL", 1.0)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class FetchOodSet(unittest.TestCase):
    def test_fetch_cut_verify_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "host" / "clip_a.mp4"
            src.parent.mkdir()
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=6",
                            "-pix_fmt", "yuv420p", str(src)], check=True)
            good = hashlib.sha256(src.read_bytes()).hexdigest()
            clips = [{"id": "ok1", "role": "easy", "license": "test", "page_url": "p", "download_url": src.as_uri(), "raw_name": "clip_a.mp4",
                      "raw_sha256": good, "window_start": 1.0, "window_dur": 3.0},
                     {"id": "bad_hash", "role": "easy", "license": "test", "page_url": "p", "download_url": src.as_uri(), "raw_name": "clip_a.mp4",
                      "raw_sha256": "0" * 64, "window_start": 0.0, "window_dur": 2.0},
                     {"id": "ctl1", "role": "control_negative", "license": "test", "page_url": "p", "download_url": src.as_uri(), "raw_name": "c.mp4",
                      "raw_sha256": good, "window_start": 0.0, "window_dur": 2.0}]
            man = td / "man.json"
            man.write_text(json.dumps({"clips": clips}))
            run = lambda *a: subprocess.run([sys.executable, str(REPO / "tools" / "fetch_ood_set.py"), str(man), str(td / "out"), *a],  # noqa: E731
                                            capture_output=True, text=True)
            r = run()
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)                      # the hash mismatch is reported, not silently accepted
            self.assertIn("bad_hash", r.stderr)
            self.assertTrue((td / "out" / "clips" / "ok1.mp4").exists() and (td / "out" / "clips" / "ctl1.mp4").exists())
            self.assertFalse((td / "out" / "clips" / "bad_hash.mp4").exists())
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=r_frame_rate,width,height:format=duration", "-of", "json",
                                    str(td / "out" / "clips" / "ok1.mp4")], capture_output=True, text=True)
            info = json.loads(probe.stdout)
            self.assertEqual(info["streams"][0]["r_frame_rate"], "15/1")
            self.assertAlmostEqual(float(info["format"]["duration"]), 3.0, delta=0.2)
            lines = (td / "out" / "clips.txt").read_text().split()
            self.assertEqual(sorted(Path(x).name for x in lines), ["ctl1.mp4", "ok1.mp4"])
            before = (td / "out" / "clips" / "ok1.mp4").read_bytes()
            r = run("--only", "ok1")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((td / "out" / "clips" / "ok1.mp4").read_bytes(), before)    # an existing clip is kept
            r = run("--role", "control_negative")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(run("--only", "bad_hash", "--no-verify").returncode, 0)


if __name__ == "__main__":
    unittest.main()
