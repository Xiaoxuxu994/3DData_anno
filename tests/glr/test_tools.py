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


@unittest.skipUnless(__import__("importlib").util.find_spec("pycolmap"), "pycolmap not installed")
class GateReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name) / "runs"
        K = np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]])
        X = _fix.scene_points(3000)
        for run, planar in (("w1/good", 0.0), ("w1/planar", 1.0), ("w2/good2", 0.0)):
            d = cls.root / run
            _fix.write_text_model(d / "sparse" / "0", _fix.look_at_rig(20, 0.25), K, 640, 360, X)
            _fix.write_db(d / "database.db", 20, planar_frac=planar)
        with open(cls.root / "labels.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "label"])
            w.writerows([["w1/good", "accurate"], ["w1/planar", "degenerate"], ["w2/good2", "accurate"]])

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def tool(self, *args):
        return subprocess.run([sys.executable, str(REPO / "tools" / "glr_gate_replay.py"), *map(str, args)], capture_output=True, text=True)

    def test_replay_report_and_output_safety(self):
        out = Path(self.td.name) / "new_dir" / "deeper" / "signals.csv"       # the directory does not exist yet
        r = self.tool(self.root, "--labels", self.root / "labels.csv", "--out", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out) as fh:
            rows = {x["run"]: x for x in csv.DictReader(fh)}
        self.assertEqual(sorted(rows), ["w1/good", "w1/planar", "w2/good2"])
        self.assertEqual(rows["w1/planar"]["verdict"], "stop")
        self.assertIn("planar_or_panoramic_pairs", rows["w1/planar"]["stop"])
        self.assertIn(rows["w1/good"]["verdict"], ("ok", "suspect"))
        self.assertAlmostEqual(float(rows["w1/good"]["focal_rel"]), 500 / 640, places=3)
        self.assertIn("catch non-acc", r.stdout)
        self.assertIn("pre-registered pass line", r.stdout)
        self.assertIn("PASS", r.stdout)                                        # the one non-accurate run is stopped, no accurate run is
        self.assertIn("2 window groups", r.stdout)
        before = out.read_text()
        r = self.tool(self.root, "--out", out)                                 # never overwrites a result
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exists", r.stderr)
        self.assertEqual(out.read_text(), before)
        self.assertNotEqual(self.tool(self.root).returncode, 0)                # --out is required: no default file in the cwd
        self.assertEqual([p.name for p in self.root.rglob("*") if p.name.startswith("database") and p.suffix not in (".db",)], [])   # read-only

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
