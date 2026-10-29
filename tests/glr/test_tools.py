import csv
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


if __name__ == "__main__":
    unittest.main()
