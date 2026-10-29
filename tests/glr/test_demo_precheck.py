"""demo_precheck.py: the demo refuses to start the slow steps when a clip lacks the GeoCalib scores or the silver reference its stand-ins are built from."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evolution" / "E15_glr" / "demo_precheck.py"


def make(td):
    pool, s1 = Path(td) / "pool", Path(td) / "s1"
    for clip, scores, ref in (("good", {"calib": {"vfov_deg": [40.0]}, "flow": {}}, {"model": {"n_reg": 30}}),
                              ("noflow", {"calib": {"vfov_deg": [40.0]}}, {"model": {"n_reg": 30}}),
                              ("nocalib", {"flow": {}}, {"model": {"n_reg": 30}}),
                              ("noref", {"calib": {}, "flow": {}}, None),
                              ("emptyref", {"calib": {}, "flow": {}}, {"model": None})):
        (pool / clip).mkdir(parents=True)
        (pool / clip / "scores.json").write_text(json.dumps(scores))
        if ref is not None:
            (s1 / clip / "ref").mkdir(parents=True)
            (s1 / clip / "ref" / "stage1.json").write_text(json.dumps(ref))
    (pool / "nostage").mkdir()                                                            # a clip whose scores.json does not exist at all
    return pool, s1


class DemoPrecheck(unittest.TestCase):
    def test_what_is_fatal_and_what_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as td:
            pool, s1 = make(td)
            sys.path.insert(0, str(SCRIPT.parent))
            try:
                import demo_precheck as P
            finally:
                sys.path.pop(0)
            self.assertEqual(P.check(pool, s1, ["good"]), ([], []))
            m, w = P.check(pool, s1, ["noflow"])
            self.assertEqual(m, [])
            self.assertEqual(len(w), 1)
            self.assertIn("no optical-flow", w[0])                                         # warning only
            m, _ = P.check(pool, s1, ["nocalib", "noref", "emptyref", "nostage"])
            text = "\n".join(m)
            self.assertIn("nocalib: no GeoCalib scores", text)
            self.assertIn("noref: no silver reference", text)
            self.assertIn("emptyref: the silver reference registered no images", text)
            self.assertIn("nostage: no GeoCalib scores", text)
            self.assertIn("nostage: no silver reference", text)

    def test_cli_exit_status(self):
        with tempfile.TemporaryDirectory() as td:
            pool, s1 = make(td)
            ok = subprocess.run([sys.executable, str(SCRIPT), str(pool), str(s1), "good", "noflow"], capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("warning", ok.stderr)
            bad = subprocess.run([sys.executable, str(SCRIPT), str(pool), str(s1), "good", "nocalib"], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("missing inputs", bad.stderr)
            self.assertNotEqual(subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
