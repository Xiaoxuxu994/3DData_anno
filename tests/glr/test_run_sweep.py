"""evolution/E15_glr/run_sweep.sh: the printed commands are the recipe of the E15 tables (same arguments as the queue that produced them)."""
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evolution" / "E15_glr" / "run_sweep.sh"


def run(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True)


class RunSweep(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.frames = Path(self.td.name) / "frames"
        self.frames.mkdir()
        self.out = Path(self.td.name) / "clip"

    def tearDown(self):
        self.td.cleanup()

    def cmds(self, r):
        return [ln[2:] for ln in r.stdout.splitlines() if ln.startswith("+ ")]

    def test_recipe_order_and_arguments(self):
        r = run(str(self.frames), str(self.out), "--focal-w", "0.8", "--hq960", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        c = self.cmds(r)
        self.assertEqual(len(c), 7)
        names = [x.split("--out ")[1].split()[0].rsplit("/", 1)[1] for x in c]
        self.assertEqual(names, ["v0a", "v0b", "v1sp", "v2prior", "v3fixed", "ref", "v0a_hq960"])
        self.assertIn("--stride 3 --long 640 --overlap 12 --guided 0 --seed 1", c[0])
        self.assertNotIn("--reuse-db", c[0])
        for k in (1, 2, 3, 4):                                            # variants reuse the matching database of v0a: only the mapper runs again
            self.assertIn(f"--reuse-db {self.out}/v0a", c[k])
        self.assertIn("--seed 2", c[1])
        self.assertIn("--db-camera SIMPLE_PINHOLE", c[2])
        self.assertIn("--focal-w 0.8", c[3])
        self.assertNotIn("--fixed-focal", c[3])
        self.assertIn("--focal-w 0.8 --fixed-focal 1", c[4])
        self.assertIn("--stride 6 --long 640 --camera-model SIMPLE_RADIAL --matcher exhaustive --mapper incremental", c[5])
        self.assertIn("--long 960", c[6])
        self.assertFalse(self.out.exists())                                # a dry run writes nothing

    def test_without_prior_the_prior_variants_are_skipped(self):
        r = run(str(self.frames), str(self.out), "--no-ref", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.cmds(r)), 3)
        self.assertIn("v2prior and v3fixed skipped", r.stdout)

    def test_refuses_existing_output_missing_frames_and_implausible_focal(self):
        self.out.mkdir()
        self.assertEqual(run(str(self.frames), str(self.out), "--dry-run").returncode, 2)
        self.assertEqual(run(str(self.frames) + "_nope", str(self.out) + "_new", "--dry-run").returncode, 2)
        self.assertEqual(run(str(self.frames), str(self.out) + "_new", "--focal-w", "850", "--dry-run").returncode, 2)     # pixels, not focal / width
        self.assertEqual(run(str(self.frames), str(self.out) + "_new", "--focal-w", "x", "--dry-run").returncode, 2)
        self.assertEqual(run(str(self.frames), str(self.out) + "_new", "--bogus", "--dry-run").returncode, 2)
        self.assertEqual(run().returncode, 2)


if __name__ == "__main__":
    unittest.main()
