import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix
import glr_merge as M


class Decide(unittest.TestCase):
    def test_filter_reject(self):
        r = M.decide({"route": "reject", "filter_reasons": "blur;cut"}, None, {"accept": "True"})
        self.assertEqual(r[2], False)
        self.assertIn("filter:blur", r[3])

    def test_glr_accepted(self):
        r = M.decide({"route": "glr"}, {"accept": "True", "tier": "T1"}, {"accept": "True"})
        self.assertEqual(r[:3], ("glr", "T1", True))
        self.assertFalse(r[4])

    def test_glr_failed_falls_back_to_v13(self):
        r = M.decide({"route": "glr"}, {"accept": "False", "reasons": "s1_planar_or_panoramic_pairs"}, {"accept": "True"})
        self.assertEqual((r[0], r[2], r[4]), ("v13", True, True))
        self.assertIn("glr:s1_planar", r[3])

    def test_glr_failed_and_v13_rejects(self):
        r = M.decide({"route": "glr"}, {"accept": "False", "reasons": "v1c_gt_1"}, {"accept": "False", "reject_flags": "jump60"})
        self.assertFalse(r[2])
        self.assertIn("v13:jump60", r[3])

    def test_glr_no_result_yet(self):
        r = M.decide({"route": "glr"}, None, {"accept": "True"})
        self.assertEqual((r[0], r[4]), ("v13", True))
        self.assertIn("no_result", r[3])

    def test_v13_route(self):
        self.assertEqual(M.decide({"route": "v13"}, None, {"accept": "True"})[:3], ("v13", "v13", True))
        self.assertFalse(M.decide({"route": "v13"}, None, {"accept": "False"})[2])
        self.assertFalse(M.decide({"route": "v13"}, None, None)[2])


class MergeFiles(unittest.TestCase):
    """final/ is a view that follows the current decisions: re-running the merge after the inputs changed must re-point or drop links."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = Path(self.td.name) / "run"
        for d in ("filter", "labels", "labels_glr"):
            (self.out / d).mkdir(parents=True)

    def tearDown(self):
        self.td.cleanup()

    def write(self, name, rows):
        with open(self.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)

    def label(self, kind, clip, tag):
        np.savez(self.out / kind / f"{clip}.npz", tag=tag)

    def merge(self):
        r = subprocess.run([sys.executable, str(_fix.PIPE / "glr_merge.py"), str(self.out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out / "final" / "results.csv") as fh:
            return {x["clip"]: x for x in csv.DictReader(fh)}

    def tag(self, clip):
        return str(np.load(self.out / "final" / "labels" / f"{clip}.npz")["tag"])

    def test_links_follow_the_decision_on_every_run(self):
        self.write("filter/glr_route.csv", [{"clip": "a", "route": "glr"}, {"clip": "b", "route": "glr"}, {"clip": "c", "route": "v13"}])
        self.write("results.csv", [{"clip": "a", "accept": "True"}, {"clip": "b", "accept": "True"}, {"clip": "c", "accept": "True"}])
        for clip in "abc":
            self.label("labels", clip, f"v13:{clip}")
        self.label("labels_glr", "a", "glr:a")
        self.label("labels_glr", "b", "glr:b")
        # run 1: the GLRMap jobs have not finished yet -> both glr clips fall back to the v1.3 labels
        rows = self.merge()
        self.assertEqual([rows[c]["annotator"] for c in "abc"], ["v13", "v13", "v13"])
        self.assertEqual([self.tag(c) for c in "abc"], ["v13:a", "v13:b", "v13:c"])
        # run 2: a finished with T1, b failed and v1.3 now rejects it, c is unchanged
        self.write("results_glr.csv", [{"clip": "a", "accept": "True", "tier": "T1", "v1c": "0.1"},
                                       {"clip": "b", "accept": "False", "reasons": "v1c_gt_1", "tier": "T3"}])
        self.write("results.csv", [{"clip": "a", "accept": "True"}, {"clip": "b", "accept": "False", "reject_flags": "jump60"},
                                   {"clip": "c", "accept": "True"}])
        rows = self.merge()
        self.assertEqual((rows["a"]["annotator"], rows["a"]["accept"]), ("glr", "True"))
        self.assertEqual(self.tag("a"), "glr:a")                                  # re-pointed to the GLRMap label
        self.assertEqual(rows["b"]["accept"], "False")
        self.assertFalse((self.out / "final" / "labels" / "b.npz").is_symlink())  # a rejected clip keeps no label
        self.assertFalse((self.out / "final" / "labels" / "b.npz").exists())
        self.assertEqual(self.tag("c"), "v13:c")
        # nothing but links was touched: every label file is intact
        self.assertEqual(sorted(p.name for p in (self.out / "labels").iterdir()), ["a.npz", "b.npz", "c.npz"])
        self.assertEqual(str(np.load(self.out / "labels" / "b.npz")["tag"]), "v13:b")
        # run 3: unchanged inputs -> same state, links stay relative so the run directory can be moved
        self.merge()
        moved = Path(self.td.name) / "moved"
        shutil.move(str(self.out), str(moved))
        self.out = moved
        self.assertEqual(self.tag("a"), "glr:a")
        self.assertFalse(os.path.isabs(os.readlink(self.out / "final" / "labels" / "a.npz")))

    def test_links_of_clips_that_left_the_route_table_are_removed_and_files_are_never_replaced(self):
        self.write("filter/glr_route.csv", [{"clip": "a", "route": "v13"}, {"clip": "b", "route": "v13"}])
        self.write("results.csv", [{"clip": "a", "accept": "True"}, {"clip": "b", "accept": "True"}])
        self.label("labels", "a", "v13:a")
        self.label("labels", "b", "v13:b")
        (self.out / "final" / "labels").mkdir(parents=True)
        (self.out / "final" / "labels" / "ghost.npz").symlink_to("../../labels/a.npz")      # a clip that is not in the route table any more
        (self.out / "final" / "labels" / "a.npz").write_bytes(b"a regular file somebody put here")
        rows = self.merge()
        self.assertFalse((self.out / "final" / "labels" / "ghost.npz").is_symlink())
        self.assertEqual((self.out / "final" / "labels" / "a.npz").read_bytes(), b"a regular file somebody put here")
        self.assertEqual(rows["a"]["accept"], "False")
        self.assertIn("final_label_is_a_file", rows["a"]["reason"])
        self.assertEqual(rows["b"]["accept"], "True")

    def test_missing_label_file_is_not_accepted(self):
        self.write("filter/glr_route.csv", [{"clip": "a", "route": "v13"}])
        self.write("results.csv", [{"clip": "a", "accept": "True"}])
        rows = self.merge()
        self.assertEqual(rows["a"]["accept"], "False")
        self.assertIn("label_file_missing", rows["a"]["reason"])

    def test_nothing_routed_to_glr_still_produces_final(self):
        self.write("filter/glr_route.csv", [{"clip": "a", "route": "v13"}])
        self.write("results.csv", [{"clip": "a", "accept": "True"}])
        self.label("labels", "a", "v13:a")
        rows = self.merge()                                                       # no results_glr.csv at all
        self.assertEqual(rows["a"]["annotator"], "v13")
        self.assertEqual(json.loads((self.out / "final" / "summary.json").read_text())["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
