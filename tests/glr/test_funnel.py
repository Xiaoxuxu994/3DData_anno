"""funnel.py: first layer that stops a clip (rules -> VLM -> stage-1 gate), how many layers would stop it alone, missing layers are 'incomplete' (never 'passed')."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evolution" / "E15_glr" / "funnel.py"
HAS_MPL = importlib.util.find_spec("matplotlib") is not None

DEC = [  # clip, rule_pass, rule_reasons, vlm_pass, reason (decide.py layout)
    ("easy_a", True, "", True, ""),
    ("easy_b", True, "", False, "vlm:no_unreconstructable"),
    ("ctl_c", False, "static_camera", True, "static_camera"),
    ("ctl_d", True, "", True, ""),
    ("ctl_e", True, "", True, ""),
    ("ctl_f", False, "cut", True, "cut"),
]
SWEEP = [("easy_a", "ok", ""), ("easy_b", "stop", "fx_fy_mismatch"), ("ctl_c", "stop", "points_per_image"), ("ctl_d", "stop", "planar_or_panoramic_pairs"),
         ("ctl_f", "suspect", "reproj_gt_1px")]       # ctl_e has no stage-1 result yet
MANIFEST = {"clips": [{"id": "easy_a", "role": "easy", "category": "drone_orbit"}, {"id": "easy_b", "role": "easy", "category": "indoor_walkthrough"},
                      {"id": "ctl_c", "role": "control_negative", "expected_failure": "static_camera"},
                      {"id": "ctl_d", "role": "control_negative", "expected_failure": "no_parallax_rotation"},
                      {"id": "ctl_e", "role": "control_negative", "expected_failure": "driving_low_parallax"},
                      {"id": "ctl_f", "role": "control_negative", "expected_failure": "cut"}]}


def write(td):
    td = Path(td)
    with open(td / "decisions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "decision", "reason", "rule_pass", "rule_reasons", "vlm_pass"])
        for clip, rp, rr, vp, reason in DEC:
            w.writerow([clip, "pass" if rp and vp else "reject", reason, rp, rr, vp])
    with open(td / "sweep_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "run", "verdict", "flags", "role"])
        for clip, v, fl in SWEEP:
            w.writerow([clip, "v0a", v, fl, ""])
        w.writerow(["easy_a", "v1sp", "stop", "x", ""])          # other runs must not leak into the v0a view
    (td / "manifest.json").write_text(json.dumps(MANIFEST))


class Funnel(unittest.TestCase):
    def test_first_stop_and_layer_counts(self):
        with tempfile.TemporaryDirectory() as td:
            write(td)
            out = Path(td) / "out"
            r = subprocess.run([sys.executable, str(SCRIPT), str(Path(td) / "decisions.csv"), str(Path(td) / "sweep_runs.csv"), str(out), "--manifest",
                                str(Path(td) / "manifest.json"), "--no-figure"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out / "funnel.csv") as fh:
                rows = {x["clip"]: x for x in csv.DictReader(fh)}
            self.assertEqual(rows["easy_a"]["first_stop"], "passed_all")
            self.assertEqual(rows["easy_b"]["first_stop"], "vlm")                  # the VLM comes before the gate, which would also have stopped it
            self.assertEqual(rows["easy_b"]["n_layers_stopping"], "2")
            self.assertEqual(rows["easy_b"]["vlm_reason"], "no_unreconstructable")
            self.assertEqual(rows["ctl_c"]["first_stop"], "rules")
            self.assertEqual(rows["ctl_c"]["rules_reason"], "static_camera")
            self.assertEqual(rows["ctl_d"]["first_stop"], "gate")
            self.assertEqual(rows["ctl_d"]["n_layers_stopping"], "1")               # only the gate catches this one: the layer a pipeline without it would miss
            self.assertEqual(rows["ctl_e"]["first_stop"], "incomplete")             # no stage-1 result: never reported as passed
            self.assertEqual(rows["ctl_e"]["gate"], "")
            self.assertEqual(rows["ctl_f"]["gate"], "pass")                         # 'suspect' passes the gate layer
            self.assertEqual(rows["ctl_f"]["first_stop"], "rules")
            self.assertEqual(rows["ctl_d"]["category"], "no_parallax_rotation")     # controls are grouped by the failure they were chosen for
            self.assertEqual(rows["easy_a"]["category"], "drone_orbit")
            md = (out / "funnel.md").read_text()
            self.assertIn("**easy** (2 clips): first stopped by rules 0, by the VLM 1, by the stage-1 gate 0; passed all 3 layers 1", md)
            self.assertIn("**control_negative** (4 clips): first stopped by rules 2, by the VLM 0, by the stage-1 gate 1; passed all 3 layers 0; incomplete", md)
            self.assertIn("Stopped by exactly one layer: rules 1, VLM 0, gate 1.", md)

    def test_router_is_a_layer_and_v13_is_not_a_rejection_of_the_clip_by_the_filter(self):
        route = [("easy_a", "glr", ""), ("easy_b", "reject", "filter"), ("ctl_c", "reject", "filter"), ("ctl_d", "glr", ""), ("ctl_e", "v13", "vlm_driving"),
                 ("ctl_f", "reject", "filter")]
        with tempfile.TemporaryDirectory() as td:
            write(td)
            with open(Path(td) / "glr_route.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["clip", "route", "reasons"])
                w.writerows(route)
            out = Path(td) / "out"
            r = subprocess.run([sys.executable, str(SCRIPT), str(Path(td) / "decisions.csv"), str(Path(td) / "sweep_runs.csv"), str(out), "--manifest",
                                str(Path(td) / "manifest.json"), "--route", str(Path(td) / "glr_route.csv"), "--no-figure"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out / "funnel.csv") as fh:
                rows = {x["clip"]: x for x in csv.DictReader(fh)}
            self.assertEqual(rows["easy_a"]["first_stop"], "passed_all")
            self.assertEqual(rows["easy_b"]["first_stop"], "vlm")                       # the filter layers come first; the router only repeats their reject
            self.assertEqual(rows["easy_b"]["n_layers_stopping"], "2")                  # vlm + gate: the router's 'reject' only repeats the filter's decision
            self.assertEqual(rows["ctl_d"]["first_stop"], "gate")
            self.assertEqual(rows["ctl_d"]["n_layers_stopping"], "1")
            self.assertEqual(rows["ctl_e"]["first_stop"], "route")                      # sent to v1.3 (driving): known without any stage-1 result
            self.assertEqual(rows["ctl_e"]["route_reason"], "vlm_driving")
            self.assertEqual(rows["ctl_f"]["first_stop"], "rules")
            md = (out / "funnel.md").read_text()
            self.assertIn("**easy** (2 clips): first stopped by rules 0, by the VLM 1, sent to the v1.3 annotator by the router 0, by the stage-1 gate 0; passed all 4 layers 1", md)
            self.assertIn("**control_negative** (4 clips): first stopped by rules 2, by the VLM 0, sent to the v1.3 annotator by the router 1, by the stage-1 gate 1; passed all 4 layers 0.", md)
            self.assertIn("Stopped by exactly one layer: rules 1, VLM 0, router 1, gate 1.", md)
            self.assertEqual(rows["ctl_c"]["n_layers_stopping"], "2")                   # rules + gate (the router's reject is not a third layer)

    @unittest.skipUnless(HAS_MPL, "matplotlib not installed")
    def test_figure_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            write(td)
            out = Path(td) / "out"
            r = subprocess.run([sys.executable, str(SCRIPT), str(Path(td) / "decisions.csv"), str(Path(td) / "sweep_runs.csv"), str(out), "--manifest",
                                str(Path(td) / "manifest.json")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((out / "fig_funnel.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
