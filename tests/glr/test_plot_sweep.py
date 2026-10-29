"""plot_sweep.py draws the sweep CSV; these tests only check that it runs on the CSV layout summarize_sweep.py writes, names the failing easy clips,
and does not crash when a run or the reference columns are missing (the figures themselves were looked at, not asserted)."""
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evolution" / "E15_glr" / "plot_sweep.py"
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
FIELDS = ["clip", "run", "n_img", "n_reg", "reg_frac", "n_points", "pts_img", "reproj", "focal_w", "fx_fy", "hfov", "planar", "tri50", "step_ratio",
          "t_total_s", "verdict", "flags", "focal_err_pct", "ate_vs_ref_pct", "n_common", "silver_accurate", "outcome", "role"]


def row(clip, run, verdict, role="", reg=1.0, pts=120.0, fxfy=1.0, planar=0.1, outcome="", ferr=None, ate=None, model=True):
    return {"clip": clip, "run": run, "n_img": 60, "n_reg": 60 * reg if model else 0, "reg_frac": reg if model else 0.0, "n_points": 7000 if model else "",
            "pts_img": pts if model else "", "reproj": 0.4 if model else "", "focal_w": 0.9 if model else "", "fx_fy": fxfy if model else "", "hfov": 60 if model else "",
            "planar": planar, "tri50": 4.0 if model else "", "step_ratio": 1.1 if model else "", "t_total_s": 50, "verdict": verdict, "flags": "",
            "focal_err_pct": "" if ferr is None else ferr, "ate_vs_ref_pct": "" if ate is None else ate, "n_common": 20, "silver_accurate": "",
            "outcome": outcome, "role": role}


class PlotSweep(unittest.TestCase):
    @unittest.skipUnless(HAS_MPL, "matplotlib not installed")
    def test_runs_on_the_summary_layout_and_writes_the_three_figures(self):
        rows = [row("mk_a", "v0a", "ok", "easy", ferr=2.0, ate=0.5, outcome="true_pass"),
                row("mk_b", "v0a", "stop", "easy", fxfy=1.3, ferr=3.0, ate=1.0, outcome="false_stop"),
                row("mk_c", "v0a", "ok", "easy", ferr=40.0, ate=9.0, outcome="silent_error"),
                row("mk_d", "v0a", "stop", "easy", model=False, outcome="true_stop"),
                row("gov_grca_x", "v1sp", "suspect", "easy", ferr=1.0, ate=0.2, outcome="true_pass"),
                row("ctl_pan_1", "v0a", "stop", "control_negative", planar=1.0, pts=3.0),
                row("ctl_static_2", "v0a", "stop", "", model=False)]                # no role column value: inferred from the 'ctl_' prefix
        with tempfile.TemporaryDirectory() as td:
            csv_path, out = Path(td) / "sweep_runs.csv", Path(td) / "figs"
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            r = subprocess.run([sys.executable, str(SCRIPT), str(csv_path), str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("fig_gate_signals.png", "fig_variants_vs_reference.png", "fig_outcomes.png"):
                png = (out / name).read_bytes()
                self.assertTrue(png.startswith(b"\x89PNG"), name)
                self.assertGreater(len(png), 5000, name)

    @unittest.skipUnless(HAS_MPL, "matplotlib not installed")
    def test_without_reference_columns_only_the_signal_figure_is_written(self):
        rows = [row("mk_a", "v0a", "ok", "easy"), row("ctl_pan_1", "v0a", "stop", "control_negative", planar=1.0)]
        with tempfile.TemporaryDirectory() as td:
            csv_path, out = Path(td) / "sweep_runs.csv", Path(td) / "figs"
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            r = subprocess.run([sys.executable, str(SCRIPT), str(csv_path), str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((out / "fig_gate_signals.png").exists())
            self.assertFalse((out / "fig_variants_vs_reference.png").exists())
            self.assertFalse((out / "fig_outcomes.png").exists())
            self.assertIn("skipped", r.stdout)

    def test_failing_side_of_each_rule_shape(self):
        spec = importlib.util.spec_from_file_location("plot_sweep_mod", SCRIPT)
        if not HAS_MPL:
            self.skipTest("matplotlib not installed")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(mod._fails(0.2, [0.5], "below"))
        self.assertFalse(mod._fails(0.6, [0.5], "below"))
        self.assertTrue(mod._fails(0.2, [0.1], "above"))
        self.assertFalse(mod._fails(0.05, [0.1], "above"))
        self.assertTrue(mod._fails(0.2, [0.3, 4.0], "outside"))
        self.assertTrue(mod._fails(5.0, [0.3, 4.0], "outside"))
        self.assertFalse(mod._fails(1.0, [0.3, 4.0], "outside"))


if __name__ == "__main__":
    unittest.main()
