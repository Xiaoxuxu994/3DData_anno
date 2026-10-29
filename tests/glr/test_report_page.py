"""build_report_page.py: the Markdown version keeps tables / lists / figure links, and the page builds from a minimal result set (missing parts say so, nothing crashes)."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evolution" / "E15_glr" / "build_report_page.py"


def load():
    spec = importlib.util.spec_from_file_location("build_report_page", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HtmlToMd(unittest.TestCase):
    def test_tables_lists_figures_and_cjk_bold(self):
        m = load()
        frag = ('<title>t</title><style>x{}</style><h2>结论<small>sub</small></h2><p>前 <b>读结果前：</b>沙箱 <code>a|b</code></p><ul><li>one <a href="http://x">link</a></li></ul>'
                '<ol><li>first</li><li>second</li></ol><div class="tw"><table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1|2</td><td><span class="b"><i>✓</i>ok</span></td></tr>'
                '</tbody></table></div><figure><img src="data:x" alt="图" data-file="f.png"><figcaption>cap</figcaption></figure><div class="gal"><div class="card"><div class="ph"><img src="data:y" alt="skipped"></div>skip me</div></div><h2>after the gallery</h2><p>still here</p>')
        md = m.html_to_md(frag, "figures")
        self.assertIn("## 结论 _(sub)_", md)
        self.assertIn("**读结果前：** 沙箱", md)                      # a space after the closing mark: CommonMark would not close it before a CJK character
        self.assertIn("`a|b`", md)
        self.assertIn("- one [link](http://x)", md)
        self.assertIn("1. first", md)
        self.assertIn("2. second", md)
        self.assertIn("| A | B |\n|---|---|\n| 1\\|2 | ✓ok |", md)
        self.assertIn("![图](figures/f.png)", md)
        self.assertNotIn("skip me", md)
        self.assertIn("## after the gallery", md)                     # a void <img> inside a skipped block must not swallow what follows
        self.assertIn("still here", md)
        self.assertNotIn("<", md.replace("<", "", 0).split("![图]")[0].replace("&lt;", ""))   # no HTML left in the text part


class BuildMinimal(unittest.TestCase):
    def test_builds_without_figures_funnel_demo(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(td / "sweep.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["clip", "run", "n_img", "n_reg", "reg_frac", "n_points", "verdict", "flags", "focal_w", "role", "outcome", "silver_accurate", "focal_err_pct", "ate_vs_ref_pct"])
                w.writerow(["a", "v0a", 60, 60, 1.0, 5000, "ok", "", 0.8, "easy", "true_pass", "True", 1.0, 0.2])
                w.writerow(["b", "v0a", 60, 0, 0.0, "", "stop", "no_model", "", "control_negative", "", "", "", ""])
            man = {"clips": [{"id": "a", "role": "easy", "category": "drone_orbit", "license": "CC0 1.0", "source": "archive.org"},
                             {"id": "b", "role": "control_negative", "expected_failure": "static_camera", "license": "Mixkit Stock Video Free License", "source": "mixkit"}]}
            (td / "man.json").write_text(json.dumps(man))
            r = subprocess.run([sys.executable, str(SCRIPT), str(td / "p.html"), "--sweep", str(td / "sweep.csv"), "--manifest", str(td / "man.json"), "--md", str(td / "p.md"),
                                "--fragment", str(td / "f.html")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            page = (td / "p.html").read_text()
            self.assertTrue(page.startswith("<!doctype html>"))
            self.assertIn("<title>", (td / "f.html").read_text()[:200])
            self.assertNotIn("<!doctype", (td / "f.html").read_text())      # the Artifact wrapper adds its own skeleton
            self.assertIn("还没有生成", page)                                # missing figures are announced, not hidden
            self.assertIn("## 结论", (td / "p.md").read_text())


if __name__ == "__main__":
    unittest.main()
