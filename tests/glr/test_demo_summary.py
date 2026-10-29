"""demo_summary.py: joins route, worker result, tier and final decision; missing pieces stay empty."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import importlib.util

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("demo_summary", REPO / "evolution" / "E15_glr" / "demo_summary.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class DemoSummary(unittest.TestCase):
    def test_rows(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_csv(out / "filter" / "glr_route.csv", [{"clip": "a", "route": "glr", "reasons": ""}, {"clip": "b", "route": "reject", "reasons": "filter"},
                                                         {"clip": "c", "route": "glr", "reasons": ""}])
            res = out / "glr_jobs" / "a" / "result"
            res.mkdir(parents=True)
            (res / "result.json").write_text(json.dumps({"gate": "ok", "status": "complete"}))
            (out / "glr_jobs" / "a" / "RESULT_DIR").write_text("result\n")
            write_csv(out / "results_glr.csv", [{"clip": "a", "tier": "T2", "reasons": ""}])
            write_csv(out / "final" / "results.csv", [{"clip": "a", "accept": "True", "annotator": "glr", "tier": "T2", "reason": ""},
                                                      {"clip": "b", "accept": "False", "annotator": "", "tier": "T3", "reason": "v13:none"}])
            rows = {r["clip"]: r for r in M.summarize(out)}
            self.assertEqual((rows["a"]["gate"], rows["a"]["status"], rows["a"]["tier"], rows["a"]["final"]), ("ok", "complete", "T2", "glr (T2)"))
            self.assertEqual((rows["b"]["route"], rows["b"]["final"], rows["b"]["why"]), ("reject (filter)", "拒绝", "v13:none"))
            self.assertEqual((rows["c"]["gate"], rows["c"]["final"]), ("", ""))             # a clip without any result: nothing invented


if __name__ == "__main__":
    unittest.main()
