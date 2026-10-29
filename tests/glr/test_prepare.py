import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _fix
import glr_common as C
import glr_prepare as P

CFG = C.load_cfg(None)
GOOD = {"flow": {"dyn_frac": 0.05, "rot_frac": 0.2, "parallax_px": 2.0, "static_pair_frac": 0.0, "mag_median_frac": 0.04},
        "rules": {"lum_mean": 100}, "vlm_qwen3vl_32b": {"scene_category": "indoor_static", "dynamic_level": "none",
                                                          "camera_motion": "mostly_translation", "reconstructable": "yes"}}


class Route(unittest.TestCase):
    def test_good_clip_goes_to_glr(self):
        r, why, f = P.route_clip(GOOD, CFG, True)
        self.assertEqual((r, why), ("glr", []))
        self.assertEqual(f["vlm_scene"], "indoor_static")

    def test_filter_reject_stays_reject(self):
        self.assertEqual(P.route_clip(GOOD, CFG, False)[0], "reject")

    def test_reasons_send_clip_to_v13(self):
        for patch, reason in (({"flow": {"rot_frac": 0.9}}, "rotation_dominant"), ({"flow": {"dyn_frac": 0.5}}, "dynamic"),
                              ({"flow": {"parallax_px": 0.1}}, "low_parallax"), ({"flow": {"static_pair_frac": 0.4}}, "slow_or_static_segments"),
                              ({"flow": {"mag_median_frac": 0.0005}}, "too_slow"),
                              ({"vlm_qwen3vl_32b": {"scene_category": "driving"}}, "vlm_driving"),
                              ({"vlm_qwen3vl_32b": {"dynamic_level": "high"}}, "vlm_dynamic")):
            s = json.loads(json.dumps(GOOD))
            for k, v in patch.items():
                s[k].update(v)
            r, why, _ = P.route_clip(s, CFG, True)
            self.assertEqual(r, "v13", reason)
            self.assertIn(reason, why)

    def test_missing_stages_never_fire(self):
        self.assertEqual(P.route_clip({}, CFG, True)[0], "glr")

    def test_force(self):
        s = json.loads(json.dumps(GOOD))
        s["flow"]["rot_frac"] = 0.9
        self.assertEqual(P.route_clip(s, CFG, True, force="glr")[0], "glr")


class Stride(unittest.TestCase):
    def test_default_without_flow(self):
        self.assertEqual(P.choose_stride(180, None, CFG), 2)

    def test_slow_motion_takes_the_larger_stride(self):
        self.assertEqual(P.choose_stride(180, 0.006, CFG), 3)  # 0.2 % of the width per frame

    def test_normal_motion(self):
        self.assertEqual(P.choose_stride(180, 0.03, CFG), 2)  # 1 % per frame

    def test_fast_motion_uses_every_frame(self):
        self.assertEqual(P.choose_stride(180, 0.09, CFG), 1)  # 3 % per frame: stride 2 would move 6 % per step

    def test_frame_count_window(self):
        for n in (120, 180, 240):
            for mag in (None, 0.003, 0.03, 0.06):
                s = P.choose_stride(n, mag, CFG)
                self.assertGreaterEqual(s, 1)
                self.assertLessEqual(s, CFG["sampling"]["max_stride"])


class Files(unittest.TestCase):
    def test_route_and_jobs_cli(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            clips = []
            for name, scores, ok in (("a", GOOD, "pass"), ("b", {**GOOD, "flow": {**GOOD["flow"], "rot_frac": 0.95}}, "pass"), ("c", GOOD, "reject")):
                fd = out / "bench" / "frames" / name
                fd.mkdir(parents=True)
                for i in range(180):
                    cv2.imwrite(str(fd / f"{i:06d}.jpg"), np.full((27, 48, 3), i % 255, np.uint8))
                (out / "pool" / name).mkdir(parents=True)
                (out / "pool" / name / "scores.json").write_text(json.dumps(scores))
                clips.append({"clip": name, "n_frames": 180, "hw": [27, 48], "fps_out": 15})
            (out / "bench" / "clips_all.json").write_text(json.dumps(clips))
            (out / "bench" / "clips.json").write_text(json.dumps(clips[:2]))
            (out / "filter").mkdir()
            with open(out / "filter" / "decisions.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["clip", "decision", "reason"])
                w.writerows([["a", "pass", ""], ["b", "pass", ""], ["c", "reject", "blur"]])
            run = lambda *a: subprocess.run([sys.executable, str(_fix.PIPE / "glr_prepare.py"), str(out), *a], capture_output=True, text=True)  # noqa: E731
            r = run("route")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), {"reject": 1, "glr": 1, "v13": 1})
            with open(out / "filter" / "glr_route.csv") as fh:
                rows = {x["clip"]: x for x in csv.DictReader(fh)}
            self.assertEqual((rows["a"]["route"], rows["b"]["route"], rows["c"]["route"]), ("glr", "v13", "reject"))
            self.assertIn("rotation_dominant", rows["b"]["reasons"])
            r = run("jobs")
            self.assertEqual(r.returncode, 0, r.stderr)
            job = json.loads((out / "glr_jobs" / "a" / "job.json").read_text())
            self.assertFalse((out / "glr_jobs" / "b").exists())
            self.assertEqual(job["stride"], 2)
            self.assertEqual(len(job["frame_idx"]), 90)
            self.assertEqual(job["image_names"][0], "000001.jpg")
            self.assertEqual(len(list((out / "glr_jobs" / "a" / "images").glob("*.jpg"))), 90)
            # the k-th job image is the frame_idx[k]-th clip frame
            im = cv2.imread(str(out / "glr_jobs" / "a" / "images" / "000010.jpg"))
            self.assertEqual(int(im[0, 0, 0]), job["frame_idx"][9] % 255)
            r2 = run("jobs")
            self.assertIn("exists, skipped", r2.stdout)  # never overwritten


if __name__ == "__main__":
    unittest.main()
