"""The GLR track copies numerical code from the v1.3 experiment scripts so that it runs without their hard-coded paths. Those scripts are
not importable (hard-coded roots, OpenEXR), so this test pulls the ORIGINAL functions out of their source files with `ast`, runs them in a
clean namespace and compares the copies with them on random inputs. A drift in either direction fails here.
Skipped item by item when an original is not in the checkout."""
import importlib.util
import ast
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import _fix  # noqa: F401  (puts pipeline/ on sys.path)
import glr_collect as GC
import glr_common as C

REPO = Path(__file__).resolve().parents[2]


def extract(rel, names, ns=None):
    """{name: function} of the top-level functions `names` of REPO/rel, executed in a namespace with numpy (+ `ns`); None if the file is missing."""
    path = REPO / rel
    if not path.exists():
        return None
    space = {"np": np, **(ns or {})}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), space)
    missing = set(names) - set(space)
    if missing:
        raise AssertionError(f"{rel}: functions {sorted(missing)} not found - was the original refactored? update this test")
    return space


def need(test, space, rel):
    if space is None:
        test.skipTest(f"{rel} is not in this checkout")
    return space


def traj(n, seed, noise=0.0):
    r = np.random.default_rng(seed)
    return np.cumsum(r.normal(size=(n, 3)) * np.array([1.0, 0.2, 0.6]), 0) + r.normal(size=(n, 3)) * noise


class Geometry(unittest.TestCase):
    def test_umeyama_and_traj_pct(self):
        ev = need(self, extract("vipe_eval/eval_vipe.py", {"umeyama"}), "vipe_eval/eval_vipe.py")
        qx = need(self, extract("qc/qc_cross.py", {"traj_pct", "depth_cmp"}, {"umeyama": ev["umeyama"]}), "qc/qc_cross.py")
        rng = np.random.default_rng(0)
        for sd in range(25):
            a = traj(90, sd)
            b = 2.7 * a @ Rot.random(random_state=sd).as_matrix().T + rng.normal(size=3) + rng.normal(size=a.shape) * 0.05
            for with_scale in (True, False):
                s1, R1, t1 = C.umeyama(a, b, with_scale)
                s2, R2, t2 = ev["umeyama"](a, b, with_scale)
                self.assertTrue(np.allclose([s1], [s2], atol=1e-12) and np.allclose(R1, R2, atol=1e-12) and np.allclose(t1, t2, atol=1e-12))
            self.assertAlmostEqual(C.traj_pct(a, b), qx["traj_pct"](a, b), places=10)
        a = traj(50, 1)
        self.assertTrue(np.isnan(C.traj_pct(a, a * 0)) and np.isnan(qx["traj_pct"](a, a * 0)))      # a degenerate (zero-length) path

    def test_depth_cmp(self):
        qx = need(self, extract("qc/qc_cross.py", {"depth_cmp", "traj_pct"}, {"umeyama": C.umeyama}), "qc/qc_cross.py")
        rng = np.random.default_rng(1)
        for sd in range(8):
            Da = rng.uniform(1, 10, (8, 64, 96)).astype(np.float32)
            Db = (Da * 1.3 * (1 + 0.1 * rng.normal(size=Da.shape))).astype(np.float32)
            Db[rng.random(Db.shape) < 0.2 * (sd % 3)] = np.nan
            st = rng.random(Da.shape) > 0.3 if sd % 2 else None
            a, b = C.depth_cmp(Da, Db, st), qx["depth_cmp"](Da, Db, st)
            self.assertTrue(np.allclose(a, b, equal_nan=True), (a, b))
        tiny = np.ones((1, 4, 4), np.float32)
        self.assertTrue(np.isnan(C.depth_cmp(tiny, tiny)[0]) and np.isnan(qx["depth_cmp"](tiny, tiny)[0]))   # < 1000 valid pixels

    def test_to_grid(self):
        import cv2
        eb = need(self, extract("bench/eval3d_bench.py", {"to_grid"}, {"cv2": cv2}), "bench/eval3d_bench.py")
        rng = np.random.default_rng(2)
        D = rng.uniform(1, 5, (3, 33, 47)).astype(np.float32)
        D[0, :5] = np.nan
        K = np.tile(np.array([[40.0, 0, 23], [0, 41, 16], [0, 0, 1]]), (3, 1, 1))
        for H, W in ((64, 96), (16, 20), (33, 47), (100, 31)):
            d1, k1 = C.to_grid(D, K, H, W)
            d2, k2 = eb["to_grid"](D, K, H, W)
            self.assertTrue(np.array_equal(d1, d2, equal_nan=True) and np.allclose(k1, k2), (H, W))

    def test_qc_grid_and_e9_flags(self):
        v13 = need(self, extract("pipeline/v13_label_qc.py", {"e9_flags", "qc_grid"}, {"LONG": 512}), "pipeline/v13_label_qc.py")
        e9 = need(self, extract("evolution/e9_hard_rules.py", {"flags"}), "evolution/e9_hard_rules.py")
        for h, w in ((1080, 1920), (1920, 1080), (480, 640), (360, 480), (721, 1281), (4000, 3000), (16, 9)):
            self.assertEqual(C.qc_grid(h, w, 512), v13["qc_grid"](h, w), (h, w))
        rng = np.random.default_rng(3)
        for turn in (0, 3, 25):
            for fps in (15, 7.5, 5, 3.75):
                n, r = 90, np.random.default_rng(turn)
                T = np.tile(np.eye(4), (n, 1, 1))
                p = np.cumsum(np.abs(r.normal(size=(n, 3))) * np.array([1, 0.1, 0.5]), 0)
                for i in range(n):
                    T[i, :3, :3] = Rot.from_euler("y", turn * i + r.normal() * 2, degrees=True).as_matrix()
                    T[i, :3, 3] = p[i]
                K = np.tile(np.array([[500.0, 0, 320], [0, 500, 180], [0, 0, 1]]), (n, 1, 1))
                D = rng.uniform(1, 10, (n, 8, 8)).astype(np.float32)
                a, b, c = GC.e9_flags(T, K, D, 640, fps), v13["e9_flags"](T, K, D, 640, fps), e9["flags"](T, K, D, 640, fps)
                for key in b:
                    same = lambda x, y: (np.isnan(x) and np.isnan(y)) or x == y  # noqa: E731
                    self.assertTrue(same(a[key], b[key]) and same(b[key], c[key]), (key, turn, fps, a[key], b[key], c[key]))


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas not installed")
class ScaleRule(unittest.TestCase):
    """glr_common.metric_scale vs evolution/e3_scale_vote.one() + votes() (column k_vipe_td = E3b)."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.pd = pd
        cls.ns = extract("evolution/e3_scale_vote.py", {"one", "votes"}, {"pd": pd, "umeyama": C.umeyama, "OTHERS": ["vipe", "da3", "pi3x"],
                                                                           "KIND": "A", "BASE": "pi3x_kgate"})

    def e3b(self, base, vipe):
        data = {"pi3x_kgate": (base[0], None, base[1], None), "vipe": (vipe[0], None, vipe[1], None),
                "da3": (vipe[0], None, vipe[1], None), "pi3x": (vipe[0], None, vipe[1], None)}
        ns = self.ns

        def loader(_clip):
            def load(m):
                if m not in data:
                    raise KeyError(m)
                return data[m]
            return load
        ns["loader"] = loader
        df = ns["votes"](self.pd.DataFrame([ns["one"]({"clip": "x", "dataset": "x"})]))
        return df.iloc[0]

    def test_identical_to_e3b_wherever_e3b_has_evidence(self):
        if self.ns is None:
            self.skipTest("evolution/e3_scale_vote.py is not in this checkout")
        warnings.filterwarnings("ignore")
        rng = np.random.default_rng(42)
        kinds = {"both": 0, "no_depth_ratio": 0}
        for t in range(150):
            n, h, w = int(rng.integers(20, 90)), 36, 64
            step = float(10 ** rng.uniform(-3.5, 0.0))                       # path from tiny to long relative to the scene depth
            P = np.cumsum(rng.normal(size=(n, 3)) * step + np.array([step, 0, 0]), 0)
            Tv = np.tile(np.eye(4), (n, 1, 1))
            Tv[:, :3, 3] = P
            Dv = (rng.uniform(2, 12) * (1 + 0.2 * rng.random((n, h, w)))).astype(np.float32)
            s = float(10 ** rng.uniform(-1.5, 1.5))
            Pg = (P @ Rot.random(random_state=int(rng.integers(1 << 30))).as_matrix().T) * s + rng.normal(size=3) \
                + rng.normal(size=P.shape) * float(rng.choice([0.0, 0.01, 0.1, 0.5])) * step * 5 * s
            Tg = np.tile(np.eye(4), (n, 1, 1))
            Tg[:, :3, 3] = Pg
            Dg = (Dv * s * float(rng.uniform(0.7, 1.4))).astype(np.float32)
            Dg[rng.random(Dg.shape) < float(rng.choice([0.0, 0.0, 0.5, 0.999, 1.0]))] = np.nan
            ref = self.e3b((Tg, Dg), (Tv, Dv))
            mine = C.metric_scale(Tg, Dg, Tv, Dv, 10)
            if np.isfinite(ref["r_vipe"]):                               # E3b has a depth ratio: the rules agree exactly
                kinds["both"] += 1
                self.assertAlmostEqual(ref["k_vipe_td"], mine["k"], delta=1e-9 * max(1, abs(mine["k"])), msg=f"case {t}")
                self.assertAlmostEqual(ref["k_vipe_t"], mine["k_traj"], delta=1e-9 * max(1, abs(mine["k_traj"])), msg=f"case {t}")
                self.assertAlmostEqual(ref["k_vipe_d"], mine["k_depth"], delta=1e-9 * max(1, abs(mine["k_depth"])), msg=f"case {t}")
            else:                                                        # E3b has no ratio, hence no scene depth and no trajectory vote: k = 1.0.
                kinds["no_depth_ratio"] += 1                             # Here the reference depth alone still gives the scene depth.
                self.assertEqual(ref["k_vipe_td"], 1.0)
                want = mine["s_vipe"] if np.isfinite(mine["s_vipe"]) else 1.0
                self.assertAlmostEqual(mine["k"], want, delta=1e-9 * max(1, abs(want)), msg=f"case {t}")
        self.assertGreater(kinds["both"], 40)
        self.assertGreater(kinds["no_depth_ratio"], 10)


class ScaleRuleDeviations(unittest.TestCase):
    """The documented differences of metric_scale from E3b - where the GLRMap label differs from a Pi3X label."""

    @staticmethod
    def scene(n=60, s=0.37, step=0.3):
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, 0, 3] = step * np.arange(n)
        T[:, 2, 3] = 0.005 * (step * np.arange(n)) ** 2
        Dv = np.full((n, 36, 64), 6.0, np.float32) * (1 + 0.1 * np.random.default_rng(0).random((n, 36, 64))).astype(np.float32)
        Tg = T.copy()
        Tg[:, :3, 3] *= s
        return T, Dv, Tg, Dv * s

    def test_depth_on_few_frames_still_gives_the_depth_ratio(self):
        T, Dv, Tg, Dg = self.scene()
        keep = np.zeros(len(Dg), bool)
        keep[3:14] = True                                    # GLR depth only on 11 of 60 frames
        Dg = np.where(keep[:, None, None], Dg, np.nan).astype(np.float32)
        m = C.metric_scale(Tg, Dg, T, Dv, 10)
        self.assertAlmostEqual(m["r_vipe"], 1 / 0.37, delta=0.05)
        self.assertAlmostEqual(m["k"], 1 / 0.37, delta=0.05)

    def test_missing_depth_uses_the_trajectory_vote_alone_and_no_evidence_is_nan(self):
        T, Dv, Tg, _ = self.scene()
        m = C.metric_scale(Tg, None, T, Dv, 10)
        self.assertAlmostEqual(m["k"], 1 / 0.37, places=4)
        self.assertAlmostEqual(m["k_depth"], m["k_traj"], places=9)
        none = C.metric_scale(Tg, None, T, None, 10, neutral=float("nan"))      # no depth anywhere: the 2 % rule cannot be evaluated
        self.assertTrue(np.isnan(none["k"]))
        self.assertEqual(C.metric_scale(Tg, None, T, None, 10)["k"], 1.0)        # E3b's neutral value is still the default


if __name__ == "__main__":
    unittest.main()
