import unittest

import _fix  # noqa: F401
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


if __name__ == "__main__":
    unittest.main()
