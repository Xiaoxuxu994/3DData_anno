"""demo_precheck.py POOL_DIR S1_ROOT CLIP [CLIP ...] - inputs the E15 plumbing demo (run_demo.sh) needs, checked before its slow steps start.

Fatal (non-zero exit, the list is printed): a clip without GeoCalib scores (filter stage s5_geocalib: the K prior of the jobs and the K witness of the stand-in) or without a
silver reference (S1_ROOT/<clip>/ref/stage1.json from run_sweep.sh: the stand-in witnesses are built from it) or without a registered reference model.
Warning only: no optical-flow scores (filter stage s4_flow): the router's rules cannot fire and the frame step falls back to its default.
"""
import json
import sys
from pathlib import Path


def check(pool, s1, clips):
    """-> (missing, warnings): lists of strings"""
    pool, s1 = Path(pool), Path(s1)
    missing, warn = [], []
    for c in clips:
        sc = pool / c / "scores.json"
        s = json.loads(sc.read_text()) if sc.exists() else {}
        if "calib" not in s:
            missing.append(f"{c}: no GeoCalib scores (filter stage s5_geocalib)")
        ref = s1 / c / "ref" / "stage1.json"
        if not ref.exists():
            missing.append(f"{c}: no silver reference ({ref.parent}; evolution/E15_glr/run_sweep.sh)")
        elif not (json.loads(ref.read_text()).get("model") or {}).get("n_reg"):
            missing.append(f"{c}: the silver reference registered no images")
        if "flow" not in s:
            warn.append(f"{c}: no optical-flow scores (filter stage s4_flow): the router's rules cannot fire and the frame step uses its default")
    return missing, warn


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    missing, warn = check(sys.argv[1], sys.argv[2], sys.argv[3:])
    for w in warn:
        print("warning:", w, file=sys.stderr)
    if missing:
        sys.exit("missing inputs:\n  " + "\n  ".join(missing))
