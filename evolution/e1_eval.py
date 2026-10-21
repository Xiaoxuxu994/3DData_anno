"""E1 evaluation: conditioned Pi3X / DA3 variants, alone and in hybrids, with the line-H machinery (run_hybrid.py).

  python e1_eval.py A|B|scannetpp|wild [--workers=N]  -> outputs/evolution/E1/raw/<kind>.csv (+ per-clip cache)
Same metrics, alignment rules and failure flags as outputs/hybrid (so numbers are directly comparable with v0).
"""

import sys
from pathlib import Path

sys.path.insert(0, "/data/home/z00533370/code/hybrid")
import run_hybrid as RH  # noqa: E402  (parses sys.argv[1] = kind at import)

KIND = RH.KIND
SINGLES = ["vipe", "pi3x", "da3", "pi3x_kvipe", "pi3x_kgeo", "pi3x_pkvipe", "da3_pkvipe"]
if KIND != "wild":
    SINGLES.append("pi3x_kgt")
HYB = ["H:vipe+pi3x+med",  # v0 (current pipeline)
       "H:vipe+pi3x_kvipe+med", "H:vipe+pi3x_kvipe+geo", "H:vipe+pi3x_kvipe+raw",
       "H:vipe+pi3x_kgeo+med",
       "H:vipe+pi3x_pkvipe+med", "H:vipe+pi3x_pkvipe+raw",
       "H:vipe+da3_pkvipe+raw", "H:vipe+da3_pkvipe+med"]
if KIND != "wild":
    HYB.append("H:vipe+pi3x_kgt+med")  # oracle K for the depth network only
RH.SPECS = SINGLES + HYB
RH.RES = Path("/data/home/z00533370/outputs/evolution/E1/raw")

if __name__ == "__main__":
    RH.main()
